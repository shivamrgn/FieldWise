"""
Offline Grounded Knowledge Base Retriever (`kb/retriever.py`).

Queries `ps1_offline_assistant/data/agri_kb.json` completely offline with
zero external dependencies beyond the Python standard library and internal
PS1 schemas. Supports exact and normalized PlantVillage / IP102 class lookup
while safely returning `None` for unknown or healthy conditions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ps1_offline_assistant.schemas.contracts import (
    DiseasePestCard,
    VisionCategory,
    VisionFinding,
)


DEFAULT_KB_PATH = Path(__file__).resolve().parents[1] / "data" / "agri_kb.json"

_CROP_SYNONYMS: Dict[str, str] = {
    "rice": "rice",
    "paddy": "rice",
    "rice paddy": "rice",
    "oryza sativa": "rice",
    "wheat": "wheat",
    "triticum aestivum": "wheat",
    "cotton": "cotton",
    "gossypium hirsutum": "cotton",
    "potato": "potato",
    "solanum tuberosum": "potato",
    "tomato": "tomato",
    "solanum lycopersicum": "tomato",
    "maize": "maize",
    "corn": "maize",
    "corn maize": "maize",
    "maize corn": "maize",
    "zea mays": "maize",
    "jowar": "jowar",
    "sorghum": "jowar",
    "jowar sorghum": "jowar",
}

_NON_CONDITION_TOKENS = {
    "",
    "none",
    "unknown",
    "uncertain",
    "healthy",
    "unclassified",
    "other",
    "background",
}


def normalize_key(text: Optional[str]) -> str:
    """Normalizes PlantVillage/IP102 class names, hyphens, and punctuation."""
    if not text:
        return ""
    cleaned = text.strip().lower()
    cleaned = cleaned.replace("___", " ").replace("__", " ")
    cleaned = re.sub(r"[_\-/(),.:]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_crop_name(crop: Optional[str]) -> Optional[str]:
    """Maps crop variants (e.g. 'Corn_(maize)', 'Paddy') to canonical KB crop ids."""
    norm = normalize_key(crop)
    if not norm or norm in {"unknown", "none", "na", "n a"}:
        return None
    return _CROP_SYNONYMS.get(norm, norm)


class OfflineKBRetriever:
    """In-memory offline index over `agri_kb.json`."""

    def __init__(self, kb_path: Path = DEFAULT_KB_PATH) -> None:
        self.kb_path = Path(kb_path)
        if not self.kb_path.is_file():
            raise FileNotFoundError(f"Offline KB file not found: {self.kb_path}")

        raw_data = json.loads(self.kb_path.read_text(encoding="utf-8"))
        self.raw_data: Dict[str, Any] = raw_data
        self.kb_schema_version: str = raw_data.get("kb_schema_version", "unknown")

        self._crops_by_id: Dict[str, Dict[str, Any]] = {}
        for crop_entry in raw_data.get("crops", []):
            cid = normalize_key(crop_entry.get("id"))
            if cid:
                self._crops_by_id[cid] = crop_entry

        self._fertilizers_by_crop: Dict[str, List[Dict[str, Any]]] = {
            normalize_key(k): v for k, v in raw_data.get("fertilizers", {}).items()
        }

        self._cards: List[DiseasePestCard] = []
        self._cards_by_id: Dict[str, DiseasePestCard] = {}
        self._index_crop_cond: Dict[Tuple[str, str], DiseasePestCard] = {}
        self._index_cond_only: Dict[str, List[DiseasePestCard]] = {}

        for raw_card in raw_data.get("disease_pest_cards", []):
            card = DiseasePestCard(**raw_card)
            self._cards.append(card)
            self._cards_by_id[card.id.lower()] = card

            crop_key = normalize_crop_name(card.crop) or card.crop.lower()
            candidate_keys = {
                normalize_key(card.id),
                normalize_key(card.condition),
                *(normalize_key(alias) for alias in card.aliases),
            }
            for cond_key in candidate_keys:
                if not cond_key:
                    continue
                self._index_crop_cond[(crop_key, cond_key)] = card
                self._index_cond_only.setdefault(cond_key, [])
                if card not in self._index_cond_only[cond_key]:
                    self._index_cond_only[cond_key].append(card)

    def list_disease_pest_cards(self) -> List[DiseasePestCard]:
        """Returns all grounded disease/pest cards loaded in the KB."""
        return list(self._cards)

    def get_crop_profile(self, crop: Optional[str]) -> Optional[Dict[str, Any]]:
        """Retrieves baseline FieldWise crop profile if present."""
        crop_key = normalize_crop_name(crop)
        if not crop_key:
            return None
        return self._crops_by_id.get(crop_key)

    def get_fertilizer_schedule(self, crop: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """Retrieves baseline FieldWise fertilizer schedule if present."""
        crop_key = normalize_crop_name(crop)
        if not crop_key:
            return None
        return self._fertilizers_by_crop.get(crop_key)

    def get_disease_pest_card(
        self,
        condition_name: str,
        crop: Optional[str] = None,
    ) -> Optional[DiseasePestCard]:
        """
        Looks up a grounded treatment card by condition name (and optional crop).

        Returns `None` safely for healthy states, unknown labels, crop mismatches,
        or unmapped classes so callers can trigger uncertainty handling.
        """
        raw_cond = (condition_name or "").strip().lower()
        norm_cond = normalize_key(condition_name)

        if (
            not norm_cond
            or norm_cond in _NON_CONDITION_TOKENS
            or norm_cond.endswith(" healthy")
        ):
            return None

        crop_key = normalize_crop_name(crop)

        # 1. Exact canonical card ID lookup
        if raw_cond in self._cards_by_id:
            card = self._cards_by_id[raw_cond]
            if crop_key is None or normalize_crop_name(card.crop) == crop_key:
                return card
            return None

        # 2. Crop-scoped exact normalized match (condition or PlantVillage/IP102 alias)
        if crop_key is not None:
            card = self._index_crop_cond.get((crop_key, norm_cond))
            if card is not None:
                return card
            # Handle PlantVillage pattern where crop prefix is duplicated inside condition_name
            if norm_cond.startswith(f"{crop_key} "):
                stripped = norm_cond[len(crop_key) :].strip()
                card = self._index_crop_cond.get((crop_key, stripped))
                if card is not None:
                    return card
            return None

        # 3. Crop omitted / unknown: resolve only by exact condition/alias key
        matches = self._index_cond_only.get(norm_cond, [])
        if matches:
            return matches[0]

        return None

    def retrieve_for_finding(self, finding: VisionFinding) -> Optional[DiseasePestCard]:
        """Retrieves the matching card for a validated VisionFinding, or None if healthy/unknown."""
        if finding.category in (VisionCategory.HEALTHY, VisionCategory.UNKNOWN):
            return None
        return self.get_disease_pest_card(
            condition_name=finding.condition_name,
            crop=finding.crop,
        )
