"""
One-Time Setup Downloader for Open-Source Pretrained PlantVillage Model.

Downloads an open-source pretrained PlantVillage MobileNetV2 / ViT image
classification model from Hugging Face into
`ps1_offline_assistant/vision/artifacts/pretrained_hf/` while internet is
available, and compiles a deterministic `kb_label_mapping.json` mapping its
classes onto `ps1_offline_assistant/data/agri_kb.json` (`disease_pest_cards`
and `healthy`).

Runtime inference (`HuggingFaceLocalVisionBackend`) strictly loads from this
local cache with `local_files_only=True` and makes ZERO network calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.kb.retriever import (  # noqa: E402
    DEFAULT_KB_PATH,
    OfflineKBRetriever,
    normalize_crop_name,
    normalize_key,
)

DEFAULT_PRETRAINED_REPO_ID = (
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)
DEFAULT_PRETRAINED_DIR = _PS1_ROOT / "vision" / "artifacts" / "pretrained_hf"


class MissingPretrainedDownloadDepsError(RuntimeError):
    """Raised when `transformers` or `huggingface_hub` is missing for one-time download."""


_KNOWN_CROP_PREFIXES: Tuple[str, ...] = (
    "corn (maize)",
    "corn maize",
    "bell pepper",
    "cedar apple",
    "tomato",
    "potato",
    "apple",
    "cherry",
    "peach",
    "grape",
    "orange",
    "squash",
    "strawberry",
    "blueberry",
    "raspberry",
    "soybean",
    "maize",
    "corn",
    "rice",
    "paddy",
    "wheat",
    "cotton",
    "jowar",
    "sorghum",
)

_PEST_KEYWORDS: Tuple[str, ...] = (
    "spider mite",
    "mites",
    "mite",
    "aphid",
    "bollworm",
    "whitefly",
    "planthopper",
    "armyworm",
    "thrips",
    "borer",
    "beetle",
    "weevil",
    "midge",
)


def _clean_crop_token(crop_raw: str) -> str:
    cleaned = re.sub(r"\bplant\b", "", crop_raw or "", flags=re.IGNORECASE).strip(" _-,()")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return normalize_crop_name(cleaned) or normalize_key(cleaned) or "general"


def _parse_hf_crop_and_condition(raw_label: str) -> Tuple[str, str, bool]:
    """
    Parses PlantVillage / Hugging Face class labels such as:
      - `Tomato___Early_blight`
      - `Corn_(maize)___Northern_Leaf_Blight`
      - `Tomato with Early Blight`
      - `Corn (Maize) with Northern Leaf Blight`
      - `Apple Scab` / `Cedar Apple Rust` / `Tomato Mosaic Virus`
      - `Healthy Tomato Plant` / `Tomato___healthy`
    into `(crop_token, condition_token, is_healthy)`.
    """
    text = str(raw_label or "").strip()
    lower = text.lower()
    is_healthy = "healthy" in lower

    if "___" in text:
        crop_part, cond_part = text.split("___", 1)
    elif " with " in lower:
        idx = lower.find(" with ")
        crop_part = text[:idx]
        cond_part = text[idx + 6 :]
    elif is_healthy:
        cleaned = re.sub(r"\bhealthy\b", "", text, flags=re.IGNORECASE)
        crop_part = cleaned
        cond_part = "Healthy"
    else:
        crop_part = "general"
        cond_part = text
        for prefix in _KNOWN_CROP_PREFIXES:
            if lower.startswith(prefix + " "):
                crop_part = "apple" if prefix == "cedar apple" else text[: len(prefix)]
                cond_part = text if prefix == "cedar apple" else text[len(prefix) :].strip()
                break

    crop_clean = _clean_crop_token(crop_part)
    cond_clean = re.sub(r"[_\-]+", " ", cond_part).strip()
    cond_clean = re.sub(r"\s+", " ", cond_clean)
    if is_healthy:
        cond_clean = "Healthy"

    return crop_clean, cond_clean, is_healthy


def map_hf_label_to_kb_entry(
    index: int,
    raw_label: str,
    retriever: OfflineKBRetriever,
) -> Dict[str, Any]:
    """Maps a single Hugging Face PlantVillage label to our KB schema structure."""
    crop_norm, cond_clean, is_healthy = _parse_hf_crop_and_condition(raw_label)

    if is_healthy:
        return {
            "index": int(index),
            "label": normalize_key(raw_label).replace(" ", "_") or f"healthy_{crop_norm}",
            "hf_label": raw_label,
            "kb_card_id": None,
            "crop": crop_norm,
            "condition_name": "Healthy",
            "category": "healthy",
        }

    # Try exact normalized lookup with crop hint, then raw label without hint
    card = retriever.get_disease_pest_card(cond_clean, crop=crop_norm)
    if card is None:
        card = retriever.get_disease_pest_card(raw_label, crop=crop_norm)
    if card is None and crop_norm == "general":
        card = retriever.get_disease_pest_card(cond_clean, crop=None)

    if card is not None:
        return {
            "index": int(index),
            "label": card.id,
            "hf_label": raw_label,
            "kb_card_id": card.id,
            "crop": card.crop,
            "condition_name": card.condition,
            "category": card.type.value,
        }

    # Unmapped PlantVillage crop/condition: preserve clean crop, condition, and pest vs disease category
    lower_cond = f"{raw_label} {cond_clean}".lower()
    inferred_category = "pest" if any(kw in lower_cond for kw in _PEST_KEYWORDS) else "disease"
    return {
        "index": int(index),
        "label": normalize_key(raw_label).replace(" ", "_"),
        "hf_label": raw_label,
        "kb_card_id": None,
        "crop": crop_norm,
        "condition_name": cond_clean or raw_label,
        "category": inferred_category,
    }


def build_hf_to_kb_mapping(
    id2label: Dict[Union[int, str], str],
    kb_path: Path = DEFAULT_KB_PATH,
) -> Dict[str, Any]:
    """
    Maps a full Hugging Face `id2label` dictionary onto `agri_kb.json` cards.
    Works 100% offline once `id2label` is loaded from local `config.json`.
    """
    retriever = OfflineKBRetriever(kb_path=kb_path)
    ordered_items = sorted(id2label.items(), key=lambda kv: int(kv[0]))
    classes: List[Dict[str, Any]] = [
        map_hf_label_to_kb_entry(int(idx), label, retriever)
        for idx, label in ordered_items
    ]
    mapped_kb_count = sum(1 for c in classes if c["kb_card_id"] is not None)
    return {
        "num_classes": len(classes),
        "mapped_kb_cards_count": mapped_kb_count,
        "classes": classes,
        "index_to_label": {str(c["index"]): c["label"] for c in classes},
    }


def download_pretrained_plantvillage(
    repo_id: str = DEFAULT_PRETRAINED_REPO_ID,
    output_dir: Path = DEFAULT_PRETRAINED_DIR,
    kb_path: Path = DEFAULT_KB_PATH,
) -> Dict[str, Any]:
    """
    Downloads an open-source pretrained PlantVillage model & image processor
    from Hugging Face once (while online) and saves `kb_label_mapping.json`
    for 100% offline runtime inference (`local_files_only=True`).
    """
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
        )
    except ImportError as exc:
        raise MissingPretrainedDownloadDepsError(
            "One-time download requires 'transformers' and 'safetensors'. "
            "Install with: pip install transformers huggingface_hub safetensors"
        ) from exc

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageClassification.from_pretrained(repo_id)
    try:
        processor = AutoImageProcessor.from_pretrained(repo_id)
    except ValueError:
        # Legacy checkpoints (e.g., MobileNetV2FeatureExtractor in transformers v4)
        # require explicit image processor class lookup in transformers v5+.
        model_type = getattr(model.config, "model_type", "")
        if model_type == "mobilenet_v2":
            from transformers import MobileNetV2ImageProcessor

            processor = MobileNetV2ImageProcessor.from_pretrained(repo_id)
        else:
            raise

    processor.save_pretrained(str(target_dir))
    model.save_pretrained(str(target_dir))
    model.eval()

    # Also export a zero-`transformers`-dependency PyTorch ExportedProgram (.pt2)
    # so global `python3` without `transformers` can still execute 100% offline.
    pt2_path: Optional[Path] = None
    pt2_export_error: Optional[str] = None
    try:
        import torch

        class _LogitsWrapper(torch.nn.Module):
            def __init__(self, hf_model: torch.nn.Module) -> None:
                super().__init__()
                self.hf_model = hf_model

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return self.hf_model(pixel_values=pixel_values).logits

        wrapper = _LogitsWrapper(model).eval()
        crop_cfg = getattr(processor, "crop_size", None) if getattr(processor, "do_center_crop", False) else None
        size_cfg = crop_cfg if crop_cfg is not None else getattr(processor, "size", {"height": 224, "width": 224})

        def _dim(obj: Any, key: str, fallback: int = 224) -> int:
            val = obj.get(key) if hasattr(obj, "get") else getattr(obj, key, None)
            if val is None and key in ("height", "width"):
                val = obj.get("shortest_edge") if hasattr(obj, "get") else getattr(obj, "shortest_edge", None)
            return int(val) if val is not None else fallback

        h = _dim(size_cfg, "height", 224)
        w = _dim(size_cfg, "width", 224)
        sample = torch.randn(1, 3, h, w, dtype=torch.float32)
        exported = torch.export.export(wrapper, (sample,))
        pt2_path = target_dir / "plantvillage_mobilenetv2.pt2"
        torch.export.save(exported, pt2_path)
    except Exception as exc:  # noqa: BLE001
        pt2_path = None
        pt2_export_error = f"{type(exc).__name__}: {exc}"

    id2label = getattr(model.config, "id2label", {})
    if not id2label:
        raise ValueError(f"Downloaded model '{repo_id}' has empty config.id2label.")

    mapping = build_hf_to_kb_mapping(id2label=id2label, kb_path=kb_path)
    mapping["source_repo_id"] = repo_id
    mapping["pt2_exported"] = pt2_path is not None
    if pt2_export_error is not None:
        mapping["pt2_export_error"] = pt2_export_error

    mapping_path = target_dir / "kb_label_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    return {
        "repo_id": repo_id,
        "output_dir": str(target_dir.resolve()),
        "mapping_path": str(mapping_path.resolve()),
        "pt2_path": str(pt2_path.resolve()) if pt2_path is not None else None,
        "pt2_export_error": pt2_export_error,
        "num_classes": mapping["num_classes"],
        "mapped_kb_cards_count": mapping["mapped_kb_cards_count"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time download of pretrained PlantVillage Hugging Face model for offline PS1 inference."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_PRETRAINED_REPO_ID,
        help="Hugging Face PlantVillage image-classification model repo ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PRETRAINED_DIR,
        help="Local cache directory for offline runtime loading",
    )
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=DEFAULT_KB_PATH,
        help="Path to agri_kb.json",
    )
    return parser.parse_args()


def main() -> int:
    if importlib.util.find_spec("transformers") is None:
        venv_python = _PS1_ROOT / ".venv" / "bin" / "python"
        if venv_python.is_file() and os.environ.get("_PS1_VENV_DELEGATED") != "1":
            env = dict(os.environ)
            env["_PS1_VENV_DELEGATED"] = "1"
            proc = subprocess.run(
                [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
                env=env,
                check=False,
            )
            return proc.returncode

    args = _parse_args()
    summary = download_pretrained_plantvillage(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        kb_path=args.kb_path,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
