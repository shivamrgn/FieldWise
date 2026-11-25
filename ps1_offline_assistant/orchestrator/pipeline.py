"""
End-to-End Dual-Pipeline Orchestrator (`orchestrator/pipeline.py`) for FieldWise PS1.

Connects:
  Pipeline A (VisionPipeline) -> Local Grounded KB (OfflineKBRetriever) ->
  Pipeline B (AdvisoryEngine) -> Offline Multilingual Localization (`i18n`) ->
  Local Offline Field-Scan Logging (JSONL or SQLite).

Preserves strict two-pipeline separation (`vision_backend` vs `advisory_backend`)
and supports instant offline language switching via `switch_language()` without
re-running Vision Pipeline A.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ps1_offline_assistant.advisory.engine import AdvisoryEngine
from ps1_offline_assistant.i18n.translations import get_ui_bundle
from ps1_offline_assistant.kb.retriever import OfflineKBRetriever
from ps1_offline_assistant.schemas.contracts import (
    AdvisoryRequest,
    AdvisoryResponse,
    StrictBool,
    StrictFiniteFloat,
    StrictNonEmptyStr,
    SupportedLanguage,
    VisionFinding,
)
from ps1_offline_assistant.vision.inference import VisionPipeline


class PipelineMetadata(BaseModel):
    """Strict telemetry and backend provenance metadata across Pipeline A & B."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_assignment=True,
    )

    vision_backend: StrictNonEmptyStr
    advisory_backend: StrictNonEmptyStr
    offline_mode: StrictBool = True
    vision_latency_ms: StrictFiniteFloat = Field(..., ge=0.0)
    advisory_ttft_ms: StrictFiniteFloat = Field(..., ge=0.0)
    advisory_tokens_per_sec: StrictFiniteFloat = Field(..., ge=0.0)
    end_to_end_latency_ms: StrictFiniteFloat = Field(..., ge=0.0)


class AssistantResult(BaseModel):
    """Complete end-to-end output emitted by `FieldAssistantOrchestrator`."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_assignment=True,
    )

    session_id: StrictNonEmptyStr
    timestamp_utc: StrictNonEmptyStr
    state_region: StrictNonEmptyStr
    vision_finding: VisionFinding
    advisory_response: AdvisoryResponse
    ui_labels: Dict[str, Any]
    pipeline_metadata: PipelineMetadata
    symptoms: List[str] = Field(default_factory=list)
    regional_advisory: Optional[str] = None
    crop_info: Optional[Dict[str, Any]] = None
    kb_card_id: Optional[str] = None
    kb_source_reference: Optional[str] = None
    top_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    vision_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    is_stub_advisory: StrictBool = False


def _is_sqlite_path(path: Path) -> bool:
    return path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}


def _append_log_record(
    log_path: Path,
    result: AssistantResult,
    event_type: str,
) -> None:
    """Appends a compact offline scan record to JSONL or SQLite storage."""
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "event_type": event_type,
        "session_id": result.session_id,
        "timestamp_utc": result.timestamp_utc,
        "state_region": result.state_region,
        "language": result.advisory_response.language.value,
        "crop": result.vision_finding.crop,
        "condition_name": result.vision_finding.condition_name,
        "category": result.vision_finding.category.value,
        "confidence": result.vision_finding.confidence,
        "severity_level": result.vision_finding.severity_level.value,
        "infected_area_pct": result.vision_finding.infected_area_pct,
        "advisory_status": result.advisory_response.status.value,
        "escalate_to_kvk": result.advisory_response.escalate_to_kvk,
        "vision_backend": result.pipeline_metadata.vision_backend,
        "advisory_backend": result.pipeline_metadata.advisory_backend,
        "end_to_end_latency_ms": result.pipeline_metadata.end_to_end_latency_ms,
    }

    if _is_sqlite_path(target):
        with contextlib.closing(sqlite3.connect(str(target))) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS field_scans (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        timestamp_utc TEXT NOT NULL,
                        state_region TEXT NOT NULL,
                        language TEXT NOT NULL,
                        crop TEXT NOT NULL,
                        condition_name TEXT NOT NULL,
                        category TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        severity_level TEXT NOT NULL,
                        infected_area_pct REAL NOT NULL,
                        advisory_status TEXT NOT NULL,
                        escalate_to_kvk INTEGER NOT NULL,
                        vision_backend TEXT NOT NULL,
                        advisory_backend TEXT NOT NULL,
                        end_to_end_latency_ms REAL NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO field_scans (
                        event_type, session_id, timestamp_utc, state_region, language,
                        crop, condition_name, category, confidence, severity_level,
                        infected_area_pct, advisory_status, escalate_to_kvk,
                        vision_backend, advisory_backend, end_to_end_latency_ms, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["event_type"],
                        record["session_id"],
                        record["timestamp_utc"],
                        record["state_region"],
                        record["language"],
                        record["crop"],
                        record["condition_name"],
                        record["category"],
                        record["confidence"],
                        record["severity_level"],
                        record["infected_area_pct"],
                        record["advisory_status"],
                        1 if record["escalate_to_kvk"] else 0,
                        record["vision_backend"],
                        record["advisory_backend"],
                        record["end_to_end_latency_ms"],
                        result.model_dump_json(),
                    ),
                )
    else:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_field_history(log_path: Path) -> List[Dict[str, Any]]:
    """Reads locally logged offline field-scan records from JSONL or SQLite."""
    target = Path(log_path)
    if not target.is_file():
        return []

    if _is_sqlite_path(target):
        with contextlib.closing(sqlite3.connect(str(target))) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT event_type, session_id, timestamp_utc, state_region, language,
                       crop, condition_name, category, confidence, severity_level,
                       infected_area_pct, advisory_status, escalate_to_kvk,
                       vision_backend, advisory_backend, end_to_end_latency_ms
                FROM field_scans ORDER BY id ASC
                """
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["escalate_to_kvk"] = bool(item["escalate_to_kvk"])
                out.append(item)
            return out

    records: List[Dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class FieldAssistantOrchestrator:
    """
    Coordinates Pipeline A (Vision) -> KB Lookup -> Pipeline B (Advisory) -> i18n
    100% offline with optional local scan history persistence.
    """

    def __init__(
        self,
        vision_pipeline: VisionPipeline,
        advisory_engine: AdvisoryEngine,
        kb_retriever: Optional[OfflineKBRetriever] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.vision_pipeline = vision_pipeline
        self.advisory_engine = advisory_engine
        self.kb_retriever = kb_retriever if kb_retriever is not None else OfflineKBRetriever()
        self.log_path = Path(log_path) if log_path is not None else None

    def _run_advisory_and_package(
        self,
        vision_finding: VisionFinding,
        state_region: str,
        language: str,
        session_id: str,
        t_start: float,
        event_type: str,
        top_predictions: Optional[List[Dict[str, Any]]] = None,
        vision_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> AssistantResult:
        ui_labels = get_ui_bundle(language)
        effective_lang = SupportedLanguage(ui_labels["language_code"])

        kb_entry = self.kb_retriever.retrieve_for_finding(vision_finding)
        advisory_request = AdvisoryRequest(
            vision_finding=vision_finding,
            state_region=state_region,
            language=effective_lang,
            kb_entry=kb_entry,
        )
        advisory_response = self.advisory_engine.advise(advisory_request)

        end_to_end_ms = round(max(0.001, (time.perf_counter() - t_start) * 1000.0), 4)
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        metadata = PipelineMetadata(
            vision_backend=vision_finding.runtime_backend,
            advisory_backend=self.advisory_engine.backend.backend_name,
            offline_mode=True,
            vision_latency_ms=vision_finding.inference_latency_ms,
            advisory_ttft_ms=advisory_response.ttft_ms,
            advisory_tokens_per_sec=advisory_response.tokens_per_sec,
            end_to_end_latency_ms=end_to_end_ms,
        )

        result = AssistantResult(
            session_id=session_id,
            timestamp_utc=timestamp_utc,
            state_region=state_region.strip(),
            vision_finding=vision_finding,
            advisory_response=advisory_response,
            ui_labels=ui_labels,
            pipeline_metadata=metadata,
            symptoms=(
                list(kb_entry.symptoms)
                if kb_entry is not None
                and advisory_response.status.value in ("SUCCESS", "SEVERE_ESCALATION")
                else []
            ),
            regional_advisory=(
                kb_entry.region_notes.get(
                    state_region.strip(),
                    kb_entry.region_notes.get("general"),
                )
                if kb_entry is not None
                and advisory_response.status.value in ("SUCCESS", "SEVERE_ESCALATION")
                else None
            ),
            crop_info=self.kb_retriever.get_crop_profile(vision_finding.crop),
            kb_card_id=kb_entry.id if kb_entry is not None else None,
            kb_source_reference=kb_entry.source_reference if kb_entry is not None else None,
            top_predictions=list(top_predictions or []),
            vision_diagnostics=dict(vision_diagnostics or {}),
            is_stub_advisory=(self.advisory_engine.backend.backend_name == "test_stub_ci_only"),
        )

        if self.log_path is not None:
            _append_log_record(self.log_path, result, event_type=event_type)

        return result

    def analyze_image(
        self,
        image_bgr: np.ndarray,
        state_region: str,
        language: str = "en",
        pest_boxes: Optional[Sequence[Sequence[float]]] = None,
        crop_hint: Optional[str] = None,
    ) -> AssistantResult:
        """
        Executes full offline scan:
          Pipeline A (Vision) -> KB Retrieval -> Pipeline B (Advisory) -> i18n -> Local Log.
        """
        t_start = time.perf_counter()
        session_id = str(uuid.uuid4())
        if hasattr(self.vision_pipeline, "predict_with_details"):
            vision_finding, top_predictions, vision_diagnostics = self.vision_pipeline.predict_with_details(
                image_bgr, pest_boxes=pest_boxes, crop_hint=crop_hint
            )
        else:
            vision_finding = self.vision_pipeline.predict(image_bgr, pest_boxes=pest_boxes)
            top_predictions = []
            vision_diagnostics = {}

        return self._run_advisory_and_package(
            vision_finding=vision_finding,
            state_region=state_region,
            language=language,
            session_id=session_id,
            t_start=t_start,
            event_type="initial_scan",
            top_predictions=top_predictions,
            vision_diagnostics=vision_diagnostics,
        )

    def switch_language(
        self,
        previous_result: AssistantResult,
        new_language: str,
        state_region: Optional[str] = None,
    ) -> AssistantResult:
        """
        Instant offline language switching: reuses `previous_result.vision_finding`
        without re-invoking Vision Pipeline A, re-running only Pipeline B + i18n.
        """
        t_start = time.perf_counter()
        effective_region = (
            state_region
            if state_region is not None and state_region.strip()
            else previous_result.state_region
        )
        return self._run_advisory_and_package(
            vision_finding=previous_result.vision_finding,
            state_region=effective_region,
            language=new_language,
            session_id=previous_result.session_id,
            t_start=t_start,
            event_type="language_switch",
            top_predictions=previous_result.top_predictions,
            vision_diagnostics=previous_result.vision_diagnostics,
        )
