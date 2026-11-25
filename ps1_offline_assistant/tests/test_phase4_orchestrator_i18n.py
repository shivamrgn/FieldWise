"""
Phase 4 Verification Suite: Offline Multilingual i18n & End-to-End Dual-Pipeline Orchestrator.

Verifies:
  1. 100% key parity and non-empty strings across `en`, `hi`, `gu`, `mr` bundles,
     KB alignment for crops/conditions, and safe `en` fallback on unsupported codes.
  2. End-to-end `FieldAssistantOrchestrator.analyze_image()` across 4 scenarios:
     blurry image, healthy leaf, moderate disease, and severe disease.
  3. Instant `switch_language()` (`en -> hi -> gu -> mr`) preserving `VisionFinding`
     without re-running Vision Pipeline A (`vision_call_count == 1`).
  4. Local offline JSONL and SQLite field-scan history logging & reading.
  5. End-to-end offline CLI runner (`orchestrator/cli.py`) in JSON and text modes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Set

import cv2
import numpy as np

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.advisory.engine import AdvisoryEngine  # noqa: E402
from ps1_offline_assistant.advisory.runtime import TestStubAdvisoryBackend  # noqa: E402
from ps1_offline_assistant.i18n.translations import (  # noqa: E402
    SUPPORTED_LANGUAGES,
    get_ui_bundle,
    localize_condition_name,
    localize_crop_name,
    localize_status_message,
)
from ps1_offline_assistant.kb.retriever import OfflineKBRetriever  # noqa: E402
from ps1_offline_assistant.orchestrator.pipeline import (  # noqa: E402
    AssistantResult,
    FieldAssistantOrchestrator,
    read_field_history,
)
from ps1_offline_assistant.schemas.contracts import (  # noqa: E402
    AdvisoryStatus,
    SeverityLevel,
    SupportedLanguage,
    VisionCategory,
)
from ps1_offline_assistant.tests.test_phase2_vision import (  # noqa: E402
    _make_synthetic_leaf,
)
from ps1_offline_assistant.vision.inference import (  # noqa: E402
    VisionBackend,
    VisionPipeline,
)
from ps1_offline_assistant.vision.train_and_export import (  # noqa: E402
    DEFAULT_KB_PATH,
    build_class_labels_from_kb,
    build_vision_model,
    export_to_torch_exported,
    save_class_labels_manifest,
)


def _collect_nested_keys_and_check_nonempty(
    obj: Dict[str, Any],
    prefix: str = "",
) -> Set[str]:
    keys: Set[str] = set()
    for key, val in obj.items():
        if key in {"requested_language", "language_fallback_used"}:
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        keys.add(full_key)
        if isinstance(val, dict):
            keys.update(_collect_nested_keys_and_check_nonempty(val, full_key))
        elif isinstance(val, str):
            if not val.strip():
                raise AssertionError(f"Empty translation string at key '{full_key}'")
    return keys


class _CountingVisionBackend(VisionBackend):
    """Counting wrapper around deterministic logits to verify Pipeline A call count."""

    def __init__(self, logits: np.ndarray, name: str = "counting_test_vision") -> None:
        self._logits = np.asarray(logits, dtype=np.float32)
        self._name = name
        self.call_count = 0

    @property
    def backend_name(self) -> str:
        return self._name

    def predict_logits(self, input_nchw: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return self._logits


class TestOfflineMultilingualI18n(unittest.TestCase):
    def test_100_percent_key_parity_and_kb_alignment(self) -> None:
        self.assertEqual(SUPPORTED_LANGUAGES, ("en", "hi", "gu", "mr", "te", "ta"))
        en_bundle = get_ui_bundle("en")
        en_keys = _collect_nested_keys_and_check_nonempty(en_bundle)

        for lang in SUPPORTED_LANGUAGES:
            with self.subTest(lang=lang):
                bundle = get_ui_bundle(lang)
                self.assertFalse(bundle["language_fallback_used"])
                lang_keys = _collect_nested_keys_and_check_nonempty(bundle)
                self.assertEqual(lang_keys, en_keys)
                self.assertEqual(bundle["kvk_escalation_messages"]["helpline_number"], "1800-180-1551")

        # Verify crop & disease/pest IDs from agri_kb.json are covered in every language
        kb_data = json.loads(DEFAULT_KB_PATH.read_text(encoding="utf-8"))
        kb_crops = {c["id"] for c in kb_data["crops"]}
        kb_cards = {c["id"] for c in kb_data["disease_pest_cards"]}
        for lang in SUPPORTED_LANGUAGES:
            bundle = get_ui_bundle(lang)
            self.assertTrue(kb_crops.issubset(bundle["crops"].keys()))
            self.assertTrue(kb_cards.issubset(bundle["conditions"].keys()))

    def test_unsupported_language_fallback_and_helpers(self) -> None:
        with self.assertLogs("ps1_offline_assistant.i18n.translations", level="WARNING"):
            fallback = get_ui_bundle("fr")
        self.assertEqual(fallback["language_code"], "en")
        self.assertTrue(fallback["language_fallback_used"])
        self.assertEqual(fallback["requested_language"], "fr")

        self.assertIn("KVK", localize_status_message("SEVERE_ESCALATION", "en"))
        self.assertIn("KVK", localize_status_message("SEVERE_ESCALATION", "hi"))
        self.assertEqual(localize_crop_name("rice", "hi"), "धान (चावल)")
        self.assertEqual(localize_condition_name("Rice Blast", "mr"), "भातावरील करपा रोग (ब्लास्ट)")


class TestDualPipelineOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.classes = build_class_labels_from_kb(DEFAULT_KB_PATH)
        self.num_classes = len(self.classes)
        self.rice_blast_idx = next(c["index"] for c in self.classes if c["label"] == "rice_blast")
        self.healthy_idx = next(c["index"] for c in self.classes if c["label"] == "healthy")
        self.retriever = OfflineKBRetriever()

    def _build_orchestrator(
        self,
        target_idx: int,
        log_path: Path | None = None,
    ) -> tuple[FieldAssistantOrchestrator, _CountingVisionBackend]:
        logits = np.full(self.num_classes, -5.0, dtype=np.float32)
        logits[target_idx] = 5.0
        vision_backend = _CountingVisionBackend(logits)
        vision_pipeline = VisionPipeline(vision_backend, self.classes)
        advisory_engine = AdvisoryEngine(TestStubAdvisoryBackend(), self.retriever)
        orchestrator = FieldAssistantOrchestrator(
            vision_pipeline=vision_pipeline,
            advisory_engine=advisory_engine,
            kb_retriever=self.retriever,
            log_path=log_path,
        )
        return orchestrator, vision_backend

    def test_four_end_to_end_synthetic_scenarios(self) -> None:
        # 1. Blurry image -> UNCERTAIN_IMAGE, zero chemicals, localized retake guidance
        orch_blur, _ = self._build_orchestrator(self.rice_blast_idx)
        blurry_leaf = cv2.GaussianBlur(_make_synthetic_leaf(0.25), (35, 35), 12.0)
        res_blur = orch_blur.analyze_image(blurry_leaf, state_region="Maharashtra", language="mr")
        self.assertEqual(res_blur.advisory_response.status, AdvisoryStatus.UNCERTAIN_IMAGE)
        self.assertFalse(res_blur.advisory_response.escalate_to_kvk)
        self.assertEqual(res_blur.advisory_response.chemical_guidance_if_approved, [])
        self.assertEqual(res_blur.ui_labels["language_code"], "mr")
        self.assertTrue(res_blur.pipeline_metadata.offline_mode)
        self.assertEqual(res_blur.pipeline_metadata.vision_backend, "counting_test_vision")
        self.assertEqual(res_blur.pipeline_metadata.advisory_backend, "test_stub_ci_only")

        # 2. Healthy leaf -> SUCCESS, preventive advice only
        orch_healthy, _ = self._build_orchestrator(self.healthy_idx)
        res_healthy = orch_healthy.analyze_image(
            _make_synthetic_leaf(0.0), state_region="Gujarat", language="gu"
        )
        self.assertEqual(res_healthy.vision_finding.category, VisionCategory.HEALTHY)
        self.assertEqual(res_healthy.advisory_response.status, AdvisoryStatus.SUCCESS)
        self.assertFalse(res_healthy.advisory_response.escalate_to_kvk)
        self.assertEqual(res_healthy.advisory_response.chemical_guidance_if_approved, [])

        # 3. Moderate disease (~25% lesion) -> SUCCESS, grounded advisory + localized labels
        orch_mod, _ = self._build_orchestrator(self.rice_blast_idx)
        res_mod = orch_mod.analyze_image(
            _make_synthetic_leaf(0.25), state_region="Punjab", language="en"
        )
        self.assertEqual(res_mod.vision_finding.severity_level, SeverityLevel.MODERATE)
        self.assertEqual(res_mod.advisory_response.status, AdvisoryStatus.SUCCESS)
        self.assertFalse(res_mod.advisory_response.escalate_to_kvk)
        self.assertGreater(len(res_mod.advisory_response.chemical_guidance_if_approved), 0)
        self.assertIn("Tricyclazole", res_mod.advisory_response.chemical_guidance_if_approved[0])
        self.assertGreater(res_mod.pipeline_metadata.end_to_end_latency_ms, 0.0)

        # 4. Severe disease (~55% lesion) -> SEVERE_ESCALATION, escalate_to_kvk=True, 1800-180-1551 notice
        orch_sev, _ = self._build_orchestrator(self.rice_blast_idx)
        res_sev = orch_sev.analyze_image(
            _make_synthetic_leaf(0.55), state_region="Punjab", language="hi"
        )
        self.assertEqual(res_sev.vision_finding.severity_level, SeverityLevel.SEVERE)
        self.assertEqual(res_sev.advisory_response.status, AdvisoryStatus.SEVERE_ESCALATION)
        self.assertTrue(res_sev.advisory_response.escalate_to_kvk)
        self.assertIn(
            "1800-180-1551",
            res_sev.ui_labels["kvk_escalation_messages"]["recommended_escalation"],
        )

    def test_instant_language_switching_reuses_vision_finding(self) -> None:
        orch, vision_backend = self._build_orchestrator(self.rice_blast_idx)
        mod_leaf = _make_synthetic_leaf(0.25)

        res_en = orch.analyze_image(mod_leaf, state_region="Punjab", language="en")
        self.assertEqual(vision_backend.call_count, 1)

        res_hi = orch.switch_language(res_en, "hi", state_region="Punjab")
        res_gu = orch.switch_language(res_hi, "gu", state_region="Punjab")
        res_mr = orch.switch_language(res_gu, "mr", state_region="Punjab")

        # Vision backend must remain at call_count == 1 across all switches
        self.assertEqual(vision_backend.call_count, 1)

        for lang_code, res in [("en", res_en), ("hi", res_hi), ("gu", res_gu), ("mr", res_mr)]:
            self.assertEqual(res.advisory_response.language, SupportedLanguage(lang_code))
            self.assertEqual(res.ui_labels["language_code"], lang_code)
            self.assertEqual(res.vision_finding, res_en.vision_finding)
            self.assertEqual(res.session_id, res_en.session_id)
            AssistantResult.model_validate_json(res.model_dump_json())

    def test_jsonl_and_sqlite_offline_field_history_logging(self) -> None:
        mod_leaf = _make_synthetic_leaf(0.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            jsonl_log = tmp / "field_scans.jsonl"
            sqlite_log = tmp / "field_scans.db"

            orch_jsonl, _ = self._build_orchestrator(self.rice_blast_idx, log_path=jsonl_log)
            r1 = orch_jsonl.analyze_image(mod_leaf, state_region="Punjab", language="en")
            orch_jsonl.switch_language(r1, "hi", state_region="Punjab")

            jsonl_records = read_field_history(jsonl_log)
            self.assertEqual(len(jsonl_records), 2)
            self.assertEqual(jsonl_records[0]["event_type"], "initial_scan")
            self.assertEqual(jsonl_records[1]["event_type"], "language_switch")
            self.assertEqual(jsonl_records[1]["language"], "hi")

            orch_db, _ = self._build_orchestrator(self.rice_blast_idx, log_path=sqlite_log)
            r2 = orch_db.analyze_image(mod_leaf, state_region="Gujarat", language="gu")
            orch_db.switch_language(r2, "mr", state_region="Gujarat")

            db_records = read_field_history(sqlite_log)
            self.assertEqual(len(db_records), 2)
            self.assertEqual(db_records[0]["language"], "gu")
            self.assertEqual(db_records[1]["language"], "mr")
            self.assertEqual(db_records[0]["vision_backend"], "counting_test_vision")
            self.assertEqual(db_records[0]["advisory_backend"], "test_stub_ci_only")


class TestOfflineCliEntrypoint(unittest.TestCase):
    def test_cli_json_and_language_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img_path = tmp / "leaf.png"
            cv2.imwrite(str(img_path), _make_synthetic_leaf(0.25))

            manifest = save_class_labels_manifest(tmp / "class_labels.json")
            model = build_vision_model(num_classes=manifest["num_classes"], arch="mobilenet_v3_small")
            pt2_path = export_to_torch_exported(model, tmp / "mobilenet_v3_small.pt2")
            log_path = tmp / "cli_history.jsonl"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ps1_offline_assistant.orchestrator.cli",
                    "--image",
                    str(img_path),
                    "--state",
                    "Punjab",
                    "--lang",
                    "hi",
                    "--vision-backend",
                    "pt2",
                    "--vision-model",
                    str(pt2_path),
                    "--class-labels",
                    str(tmp / "class_labels.json"),
                    "--advisory-backend",
                    "test-stub",
                    "--switch-lang",
                    "mr",
                    "--log-path",
                    str(log_path),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            data = json.loads(proc.stdout)
            self.assertIn("initial", data)
            self.assertIn("switched", data)
            self.assertEqual(data["initial"]["advisory_response"]["language"], "hi")
            self.assertEqual(data["switched"]["advisory_response"]["language"], "mr")
            self.assertEqual(data["initial"]["vision_finding"], data["switched"]["vision_finding"])
            self.assertEqual(len(read_field_history(log_path)), 2)

            # Also verify human-readable localized stdout mode
            proc_txt = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ps1_offline_assistant.orchestrator.cli",
                    "--image",
                    str(img_path),
                    "--state",
                    "Gujarat",
                    "--lang",
                    "gu",
                    "--vision-backend",
                    "pt2",
                    "--vision-model",
                    str(pt2_path),
                    "--advisory-backend",
                    "test-stub",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("ગુજરાતી", proc_txt.stdout)
            self.assertIn("[Telemetry]", proc_txt.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
