"""
Phase 1 Verification Suite: Offline Knowledge Base, Retriever, and Strict Contracts.

Covers:
  1. Deterministic compilation of `agri_kb.json` from read-only FieldWise JS files.
  2. Strict Pydantic validation and rejection of malformed inputs (NaN, Inf, bools,
     out-of-range scores, invalid enums, extra fields, and unsafe advisory states).
  3. Exact and normalized offline retrieval across PlantVillage / IP102 classes and
     safe `None` handling for unknown or healthy inputs.
  4. Zero-cloud / zero-network-import compliance across `ps1_offline_assistant/`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.kb.retriever import OfflineKBRetriever  # noqa: E402
from ps1_offline_assistant.schemas.contracts import (  # noqa: E402
    AdvisoryRequest,
    AdvisoryResponse,
    AdvisoryStatus,
    DiseasePestCard,
    SeverityLevel,
    SupportedLanguage,
    VisionCategory,
    VisionFinding,
)
from ps1_offline_assistant.scripts.build_offline_kb import (  # noqa: E402
    DEFAULT_CROP_DB_JS,
    DEFAULT_FERT_DB_JS,
    DEFAULT_OUTPUT_JSON,
    _extract_js_exports_via_node,
    _extract_js_literal_python,
    build_offline_kb,
)


def _valid_vision_finding(**overrides) -> dict:
    payload = {
        "crop": "rice",
        "condition_name": "Rice Blast",
        "category": "disease",
        "confidence": 0.91,
        "severity_level": "moderate",
        "infected_area_pct": 14.5,
        "image_quality_ok": True,
        "blur_score": 142.3,
        "inference_latency_ms": 18.4,
        "runtime_backend": "hailort_npu",
    }
    payload.update(overrides)
    return payload


def _valid_advisory_response(**overrides) -> dict:
    payload = {
        "status": "SUCCESS",
        "explanation": "Moderate Rice Blast detected (~14.5% leaf area). Follow cultural and threshold-based IPM steps.",
        "immediate_actions": [
            "Avoid additional urea top-dressing immediately.",
            "Scout neighboring tillers and panicles for neck blast.",
        ],
        "organic_cultural_steps": [
            "Maintain field bund sanitation and avoid moisture stress.",
        ],
        "chemical_guidance_if_approved": [
            "Tricyclazole 75% WP at CIB&RC label rate verified with local KVK/DAO.",
        ],
        "preventive_measures": [
            "Adopt blast-tolerant paddy varieties and balanced NPK next season.",
        ],
        "escalate_to_kvk": False,
        "disclaimer": "Verify CIB&RC label dosage and PHI with local KVK before spraying.",
        "language": "en",
        "ttft_ms": 115.0,
        "tokens_per_sec": 24.5,
    }
    payload.update(overrides)
    return payload


class TestOfflineKBBuildAndDeterminism(unittest.TestCase):
    def test_deterministic_build_and_js_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_a = Path(tmpdir) / "kb_a.json"
            out_b = Path(tmpdir) / "kb_b.json"

            build_offline_kb(DEFAULT_CROP_DB_JS, DEFAULT_FERT_DB_JS, out_a)
            build_offline_kb(DEFAULT_CROP_DB_JS, DEFAULT_FERT_DB_JS, out_b)

            bytes_a = out_a.read_bytes()
            bytes_b = out_b.read_bytes()
            self.assertEqual(bytes_a, bytes_b, "KB build must be byte-for-byte deterministic.")

            committed_bytes = DEFAULT_OUTPUT_JSON.read_bytes()
            self.assertEqual(
                bytes_a,
                committed_bytes,
                "Committed data/agri_kb.json must match deterministic build output.",
            )

            # Verify pure-Python JS fallback matches Node.js extraction identically
            node_extracted = _extract_js_exports_via_node(
                DEFAULT_CROP_DB_JS, DEFAULT_FERT_DB_JS
            )
            py_crops = _extract_js_literal_python(
                DEFAULT_CROP_DB_JS.read_text(encoding="utf-8"), "crops"
            )
            py_ferts = _extract_js_literal_python(
                DEFAULT_FERT_DB_JS.read_text(encoding="utf-8"), "fertilizers"
            )
            self.assertEqual(node_extracted["crops"], py_crops)
            self.assertEqual(node_extracted["fertilizers"], py_ferts)

    def test_kb_schema_coverage_and_card_fields(self) -> None:
        kb_data = json.loads(DEFAULT_OUTPUT_JSON.read_text(encoding="utf-8"))

        # Provenance hashes match existing read-only JS files
        crop_sha = hashlib.sha256(DEFAULT_CROP_DB_JS.read_bytes()).hexdigest()
        fert_sha = hashlib.sha256(DEFAULT_FERT_DB_JS.read_bytes()).hexdigest()
        self.assertEqual(kb_data["provenance"]["crop_database_sha256"], crop_sha)
        self.assertEqual(kb_data["provenance"]["fertilizer_database_sha256"], fert_sha)

        self.assertEqual(len(kb_data["crops"]), 16)
        self.assertEqual(len(kb_data["fertilizers"]), 16)

        cards = kb_data["disease_pest_cards"]
        self.assertGreaterEqual(len(cards), 12)

        crops_covered = {c["crop"] for c in cards}
        for required_crop in ("rice", "wheat", "cotton", "potato", "tomato", "maize"):
            self.assertIn(required_crop, crops_covered)

        required_card_keys = {
            "id",
            "crop",
            "condition",
            "type",
            "symptoms",
            "severity_thresholds",
            "cultural_organic_control",
            "approved_chemical_options",
            "region_notes",
            "kvk_escalation_trigger",
        }
        for raw_card in cards:
            self.assertTrue(required_card_keys.issubset(raw_card.keys()))
            validated = DiseasePestCard(**raw_card)
            self.assertGreaterEqual(len(validated.approved_chemical_options), 1)
            for opt in validated.approved_chemical_options:
                self.assertTrue(opt.safety_warning)


class TestStrictContracts(unittest.TestCase):
    def test_valid_contracts_and_json_roundtrip(self) -> None:
        retriever = OfflineKBRetriever()
        card = retriever.get_disease_pest_card("Rice___Blast", crop="Rice")
        self.assertIsNotNone(card)

        finding = VisionFinding(**_valid_vision_finding())
        request = AdvisoryRequest(
            vision_finding=finding,
            state_region="Punjab",
            language=SupportedLanguage.HI,
            kb_entry=card,
        )
        response = AdvisoryResponse(**_valid_advisory_response(language="hi"))

        # Round-trip serialization
        req_roundtrip = AdvisoryRequest.model_validate_json(request.model_dump_json())
        resp_roundtrip = AdvisoryResponse.model_validate_json(response.model_dump_json())
        self.assertEqual(req_roundtrip.vision_finding.crop, "rice")
        self.assertEqual(req_roundtrip.kb_entry.id, "rice_blast")
        self.assertEqual(resp_roundtrip.status, AdvisoryStatus.SUCCESS)

    def test_vision_finding_rejects_nan_inf_and_out_of_range(self) -> None:
        for bad_val in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01):
            with self.assertRaises(ValidationError):
                VisionFinding(**_valid_vision_finding(confidence=bad_val))

        for bad_pct in (float("nan"), -1.0, 100.01):
            with self.assertRaises(ValidationError):
                VisionFinding(**_valid_vision_finding(infected_area_pct=bad_pct))

        for bad_nonneg_field in ("blur_score", "inference_latency_ms"):
            with self.assertRaises(ValidationError):
                VisionFinding(**_valid_vision_finding(**{bad_nonneg_field: -0.5}))
            with self.assertRaises(ValidationError):
                VisionFinding(**_valid_vision_finding(**{bad_nonneg_field: float("nan")}))

    def test_vision_finding_rejects_invalid_enums_and_strict_types(self) -> None:
        with self.assertRaises(ValidationError):
            VisionFinding(**_valid_vision_finding(category="weed"))
        with self.assertRaises(ValidationError):
            VisionFinding(**_valid_vision_finding(severity_level="critical"))
        with self.assertRaises(ValidationError):
            VisionFinding(**_valid_vision_finding(confidence="0.85"))
        with self.assertRaises(ValidationError):
            VisionFinding(**_valid_vision_finding(confidence=True))
        with self.assertRaises(ValidationError):
            VisionFinding(**_valid_vision_finding(image_quality_ok=1))
        with self.assertRaises(ValidationError):
            VisionFinding(**_valid_vision_finding(image_quality_ok="true"))
        with self.assertRaises(ValidationError):
            VisionFinding(**_valid_vision_finding(crop="   "))
        with self.assertRaises(ValidationError):
            VisionFinding(**_valid_vision_finding(unexpected_extra="bad"))

    def test_advisory_request_and_response_validation_and_safety_invariants(self) -> None:
        finding = VisionFinding(**_valid_vision_finding())

        # Invalid language rejected
        with self.assertRaises(ValidationError):
            AdvisoryRequest(
                vision_finding=finding,
                state_region="Gujarat",
                language="fr",
                kb_entry=None,
            )

        # Invalid response status rejected
        with self.assertRaises(ValidationError):
            AdvisoryResponse(**_valid_advisory_response(status="FAILED"))

        # NaN / negative telemetry rejected
        with self.assertRaises(ValidationError):
            AdvisoryResponse(**_valid_advisory_response(ttft_ms=float("nan")))
        with self.assertRaises(ValidationError):
            AdvisoryResponse(**_valid_advisory_response(tokens_per_sec=-2.0))

        # Bare string instead of List[str] rejected
        with self.assertRaises(ValidationError):
            AdvisoryResponse(
                **_valid_advisory_response(immediate_actions="Spray water")
            )

        # Strict bool for escalate_to_kvk
        with self.assertRaises(ValidationError):
            AdvisoryResponse(**_valid_advisory_response(escalate_to_kvk=1))

        # Safety invariant: SEVERE_ESCALATION requires escalate_to_kvk=True
        with self.assertRaises(ValidationError):
            AdvisoryResponse(
                **_valid_advisory_response(
                    status="SEVERE_ESCALATION", escalate_to_kvk=False
                )
            )
        severe_ok = AdvisoryResponse(
            **_valid_advisory_response(
                status="SEVERE_ESCALATION", escalate_to_kvk=True
            )
        )
        self.assertTrue(severe_ok.escalate_to_kvk)

        # Safety invariant: UNCERTAIN_IMAGE / LOW_CONFIDENCE must withhold chemical guidance
        for uncertain_status in ("UNCERTAIN_IMAGE", "LOW_CONFIDENCE"):
            with self.assertRaises(ValidationError):
                AdvisoryResponse(
                    **_valid_advisory_response(
                        status=uncertain_status,
                        chemical_guidance_if_approved=["Tricyclazole 75% WP"],
                    )
                )
            safe_uncertain = AdvisoryResponse(
                **_valid_advisory_response(
                    status=uncertain_status,
                    chemical_guidance_if_approved=[],
                )
            )
            self.assertEqual(safe_uncertain.chemical_guidance_if_approved, [])


class TestOfflineKBRetriever(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = OfflineKBRetriever()

    def test_exact_and_normalized_lookup(self) -> None:
        cases = [
            ("rice", "rice_blast", "rice_blast"),
            ("Rice", "Rice___Blast", "rice_blast"),
            ("paddy", "Brown Spot", "rice_brown_spot"),
            ("Wheat", "Wheat Rust", "wheat_leaf_rust"),
            ("wheat", "Stripe rust", "wheat_yellow_rust"),
            ("Wheat", "aphids", "wheat_aphids"),
            ("Cotton", "aphids", "cotton_aphids"),
            ("Cotton", "cotton bollworm", "cotton_bollworm"),
            ("Potato", "Potato___Late_blight", "potato_late_blight"),
            ("Tomato", "Tomato___Early_blight", "tomato_early_blight"),
            ("Maize", "Fall Armyworm", "maize_fall_armyworm"),
            (
                "Corn_(maize)",
                "Corn_(maize)___Northern_Leaf_Blight",
                "maize_northern_leaf_blight",
            ),
            (None, "Potato Late Blight", "potato_late_blight"),
        ]
        for crop, cond, expected_id in cases:
            with self.subTest(crop=crop, cond=cond):
                card = self.retriever.get_disease_pest_card(cond, crop=crop)
                self.assertIsNotNone(card)
                self.assertEqual(card.id, expected_id)

    def test_unknown_healthy_and_mismatch_return_none(self) -> None:
        for crop, cond in [
            ("rice", "healthy"),
            ("tomato", "Tomato___healthy"),
            ("rice", "unknown"),
            ("rice", ""),
            ("rice", "unknown_alien_fungus"),
            ("wheat", "apple_scab"),
            ("rice", "Potato___Late_blight"),  # crop mismatch must not cross-match
            (None, "unknown_pest_xyz"),
        ]:
            with self.subTest(crop=crop, cond=cond):
                self.assertIsNone(self.retriever.get_disease_pest_card(cond, crop=crop))

        healthy_finding = VisionFinding(
            **_valid_vision_finding(
                crop="tomato",
                condition_name="Healthy",
                category=VisionCategory.HEALTHY,
                severity_level=SeverityLevel.LOW,
                infected_area_pct=0.0,
            )
        )
        unknown_finding = VisionFinding(
            **_valid_vision_finding(
                crop="tomato",
                condition_name="Tomato___Early_blight",
                category=VisionCategory.UNKNOWN,
                confidence=0.25,
                severity_level=SeverityLevel.UNDETERMINED,
                infected_area_pct=0.0,
            )
        )
        self.assertIsNone(self.retriever.retrieve_for_finding(healthy_finding))
        self.assertIsNone(self.retriever.retrieve_for_finding(unknown_finding))

    def test_crop_profile_and_fertilizer_lookup(self) -> None:
        rice_profile = self.retriever.get_crop_profile("Paddy")
        self.assertIsNotNone(rice_profile)
        self.assertEqual(rice_profile["id"], "rice")

        maize_fert = self.retriever.get_fertilizer_schedule("Corn_(maize)")
        self.assertIsNotNone(maize_fert)
        self.assertGreater(len(maize_fert), 0)
        self.assertIsNone(self.retriever.get_crop_profile("nonexistent_crop"))


class TestZeroCloudCompliance(unittest.TestCase):
    def test_no_cloud_or_http_imports(self) -> None:
        always_forbidden = {
            "supabase",
            "firebase",
            "firebase_admin",
            "google",
            "openai",
            "anthropic",
            "requests",
            "httpx",
            "aiohttp",
            "urllib.request",
        }
        pipeline_forbidden = always_forbidden | {"urllib", "http", "socket"}
        app_server_allowed_http_modules = {"http", "http.server", "urllib.parse"}

        for py_file in _PS1_ROOT.rglob("*.py"):
            if ".venv" in py_file.parts or "__pycache__" in py_file.parts:
                continue
            rel_path = py_file.relative_to(_PS1_ROOT).as_posix()
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                imported_names = []
                if isinstance(node, ast.Import):
                    imported_names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_names = [node.module]

                for full_name in imported_names:
                    root = full_name.split(".")[0]
                    self.assertNotIn(
                        root,
                        always_forbidden,
                        f"Forbidden cloud/HTTP client import '{full_name}' in {rel_path}",
                    )
                    self.assertNotIn(
                        full_name,
                        always_forbidden,
                        f"Forbidden cloud/HTTP client import '{full_name}' in {rel_path}",
                    )

                    if rel_path == "app/server.py":
                        if root in {"http", "urllib", "socket"}:
                            self.assertIn(
                                full_name,
                                app_server_allowed_http_modules,
                                f"Disallowed network import '{full_name}' in {rel_path}",
                            )
                    elif rel_path in {
                        "tests/test_phase5_app.py",
                        "tests/test_phase7_crop_fertilizer_and_vision.py",
                    }:
                        if root in {"http", "urllib", "socket"}:
                            self.assertEqual(
                                full_name,
                                "http.client",
                                f"Only http.client loopback testing allowed in {rel_path}, got '{full_name}'",
                            )
                    else:
                        self.assertNotIn(
                            root,
                            pipeline_forbidden,
                            f"Forbidden network/cloud import '{full_name}' in {rel_path}",
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
