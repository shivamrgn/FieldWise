"""
Phase 7 Verification Suite: Offline V2 Crop Recommendation, Fertilizer Advisory & HF Local Loader.

Verifies:
  1. `POST /api/recommend-crop` delegates to frozen FieldWise V2 `CropRecommendationEngine`
     (Punjab + June + Loam -> Rice/Maize/Cotton eligible, Wheat excluded; stoichiometric
     DAP/Urea/MOP with SHC test; strict HTTP 400 on missing/invalid inputs).
  2. `GET /api/fertilizer-guide` scales hectare-based dosages accurately from `agri_kb.json`
     while preserving non-hectare concentration rates (`g/L`, `g/10 kg seed`).
  3. `index.html` retains strict Zero-Network compliance (0 `http://`/`https://`) while
     exposing all 3 tabs and untrained smoke warning banner.
  4. `HuggingFaceLocalVisionBackend` enforces `local_files_only=True`, fails cleanly when
     weights are missing, and maps 38-class PlantVillage labels onto `agri_kb.json`.
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import patch

import numpy as np
import torch

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.advisory.engine import AdvisoryEngine  # noqa: E402
from ps1_offline_assistant.advisory.runtime import TestStubAdvisoryBackend  # noqa: E402
from ps1_offline_assistant.app.server import create_offline_server  # noqa: E402
from ps1_offline_assistant.kb.retriever import OfflineKBRetriever  # noqa: E402
from ps1_offline_assistant.orchestrator.pipeline import FieldAssistantOrchestrator  # noqa: E402
from ps1_offline_assistant.vision.download_pretrained import build_hf_to_kb_mapping  # noqa: E402
from ps1_offline_assistant.vision.inference import (  # noqa: E402
    HuggingFaceLocalVisionBackend,
    MissingBackendDependencyError,
    PyTorchVisionBackend,
    VisionPipeline,
)
from ps1_offline_assistant.vision.train_and_export import (  # noqa: E402
    DEFAULT_KB_PATH,
    build_class_labels_from_kb,
    build_vision_model,
)


class TestPhase7WebAppV2AndFertilizer(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        classes = build_class_labels_from_kb(DEFAULT_KB_PATH)
        model = build_vision_model(num_classes=len(classes))
        # Simulate untrained smoke backend so /api/health reports vision_weights_trained=False
        v_backend = PyTorchVisionBackend(model, is_untrained_smoke=True)
        retriever = OfflineKBRetriever()
        orchestrator = FieldAssistantOrchestrator(
            vision_pipeline=VisionPipeline(v_backend, classes),
            advisory_engine=AdvisoryEngine(TestStubAdvisoryBackend(), retriever),
            kb_retriever=retriever,
            log_path=Path(self._tmpdir.name) / "scans.jsonl",
        )
        self.server = create_offline_server("127.0.0.1", 0, orchestrator)
        self.host, self.port = self.server.server_address
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

        # Probe loopback TCP using http.client (allowed by TestZeroCloudCompliance)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=2)
        try:
            conn.connect()
        except PermissionError:
            self.server.shutdown()
            self.server.server_close()
            self._thread.join(timeout=5)
            self._tmpdir.cleanup()
            self.skipTest("Loopback TCP blocked in this environment (sandbox/CI restriction)")
        finally:
            conn.close()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)
        self._tmpdir.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: Dict[str, str] | None = None,
    ) -> Tuple[int, str, bytes]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            raw = resp.read()
            ctype = resp.getheader("Content-Type", "")
            return resp.status, ctype, raw
        finally:
            conn.close()

    def test_zero_network_three_tab_ui_and_health(self) -> None:
        st, _, raw_html = self._request("GET", "/")
        self.assertEqual(st, 200)
        html = raw_html.decode("utf-8")

        self.assertNotRegex(html, re.compile(r"https?://", re.IGNORECASE))
        self.assertNotRegex(html, re.compile(r'<(?:script|link|img)[^>]+(?:src|href)=["\']//', re.IGNORECASE))

        for marker in (
            "panel-disease",
            "panel-recommend",
            "panel-fertilizer",
            "/api/recommend-crop",
            "/api/fertilizer-guide",
            "untrainedBanner",
            "advisoryModeBanner",
            "symptomsBlock",
            "analyzeLoadingBox",
        ):
            self.assertIn(marker, html)

        hst, _, hraw = self._request("GET", "/api/health")
        self.assertEqual(hst, 200)
        health = json.loads(hraw.decode("utf-8"))
        self.assertFalse(health["vision_weights_trained"])
        self.assertTrue(health["advisory_is_stub"])
        self.assertFalse(health["llama_executorch_active"])
        self.assertTrue(health["host_platform"])
        self.assertTrue(health["v2_crop_engine_ready"])
        self.assertEqual(health["crops_count"], 16)

    def test_recommend_crop_v2_punjab_june_loam_and_validation(self) -> None:
        # 1. Valid Kharif June Punjab query with SHC test
        req_payload = {
            "state": "Punjab",
            "sowing_month": 6,
            "irrigation_access": "Tubewell_Borewell",
            "soil_texture": "Loamy",
            "soil_ph": 6.8,
            "temperature_mean": 28.0,
            "relative_humidity": 72.0,
            "has_soil_test": True,
            "nutrient_basis": "SHC_Available_kg_ha",
            "soil_nitrogen": 80.0,
            "soil_phosphorus": 40.0,
            "soil_potassium": 40.0,
            "language": "hi",
        }
        st, _, raw = self._request(
            "POST",
            "/api/recommend-crop",
            body=json.dumps(req_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(st, 200)
        data = json.loads(raw.decode("utf-8"))
        self.assertTrue(data["offline_mode"])
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["ui_labels"]["language_code"], "hi")

        rec_ids = {r["crop_id"] for r in data["recommendations"]}
        excl_ids = {e["crop_id"] for e in data["excluded_candidates"]}
        self.assertTrue({"rice", "maize", "cotton"}.intersection(rec_ids))
        self.assertIn("wheat", excl_ids)  # Rabi wheat excluded for June sowing

        top_rec = data["recommendations"][0]
        self.assertEqual(top_rec["fertilizer_prescription"]["status"], "BASELINE_RDF_ARITHMETIC")
        self.assertIn("dap_kg_ha", top_rec["fertilizer_prescription"]["commercial_products"])
        self.assertIn("urea_supplementary_kg_ha", top_rec["fertilizer_prescription"]["commercial_products"])

        # 2. Missing mandatory state or sowing_month -> 400 Bad Request
        for bad_payload in (
            {"sowing_month": 6},
            {"state": "Punjab"},
            {"state": "Punjab", "sowing_month": "6"},
            {"state": "Punjab", "sowing_month": 13},
            {"state": "Punjab", "sowing_month": 6, "has_soil_test": "true"},
        ):
            bad_st, _, _ = self._request(
                "POST",
                "/api/recommend-crop",
                body=json.dumps(bad_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(bad_st, 400)

    def test_fertilizer_guide_scaling_and_validation(self) -> None:
        st, _, raw = self._request("GET", "/api/fertilizer-guide?crop=rice&area_ha=2.0&lang=gu")
        self.assertEqual(st, 200)
        data = json.loads(raw.decode("utf-8"))
        self.assertTrue(data["offline_mode"])
        self.assertEqual(data["crop_id"], "rice")
        self.assertEqual(data["area_ha"], 2.0)
        self.assertEqual(data["ui_labels"]["language_code"], "gu")

        urea = next(f for f in data["fertilizer_schedule"] if f["name"] == "Urea")
        self.assertTrue(urea["scaling_applied"])
        self.assertEqual(urea["scaled_min"], 200.0)
        self.assertEqual(urea["scaled_max"], 240.0)
        self.assertIn("200–240 kg", urea["scaled_dosage"])

        # Tomato includes Calcium Nitrate in g/L (foliar concentration must NOT scale by area)
        st_tom, _, raw_tom = self._request("GET", "/api/fertilizer-guide?crop=tomato&area_ha=3.0")
        self.assertEqual(st_tom, 200)
        tom_data = json.loads(raw_tom.decode("utf-8"))
        ca_nitrate = next(f for f in tom_data["fertilizer_schedule"] if "Calcium" in f["name"])
        self.assertFalse(ca_nitrate["scaling_applied"])
        self.assertIn("5 g/L", ca_nitrate["scaled_dosage"])

        # Error cases
        self.assertEqual(self._request("GET", "/api/fertilizer-guide")[0], 400)
        self.assertEqual(self._request("GET", "/api/fertilizer-guide?crop=rice&area_ha=0")[0], 400)
        self.assertEqual(self._request("GET", "/api/fertilizer-guide?crop=dragonfruit&area_ha=1")[0], 404)


class TestPretrainedPlantVillageMappingAndBackend(unittest.TestCase):
    def test_hf_label_mapping_offline(self) -> None:
        id2label = {
            "0": "Tomato___Early_blight",
            "1": "Potato___Late_blight",
            "2": "Corn (maize) with Northern Leaf Blight",
            "3": "Tomato___healthy",
            "4": "Apple___Apple_scab",
            "5": "Healthy Potato",
            "6": "Apple Scab",
            "7": "Tomato with Spider Mites or Two-spotted Spider Mite",
            "8": "Healthy Corn (Maize) Plant",
        }
        mapping = build_hf_to_kb_mapping(id2label)
        self.assertEqual(mapping["num_classes"], 9)
        self.assertEqual(mapping["mapped_kb_cards_count"], 3)

        classes = {c["index"]: c for c in mapping["classes"]}
        self.assertEqual(classes[0]["kb_card_id"], "tomato_early_blight")
        self.assertEqual(classes[1]["kb_card_id"], "potato_late_blight")
        self.assertEqual(classes[2]["kb_card_id"], "maize_northern_leaf_blight")
        self.assertEqual(classes[3]["category"], "healthy")
        self.assertEqual(classes[3]["crop"], "tomato")
        self.assertEqual(classes[4]["kb_card_id"], None)
        self.assertEqual(classes[4]["crop"], "apple")
        self.assertEqual(classes[5]["category"], "healthy")
        self.assertEqual(classes[5]["crop"], "potato")
        self.assertEqual(classes[6]["crop"], "apple")
        self.assertEqual(classes[7]["crop"], "tomato")
        self.assertEqual(classes[7]["category"], "pest")
        self.assertEqual(classes[8]["crop"], "maize")
        self.assertEqual(classes[8]["category"], "healthy")

    def test_hf_local_backend_missing_dir_and_strict_offline_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_dir = Path(tmpdir) / "does_not_exist"
            with self.assertRaises(MissingBackendDependencyError):
                HuggingFaceLocalVisionBackend(missing_dir)

            # Create synthetic local cache layout and verify local_files_only=True is enforced
            cache_dir = Path(tmpdir) / "pretrained_hf"
            cache_dir.mkdir()
            (cache_dir / "config.json").write_text('{"id2label": {"0": "Tomato___Early_blight", "1": "Tomato___healthy"}}', encoding="utf-8")
            (cache_dir / "model.safetensors").write_bytes(b"stub-safetensors")

            recorded_calls = []

            class _FakeProcessor:
                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    recorded_calls.append(("processor", path, kwargs))
                    return cls()

                def __call__(self, images=None, return_tensors="pt"):
                    return {"pixel_values": torch.zeros(1, 3, 224, 224)}

            class _FakeConfig:
                id2label = {0: "Tomato___Early_blight", 1: "Tomato___healthy"}

            class _FakeModel:
                config = _FakeConfig()

                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    recorded_calls.append(("model", path, kwargs))
                    return cls()

                def eval(self):
                    return self

                def __call__(self, **kwargs):
                    out = types.SimpleNamespace()
                    out.logits = torch.tensor([[5.0, -2.0]], dtype=torch.float32)
                    return out

            fake_tf = types.SimpleNamespace(
                AutoImageProcessor=_FakeProcessor,
                AutoModelForImageClassification=_FakeModel,
            )
            with patch.dict(sys.modules, {"transformers": fake_tf}):
                backend = HuggingFaceLocalVisionBackend(cache_dir)
                self.assertEqual(backend.backend_name, "huggingface_local_offline")
                self.assertFalse(backend.is_untrained_smoke)
                self.assertEqual(len(recorded_calls), 2)
                for kind, _, kwargs in recorded_calls:
                    self.assertTrue(
                        kwargs.get("local_files_only") is True,
                        f"{kind} must pass local_files_only=True",
                    )

                pipeline = VisionPipeline(backend)
                img = np.full((240, 240, 3), (35, 165, 45), dtype=np.uint8)
                img[0:20, 0:20] = (200, 200, 200)
                logits = backend.predict_bgr_logits(img)
                self.assertEqual(logits.shape, (2,))
                self.assertEqual(pipeline.classes[0]["kb_card_id"], "tomato_early_blight")

    def test_hf_local_backend_pt2_fallback_without_transformers(self) -> None:
        class _TinyPlantModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor([[6.0, -3.0]], dtype=torch.float32))

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                pooled = pixel_values.mean(dim=(2, 3)) * 0.0
                return pooled[:, :2] + self.bias

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "pretrained_hf"
            cache_dir.mkdir()
            (cache_dir / "config.json").write_text(
                json.dumps({"id2label": {"0": "Tomato___Early_blight", "1": "Tomato___healthy"}}),
                encoding="utf-8",
            )
            (cache_dir / "preprocessor_config.json").write_text(
                json.dumps(
                    {
                        "crop_size": {"height": 224, "width": 224},
                        "do_center_crop": True,
                        "size": {"shortest_edge": 256},
                        "image_mean": [0.5, 0.5, 0.5],
                        "image_std": [0.5, 0.5, 0.5],
                    }
                ),
                encoding="utf-8",
            )
            pt2_path = cache_dir / "plantvillage_mobilenetv2.pt2"
            exported = torch.export.export(
                _TinyPlantModel().eval(),
                (torch.randn(1, 3, 224, 224, dtype=torch.float32),),
            )
            torch.export.save(exported, pt2_path)

            with patch.dict(sys.modules, {"transformers": None}):
                backend = HuggingFaceLocalVisionBackend(cache_dir)
                self.assertTrue(backend._use_pt2_fallback)
                self.assertEqual(backend.backend_name, "huggingface_local_offline")
                self.assertFalse(backend.is_untrained_smoke)

                rng = np.random.default_rng(7)
                leaf = rng.integers(45, 165, (256, 256, 3), dtype=np.uint8)
                leaf[:, :, 1] = np.clip(leaf[:, :, 1] + 45, 0, 255)
                finding = VisionPipeline(backend).predict(leaf)
                self.assertEqual(finding.crop, "tomato")
                self.assertEqual(finding.condition_name, "Tomato Early Blight")
                self.assertGreater(finding.confidence, 0.9)
                self.assertEqual(finding.runtime_backend, "huggingface_local_offline")


if __name__ == "__main__":
    unittest.main(verbosity=2)
