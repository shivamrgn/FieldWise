"""
Phase 5 Verification Suite: 100% Offline Local HTTP Server & Zero-Network UI.

Spins up `OfflineAssistantHTTPServer` on an ephemeral localhost port (`127.0.0.1:0`)
in a daemon thread and verifies:
  1. `GET /` serves `index.html` (200 OK) with ZERO `http://` or `https://` URLs
     and ZERO external CDN/font/script links.
  2. `GET /api/health` returns `offline_mode == True`, active `vision_backend`,
     `advisory_backend`, `supported_languages`, and `kb_cards_count`.
  3. `POST /api/analyze` accepts base64 (and raw image bytes), returns valid
     `AssistantResult` JSON with `session_id`, `vision_finding`, `advisory_response`,
     and `ui_labels`.
  4. `POST /api/switch-language` switches `en -> hi -> gu` via `session_id`
     without re-running Vision Pipeline A (`vision_call_count == 1`).
  5. `GET /api/history` returns persisted local JSONL scan records.
"""

from __future__ import annotations

import base64
import http.client
import json
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.advisory.engine import AdvisoryEngine  # noqa: E402
from ps1_offline_assistant.advisory.runtime import TestStubAdvisoryBackend  # noqa: E402
from ps1_offline_assistant.app.server import create_offline_server  # noqa: E402
from ps1_offline_assistant.kb.retriever import OfflineKBRetriever  # noqa: E402
from ps1_offline_assistant.orchestrator.pipeline import (  # noqa: E402
    AssistantResult,
    FieldAssistantOrchestrator,
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
)


class _CountingAppVisionBackend(VisionBackend):
    def __init__(self, logits: np.ndarray) -> None:
        self._logits = np.asarray(logits, dtype=np.float32)
        self.call_count = 0

    @property
    def backend_name(self) -> str:
        return "counting_app_vision"

    def predict_logits(self, input_nchw: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return self._logits


class TestOfflineWebApp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmpdir.name) / "app_scans.jsonl"

        classes = build_class_labels_from_kb(DEFAULT_KB_PATH)
        rice_blast_idx = next(c["index"] for c in classes if c["label"] == "rice_blast")
        logits = np.full(len(classes), -5.0, dtype=np.float32)
        logits[rice_blast_idx] = 5.0

        self.vision_backend = _CountingAppVisionBackend(logits)
        retriever = OfflineKBRetriever()
        orchestrator = FieldAssistantOrchestrator(
            vision_pipeline=VisionPipeline(self.vision_backend, classes),
            advisory_engine=AdvisoryEngine(TestStubAdvisoryBackend(), retriever),
            kb_retriever=retriever,
            log_path=self.log_path,
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

    def test_index_html_zero_network_audit(self) -> None:
        status, ctype, raw = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)

        html = raw.decode("utf-8")
        self.assertIn("OFFLINE MODE", html)
        self.assertIn("/api/analyze", html)
        self.assertIn("/api/switch-language", html)

        # Strict Zero-Network Audit: ZERO external http://, https://, or //cdn URLs
        self.assertNotRegex(html, re.compile(r"https?://", re.IGNORECASE))
        self.assertNotRegex(html, re.compile(r'<(?:script|link|img)[^>]+(?:src|href)=["\']//', re.IGNORECASE))

    def test_health_analyze_switch_language_and_history_flow(self) -> None:
        # 1. GET /api/health
        h_status, _, h_raw = self._request("GET", "/api/health")
        self.assertEqual(h_status, 200)
        health = json.loads(h_raw.decode("utf-8"))
        self.assertTrue(health["offline_mode"])
        self.assertEqual(health["vision_backend"], "counting_app_vision")
        self.assertEqual(health["advisory_backend"], "test_stub_ci_only")
        self.assertEqual(health["supported_languages"], ["en", "hi", "gu", "mr", "te", "ta"])
        self.assertGreaterEqual(health["kb_cards_count"], 12)

        # 2. POST /api/analyze (JSON base64 payload with Data URI prefix)
        leaf_img = _make_synthetic_leaf(lesion_fraction=0.25)
        ok, png_buf = cv2.imencode(".png", leaf_img)
        self.assertTrue(ok)
        b64_data_uri = "data:image/png;base64," + base64.b64encode(png_buf.tobytes()).decode("ascii")

        analyze_payload = json.dumps(
            {
                "image_base64": b64_data_uri,
                "state_region": "Punjab",
                "language": "en",
            }
        ).encode("utf-8")
        a_status, _, a_raw = self._request(
            "POST",
            "/api/analyze",
            body=analyze_payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(a_status, 200)
        res_en = AssistantResult.model_validate_json(a_raw.decode("utf-8"))
        self.assertEqual(self.vision_backend.call_count, 1)
        self.assertEqual(res_en.advisory_response.language.value, "en")
        self.assertEqual(res_en.ui_labels["language_code"], "en")
        self.assertEqual(res_en.vision_finding.crop, "rice")

        # 3. POST /api/switch-language (en -> hi -> gu) without re-running vision
        for target_lang in ("hi", "gu"):
            sw_payload = json.dumps(
                {
                    "session_id": res_en.session_id,
                    "new_language": target_lang,
                    "state_region": "Punjab",
                }
            ).encode("utf-8")
            sw_status, _, sw_raw = self._request(
                "POST",
                "/api/switch-language",
                body=sw_payload,
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(sw_status, 200)
            res_sw = AssistantResult.model_validate_json(sw_raw.decode("utf-8"))
            self.assertEqual(res_sw.session_id, res_en.session_id)
            self.assertEqual(res_sw.advisory_response.language.value, target_lang)
            self.assertEqual(res_sw.ui_labels["language_code"], target_lang)
            self.assertEqual(res_sw.vision_finding, res_en.vision_finding)

        # Vision Pipeline A must remain called ONLY ONCE!
        self.assertEqual(self.vision_backend.call_count, 1)

        # 4. Also verify raw binary image upload mode works
        raw_status, _, raw_resp = self._request(
            "POST",
            "/api/analyze?state_region=Gujarat&language=gu",
            body=png_buf.tobytes(),
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(raw_status, 200)
        res_raw = AssistantResult.model_validate_json(raw_resp.decode("utf-8"))
        self.assertEqual(res_raw.advisory_response.language.value, "gu")
        self.assertEqual(self.vision_backend.call_count, 2)

        # 5. GET /api/history returns all 4 logged events (2 scans + 2 language switches)
        hist_status, _, hist_raw = self._request("GET", "/api/history")
        self.assertEqual(hist_status, 200)
        hist_payload: Dict[str, Any] = json.loads(hist_raw.decode("utf-8"))
        self.assertTrue(hist_payload["log_configured"])
        self.assertEqual(len(hist_payload["records"]), 4)
        self.assertEqual(hist_payload["records"][0]["language"], "en")
        self.assertEqual(hist_payload["records"][1]["language"], "hi")
        self.assertEqual(hist_payload["records"][2]["language"], "gu")
        self.assertEqual(hist_payload["records"][3]["state_region"], "Gujarat")

    def test_error_handling_for_unknown_session_and_bad_image(self) -> None:
        bad_sw = json.dumps({"session_id": "missing-id", "new_language": "hi"}).encode("utf-8")
        st_404, _, _ = self._request(
            "POST",
            "/api/switch-language",
            body=bad_sw,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(st_404, 404)

        bad_img = json.dumps(
            {"image_base64": "not-valid-base64!!!", "state_region": "Punjab", "language": "en"}
        ).encode("utf-8")
        st_400, _, _ = self._request(
            "POST",
            "/api/analyze",
            body=bad_img,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(st_400, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
