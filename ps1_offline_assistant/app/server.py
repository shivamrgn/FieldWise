"""
100% Offline Local HTTP Server for FieldWise PS1 (Arm Edge / RPi 5 / Hotspot UI).

Uses Python's built-in `http.server.ThreadingHTTPServer` (zero extra web
framework dependencies, zero outbound network calls).

Endpoints:
  - `GET /`                    -> Serves self-contained `static/index.html`
  - `GET /api/health`          -> Returns active backends, KB card count, offline_mode=True
  - `GET /api/ui-bundle`       -> Returns localized UI strings for `en|hi|gu|mr`
  - `GET /api/fertilizer-guide`-> Returns area-scaled fertilizer & crop irrigation schedule
  - `POST /api/analyze`        -> Runs Pipeline A + KB + Pipeline B, caches session
  - `POST /api/switch-language`-> Re-runs Pipeline B + i18n only (reuses VisionFinding)
  - `POST /api/recommend-crop` -> Runs frozen FieldWise V2 CropRecommendationEngine offline
  - `GET /api/history`         -> Returns locally logged scans via `read_field_history`
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import re
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ml_pipeline.recommendation_engine.adapter_cli import (  # noqa: E402
    AdapterValidationError,
    build_query,
    serialize_result,
)
from ml_pipeline.recommendation_engine.engine import CropRecommendationEngine  # noqa: E402
from ml_pipeline.recommendation_engine.input_validator import InputValidationError  # noqa: E402
from ml_pipeline.recommendation_engine.schemas import ExecutionStatus  # noqa: E402
from ps1_offline_assistant.advisory.engine import AdvisoryEngine  # noqa: E402
from ps1_offline_assistant.i18n.translations import (  # noqa: E402
    SUPPORTED_LANGUAGES,
    get_ui_bundle,
)
from ps1_offline_assistant.kb.retriever import OfflineKBRetriever  # noqa: E402
from ps1_offline_assistant.orchestrator.cli import (  # noqa: E402
    _resolve_advisory_backend,
    _resolve_vision_backend,
)
from ps1_offline_assistant.orchestrator.pipeline import (  # noqa: E402
    AssistantResult,
    FieldAssistantOrchestrator,
    read_field_history,
)
from ps1_offline_assistant.vision.inference import VisionPipeline  # noqa: E402
from ps1_offline_assistant.vision.train_and_export import build_class_labels_from_kb  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024  # 16 MiB
MAX_CACHED_SESSIONS = 128


def _decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decodes raw encoded image bytes (JPEG/PNG/WEBP) into an OpenCV BGR array."""
    if not raw_bytes:
        raise ValueError("Empty image payload.")
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Unable to decode image bytes with OpenCV.")
    return image_bgr


def _decode_base64_image(image_base64: str) -> np.ndarray:
    """Decodes a base64 string (with optional Data URI prefix) into OpenCV BGR."""
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ValueError("`image_base64` must be a non-empty string.")
    raw = image_base64.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid base64 image encoding: {exc}") from exc
    return _decode_image_bytes(decoded)


_HA_DOSAGE_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)(?:\s*[–\-]\s*([0-9]+(?:\.[0-9]+)?))?\s*([a-zA-Z]+)\s*/\s*ha(?:\s*/\s*([a-zA-Z]+))?(.*)$",
    re.IGNORECASE,
)


def _format_num(val: float) -> str:
    rounded = round(val, 2)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _scale_fertilizer_dosage(dosage_str: str, area_ha: float) -> Dict[str, Any]:
    """
    Scales per-hectare fertilizer dosages accurately by `area_ha`.
    Preserves non-area rates (foliar g/L, seed-treatment g/10 kg seed, kg/tree/year)
    without unsafe multiplication.
    """
    match = _HA_DOSAGE_RE.match(dosage_str.strip())
    if not match:
        return {
            "base_dosage": dosage_str,
            "scaled_dosage": f"{dosage_str} (rate unchanged; non-hectare basis)",
            "scaled_min": None,
            "scaled_max": None,
            "unit": None,
            "scaling_applied": False,
        }

    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) is not None else low
    unit = match.group(3)
    per_period = match.group(4)
    suffix = match.group(5).strip()

    s_low = round(low * area_ha, 2)
    s_high = round(high * area_ha, 2)
    qty_str = (
        f"{_format_num(s_low)}–{_format_num(s_high)}"
        if match.group(2) is not None
        else _format_num(s_low)
    )
    period_part = f"/{per_period}" if per_period else ""
    extra_part = f" {suffix}" if suffix else ""
    scaled_text = f"{qty_str} {unit}{period_part}{extra_part} (for {_format_num(area_ha)} ha)"

    return {
        "base_dosage": dosage_str,
        "scaled_dosage": scaled_text,
        "scaled_min": s_low,
        "scaled_max": s_high,
        "unit": f"{unit}{period_part}",
        "scaling_applied": True,
    }


class OfflineAssistantHTTPServer(ThreadingHTTPServer):
    """Stateful local HTTP server holding the orchestrator and bounded session cache."""

    daemon_threads = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        orchestrator: FieldAssistantOrchestrator,
        static_dir: Path = DEFAULT_STATIC_DIR,
    ) -> None:
        self.orchestrator = orchestrator
        self.crop_engine = CropRecommendationEngine()
        self.static_dir = Path(static_dir)
        self.index_html_path = self.static_dir / "index.html"
        if not self.index_html_path.is_file():
            raise FileNotFoundError(
                f"Offline UI index.html not found at {self.index_html_path}"
            )

        self._lock = threading.Lock()
        self._sessions: Dict[str, AssistantResult] = {}
        super().__init__(server_address, _OfflineRequestHandler)

    def cache_session(self, result: AssistantResult) -> None:
        with self._lock:
            if result.session_id not in self._sessions and len(self._sessions) >= MAX_CACHED_SESSIONS:
                oldest_key = next(iter(self._sessions))
                del self._sessions[oldest_key]
            self._sessions[result.session_id] = result

    def get_session(self, session_id: str) -> Optional[AssistantResult]:
        with self._lock:
            return self._sessions.get(session_id)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def run_analyze(
        self,
        image_bgr: np.ndarray,
        state_region: str,
        language: str,
        crop_hint: Optional[str] = None,
    ) -> AssistantResult:
        with self._lock:
            result = self.orchestrator.analyze_image(
                image_bgr=image_bgr,
                state_region=state_region,
                language=language,
                crop_hint=crop_hint,
            )
            if len(self._sessions) >= MAX_CACHED_SESSIONS:
                oldest_key = next(iter(self._sessions))
                del self._sessions[oldest_key]
            self._sessions[result.session_id] = result
            return result

    def run_switch_language(
        self,
        session_id: str,
        new_language: str,
        state_region: Optional[str] = None,
    ) -> Optional[AssistantResult]:
        with self._lock:
            prev = self._sessions.get(session_id)
            if prev is None:
                return None
            updated = self.orchestrator.switch_language(
                previous_result=prev,
                new_language=new_language,
                state_region=state_region,
            )
            self._sessions[session_id] = updated
            return updated

    def run_recommend_crop(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes the frozen offline FieldWise V2 CropRecommendationEngine read-only."""
        if not isinstance(payload, dict):
            raise AdapterValidationError("Request payload must be a JSON object.")

        language = str(payload.get("language", "en")).strip() or "en"
        # Strip empty-string optional entries while strictly preserving explicit
        # booleans (False), zero numbers (0/0.0), and mandatory fields for V2 checks.
        cleaned_payload: Dict[str, Any] = {}
        for k, v in payload.items():
            if k == "language":
                continue
            if v is None:
                continue
            if isinstance(v, str) and not v.strip() and k not in {"state"}:
                continue
            cleaned_payload[k] = v

        query = build_query(cleaned_payload)
        with self._lock:
            v2_result = self.crop_engine.recommend(query)

        if v2_result.status == ExecutionStatus.VALIDATION_ERROR:
            err_msg = v2_result.diagnostics.get("error", "Crop engine validation failed.")
            raise AdapterValidationError(err_msg)

        serialized = serialize_result(v2_result)
        serialized["offline_mode"] = True
        serialized["ui_labels"] = get_ui_bundle(language)
        return serialized


class _OfflineRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving the offline frontend and local JSON APIs."""

    server: OfflineAssistantHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, payload: Any) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_bytes: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html_bytes)

    def _read_body_bytes(self) -> Optional[bytes]:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            self._send_json(
                HTTPStatus.LENGTH_REQUIRED,
                {"error": "Content-Length header is required."},
            )
            return None
        try:
            content_len = int(raw_len)
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length header."})
            return None

        if content_len <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Empty request body."})
            return None
        if content_len > MAX_REQUEST_BODY_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"Request exceeds {MAX_REQUEST_BODY_BYTES} byte limit."},
            )
            return None

        return self.rfile.read(content_len)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            html_bytes = self.server.index_html_path.read_bytes()
            self._send_html(html_bytes)
            return

        if route == "/api/health":
            import platform

            orch = self.server.orchestrator
            raw_crops = orch.kb_retriever.raw_data.get("crops", [])
            is_untrained = bool(
                getattr(orch.vision_pipeline.backend, "is_untrained_smoke", False)
            )
            adv_name = orch.advisory_engine.backend.backend_name
            num_classes = len(orch.vision_pipeline.classes)
            payload = {
                "status": "ok",
                "offline_mode": True,
                "host_platform": f"{platform.system()}-{platform.machine()}",
                "vision_backend": orch.vision_pipeline.backend.backend_name,
                "vision_weights_trained": not is_untrained,
                "vision_num_classes": num_classes,
                "vision_model_info": (
                    f"MobileNetV2 PlantVillage ({num_classes} classes, cached offline)"
                    if orch.vision_pipeline.backend.backend_name == "huggingface_local_offline"
                    else (
                        f"Untrained MobileNetV3 Smoke ({num_classes} classes)"
                        if is_untrained
                        else f"Trained Local Model ({num_classes} classes)"
                    )
                ),
                "advisory_backend": adv_name,
                "advisory_is_stub": adv_name == "test_stub_ci_only",
                "llama_executorch_active": adv_name.startswith("executorch"),
                "v2_crop_engine_ready": True,
                "supported_languages": list(SUPPORTED_LANGUAGES),
                "kb_cards_count": len(orch.kb_retriever.list_disease_pest_cards()),
                "crops_count": len(raw_crops),
                "available_crops": [
                    {"id": c["id"], "name": c["name"], "season": c["season"]}
                    for c in raw_crops
                ],
                "cached_sessions": self.server.session_count(),
            }
            self._send_json(HTTPStatus.OK, payload)
            return

        if route == "/api/ui-bundle":
            params = parse_qs(parsed.query)
            lang = str(params.get("lang", ["en"])[0]).strip() or "en"
            self._send_json(HTTPStatus.OK, get_ui_bundle(lang))
            return

        if route == "/api/fertilizer-guide":
            self._handle_fertilizer_guide(parsed.query)
            return

        if route == "/api/history":
            log_path = self.server.orchestrator.log_path
            if log_path is None:
                self._send_json(
                    HTTPStatus.OK,
                    {"log_configured": False, "records": []},
                )
                return
            records = read_field_history(log_path)
            self._send_json(
                HTTPStatus.OK,
                {
                    "log_configured": True,
                    "log_path": str(log_path),
                    "records": records,
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown GET endpoint: {route}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/analyze":
            self._handle_analyze(parsed.query)
            return

        if route == "/api/switch-language":
            self._handle_switch_language()
            return

        if route == "/api/recommend-crop":
            self._handle_recommend_crop()
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown POST endpoint: {route}"})

    def _handle_analyze(self, query_string: str) -> None:
        raw_body = self._read_body_bytes()
        if raw_body is None:
            return

        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        params = parse_qs(query_string)

        try:
            if content_type in ("application/json", ""):
                data = json.loads(raw_body.decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("JSON payload must be an object.")
                image_b64 = data.get("image_base64")
                state_region = str(data.get("state_region", "")).strip()
                language = str(data.get("language", "en")).strip() or "en"
                crop_hint = str(data.get("crop_hint", "")).strip() or None
                image_bgr = _decode_base64_image(image_b64)
            elif content_type.startswith("image/") or content_type == "application/octet-stream":
                state_region = str(params.get("state_region", [""])[0]).strip()
                language = str(params.get("language", ["en"])[0]).strip() or "en"
                crop_hint = str(params.get("crop_hint", [""])[0]).strip() or None
                image_bgr = _decode_image_bytes(raw_body)
            else:
                self._send_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"error": f"Unsupported Content-Type: {content_type}"},
                )
                return

            if not state_region:
                raise ValueError("`state_region` is required.")

            result = self.server.run_analyze(
                image_bgr=image_bgr,
                state_region=state_region,
                language=language,
                crop_hint=crop_hint,
            )
            self._send_json(HTTPStatus.OK, result.model_dump(mode="json"))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Unexpected error during /api/analyze")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_switch_language(self) -> None:
        raw_body = self._read_body_bytes()
        if raw_body is None:
            return

        try:
            data = json.loads(raw_body.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON payload must be an object.")
            session_id = str(data.get("session_id", "")).strip()
            new_language = str(data.get("new_language", "")).strip()
            state_region = data.get("state_region")
            if state_region is not None:
                state_region = str(state_region).strip() or None

            if not session_id:
                raise ValueError("`session_id` is required.")
            if not new_language:
                raise ValueError("`new_language` is required.")

            updated = self.server.run_switch_language(
                session_id=session_id,
                new_language=new_language,
                state_region=state_region,
            )
            if updated is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"Session '{session_id}' not found in local cache."},
                )
                return

            self._send_json(HTTPStatus.OK, updated.model_dump(mode="json"))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Unexpected error during /api/switch-language")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_recommend_crop(self) -> None:
        raw_body = self._read_body_bytes()
        if raw_body is None:
            return

        try:
            data = json.loads(raw_body.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON payload must be an object.")
            result_payload = self.server.run_recommend_crop(data)
            self._send_json(HTTPStatus.OK, result_payload)
        except (AdapterValidationError, InputValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Unexpected error during /api/recommend-crop")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_fertilizer_guide(self, query_string: str) -> None:
        params = parse_qs(query_string)
        crop_raw = str(params.get("crop", [""])[0]).strip()
        if not crop_raw:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Query parameter 'crop' is required (e.g., ?crop=rice&area_ha=1.0)."},
            )
            return

        raw_area = str(params.get("area_ha", ["1.0"])[0]).strip()
        try:
            area_ha = float(raw_area)
        except ValueError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"Invalid numeric 'area_ha': {raw_area!r}."},
            )
            return

        if not math.isfinite(area_ha) or area_ha <= 0.0 or area_ha > 10000.0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"'area_ha' must be a finite number in (0, 10000], got {area_ha!r}."},
            )
            return

        lang = str(params.get("lang", ["en"])[0]).strip() or "en"
        retriever = self.server.orchestrator.kb_retriever
        crop_profile = retriever.get_crop_profile(crop_raw)
        fert_schedule = retriever.get_fertilizer_schedule(crop_raw)

        if crop_profile is None or fert_schedule is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"No offline fertilizer/crop entry found for '{crop_raw}' in agri_kb.json."},
            )
            return

        scaled_schedule: List[Dict[str, Any]] = []
        for entry in fert_schedule:
            scaling = _scale_fertilizer_dosage(str(entry["dosage"]), area_ha)
            scaled_schedule.append(
                {
                    "name": entry["name"],
                    "nutrients": entry["nutrients"],
                    "applicationTime": entry["applicationTime"],
                    **scaling,
                }
            )

        payload = {
            "offline_mode": True,
            "crop_id": crop_profile["id"],
            "crop_name": crop_profile["name"],
            "area_ha": round(area_ha, 4),
            "fertilizer_schedule": scaled_schedule,
            "crop_profile": crop_profile,
            "safety_disclaimer": retriever.raw_data.get("provenance", {}).get(
                "fertilizer_safety_policy",
                "Baseline reference only; verify Soil Health Card (SHC) readings before application.",
            ),
            "ui_labels": get_ui_bundle(lang),
        }
        self._send_json(HTTPStatus.OK, payload)


def create_offline_server(
    host: str,
    port: int,
    orchestrator: FieldAssistantOrchestrator,
    static_dir: Path = DEFAULT_STATIC_DIR,
) -> OfflineAssistantHTTPServer:
    """Creates an `OfflineAssistantHTTPServer` bound to `(host, port)`."""
    return OfflineAssistantHTTPServer(
        server_address=(host, int(port)),
        orchestrator=orchestrator,
        static_dir=static_dir,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch 100% offline FieldWise PS1 local web application."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--vision-backend",
        choices=["pytorch", "pt2", "onnx", "executorch", "hailort", "hf-local"],
        default="pytorch",
    )
    parser.add_argument("--vision-model", type=Path, default=None)
    parser.add_argument("--class-labels", type=Path, default=None)
    parser.add_argument("--allow-untrained-smoke", action="store_true")
    parser.add_argument(
        "--advisory-backend",
        choices=["executorch-worker", "executorch-cli", "executorch-py", "test-stub"],
        default="executorch-cli",
    )
    parser.add_argument("--llama-bin", type=Path, default=None)
    parser.add_argument("--llama-pte", type=Path, default=None)
    parser.add_argument("--llama-tokenizer", type=Path, default=None)
    parser.add_argument("--llama-python", type=Path, default=None)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=_PS1_ROOT / "data" / "field_scans.jsonl",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    class_labels: Any = None
    if args.class_labels is not None:
        class_labels = args.class_labels
        num_classes = len(json.loads(args.class_labels.read_text(encoding="utf-8"))["classes"])
    else:
        kb_classes = build_class_labels_from_kb()
        num_classes = len(kb_classes)
        if args.vision_backend != "hf-local":
            class_labels = kb_classes

    vision_backend = _resolve_vision_backend(
        backend_kind=args.vision_backend,
        model_path=args.vision_model,
        num_classes=num_classes,
        allow_untrained_smoke=args.allow_untrained_smoke,
    )
    advisory_backend = _resolve_advisory_backend(
        backend_kind=args.advisory_backend,
        llama_bin=args.llama_bin,
        llama_pte=args.llama_pte,
        llama_tokenizer=args.llama_tokenizer,
        llama_python=args.llama_python,
    )

    retriever = OfflineKBRetriever()
    orchestrator = FieldAssistantOrchestrator(
        vision_pipeline=VisionPipeline(vision_backend, class_labels),
        advisory_engine=AdvisoryEngine(advisory_backend, retriever),
        kb_retriever=retriever,
        log_path=args.log_path,
    )

    server = create_offline_server(args.host, args.port, orchestrator)
    actual_host, actual_port = server.server_address
    print(
        f"FieldWise PS1 Offline Assistant listening on http://{actual_host}:{actual_port}/ "
        f"(vision={vision_backend.backend_name}, advisory={advisory_backend.backend_name}, offline=True)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down FieldWise PS1 Offline Server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
