"""
Offline CLI Entrypoint for FieldWise PS1 Dual-Pipeline Assistant.

Example:
    python3 -m ps1_offline_assistant.orchestrator.cli \\
        --image path/to/leaf.jpg \\
        --state Punjab \\
        --lang hi \\
        --vision-backend pytorch \\
        --vision-model path/to/mobilenet_v3_small.pt2 \\
        --advisory-backend test-stub \\
        --switch-lang gu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.advisory.engine import AdvisoryEngine  # noqa: E402
from ps1_offline_assistant.advisory.runtime import (  # noqa: E402
    DEFAULT_EXECUTORCH_PYTHON,
    DEFAULT_LLAMA_PTE,
    DEFAULT_LLAMA_TOKENIZER,
    ExecuTorchLlamaCliBackend,
    ExecuTorchPythonRuntimeBackend,
    ExecuTorchWorkerAdvisoryBackend,
    LlamaAdvisoryBackend,
    TestStubAdvisoryBackend,
)
from ps1_offline_assistant.i18n.translations import (  # noqa: E402
    localize_condition_name,
    localize_crop_name,
    localize_status_message,
)
from ps1_offline_assistant.kb.retriever import OfflineKBRetriever  # noqa: E402
from ps1_offline_assistant.orchestrator.pipeline import (  # noqa: E402
    AssistantResult,
    FieldAssistantOrchestrator,
)
from ps1_offline_assistant.vision.download_pretrained import (  # noqa: E402
    DEFAULT_PRETRAINED_DIR,
)
from ps1_offline_assistant.vision.inference import (  # noqa: E402
    ExecuTorchVisionBackend,
    HailoRTVisionBackend,
    HuggingFaceLocalVisionBackend,
    OnnxVisionBackend,
    PyTorchVisionBackend,
    VisionBackend,
    VisionPipeline,
)
from ps1_offline_assistant.vision.train_and_export import (  # noqa: E402
    build_class_labels_from_kb,
    build_vision_model,
)


def _resolve_vision_backend(
    backend_kind: str,
    model_path: Optional[Path],
    num_classes: int,
    allow_untrained_smoke: bool,
) -> VisionBackend:
    if backend_kind == "hailort":
        if model_path is None:
            raise ValueError("--vision-model (.hef) is required for hailort backend.")
        return HailoRTVisionBackend(model_path)
    if backend_kind == "executorch":
        if model_path is None:
            raise ValueError("--vision-model (.pte) is required for executorch backend.")
        return ExecuTorchVisionBackend(model_path)
    if backend_kind == "onnx":
        if model_path is None:
            raise ValueError("--vision-model (.onnx) is required for onnx backend.")
        return OnnxVisionBackend(model_path)
    if backend_kind == "hf-local":
        target_dir = model_path if model_path is not None else DEFAULT_PRETRAINED_DIR
        return HuggingFaceLocalVisionBackend(target_dir)
    if backend_kind in ("pytorch", "pt2"):
        if model_path is not None:
            if model_path.suffix.lower() == ".pt2":
                return PyTorchVisionBackend(model_path, is_untrained_smoke=False)
            model = build_vision_model(num_classes=num_classes, weights_path=model_path)
            return PyTorchVisionBackend(model, is_untrained_smoke=False)
        if allow_untrained_smoke:
            model = build_vision_model(num_classes=num_classes, weights_path=None)
            return PyTorchVisionBackend(model, is_untrained_smoke=True)
        raise ValueError(
            "--vision-model (.pt2 or .pt) is required unless --allow-untrained-smoke is passed."
        )
    raise ValueError(f"Unsupported vision backend: {backend_kind}")


def _resolve_advisory_backend(
    backend_kind: str,
    llama_bin: Optional[Path],
    llama_pte: Optional[Path],
    llama_tokenizer: Optional[Path],
    llama_python: Optional[Path] = None,
) -> LlamaAdvisoryBackend:
    if backend_kind == "executorch-worker":
        return ExecuTorchWorkerAdvisoryBackend(
            pte_path=llama_pte or DEFAULT_LLAMA_PTE,
            tokenizer_path=llama_tokenizer or DEFAULT_LLAMA_TOKENIZER,
            python_bin=llama_python or DEFAULT_EXECUTORCH_PYTHON,
        )
    if backend_kind == "executorch-cli":
        if not (llama_bin and llama_pte and llama_tokenizer):
            raise ValueError(
                "--llama-bin, --llama-pte, and --llama-tokenizer are required for executorch-cli. "
                "When running locally without Llama 3.2 .pte weights (e.g., to use the offline "
                "PlantVillage Vision, V2 Crop Recommendation, and Fertilizer tabs), pass: "
                "--advisory-backend test-stub"
            )
        return ExecuTorchLlamaCliBackend(
            binary_path=llama_bin,
            pte_path=llama_pte,
            tokenizer_path=llama_tokenizer,
        )
    if backend_kind == "executorch-py":
        if not (llama_pte and llama_tokenizer):
            raise ValueError(
                "--llama-pte and --llama-tokenizer are required for executorch-py. "
                "Pass --advisory-backend test-stub when running locally without Llama 3.2 .pte weights."
            )
        return ExecuTorchPythonRuntimeBackend(
            pte_path=llama_pte,
            tokenizer_path=llama_tokenizer,
        )
    if backend_kind == "test-stub":
        return TestStubAdvisoryBackend()
    raise ValueError(f"Unsupported advisory backend: {backend_kind}")


def format_localized_summary(result: AssistantResult) -> str:
    """Formats a readable localized terminal report from an `AssistantResult`."""
    ui = result.ui_labels
    headers = ui["headers"]
    vf = result.vision_finding
    adv = result.advisory_response
    meta = result.pipeline_metadata
    lang = adv.language.value

    crop_disp = localize_crop_name(vf.crop, lang)
    cond_disp = localize_condition_name(vf.condition_name, lang)
    sev_disp = ui["severity_badges"].get(vf.severity_level.value, vf.severity_level.value)
    status_banner = localize_status_message(adv.status.value, lang)

    lines = [
        f"=== FieldWise Offline Assistant [{ui['language_name']} ({lang})] ===",
        f"{headers['status']}: {adv.status.value} — {status_banner}",
        f"{headers['crop']}: {crop_disp} | {headers['detected_condition']}: {cond_disp}",
        f"{headers['severity_level']}: {sev_disp} | {headers['infected_area']}: {vf.infected_area_pct:.1f}% | {headers['confidence']}: {vf.confidence:.2f}",
        "",
        f"{headers['explanation']}:",
        f"  {adv.explanation}",
        "",
        f"{headers['immediate_actions']}:",
        *[f"  - {item}" for item in adv.immediate_actions],
        "",
        f"{headers['organic_cultural_control']}:",
        *[f"  - {item}" for item in adv.organic_cultural_steps],
    ]

    if adv.chemical_guidance_if_approved:
        lines.extend(
            [
                "",
                f"{headers['approved_chemical_guidance']}:",
                *[f"  - {item}" for item in adv.chemical_guidance_if_approved],
            ]
        )

    lines.extend(
        [
            "",
            f"{headers['preventive_measures']}:",
            *[f"  - {item}" for item in adv.preventive_measures],
            "",
            f"{headers['kvk_escalation']}: {'YES' if adv.escalate_to_kvk else 'No'} ({ui['kvk_escalation_messages']['helpline_number']})",
            f"{headers['disclaimer']}: {adv.disclaimer}",
            (
                f"[Telemetry] vision={meta.vision_backend} ({meta.vision_latency_ms:.2f} ms), "
                f"advisory={meta.advisory_backend} (TTFT={meta.advisory_ttft_ms:.2f} ms, "
                f"tok/s={meta.advisory_tokens_per_sec:.2f}), "
                f"e2e={meta.end_to_end_latency_ms:.2f} ms, offline={meta.offline_mode}"
            ),
        ]
    )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FieldWise PS1 Offline Dual-Pipeline Assistant CLI."
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to leaf/pest image")
    parser.add_argument("--state", type=str, required=True, help="Farmer state/region (e.g., Punjab)")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Response language (en | hi | gu | mr)",
    )
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
    parser.add_argument("--log-path", type=Path, default=None, help="Optional .jsonl or .db history file")
    parser.add_argument("--switch-lang", type=str, default=None, help="Optional instant switch language")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise ValueError(f"Failed to decode image via OpenCV: {args.image}")

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

    initial_result = orchestrator.analyze_image(
        image_bgr=image_bgr,
        state_region=args.state,
        language=args.lang,
    )

    switched_result: Optional[AssistantResult] = None
    if args.switch_lang:
        switched_result = orchestrator.switch_language(
            previous_result=initial_result,
            new_language=args.switch_lang,
            state_region=args.state,
        )

    if args.json:
        payload: Dict[str, Any] = (
            {
                "initial": initial_result.model_dump(mode="json"),
                "switched": switched_result.model_dump(mode="json"),
            }
            if switched_result is not None
            else initial_result.model_dump(mode="json")
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_localized_summary(initial_result))
        if switched_result is not None:
            print("\n--- Instant Language Switch ---")
            print(format_localized_summary(switched_result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
