"""
On-Device Benchmarking Suite for FieldWise PS1 (Arm CPU / Hailo NPU / Local Host).

Measures reproducible latency, throughput, peak RSS memory, artifact disk
footprint, and optional Raspberry Pi 5 thermal/PMIC telemetry without fabricating
any target hardware numbers.

Example:
    python3 -m ps1_offline_assistant.benchmarks.run_benchmarks \\
        --runs 5 --output-dir ps1_offline_assistant/benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.advisory.engine import AdvisoryEngine  # noqa: E402
from ps1_offline_assistant.advisory.runtime import TestStubAdvisoryBackend  # noqa: E402
from ps1_offline_assistant.kb.retriever import (  # noqa: E402
    DEFAULT_KB_PATH,
    OfflineKBRetriever,
)
from ps1_offline_assistant.orchestrator.cli import (  # noqa: E402
    _resolve_advisory_backend,
    _resolve_vision_backend,
)
from ps1_offline_assistant.orchestrator.pipeline import (  # noqa: E402
    AssistantResult,
    FieldAssistantOrchestrator,
)
from ps1_offline_assistant.schemas.contracts import (  # noqa: E402
    SeverityLevel,
    VisionCategory,
    VisionFinding,
)
from ps1_offline_assistant.vision.download_pretrained import DEFAULT_PRETRAINED_DIR  # noqa: E402
from ps1_offline_assistant.vision.inference import (  # noqa: E402
    VisionBackend,
    VisionPipeline,
)
from ps1_offline_assistant.vision.train_and_export import build_class_labels_from_kb  # noqa: E402

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _make_benchmark_synthetic_leaf(lesion_fraction: float = 0.25, size: int = 320) -> np.ndarray:
    """Generates a deterministic 320x320 BGR synthetic diseased leaf when no --image is passed."""
    img = np.full((size, size, 3), (215, 215, 215), dtype=np.uint8)
    center = (size // 2, size // 2)
    axes = (int(size * 0.34), int(size * 0.22))
    leaf_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(leaf_mask, center, axes, 0, 0, 360, 255, thickness=-1)
    img[leaf_mask > 0] = (35, 165, 45)
    cv2.line(
        img,
        (center[0] - axes[0] + 6, center[1]),
        (center[0] + axes[0] - 6, center[1]),
        (20, 125, 30),
        thickness=3,
    )
    if lesion_fraction > 0.0:
        ys, xs = np.where(leaf_mask > 0)
        target_px = int(round(len(ys) * lesion_fraction))
        order = np.argsort(xs + 0.15 * ys)
        lesion_idx = order[:target_px]
        img[ys[lesion_idx], xs[lesion_idx]] = (30, 68, 128)
    return img


def _percentile(values: List[float], pct: float) -> float:
    """Computes linear-interpolated percentile from a non-empty list of floats."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _summarize_series(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": round(float(statistics.mean(values)), 4),
        "p50": round(_percentile(values, 50.0), 4),
        "p95": round(_percentile(values, 95.0), 4),
    }


def get_peak_rss_mb() -> float:
    """
    Returns peak process resident set size (MB) via `resource.getrusage`.
    Handles macOS (bytes) vs Linux (kilobytes) units accurately.
    """
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(raw_rss / (1024.0 * 1024.0), 3)
    return round(raw_rss / 1024.0, 3)


def read_arm_thermal_and_power() -> Dict[str, Any]:
    """
    Safely collects Raspberry Pi 5 / Arm Linux thermal and PMIC telemetry if available.
    Returns `None` (`null` in JSON) cleanly on macOS or non-RPi hosts.
    """
    cpu_temp_c: Optional[float] = None
    thermal_sysfs = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_sysfs.is_file():
        try:
            raw_milli = thermal_sysfs.read_text(encoding="utf-8").strip()
            cpu_temp_c = round(int(raw_milli) / 1000.0, 2)
        except (OSError, ValueError):
            cpu_temp_c = None

    vcgencmd_temp_c: Optional[float] = None
    vcgencmd_pmic_raw: Optional[str] = None
    vcgencmd_bin = shutil.which("vcgencmd")
    if vcgencmd_bin:
        try:
            temp_proc = subprocess.run(
                [vcgencmd_bin, "measure_temp"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            m = re.search(r"temp=([0-9]+(?:\.[0-9]+)?)", temp_proc.stdout)
            if m:
                vcgencmd_temp_c = float(m.group(1))
        except (subprocess.SubprocessError, OSError, ValueError):
            vcgencmd_temp_c = None

        try:
            pmic_proc = subprocess.run(
                [vcgencmd_bin, "pmic_read_adc"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if pmic_proc.returncode == 0 and pmic_proc.stdout.strip():
                vcgencmd_pmic_raw = pmic_proc.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            vcgencmd_pmic_raw = None

    return {
        "cpu_temp_c": cpu_temp_c,
        "vcgencmd_temp_c": vcgencmd_temp_c,
        "vcgencmd_pmic_raw": vcgencmd_pmic_raw,
    }


def _collect_artifact_footprint(
    kb_path: Path,
    vision_model_path: Optional[Path] = None,
    llama_pte_path: Optional[Path] = None,
) -> Dict[str, Any]:
    kb_bytes = kb_path.stat().st_size if kb_path.is_file() else 0
    if vision_model_path is not None and vision_model_path.is_dir():
        vision_bytes = sum(
            f.stat().st_size for f in vision_model_path.rglob("*") if f.is_file()
        )
    elif vision_model_path is not None and vision_model_path.is_file():
        vision_bytes = vision_model_path.stat().st_size
    else:
        vision_bytes = 0
    llama_bytes = (
        llama_pte_path.stat().st_size
        if llama_pte_path is not None and llama_pte_path.is_file()
        else 0
    )
    total_bytes = kb_bytes + vision_bytes + llama_bytes
    return {
        "agri_kb_bytes": kb_bytes,
        "agri_kb_kb": round(kb_bytes / 1024.0, 2),
        "vision_model_bytes": vision_bytes if vision_bytes > 0 else None,
        "vision_model_mb": round(vision_bytes / (1024.0 * 1024.0), 3) if vision_bytes > 0 else None,
        "llama_pte_bytes": llama_bytes if llama_bytes > 0 else None,
        "llama_pte_mb": round(llama_bytes / (1024.0 * 1024.0), 3) if llama_bytes > 0 else None,
        "total_measured_bytes": total_bytes,
        "total_measured_mb": round(total_bytes / (1024.0 * 1024.0), 3),
    }


def _format_summary_markdown(results: Dict[str, Any]) -> str:
    meta = results["platform_metadata"]
    metrics = results["latency_and_throughput"]
    lang_sw = results["language_switch_benchmark"]
    mem = results["memory_and_storage"]
    therm = results["arm_thermal_and_power"]

    lines = [
        "# FieldWise PS1 On-Device Benchmark Summary",
        "",
        "## Platform & Configuration",
        "",
        "| Field | Value |",
        "| :--- | :--- |",
        f"| Host Platform | `{meta['host_platform']}` (`{meta['os']}` {meta['os_release']}) |",
        f"| CPU Architecture | `{meta['cpu_arch']}` |",
        f"| Python Version | `{meta['python_version']}` |",
        f"| Vision Backend | `{meta['vision_backend']}` ({meta['vision_model_source']}) |",
        f"| Advisory Backend | `{meta['advisory_backend']}` |",
        f"| Is Stub Advisory | `{meta['is_stub_advisory']}` |",
        f"| Warmup / Timed Runs | `{meta['warmup_runs']}` / `{meta['timed_runs']}` |",
        f"| Input Image Source | `{meta['image_source']}` |",
        "",
    ]

    if meta["is_stub_advisory"]:
        lines.extend(
            [
                "> **Honesty Notice**: `is_stub_advisory=true` (`test_stub_ci_only`). Advisory TTFT and "
                "tokens/sec reflect local host Python/stub harness timings, NOT real Llama 3.2 `.pte` "
                "quantized inference on Arm CPU.",
                "",
            ]
        )

    lines.extend(
        [
            "## Pipeline Latency & Throughput",
            "",
            "| Metric | Mean | P50 | P95 |",
            "| :--- | ---: | ---: | ---: |",
            f"| Vision Latency (ms) | {metrics['vision_latency_ms']['mean']:.3f} | {metrics['vision_latency_ms']['p50']:.3f} | {metrics['vision_latency_ms']['p95']:.3f} |",
            f"| Advisory TTFT (ms) | {metrics['advisory_ttft_ms']['mean']:.3f} | {metrics['advisory_ttft_ms']['p50']:.3f} | {metrics['advisory_ttft_ms']['p95']:.3f} |",
            f"| Advisory Speed (tokens/s) | {metrics['advisory_tokens_per_sec']['mean']:.3f} | {metrics['advisory_tokens_per_sec']['p50']:.3f} | {metrics['advisory_tokens_per_sec']['p95']:.3f} |",
            f"| End-to-End Latency (ms) | {metrics['end_to_end_latency_ms']['mean']:.3f} | {metrics['end_to_end_latency_ms']['p50']:.3f} | {metrics['end_to_end_latency_ms']['p95']:.3f} |",
            "",
            "## Instant Language Switch Latency (`en -> hi -> gu -> mr`)",
            "",
            f"- **Vision Re-Executed**: `{lang_sw['vision_re_executed']}` (zero Pipeline A rerun)",
            f"- **Mean Switch Latency**: `{lang_sw['summary_ms']['mean']:.3f} ms` (P50 `{lang_sw['summary_ms']['p50']:.3f} ms`, P95 `{lang_sw['summary_ms']['p95']:.3f} ms`)",
            "",
            "| Transition | Latency (ms) |",
            "| :--- | ---: |",
        ]
    )

    for step_name, val in lang_sw["per_transition_ms"].items():
        lines.append(f"| `{step_name}` | {val:.3f} |")

    lines.extend(
        [
            "",
            "## Memory, Disk Footprint & Arm Thermal Telemetry",
            "",
            "| Resource | Measured Value |",
            "| :--- | :--- |",
            f"| Peak Process RSS | `{mem['peak_rss_mb']} MB` |",
            f"| `agri_kb.json` Footprint | `{mem['artifact_footprint']['agri_kb_kb']} KB` ({mem['artifact_footprint']['agri_kb_bytes']} bytes) |",
            f"| Vision Model Size | `{mem['artifact_footprint']['vision_model_mb']} MB` |",
            f"| Llama `.pte` Size | `{mem['artifact_footprint']['llama_pte_mb']} MB` |",
            f"| Total Measured Artifact Footprint | `{mem['artifact_footprint']['total_measured_mb']} MB` |",
            f"| Linux Thermal Zone (`temp_c`) | `{therm['cpu_temp_c']}` |",
            f"| RPi `vcgencmd` Temp (`temp_c`) | `{therm['vcgencmd_temp_c']}` |",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark_suite(
    runs: int = 5,
    warmup: int = 1,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    vision_backend_kind: str = "pytorch",
    vision_model_path: Optional[Path] = None,
    advisory_backend_kind: str = "test-stub",
    llama_bin: Optional[Path] = None,
    llama_pte: Optional[Path] = None,
    llama_tokenizer: Optional[Path] = None,
    image_path: Optional[Path] = None,
    state_region: str = "Punjab",
) -> Dict[str, Any]:
    """
    Executes warmup + timed runs of the dual-pipeline orchestrator and language
    switcher, saving `benchmark_results.json` and `benchmark_summary.md`.
    """
    if runs < 1:
        raise ValueError(f"--runs must be >= 1, got {runs}")
    if warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {warmup}")

    if vision_backend_kind == "hf-local":
        effective_vision_dir = vision_model_path if vision_model_path is not None else DEFAULT_PRETRAINED_DIR
        allow_untrained = False
        pipeline_class_labels = None
        v_backend = _resolve_vision_backend(
            backend_kind="hf-local",
            model_path=effective_vision_dir,
            num_classes=38,
            allow_untrained_smoke=False,
        )
        num_classes = len(getattr(v_backend, "class_labels", [])) or 38
    else:
        effective_vision_dir = vision_model_path
        class_labels = build_class_labels_from_kb(DEFAULT_KB_PATH)
        num_classes = len(class_labels)
        pipeline_class_labels = class_labels
        allow_untrained = vision_model_path is None and vision_backend_kind in ("pytorch", "pt2")
        v_backend = _resolve_vision_backend(
            backend_kind=vision_backend_kind,
            model_path=vision_model_path,
            num_classes=num_classes,
            allow_untrained_smoke=allow_untrained,
        )

    a_backend = _resolve_advisory_backend(
        backend_kind=advisory_backend_kind,
        llama_bin=llama_bin,
        llama_pte=llama_pte,
        llama_tokenizer=llama_tokenizer,
    )

    retriever = OfflineKBRetriever(DEFAULT_KB_PATH)
    vision_pipeline = VisionPipeline(v_backend, pipeline_class_labels)
    advisory_engine = AdvisoryEngine(a_backend, retriever)
    orchestrator = FieldAssistantOrchestrator(
        vision_pipeline=vision_pipeline,
        advisory_engine=advisory_engine,
        kb_retriever=retriever,
        log_path=None,
    )

    if image_path is not None:
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Benchmark image not found: {image_path}")
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Failed to decode image via OpenCV: {image_path}")
        image_source = str(Path(image_path).resolve())
    else:
        image_bgr = _make_benchmark_synthetic_leaf(lesion_fraction=0.25, size=320)
        image_source = "synthetic_320x320_leaf_25pct_lesion"

    # Warmup iterations
    for _ in range(warmup):
        orchestrator.analyze_image(image_bgr, state_region=state_region, language="en")

    vision_latencies: List[float] = []
    advisory_ttfts: List[float] = []
    advisory_tpss: List[float] = []
    e2e_latencies: List[float] = []
    last_result: Optional[AssistantResult] = None

    for _ in range(runs):
        res = orchestrator.analyze_image(image_bgr, state_region=state_region, language="en")
        last_result = res
        vision_latencies.append(res.pipeline_metadata.vision_latency_ms)
        advisory_ttfts.append(res.pipeline_metadata.advisory_ttft_ms)
        advisory_tpss.append(res.pipeline_metadata.advisory_tokens_per_sec)
        e2e_latencies.append(res.pipeline_metadata.end_to_end_latency_ms)

    # Benchmark instant language switching (en -> hi -> gu -> mr) without re-running vision.
    # If an untrained smoke vision model yielded low-confidence/unknown, seed a confident
    # VisionFinding fixture strictly for exercising Pipeline B language-switch timing.
    assert last_result is not None
    seed_finding = last_result.vision_finding
    used_seeded_finding = False
    if seed_finding.category == VisionCategory.UNKNOWN:
        used_seeded_finding = True
        seed_finding = VisionFinding(
            crop="rice",
            condition_name="Rice Blast",
            category=VisionCategory.DISEASE,
            confidence=0.92,
            severity_level=SeverityLevel.MODERATE,
            infected_area_pct=25.0,
            image_quality_ok=True,
            blur_score=seed_finding.blur_score,
            inference_latency_ms=seed_finding.inference_latency_ms,
            runtime_backend=seed_finding.runtime_backend,
        )

    seed_result = AssistantResult(
        session_id=last_result.session_id,
        timestamp_utc=last_result.timestamp_utc,
        state_region=state_region,
        vision_finding=seed_finding,
        advisory_response=last_result.advisory_response,
        ui_labels=last_result.ui_labels,
        pipeline_metadata=last_result.pipeline_metadata,
    )

    per_transition_ms: Dict[str, float] = {}
    switch_times: List[float] = []
    switch_ttfts: List[float] = []
    switch_tpss: List[float] = []
    current = seed_result
    transitions = [("en", "hi"), ("hi", "gu"), ("gu", "mr")]
    for src_lang, dst_lang in transitions:
        t0 = time.perf_counter()
        current = orchestrator.switch_language(current, new_language=dst_lang, state_region=state_region)
        elapsed_ms = round(max(0.001, (time.perf_counter() - t0) * 1000.0), 4)
        per_transition_ms[f"{src_lang}->{dst_lang}"] = elapsed_ms
        switch_times.append(elapsed_ms)
        switch_ttfts.append(current.pipeline_metadata.advisory_ttft_ms)
        switch_tpss.append(current.pipeline_metadata.advisory_tokens_per_sec)

    is_stub = isinstance(a_backend, TestStubAdvisoryBackend)
    artifact_footprint = _collect_artifact_footprint(
        kb_path=DEFAULT_KB_PATH,
        vision_model_path=effective_vision_dir,
        llama_pte_path=llama_pte,
    )
    thermal_and_power = read_arm_thermal_and_power()
    peak_rss_mb = get_peak_rss_mb()

    if vision_backend_kind == "hf-local":
        try:
            disp_dir = Path(effective_vision_dir).resolve().relative_to(_MODEL_ROOT).as_posix()
        except ValueError:
            disp_dir = str(Path(effective_vision_dir).resolve())
        vision_model_source = (
            f"linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification "
            f"({num_classes}-class MobileNetV2 PlantVillage @ {disp_dir})"
        )
    elif vision_model_path is not None:
        vision_model_source = str(Path(vision_model_path).resolve())
    else:
        vision_model_source = "untrained_mobilenet_v3_small_smoke"

    results: Dict[str, Any] = {
        "platform_metadata": {
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu_arch": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python_version": platform.python_version(),
            "host_platform": f"{platform.system()}-{platform.machine()}",
            "vision_backend": v_backend.backend_name,
            "advisory_backend": a_backend.backend_name,
            "is_stub_advisory": is_stub,
            "vision_weights_trained": not getattr(v_backend, "is_untrained_smoke", False),
            "vision_model_source": vision_model_source,
            "image_source": image_source,
            "warmup_runs": warmup,
            "timed_runs": runs,
            "language_switch_used_seeded_confident_finding": used_seeded_finding,
        },
        "latency_and_throughput": {
            "vision_latency_ms": _summarize_series(vision_latencies),
            "advisory_ttft_ms": _summarize_series(advisory_ttfts),
            "advisory_tokens_per_sec": _summarize_series(advisory_tpss),
            "end_to_end_latency_ms": _summarize_series(e2e_latencies),
        },
        "language_switch_benchmark": {
            "vision_re_executed": False,
            "per_transition_ms": per_transition_ms,
            "summary_ms": _summarize_series(switch_times),
            "advisory_ttft_ms": _summarize_series(switch_ttfts),
            "advisory_tokens_per_sec": _summarize_series(switch_tpss),
        },
        "memory_and_storage": {
            "peak_rss_mb": peak_rss_mb,
            "artifact_footprint": artifact_footprint,
        },
        "arm_thermal_and_power": thermal_and_power,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_results.json"
    md_path = out_dir / "benchmark_summary.md"

    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_summary_markdown(results), encoding="utf-8")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FieldWise PS1 reproducible on-device benchmark suite."
    )
    parser.add_argument("--runs", type=int, default=5, help="Timed analyze_image runs")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for benchmark_results.json and benchmark_summary.md",
    )
    parser.add_argument(
        "--vision-backend",
        choices=["hf-local", "pytorch", "pt2", "onnx", "executorch", "hailort"],
        default="hf-local" if (DEFAULT_PRETRAINED_DIR / "config.json").is_file() else "pytorch",
    )
    parser.add_argument("--vision-model", type=Path, default=None)
    parser.add_argument(
        "--advisory-backend",
        choices=["executorch-worker", "executorch-cli", "executorch-py", "test-stub"],
        default="test-stub",
    )
    parser.add_argument("--llama-bin", type=Path, default=None)
    parser.add_argument("--llama-pte", type=Path, default=None)
    parser.add_argument("--llama-tokenizer", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--state", type=str, default="Punjab")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    results = run_benchmark_suite(
        runs=args.runs,
        warmup=args.warmup,
        output_dir=args.output_dir,
        vision_backend_kind=args.vision_backend,
        vision_model_path=args.vision_model,
        advisory_backend_kind=args.advisory_backend,
        llama_bin=args.llama_bin,
        llama_pte=args.llama_pte,
        llama_tokenizer=args.llama_tokenizer,
        image_path=args.image,
        state_region=args.state,
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
