"""
Phase 6 Verification Suite: On-Device Benchmarks & PS1 Documentation.

Verifies:
  1. `run_benchmarks.py` executes cleanly via API and CLI, producing valid
     `benchmark_results.json` and `benchmark_summary.md` with positive latencies,
     non-zero peak RSS memory, accurate `host_platform` + `is_stub_advisory` flags,
     and non-re-executed vision during language switching.
  2. `docs/TECHNICAL_REPORT.md` and `docs/DEMO_SCRIPT.md` exist and contain all
     required PS1 architecture, hardware split, safety, and demo sections.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.benchmarks.run_benchmarks import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    get_peak_rss_mb,
    read_arm_thermal_and_power,
    run_benchmark_suite,
)


class TestOnDeviceBenchmarkSuite(unittest.TestCase):
    def test_benchmark_runner_cli_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "bench_out"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ps1_offline_assistant.benchmarks.run_benchmarks",
                    "--runs",
                    "2",
                    "--warmup",
                    "1",
                    "--output-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            cli_json = json.loads(proc.stdout)

            json_file = out_dir / "benchmark_results.json"
            md_file = out_dir / "benchmark_summary.md"
            self.assertTrue(json_file.is_file())
            self.assertTrue(md_file.is_file())

            saved = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["platform_metadata"]["timed_runs"], 2)
            self.assertTrue(saved["platform_metadata"]["is_stub_advisory"])
            self.assertTrue(saved["platform_metadata"]["host_platform"])
            self.assertEqual(saved["platform_metadata"], cli_json["platform_metadata"])

            lat = saved["latency_and_throughput"]
            self.assertGreater(lat["vision_latency_ms"]["mean"], 0.0)
            self.assertGreater(lat["vision_latency_ms"]["p50"], 0.0)
            self.assertGreater(lat["vision_latency_ms"]["p95"], 0.0)
            self.assertGreater(lat["end_to_end_latency_ms"]["mean"], 0.0)

            sw = saved["language_switch_benchmark"]
            self.assertFalse(sw["vision_re_executed"])
            self.assertEqual(set(sw["per_transition_ms"].keys()), {"en->hi", "hi->gu", "gu->mr"})
            self.assertGreater(sw["summary_ms"]["mean"], 0.0)

            mem = saved["memory_and_storage"]
            self.assertGreater(mem["peak_rss_mb"], 0.0)
            self.assertGreater(mem["artifact_footprint"]["agri_kb_bytes"], 0)

            md_text = md_file.read_text(encoding="utf-8")
            self.assertIn("FieldWise PS1 On-Device Benchmark Summary", md_text)
            self.assertIn("Peak Process RSS", md_text)

    def test_committed_results_and_system_helpers(self) -> None:
        self.assertGreater(get_peak_rss_mb(), 0.0)
        therm = read_arm_thermal_and_power()
        self.assertEqual(
            set(therm.keys()),
            {"cpu_temp_c", "vcgencmd_temp_c", "vcgencmd_pmic_raw"},
        )

        # Verify direct API call also returns valid metrics
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_benchmark_suite(runs=1, warmup=0, output_dir=Path(tmpdir))
            self.assertGreater(res["memory_and_storage"]["peak_rss_mb"], 0.0)

        # Committed benchmark results directory matches schema
        committed_json = DEFAULT_RESULTS_DIR / "benchmark_results.json"
        committed_md = DEFAULT_RESULTS_DIR / "benchmark_summary.md"
        self.assertTrue(committed_json.is_file())
        self.assertTrue(committed_md.is_file())
        parsed = json.loads(committed_json.read_text(encoding="utf-8"))
        self.assertIn("latency_and_throughput", parsed)


class TestPS1DocumentationDeliverables(unittest.TestCase):
    def test_technical_report_required_sections_and_no_ascii_boxes(self) -> None:
        tech_report = _PS1_ROOT / "docs" / "TECHNICAL_REPORT.md"
        self.assertTrue(tech_report.is_file())
        text = tech_report.read_text(encoding="utf-8")

        for required in (
            "ExecuTorch",
            "Llama 3.2",
            "HailoRT",
            "XNNPACK",
            "KleidiAI",
            "8da4w",
            "PlantVillage",
            "IP102",
            "MobileNetV3",
            "agri_kb.json",
            "cropDatabase.js",
            "fertilizerDatabase.js",
            "1800-180-1551",
            "PENDING_TARGET_ARM_MEASUREMENT",
            "SEVERE_ESCALATION",
            "UNCERTAIN_IMAGE",
            "LOW_CONFIDENCE",
        ):
            self.assertIn(required, text, f"Missing '{required}' in TECHNICAL_REPORT.md")

        # Verify ASCII-free clean diagramming (Mermaid + tables)
        for box_char in ("┌", "│", "└", "├", "+---"):
            self.assertNotIn(box_char, text)

    def test_demo_script_required_sections(self) -> None:
        demo_script = _PS1_ROOT / "docs" / "DEMO_SCRIPT.md"
        self.assertTrue(demo_script.is_file())
        text = demo_script.read_text(encoding="utf-8")

        for required in (
            "Airplane Mode",
            "127.0.0.1:8080",
            "OFFLINE MODE",
            "1800-180-1551",
            "UNCERTAIN_IMAGE",
            "SEVERE_ESCALATION",
            "हिन्दी",
            "ગુજરાતી",
            "मराठी",
        ):
            self.assertIn(required, text, f"Missing '{required}' in DEMO_SCRIPT.md")


if __name__ == "__main__":
    unittest.main(verbosity=2)
