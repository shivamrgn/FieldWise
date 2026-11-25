#!/usr/bin/env python3
"""
FieldWise Unified Offline Launcher & Project Entrypoint.

Binds all active FieldWise components into one standalone offline project:
  1. PS1 Offline Edge-AI Web App & Vision Pipeline (`ps1_offline_assistant/`)
     - Crop disease/pest detection (MobileNetV2 PlantVillage + OpenCV severity)
     - 100% offline KB & multilingual i18n (`en`, `hi`, `gu`, `mr`)
  2. FieldWise V2 Rule-Based Crop Recommendation (`ml_pipeline/recommendation_engine/`)
     - 2-tier agronomic eligibility & Liebig scoring + STCR fertilizer math
  3. FieldWise V1 JavaScript Databases (`cropDatabase.js`, `fertilizerDatabase.js`)
     - Source baseline databases compiled into `agri_kb.json`

Usage:
  python3 app.py serve                     # Launch offline 3-tab web UI on 127.0.0.1:8080
  python3 app.py scan --image leaf.jpg     # Run offline disease/pest CLI scan
  python3 app.py crop < crop_query.json    # Run V2 crop recommendation CLI
  python3 app.py rebuild-kb                # Recompile agri_kb.json from JS DBs
  python3 app.py test                      # Run all PS1, V2, and V1 checks
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent
PS1_DIR = ROOT_DIR / "ps1_offline_assistant"
PRETRAINED_CONFIG = PS1_DIR / "vision" / "artifacts" / "pretrained_hf" / "config.json"
EXECUTORCH_PYTHON = ROOT_DIR / "executorch_env" / "bin" / "python"
LLAMA_PTE = PS1_DIR / "advisory" / "artifacts" / "xnnpack" / "Llama-3.2-1B-Instruct-8da4w-gptq-2k.pte"
LLAMA_TOKENIZER = PS1_DIR / "advisory" / "artifacts" / "tokenizer.json"
V2_TESTS_DIR = ROOT_DIR / "ml_pipeline" / "recommendation_engine" / "tests"


def _run_module(module: str, args: List[str], cwd: Path = ROOT_DIR) -> int:
    """Executes a Python module using the active interpreter with ROOT_DIR on PYTHONPATH."""
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    proc = subprocess.run([sys.executable, "-m", module, *args], cwd=str(cwd), check=False)
    return proc.returncode


def _has_flag(args: List[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in args)


def _with_local_defaults(extra_args: List[str]) -> List[str]:
    """Supplies safe offline defaults when local Llama or vision flags are omitted."""
    resolved = list(extra_args)
    if not _has_flag(resolved, "--vision-backend"):
        if PRETRAINED_CONFIG.is_file():
            resolved.extend(["--vision-backend", "hf-local"])
        else:
            resolved.extend(["--vision-backend", "pytorch"])
            if not _has_flag(resolved, "--allow-untrained-smoke") and not _has_flag(resolved, "--vision-model"):
                resolved.append("--allow-untrained-smoke")

    has_llama_args = (
        _has_flag(resolved, "--llama-bin")
        and _has_flag(resolved, "--llama-pte")
        and _has_flag(resolved, "--llama-tokenizer")
    )
    if not _has_flag(resolved, "--advisory-backend") and not has_llama_args:
        if EXECUTORCH_PYTHON.is_file() and LLAMA_PTE.is_file() and LLAMA_TOKENIZER.is_file():
            resolved.extend(["--advisory-backend", "executorch-worker"])
        else:
            resolved.extend(["--advisory-backend", "test-stub"])
    return resolved


def cmd_serve(extra_args: List[str]) -> int:
    """Starts the 100% offline 3-tab local web server."""
    args = _with_local_defaults(extra_args)
    return _run_module("ps1_offline_assistant.app.server", args)


def cmd_scan(extra_args: List[str]) -> int:
    """Runs offline leaf disease/pest inference and localized advisory via CLI."""
    args = _with_local_defaults(extra_args)
    return _run_module("ps1_offline_assistant.orchestrator.cli", args)


def cmd_crop(extra_args: List[str]) -> int:
    """Runs the FieldWise V2 rule-based crop recommendation CLI (stdin JSON or --state/--sowing-month flags)."""
    if not extra_args:
        return _run_module("ml_pipeline.recommendation_engine.adapter_cli", [])

    p = argparse.ArgumentParser(prog="app.py crop")
    p.add_argument("--state", required=True)
    p.add_argument("--district", default=None)
    p.add_argument("--sowing-month", type=int, required=True)
    p.add_argument("--soil-texture", default="Loam")
    p.add_argument("--irrigation-access", default="Canal_River")
    p.add_argument("--soil-ph", type=float, default=None)
    ns = p.parse_args(extra_args)

    import json

    payload = {
        "state": ns.state,
        "sowing_month": ns.sowing_month,
        "soil_texture": ns.soil_texture,
        "irrigation_access": ns.irrigation_access,
    }
    if ns.district:
        payload["district"] = ns.district
    if ns.soil_ph is not None:
        payload["soil_ph"] = ns.soil_ph

    proc = subprocess.run(
        [sys.executable, "-m", "ml_pipeline.recommendation_engine.adapter_cli"],
        input=json.dumps(payload),
        text=True,
        cwd=str(ROOT_DIR),
        check=False,
    )
    return proc.returncode


def cmd_rebuild_kb(extra_args: List[str]) -> int:
    """Rebuilds `agri_kb.json` from V1 `cropDatabase.js` and `fertilizerDatabase.js`."""
    return _run_module("ps1_offline_assistant.scripts.build_offline_kb", extra_args)


def cmd_test() -> int:
    """Executes all PS1 offline assistant tests, V2 engine tests, and V1 JS sanity check."""
    checks: List[tuple[str, List[str]]] = [
        (
            "PS1 Offline Edge-AI Suite (Phases 1-7)",
            [sys.executable, "-m", "unittest", "discover", "-s", "ps1_offline_assistant/tests", "-p", "test_*.py", "-v"],
        ),
        (
            "FieldWise V2 Core & Regressions (62 tests)",
            [sys.executable, str(V2_TESTS_DIR / "run_all_tests.py")],
        ),
        (
            "FieldWise V2 CLI Adapter & Frozen Hash Audit (17 tests)",
            [sys.executable, str(V2_TESTS_DIR / "test_adapter_cli.py")],
        ),
    ]
    if shutil.which("node"):
        checks.append(
            (
                "FieldWise V1 NodeJS Sanity Check",
                ["node", "-e", "const p = require('./predictor'); if (typeof p.predictCrops !== 'function') process.exit(1); console.log('V1 JS OK');"],
            )
        )

    for label, cmd in checks:
        print(f"\n=== Running: {label} ===")
        res = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False)
        if res.returncode != 0:
            print(f"FAILED: {label} (exit code {res.returncode})")
            return res.returncode

    print("\nALL FIELDWISE CHECKS PASSED.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0].startswith("-"):
        # Default bare `python3 app.py [--port ...]` to `serve`
        return cmd_serve(raw)

    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Unified FieldWise Offline AI Project Launcher",
    )
    parser.add_argument(
        "command",
        choices=["serve", "scan", "crop", "rebuild-kb", "test"],
        help="Action to execute",
    )
    ns, remainder = parser.parse_known_args(raw)

    dispatch = {
        "serve": lambda: cmd_serve(remainder),
        "scan": lambda: cmd_scan(remainder),
        "crop": lambda: cmd_crop(remainder),
        "rebuild-kb": lambda: cmd_rebuild_kb(remainder),
        "test": cmd_test,
    }
    return dispatch[ns.command]()


if __name__ == "__main__":
    raise SystemExit(main())
