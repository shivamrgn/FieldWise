"""
Reproducible Llama 3.2 ExecuTorch (.pte) Export Driver for FieldWise PS1.

Validates the XNNPACK + KleidiAI 8da4w quantization YAML configuration and local
Meta Llama 3.2 checkpoint files (`consolidated.00.pth`, `params.json`,
`tokenizer.model`), invokes `executorch.extension.llm.export.export_llm` (or prints
the reproducible command in `--dry-run` mode), and records artifact metadata
(`.pte` byte size and SHA-256) in a manifest JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_PS1_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _PS1_ROOT / "configs" / "llama3_2_1b_xnnpack_8da4w.yaml"
DEFAULT_OUTPUT_DIR = _PS1_ROOT / "advisory" / "artifacts"

REQUIRED_CHECKPOINT_FILES = ("consolidated.00.pth", "params.json", "tokenizer.model")


class LlamaExportConfigError(ValueError):
    """Raised when the ExecuTorch LLM export configuration is invalid."""


class MissingLlamaExportDependencyError(RuntimeError):
    """Raised when `executorch` LLM export tooling is not installed."""


def sha256_file(path: Path) -> str:
    """Computes deterministic SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_export_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads and validates the Llama 3.2 Arm XNNPACK ExecuTorch export YAML config.
    """
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Export config file not found: {cfg_file}")

    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise LlamaExportConfigError("Export config YAML must parse to a mapping.")

    base = cfg.get("base", {})
    model = cfg.get("model", {})
    quant = cfg.get("quantization", {})
    backend = cfg.get("backend", {})

    if base.get("model_class") != "llama3_2":
        raise LlamaExportConfigError(
            f"Expected base.model_class == 'llama3_2', got {base.get('model_class')!r}"
        )
    if quant.get("qmode") != "torchao:8da4w":
        raise LlamaExportConfigError(
            f"Expected quantization.qmode == 'torchao:8da4w', got {quant.get('qmode')!r}"
        )
    if quant.get("group_size") != 64:
        raise LlamaExportConfigError(
            f"Expected quantization.group_size == 64, got {quant.get('group_size')!r}"
        )
    if quant.get("embedding_quantize") != "4,32":
        raise LlamaExportConfigError(
            f"Expected quantization.embedding_quantize == '4,32', got {quant.get('embedding_quantize')!r}"
        )
    if model.get("use_sdpa_with_kv_cache") is not True:
        raise LlamaExportConfigError("Expected model.use_sdpa_with_kv_cache == true")
    if model.get("max_seq_length") != 1024:
        raise LlamaExportConfigError(
            f"Expected model.max_seq_length == 1024, got {model.get('max_seq_length')!r}"
        )
    if model.get("max_context_length") != 1024:
        raise LlamaExportConfigError(
            f"Expected model.max_context_length == 1024, got {model.get('max_context_length')!r}"
        )
    if not backend.get("xnnpack", {}).get("enabled", False):
        raise LlamaExportConfigError("Expected backend.xnnpack.enabled == true")

    return cfg


def validate_checkpoint_dir(checkpoint_dir: Path) -> Dict[str, Path]:
    """
    Verifies local Meta Llama 3.2 checkpoint files exist and are non-empty:
      - `consolidated.00.pth`
      - `params.json`
      - `tokenizer.model`
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Llama checkpoint directory not found: {ckpt_dir}")

    resolved: Dict[str, Path] = {}
    for fname in REQUIRED_CHECKPOINT_FILES:
        fpath = ckpt_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Required Llama 3.2 file '{fname}' missing in {ckpt_dir}"
            )
        if fpath.stat().st_size == 0:
            raise LlamaExportConfigError(f"Checkpoint file '{fpath}' is empty.")
        resolved[fname] = fpath.resolve()

    # Validate params.json parses cleanly as JSON mapping
    try:
        params_obj = json.loads(resolved["params.json"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LlamaExportConfigError(f"Invalid JSON in {resolved['params.json']}: {exc}") from exc
    if not isinstance(params_obj, dict):
        raise LlamaExportConfigError("params.json must contain a JSON object.")

    return resolved


def build_export_command(
    config_path: Path,
    output_pte_path: Path,
    checkpoint_paths: Optional[Dict[str, Path]] = None,
) -> List[str]:
    """Builds the deterministic `executorch.extension.llm.export.export_llm` argv command."""
    ckpt = str(checkpoint_paths["consolidated.00.pth"]) if checkpoint_paths else "<CHECKPOINT_DIR>/consolidated.00.pth"
    params = str(checkpoint_paths["params.json"]) if checkpoint_paths else "<CHECKPOINT_DIR>/params.json"
    tok = str(checkpoint_paths["tokenizer.model"]) if checkpoint_paths else "<CHECKPOINT_DIR>/tokenizer.model"

    return [
        sys.executable,
        "-m",
        "executorch.extension.llm.export.export_llm",
        "--config",
        str(Path(config_path).resolve()),
        f"+base.checkpoint={ckpt}",
        f"+base.params={params}",
        f"+base.tokenizer_path={tok}",
        f"+export.output_name={Path(output_pte_path).resolve()}",
    ]


def write_artifact_manifest(
    pte_path: Path,
    config_path: Path,
    manifest_path: Optional[Path] = None,
    checkpoint_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """Records exported `.pte` size and SHA-256 alongside config/checkpoint metadata."""
    pte_file = Path(pte_path)
    if not pte_file.is_file():
        raise FileNotFoundError(f"Exported .pte artifact not found: {pte_file}")

    cfg = validate_export_config(config_path)
    manifest = {
        "model_class": cfg["base"]["model_class"],
        "quantization": cfg["quantization"],
        "max_seq_length": cfg["model"]["max_seq_length"],
        "max_context_length": cfg["model"]["max_context_length"],
        "use_sdpa_with_kv_cache": cfg["model"]["use_sdpa_with_kv_cache"],
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(Path(config_path)),
        "pte_path": str(pte_file.resolve()),
        "pte_size_bytes": pte_file.stat().st_size,
        "pte_sha256": sha256_file(pte_file),
        "source_params_sha256": (
            sha256_file(checkpoint_paths["params.json"]) if checkpoint_paths else None
        ),
    }

    target_manifest = manifest_path or pte_file.with_suffix(".manifest.json")
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_export(
    config_path: Path = DEFAULT_CONFIG_PATH,
    checkpoint_dir: Optional[Path] = None,
    output_pte_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Validates configuration/checkpoints and either prints the exact `export_llm`
    command (`dry_run=True`) or executes ExecuTorch `.pte` export and writes the manifest.
    """
    cfg = validate_export_config(config_path)
    default_name = cfg.get("export", {}).get("output_name", "llama3_2_1b_xnnpack_8da4w.pte")
    pte_target = Path(output_pte_path) if output_pte_path else (DEFAULT_OUTPUT_DIR / default_name)

    resolved_ckpts: Optional[Dict[str, Path]] = None
    if checkpoint_dir is not None:
        resolved_ckpts = validate_checkpoint_dir(checkpoint_dir)
    elif not dry_run:
        raise ValueError(
            "--checkpoint-dir containing consolidated.00.pth, params.json, and "
            "tokenizer.model is required when running live export."
        )

    cmd = build_export_command(
        config_path=config_path,
        output_pte_path=pte_target,
        checkpoint_paths=resolved_ckpts,
    )

    if dry_run:
        return {
            "dry_run": True,
            "config_valid": True,
            "checkpoint_validated": resolved_ckpts is not None,
            "command": cmd,
            "command_str": " ".join(cmd),
            "target_pte_path": str(pte_target.resolve()),
        }

    if (
        importlib.util.find_spec("executorch") is None
        or importlib.util.find_spec("executorch.extension.llm.export.export_llm") is None
    ):
        raise MissingLlamaExportDependencyError(
            "ExecuTorch LLM export tooling ('executorch.extension.llm.export.export_llm') "
            "is not installed in this environment."
        )

    pte_target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, shell=False, check=True)
    manifest = write_artifact_manifest(
        pte_path=pte_target,
        config_path=config_path,
        checkpoint_paths=resolved_ckpts,
    )
    return {
        "dry_run": False,
        "config_valid": True,
        "checkpoint_validated": True,
        "command": cmd,
        "command_str": " ".join(cmd),
        "manifest": manifest,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate & export Llama 3.2 1B-Instruct to ExecuTorch .pte for Arm CPU."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to ExecuTorch Llama export YAML configuration",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory holding Meta consolidated.00.pth, params.json, tokenizer.model",
    )
    parser.add_argument(
        "--output-pte",
        type=Path,
        default=None,
        help="Output .pte file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print exact export_llm command without executing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_export(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_pte_path=args.output_pte,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
