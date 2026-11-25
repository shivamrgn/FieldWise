"""
Isolated ExecuTorch Llama 3.2 Worker Process (`executorch_worker.py`).

Intended to be executed directly inside the dedicated ExecuTorch virtual
environment (`executorch_env/bin/python -u .../executorch_worker.py ...`) so the
main FieldWise Python environment remains unmodified while genuine C++
`TextLLMRunner` (.pte + tokenizer.json) inference stays warm in memory across
requests over a strict JSON-lines stdin/stdout pipe.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import resource
import sys
import time
from typing import Any, Dict, List


def _load_executorch_llm_runner():
    """Imports ExecuTorch runtime and registers Llama/quantized/TorchAO kernels."""
    import executorch  # type: ignore
    import executorch.extension.pybindings.portable_lib  # type: ignore  # noqa: F401
    import executorch.extension.llm.custom_ops.custom_ops  # type: ignore  # noqa: F401
    import executorch.kernels.quantized  # type: ignore  # noqa: F401

    et_root = pathlib.Path(list(executorch.__path__)[0])
    for rel_lib in (
        "lib/libexecutorch_kernels_quantized.dylib",
        "lib/libexecutorch_kernels_torchao.dylib",
        "lib/libexecutorch_kernels_optimized.dylib",
        "lib/libexecutorch_kernels_quantized.so",
        "lib/libexecutorch_kernels_torchao.so",
        "lib/libexecutorch_kernels_optimized.so",
    ):
        candidate = et_root / rel_lib
        if candidate.is_file():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)

    from executorch.extension.llm.runner import (  # type: ignore
        GenerationConfig,
        TextLLMRunner,
    )

    return GenerationConfig, TextLLMRunner


def _peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(raw / (1024.0 * 1024.0), 3)
    return round(raw / 1024.0, 3)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent ExecuTorch Llama 3.2 worker.")
    parser.add_argument("--pte", required=True, help="Path to .pte model artifact.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer (.json or .model).")
    parser.add_argument("--seq-len", type=int, default=2048, help="Max total sequence length (-1 for auto).")
    parser.add_argument("--eager-load", action="store_true", help="Run a zero-token load warmup at startup.")
    args = parser.parse_args(argv)

    # Reserve fd 1 strictly for JSON-lines IPC; forward C++ stdout logs/tokens to stderr.
    ipc_fd = os.dup(1)
    os.dup2(2, 1)
    ipc_out = os.fdopen(ipc_fd, "w", encoding="utf-8", buffering=1)

    try:
        GenerationConfig, TextLLMRunner = _load_executorch_llm_runner()
        t_init0 = time.perf_counter()
        runner = TextLLMRunner(str(args.pte), str(args.tokenizer))
        runner_init_ms = round((time.perf_counter() - t_init0) * 1000.0, 3)
    except Exception as exc:  # noqa: BLE001
        ipc_out.write(json.dumps({"ready": False, "error": str(exc)}) + "\n")
        ipc_out.flush()
        return 1

    ipc_out.write(
        json.dumps(
            {
                "ready": True,
                "runner_init_ms": runner_init_ms,
                "peak_rss_mb": _peak_rss_mb(),
            }
        )
        + "\n"
    )
    ipc_out.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            ipc_out.write(json.dumps({"ok": False, "error": f"Invalid JSON request: {exc}"}) + "\n")
            ipc_out.flush()
            continue

        op = str(req.get("op", "generate"))
        if op == "shutdown":
            ipc_out.write(json.dumps({"ok": True, "shutdown": True}) + "\n")
            ipc_out.flush()
            break
        if op == "ping":
            ipc_out.write(json.dumps({"ok": True, "pong": True, "peak_rss_mb": _peak_rss_mb()}) + "\n")
            ipc_out.flush()
            continue

        prompt = str(req.get("prompt", ""))
        max_new_tokens = int(req.get("max_new_tokens", 512))
        temperature = float(req.get("temperature", 0.0))

        tokens: List[str] = []
        captured_stats: List[Dict[str, Any]] = []
        first_token_wall: List[float] = []
        t_start = time.perf_counter()

        def _on_token(piece: str) -> None:
            if not first_token_wall:
                first_token_wall.append(time.perf_counter())
            tokens.append(piece)

        def _on_stats(stats_obj: Any) -> None:
            try:
                captured_stats.append(json.loads(stats_obj.to_json_string()))
            except Exception:  # noqa: BLE001
                pass

        try:
            cfg = GenerationConfig(
                echo=False,
                max_new_tokens=max_new_tokens,
                seq_len=-1,
                temperature=temperature,
                num_bos=0,
                num_eos=0,
                warming=False,
            )
            runner.generate(prompt, cfg, _on_token, _on_stats)
            wall_total_s = max(0.001, time.perf_counter() - t_start)

            stat = captured_stats[-1] if captured_stats else {}
            inf_start = float(stat.get("inference_start_ms", 0.0) or 0.0)
            first_ms = float(stat.get("first_token_ms", 0.0) or 0.0)
            load_start = float(stat.get("model_load_start_ms", 0.0) or 0.0)
            load_end = float(stat.get("model_load_end_ms", 0.0) or 0.0)
            decode_tps = float(stat.get("decode_token_per_sec", 0.0) or 0.0)
            prefill_tps = float(stat.get("prefill_token_per_sec", 0.0) or 0.0)
            prompt_tokens = int(stat.get("prompt_tokens", 0) or 0)
            generated_tokens = int(stat.get("generated_tokens", len(tokens)) or len(tokens))

            if inf_start > 0.0 and first_ms >= inf_start:
                ttft_ms = first_ms - inf_start
            elif first_token_wall:
                ttft_ms = (first_token_wall[0] - t_start) * 1000.0
            else:
                ttft_ms = wall_total_s * 1000.0

            if decode_tps <= 0.0:
                decode_tps = max(0.001, len(tokens) / wall_total_s)

            model_load_ms = max(0.0, load_end - load_start) if (load_end > load_start > 0.0) else 0.0
            raw_text = "".join(tokens).strip()
            for stop_tok in ("<|eot_id|>", "<|end_of_text|>"):
                if raw_text.endswith(stop_tok):
                    raw_text = raw_text[: -len(stop_tok)].strip()

            resp = {
                "ok": True,
                "text": raw_text,
                "ttft_ms": round(float(ttft_ms), 4),
                "tokens_per_sec": round(float(decode_tps), 4),
                "prefill_tokens_per_sec": round(float(prefill_tps), 4),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "model_load_ms": round(float(model_load_ms), 3),
                "peak_rss_mb": _peak_rss_mb(),
            }
        except Exception as exc:  # noqa: BLE001
            resp = {"ok": False, "error": str(exc), "peak_rss_mb": _peak_rss_mb()}
        finally:
            try:
                runner.reset()
            except Exception:  # noqa: BLE001
                pass

        ipc_out.write(json.dumps(resp, ensure_ascii=False) + "\n")
        ipc_out.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
