"""
Pluggable Offline Llama 3.2 ExecuTorch Runtime Runners for FieldWise PS1 (Pipeline B).

Provides a unified `LlamaAdvisoryBackend` interface returning:
    `(raw_text: str, ttft_ms: float, tokens_per_sec: float)`

Implementations:
  - `ExecuTorchLlamaCliBackend`: Invokes compiled ExecuTorch C++ `llama_main`
    via `subprocess.Popen` (`shell=False`) and measures wall-clock TTFT & tok/s.
  - `ExecuTorchPythonRuntimeBackend`: Uses `executorch.runtime.Runtime` +
    `pytorch_tokenizers` for native Python `.pte` decode.
  - `TestStubAdvisoryBackend`: Strictly test-only deterministic backend for CI/unit
    verification when Llama weights and `executorch` are absent. NEVER presented
    as real LLM inference.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import selectors
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
DEFAULT_EXECUTORCH_PYTHON = _MODEL_ROOT / "executorch_env" / "bin" / "python"
DEFAULT_LLAMA_PTE = _PS1_ROOT / "advisory" / "artifacts" / "xnnpack" / "Llama-3.2-1B-Instruct-8da4w-gptq-2k.pte"
DEFAULT_LLAMA_TOKENIZER = _PS1_ROOT / "advisory" / "artifacts" / "tokenizer.json"
DEFAULT_EXECUTORCH_WORKER = _PS1_ROOT / "advisory" / "executorch_worker.py"


LLAMA3_EOS_IDS = {128001, 128009}


class MissingLlamaRuntimeError(RuntimeError):
    """Raised when ExecuTorch runtime, compiled binary, tokenizer, or .pte is unavailable."""


class LlamaAdvisoryBackend(ABC):
    """Abstract runtime backend for on-device Llama advisory generation."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human/telemetry identifier for the runtime backend."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[str, float, float]:
        """Generates `(raw_text, ttft_ms, tokens_per_sec)` from a Llama 3.2 prompt."""


def _estimate_token_count(text: str) -> int:
    """Rough subword token approximation when exact tokenizer counts aren't exposed by CLI."""
    words = re.findall(r"\S+", text)
    if not words:
        return 0
    return max(len(words), int(round(len(text) / 4.0)))


class ExecuTorchLlamaCliBackend(LlamaAdvisoryBackend):
    """
    Runs ExecuTorch's compiled C++ `llama_main` binary (`shell=False`, strict argv list),
    streaming stdout to capture exact wall-clock TTFT and decode speed.
    """

    def __init__(
        self,
        binary_path: Path,
        pte_path: Path,
        tokenizer_path: Path,
        seq_len: int = 1024,
        timeout_sec: float = 120.0,
    ) -> None:
        self.binary_path = Path(binary_path)
        self.pte_path = Path(pte_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.seq_len = int(seq_len)
        self.timeout_sec = float(timeout_sec)

        if not self.binary_path.is_file() or not os.access(self.binary_path, os.X_OK):
            raise MissingLlamaRuntimeError(
                f"ExecuTorch C++ llama_main binary not found or not executable: {self.binary_path}"
            )
        if not self.pte_path.is_file():
            raise MissingLlamaRuntimeError(
                f"Llama .pte model artifact not found: {self.pte_path}"
            )
        if not self.tokenizer_path.is_file():
            raise MissingLlamaRuntimeError(
                f"Llama tokenizer model not found: {self.tokenizer_path}"
            )

    @property
    def backend_name(self) -> str:
        return "executorch_llama_cli"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[str, float, float]:
        argv = [
            str(self.binary_path),
            "--model_path",
            str(self.pte_path),
            "--tokenizer_path",
            str(self.tokenizer_path),
            "--prompt",
            prompt,
            "--seq_len",
            str(min(self.seq_len, max_new_tokens + _estimate_token_count(prompt) + 64)),
            "--temperature",
            f"{float(temperature):.4f}",
        ]

        t_start = time.perf_counter()
        first_token_time: Optional[float] = None
        chunks: List[str] = []

        proc = subprocess.Popen(
            argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        deadline = t_start + self.timeout_sec

        try:
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    proc.kill()
                    proc.wait(timeout=5)
                    raise TimeoutError(
                        f"ExecuTorch llama_main exceeded {self.timeout_sec}s timeout."
                    )

                events = selector.select(timeout=min(0.25, remaining))
                if not events:
                    if proc.poll() is not None:
                        break
                    continue

                line = proc.stdout.readline()
                if line == "":
                    if proc.poll() is not None:
                        break
                    continue

                if first_token_time is None and line.strip():
                    first_token_time = time.perf_counter()
                chunks.append(line)

            stderr_output = proc.stderr.read()
            return_code = proc.wait(timeout=5)
        finally:
            selector.close()
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()

        t_end = time.perf_counter()

        if return_code != 0:
            raise RuntimeError(
                f"llama_main exited with code {return_code}: {stderr_output.strip()}"
            )

        raw_output = "".join(chunks).strip()
        if raw_output.startswith(prompt):
            raw_output = raw_output[len(prompt) :].strip()
        raw_output = raw_output.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

        ttft_ms = (
            ((first_token_time - t_start) * 1000.0)
            if first_token_time is not None
            else ((t_end - t_start) * 1000.0)
        )
        decode_sec = max(
            1e-4,
            (t_end - first_token_time) if first_token_time is not None else (t_end - t_start),
        )
        tok_count = max(1, _estimate_token_count(raw_output))
        tokens_per_sec = tok_count / decode_sec

        return raw_output, round(ttft_ms, 3), round(tokens_per_sec, 3)


class ExecuTorchPythonRuntimeBackend(LlamaAdvisoryBackend):
    """
    Python-native ExecuTorch `.pte` runner using `executorch.runtime.Runtime`
    and `pytorch_tokenizers`.
    """

    def __init__(
        self,
        pte_path: Path,
        tokenizer_path: Path,
        max_seq_len: int = 1024,
    ) -> None:
        self.pte_path = Path(pte_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.max_seq_len = int(max_seq_len)

        if not self.pte_path.is_file():
            raise MissingLlamaRuntimeError(
                f"Llama .pte artifact not found: {self.pte_path}"
            )
        if not self.tokenizer_path.is_file():
            raise MissingLlamaRuntimeError(
                f"Llama tokenizer artifact not found: {self.tokenizer_path}"
            )

        try:
            from executorch.runtime import Runtime
        except ImportError as exc:
            raise MissingLlamaRuntimeError(
                "ExecuTorchPythonRuntimeBackend requires 'executorch' Python runtime."
            ) from exc

        try:
            from pytorch_tokenizers import get_tokenizer
        except ImportError as exc:
            raise MissingLlamaRuntimeError(
                "ExecuTorchPythonRuntimeBackend requires 'pytorch_tokenizers' package."
            ) from exc

        runtime = Runtime.get()
        self._program = runtime.load_program(str(self.pte_path))
        self._method = self._program.load_method("forward")
        self._tokenizer = get_tokenizer(str(self.tokenizer_path))

    @property
    def backend_name(self) -> str:
        return "executorch_python_runtime"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[str, float, float]:
        import torch

        prompt_ids: List[int] = self._tokenizer.encode(prompt, bos=False, eos=False)
        if len(prompt_ids) >= self.max_seq_len:
            raise ValueError(
                f"Prompt token length ({len(prompt_ids)}) exceeds max_seq_len ({self.max_seq_len})."
            )

        t_start = time.perf_counter()
        generated_ids: List[int] = []

        # Prefill phase
        pos = 0
        next_token = prompt_ids[0]
        for tok in prompt_ids:
            logits = self._method.execute(
                [
                    torch.tensor([[tok]], dtype=torch.long),
                    torch.tensor([pos], dtype=torch.long),
                ]
            )[0]
            pos += 1
            next_token = int(torch.argmax(logits[0, -1, :]).item())

        t_first = time.perf_counter()
        ttft_ms = (t_first - t_start) * 1000.0

        # Autoregressive KV-cache decode loop
        budget = min(max_new_tokens, self.max_seq_len - pos)
        for _ in range(max(0, budget)):
            if next_token in LLAMA3_EOS_IDS:
                break
            generated_ids.append(next_token)
            logits = self._method.execute(
                [
                    torch.tensor([[next_token]], dtype=torch.long),
                    torch.tensor([pos], dtype=torch.long),
                ]
            )[0]
            pos += 1
            if temperature > 0.0:
                probs = torch.softmax(logits[0, -1, :] / float(temperature), dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_token = int(torch.argmax(logits[0, -1, :]).item())

        t_end = time.perf_counter()
        decode_sec = max(1e-4, t_end - t_first)
        tokens_per_sec = max(1, len(generated_ids)) / decode_sec
        decoded_text = self._tokenizer.decode(generated_ids).strip()

        return decoded_text, round(ttft_ms, 3), round(tokens_per_sec, 3)


class ExecuTorchWorkerAdvisoryBackend(LlamaAdvisoryBackend):
    """
    Persistent out-of-process ExecuTorch Llama 3.2 worker backend running
    `executorch_worker.py` inside the isolated `executorch_env` environment.
    Keeps the quantized `.pte` model warm in memory across requests via
    thread-safe JSON-lines IPC (`shell=False`).
    """

    def __init__(
        self,
        pte_path: Path = DEFAULT_LLAMA_PTE,
        tokenizer_path: Path = DEFAULT_LLAMA_TOKENIZER,
        python_bin: Path = DEFAULT_EXECUTORCH_PYTHON,
        worker_script: Path = DEFAULT_EXECUTORCH_WORKER,
        timeout_sec: float = 120.0,
    ) -> None:
        self.pte_path = Path(pte_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.python_bin = Path(python_bin)
        self.worker_script = Path(worker_script)
        self.timeout_sec = float(timeout_sec)

        if not self.python_bin.is_file():
            raise MissingLlamaRuntimeError(
                f"ExecuTorch Python environment interpreter not found: {self.python_bin}"
            )
        if not self.pte_path.is_file():
            raise MissingLlamaRuntimeError(f"Llama .pte artifact not found: {self.pte_path}")
        if not self.tokenizer_path.is_file():
            raise MissingLlamaRuntimeError(f"Llama tokenizer not found: {self.tokenizer_path}")
        if not self.worker_script.is_file():
            raise MissingLlamaRuntimeError(f"ExecuTorch worker script not found: {self.worker_script}")

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[str]] = None
        atexit.register(self.close)

    @property
    def backend_name(self) -> str:
        return "executorch_llama_worker"

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        proc = subprocess.Popen(
            [
                str(self.python_bin),
                "-u",
                str(self.worker_script),
                "--pte",
                str(self.pte_path),
                "--tokenizer",
                str(self.tokenizer_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
        assert proc.stdout is not None
        ready_line = proc.stdout.readline()
        if not ready_line:
            stderr_tail = proc.stderr.read() if proc.stderr is not None else ""
            proc.kill()
            raise MissingLlamaRuntimeError(
                f"ExecuTorch worker failed during startup: {stderr_tail.strip()}"
            )
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            proc.kill()
            raise MissingLlamaRuntimeError(
                f"ExecuTorch worker emitted invalid startup line: {ready_line!r}"
            ) from exc

        if not ready.get("ready"):
            proc.kill()
            raise MissingLlamaRuntimeError(
                f"ExecuTorch worker failed to load model: {ready.get('error', 'unknown error')}"
            )

        self._proc = proc
        return proc

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[str, float, float]:
        with self._lock:
            proc = self._ensure_started()
            assert proc.stdin is not None and proc.stdout is not None
            req = {
                "op": "generate",
                "prompt": prompt,
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
            }
            proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            raw_line = proc.stdout.readline()
            if not raw_line:
                self.close()
                raise RuntimeError("ExecuTorch worker terminated unexpectedly during generation.")
            resp = json.loads(raw_line)
            if not resp.get("ok"):
                raise RuntimeError(f"ExecuTorch worker error: {resp.get('error', 'unknown')}")
            return (
                str(resp.get("text", "")),
                round(float(resp.get("ttft_ms", 0.0)), 3),
                round(float(resp.get("tokens_per_sec", 0.0)), 3),
            )

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=3)
        except Exception:
            pass
        finally:
            if proc.poll() is None:
                proc.kill()


class TestStubAdvisoryBackend(LlamaAdvisoryBackend):
    """
    TEST-ONLY DETERMINISTIC STUB BACKEND (CI & Unit Tests Only).

    Extracts the grounded `kb_entry` JSON embedded inside `prompt` and formats a
    deterministic JSON advisory payload without invoking any LLM weights.
    NEVER report outputs from this backend as real Llama 3.2 inference metrics.
    """

    def __init__(
        self,
        override_raw_output: Optional[str] = None,
        override_fn: Optional[Callable[[str], str]] = None,
        simulated_ttft_ms: float = 12.5,
        simulated_tokens_per_sec: float = 42.0,
    ) -> None:
        self._override_raw_output = override_raw_output
        self._override_fn = override_fn
        self._ttft_ms = float(simulated_ttft_ms)
        self._tps = float(simulated_tokens_per_sec)

    @property
    def backend_name(self) -> str:
        return "test_stub_ci_only"

    @staticmethod
    def _extract_prompt_section_json(prompt: str, header: str, end_marker: str) -> Dict[str, Any]:
        start_idx = prompt.find(header)
        if start_idx == -1:
            return {}
        after = prompt[start_idx + len(header) :]
        end_idx = after.find(end_marker)
        raw_block = after[:end_idx] if end_idx != -1 else after
        return json.loads(raw_block.strip())

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[str, float, float]:
        if self._override_fn is not None:
            return self._override_fn(prompt), self._ttft_ms, self._tps
        if self._override_raw_output is not None:
            return self._override_raw_output, self._ttft_ms, self._tps

        lang_match = re.search(r"Requested Response Language:\s*(en|hi|gu|mr|te|ta)\b", prompt)
        language = lang_match.group(1) if lang_match else "en"

        finding = self._extract_prompt_section_json(
            prompt, "VISION FINDING JSON:\n", "\n\nLOCAL GROUNDED KB TREATMENT CARD"
        )
        kb_data = self._extract_prompt_section_json(
            prompt,
            "LOCAL GROUNDED KB TREATMENT CARD (kb_entry):\n",
            "\n\nGenerate the grounded JSON advisory response now.",
        )

        condition = kb_data.get("condition", finding.get("condition_name", "Condition"))
        crop = kb_data.get("crop", finding.get("crop", "crop"))
        severity = finding.get("severity_level", "moderate")
        confidence = finding.get("confidence", 0.0)
        area_pct = finding.get("infected_area_pct", 0.0)

        cultural = list(kb_data.get("cultural_organic_control", []))
        chem_opts = kb_data.get("approved_chemical_options", [])
        chem_guidance = [
            f"{opt['active_ingredient']} ({opt['target_stage']}): {opt['safety_warning']}"
            for opt in chem_opts
            if isinstance(opt, dict) and "active_ingredient" in opt
        ]

        lang_intro = {
            "en": (
                f"Visual screening indicates likely {condition} on {crop} "
                f"(confidence {confidence:.2f}, {severity} severity, ~{area_pct}% area). "
                "This is an offline screening aid, not a laboratory confirmation."
            ),
            "hi": (
                f"कैमरा जांच में {crop} में {condition} के संकेत मिले हैं "
                f"(विश्वास {confidence:.2f}, गंभीरता {severity}, ~{area_pct}% क्षेत्र)। "
                "यह प्रयोगशाला जांच का विकल्प नहीं है।"
            ),
            "gu": (
                f"કેમેરા તપાસમાં {crop} માં {condition} ના લક્ષણો દેખાય છે "
                f"(વિશ્વાસ {confidence:.2f}, તીવ્રતા {severity}, ~{area_pct}% વિસ્તાર). "
                "આ પ્રયોગશાળા નિદાન નથી."
            ),
            "mr": (
                f"कॅमेरा तपासणीत {crop} मध्ये {condition} ची लक्षणे आढळली आहेत "
                f"(विश्वास {confidence:.2f}, तीव्रता {severity}, ~{area_pct}% क्षेत्र). "
                "ही प्रयोगशाळा तपासणी नाही."
            ),
            "te": (
                f"కెమెరా పరీక్షలో {crop} పంటపై {condition} లక్షణాలు కనిపించాయి "
                f"(నమ్మకం {confidence:.2f}, తీవ్రత {severity}, ~{area_pct}% ప్రాంతం). "
                "ఇది ప్రయోగశాల నిర్ధారణ కాదు."
            ),
            "ta": (
                f"கேமரா ஆய்வில் {crop} பயிரில் {condition} அறிகுறிகள் தென்படுகின்றன "
                f"(நம்பிக்கை {confidence:.2f}, தீவிரம் {severity}, ~{area_pct}% பரப்பு). "
                "இது ஆய்வக உறுதிப்படுத்தல் அல்ல."
            ),
        }

        payload = {
            "explanation": lang_intro[language],
            "immediate_actions": cultural[:1]
            or ["Inspect neighboring plants and isolate affected foliage."],
            "organic_cultural_steps": cultural
            or ["Follow recommended sanitation and cultural practices."],
            "chemical_guidance_if_approved": chem_guidance,
            "preventive_measures": [
                f"Monitor weekly for symptoms: {kb_data.get('symptoms', ['foliar lesions'])[0]}",
                kb_data.get("kvk_escalation_trigger", "Consult local KVK if spread continues."),
            ],
            "disclaimer": (
                "Offline AI screening only (not a lab diagnosis). Verify CIB&RC label "
                "dosage and Pre-Harvest Interval (PHI) with local KVK before spraying."
            ),
        }
        return json.dumps(payload, ensure_ascii=False), self._ttft_ms, self._tps
