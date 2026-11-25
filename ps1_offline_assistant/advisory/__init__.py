"""Offline Advisory Pipeline (Pipeline B — Llama 3.2 + ExecuTorch) for FieldWise PS1."""

from .engine import AdvisoryEngine
from .prompt_builder import build_llama3_prompt
from .runtime import (
    ExecuTorchLlamaCliBackend,
    ExecuTorchPythonRuntimeBackend,
    ExecuTorchWorkerAdvisoryBackend,
    LlamaAdvisoryBackend,
    MissingLlamaRuntimeError,
    TestStubAdvisoryBackend,
)

__all__ = [
    "AdvisoryEngine",
    "ExecuTorchLlamaCliBackend",
    "ExecuTorchPythonRuntimeBackend",
    "ExecuTorchWorkerAdvisoryBackend",
    "LlamaAdvisoryBackend",
    "MissingLlamaRuntimeError",
    "TestStubAdvisoryBackend",
    "build_llama3_prompt",
]
