"""Offline Vision Pipeline (Pipeline A) for FieldWise PS1."""

from typing import Any

__all__ = [
    "ExecuTorchVisionBackend",
    "HailoRTVisionBackend",
    "HuggingFaceLocalVisionBackend",
    "ImageQualityReport",
    "MissingBackendDependencyError",
    "OnnxVisionBackend",
    "PyTorchVisionBackend",
    "SeverityEstimate",
    "VisionBackend",
    "VisionPipeline",
    "assess_image_quality",
    "estimate_severity",
    "preprocess_for_model",
]

_PREPROCESS_EXPORTS = {
    "ImageQualityReport",
    "assess_image_quality",
    "preprocess_for_model",
}
_SEVERITY_EXPORTS = {"SeverityEstimate", "estimate_severity"}
_INFERENCE_EXPORTS = {
    "ExecuTorchVisionBackend",
    "HailoRTVisionBackend",
    "HuggingFaceLocalVisionBackend",
    "MissingBackendDependencyError",
    "OnnxVisionBackend",
    "PyTorchVisionBackend",
    "VisionBackend",
    "VisionPipeline",
}


def __getattr__(name: str) -> Any:
    if name in _PREPROCESS_EXPORTS:
        from . import preprocess

        return getattr(preprocess, name)
    if name in _SEVERITY_EXPORTS:
        from . import severity

        return getattr(severity, name)
    if name in _INFERENCE_EXPORTS:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
