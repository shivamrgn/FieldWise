"""
Pluggable Offline Inference Engine for FieldWise PS1 (Pipeline A).

Supports multiple hardware backends behind a unified `VisionBackend` interface:
  - `ExecuTorchVisionBackend`: `.pte` execution on Arm CPU (ExecuTorch Runtime)
  - `OnnxVisionBackend`: `.onnx` execution via `onnxruntime` CPUExecutionProvider
  - `HailoRTVisionBackend`: `.hef` execution on Raspberry Pi 5 + AI HAT+ (Hailo NPU)
  - `PyTorchVisionBackend`: `.pt2` / `nn.Module` local CPU inference

Enforces deterministic image quality gating, low-confidence (< 0.65) uncertainty
routing (`category="unknown"`), severity estimation, and strict `VisionFinding` output.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from ps1_offline_assistant.schemas.contracts import (
    SeverityLevel,
    VisionCategory,
    VisionFinding,
)
from ps1_offline_assistant.vision.download_pretrained import (
    DEFAULT_PRETRAINED_DIR,
    build_hf_to_kb_mapping,
)
from ps1_offline_assistant.vision.preprocess import (
    DEFAULT_MAX_BRIGHTNESS,
    DEFAULT_MIN_BLUR_SCORE,
    DEFAULT_MIN_BRIGHTNESS,
    assess_image_quality,
    preprocess_for_model,
)
from ps1_offline_assistant.vision.severity import (
    assess_leaf_foliar_coverage,
    estimate_severity,
)


DEFAULT_CONFIDENCE_THRESHOLD = 0.65


class MissingBackendDependencyError(RuntimeError):
    """Raised when a hardware-specific runtime (ExecuTorch, ONNX Runtime, HailoRT) is absent."""


class VisionBackend(ABC):
    """Abstract runtime backend returning raw 1D class logits for a `(1, 3, 224, 224)` input."""

    is_untrained_smoke: bool = False

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Identifier recorded in `VisionFinding.runtime_backend`."""

    @abstractmethod
    def predict_logits(self, input_nchw: np.ndarray) -> np.ndarray:
        """Executes forward pass and returns 1D float32 logits `(num_classes,)`."""


class ExecuTorchVisionBackend(VisionBackend):
    """Executes `.pte` models on Arm CPU using the ExecuTorch Python Runtime."""

    def __init__(self, pte_path: Path, method_name: str = "forward") -> None:
        self.pte_path = Path(pte_path)
        self.method_name = method_name
        if not self.pte_path.is_file():
            raise FileNotFoundError(f"ExecuTorch .pte artifact not found: {self.pte_path}")

        try:
            from executorch.runtime import Runtime
        except ImportError as exc:
            raise MissingBackendDependencyError(
                "ExecuTorchVisionBackend requires the 'executorch' runtime package. "
                "Install executorch on Arm64 / Linux to load .pte artifacts."
            ) from exc

        runtime = Runtime.get()
        self._program = runtime.load_program(str(self.pte_path))
        self._method = self._program.load_method(self.method_name)

    @property
    def backend_name(self) -> str:
        return "executorch_xnnpack_cpu"

    def predict_logits(self, input_nchw: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(input_nchw, dtype=np.float32))
        outputs = self._method.execute([tensor])
        logits = outputs[0].detach().cpu().numpy()
        return np.asarray(logits, dtype=np.float32).reshape(-1)


class OnnxVisionBackend(VisionBackend):
    """Executes static `.onnx` models locally via `onnxruntime` CPUExecutionProvider."""

    def __init__(self, onnx_path: Path) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise MissingBackendDependencyError(
                "OnnxVisionBackend requires 'onnxruntime'. Install onnxruntime to run .onnx inference."
            ) from exc

        self._session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    @property
    def backend_name(self) -> str:
        return "onnxruntime_cpu"

    def predict_logits(self, input_nchw: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(input_nchw, dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: arr})
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)


class HailoRTVisionBackend(VisionBackend):
    """
    Executes compiled `.hef` models on Raspberry Pi 5 + AI HAT+ (Hailo-8 / Hailo-8L NPU)
    via the `hailo_platform` HailoRT SDK.
    """

    def __init__(self, hef_path: Path) -> None:
        self.hef_path = Path(hef_path)
        if not self.hef_path.is_file():
            raise FileNotFoundError(f"Hailo .hef artifact not found: {self.hef_path}")

        try:
            from hailo_platform import (
                ConfigureParams,
                FormatType,
                HEF,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as exc:
            raise MissingBackendDependencyError(
                "HailoRTVisionBackend requires 'hailo_platform' (HailoRT SDK) and a "
                "connected Hailo-8/8L AI HAT+ NPU. Cannot initialize Hailo NPU on this host."
            ) from exc

        self._hef = HEF(str(self.hef_path))
        self._vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=self._hef,
            interface=HailoStreamInterface.PCIe,
        )
        self._network_group = self._vdevice.configure(self._hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()
        self._input_vstream_info = self._hef.get_input_vstream_infos()[0]
        self._output_vstream_info = self._hef.get_output_vstream_infos()[0]
        self._input_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        self._output_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        self._infer_vstreams_cls = InferVStreams

    @property
    def backend_name(self) -> str:
        return "hailort_npu"

    def predict_logits(self, input_nchw: np.ndarray) -> np.ndarray:
        # Hailo input streams use NHWC layout by default
        nhwc = np.ascontiguousarray(np.transpose(input_nchw, (0, 2, 3, 1)), dtype=np.float32)
        with self._network_group.activate(self._network_group_params):
            with self._infer_vstreams_cls(
                self._network_group, self._input_params, self._output_params
            ) as infer_pipeline:
                raw_outputs = infer_pipeline.infer({self._input_vstream_info.name: nhwc})
        logits = raw_outputs[self._output_vstream_info.name]
        return np.asarray(logits, dtype=np.float32).reshape(-1)


class PyTorchVisionBackend(VisionBackend):
    """
    Executes an in-memory PyTorch `nn.Module` or a saved `.pt2` `ExportedProgram`
    locally on CPU for offline development and integration verification.
    """

    def __init__(
        self,
        model_or_pt2: Union[nn.Module, Path],
        is_untrained_smoke: bool = False,
    ) -> None:
        self.is_untrained_smoke = bool(is_untrained_smoke)
        if isinstance(model_or_pt2, (str, Path)):
            pt2_path = Path(model_or_pt2)
            if not pt2_path.is_file():
                raise FileNotFoundError(f"PyTorch .pt2 artifact not found: {pt2_path}")
            exported = torch.export.load(pt2_path)
            self._callable = exported.module()
            self._is_exported = True
        elif isinstance(model_or_pt2, nn.Module):
            model_or_pt2.eval()
            self._callable = model_or_pt2
            self._is_exported = False
        else:
            raise TypeError(
                f"Expected nn.Module or .pt2 Path, got {type(model_or_pt2).__name__}."
            )

    @property
    def backend_name(self) -> str:
        return "pytorch_export_cpu" if self._is_exported else "pytorch_cpu"

    def predict_logits(self, input_nchw: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(input_nchw, dtype=np.float32))
        with torch.inference_mode():
            logits = self._callable(tensor)
        return logits.detach().cpu().numpy().astype(np.float32).reshape(-1)


class HuggingFaceLocalVisionBackend(VisionBackend):
    """
    100% offline PlantVillage vision backend loading pretrained Hugging Face
    checkpoints strictly from disk (`local_files_only=True`, zero network calls).
    Supports both `transformers` local loading and zero-`transformers` native
    `.pt2` fallback when `plantvillage_mobilenetv2.pt2` was exported during setup.
    """

    def __init__(self, model_dir: Union[str, Path] = DEFAULT_PRETRAINED_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.is_untrained_smoke = False
        self._use_pt2_fallback = False

        config_file = self.model_dir / "config.json"
        pt2_file = self.model_dir / "plantvillage_mobilenetv2.pt2"
        has_weights = (
            (self.model_dir / "model.safetensors").is_file()
            or (self.model_dir / "pytorch_model.bin").is_file()
            or pt2_file.is_file()
        )
        if not self.model_dir.is_dir() or not config_file.is_file() or not has_weights:
            raise MissingBackendDependencyError(
                f"Local pretrained PlantVillage model not found in '{self.model_dir}'. "
                "Run 'python3 ps1_offline_assistant/vision/download_pretrained.py' once "
                "while online to cache weights for 100% offline inference."
            )

        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForImageClassification,
            )

            self._model = AutoModelForImageClassification.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            )
            self._model.eval()
            try:
                self._processor = AutoImageProcessor.from_pretrained(
                    str(self.model_dir),
                    local_files_only=True,
                )
            except ValueError:
                if getattr(self._model.config, "model_type", "") == "mobilenet_v2":
                    from transformers import MobileNetV2ImageProcessor

                    self._processor = MobileNetV2ImageProcessor.from_pretrained(
                        str(self.model_dir),
                        local_files_only=True,
                    )
                else:
                    raise
            raw_id2label = getattr(self._model.config, "id2label", {})
        except ImportError as exc:
            if not pt2_file.is_file():
                raise MissingBackendDependencyError(
                    "HuggingFaceLocalVisionBackend requires 'transformers' and 'safetensors' "
                    "(or a locally exported 'plantvillage_mobilenetv2.pt2'). "
                    "Run 'ps1_offline_assistant/.venv/bin/python -m ps1_offline_assistant.vision.download_pretrained' once."
                ) from exc

            self._use_pt2_fallback = True
            self._pt2_module = torch.export.load(pt2_file).module()
            cfg_json = json.loads(config_file.read_text(encoding="utf-8"))
            raw_id2label = cfg_json.get("id2label", {})
            preproc_path = self.model_dir / "preprocessor_config.json"
            preproc = (
                json.loads(preproc_path.read_text(encoding="utf-8"))
                if preproc_path.is_file()
                else {}
            )
            self._mean = np.array(preproc.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            self._std = np.array(preproc.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
            self._do_center_crop = bool(preproc.get("do_center_crop", False))
            size_obj = preproc.get("size", {"height": 224, "width": 224})
            crop_obj = preproc.get("crop_size", {})
            self._resize_short_edge: Optional[int] = (
                int(size_obj["shortest_edge"])
                if isinstance(size_obj, dict) and size_obj.get("shortest_edge") is not None
                else None
            )
            if self._do_center_crop and isinstance(crop_obj, dict) and crop_obj.get("height") and crop_obj.get("width"):
                self._target_w = int(crop_obj["width"])
                self._target_h = int(crop_obj["height"])
            elif isinstance(size_obj, dict):
                self._target_w = int(size_obj.get("width", size_obj.get("shortest_edge", 224)))
                self._target_h = int(size_obj.get("height", size_obj.get("shortest_edge", 224)))
            else:
                self._target_w = self._target_h = 224

        mapping_file = self.model_dir / "kb_label_mapping.json"
        if mapping_file.is_file():
            raw_map = json.loads(mapping_file.read_text(encoding="utf-8"))
            self.class_labels: List[Dict[str, Any]] = raw_map["classes"]
        else:
            self.class_labels = build_hf_to_kb_mapping(raw_id2label)["classes"]

    @property
    def backend_name(self) -> str:
        return "huggingface_local_offline"

    def predict_bgr_logits(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image

        if self._use_pt2_fallback:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_rgb = Image.fromarray(rgb)
            if self._do_center_crop and self._resize_short_edge is not None:
                w_in, h_in = pil_rgb.size
                short = min(h_in, w_in)
                scale = float(self._resize_short_edge) / float(max(short, 1))
                new_w = max(self._target_w, int(round(w_in * scale)))
                new_h = max(self._target_h, int(round(h_in * scale)))
                resized_pil = pil_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
                y0 = max(0, (new_h - self._target_h) // 2)
                x0 = max(0, (new_w - self._target_w) // 2)
                cropped_pil = resized_pil.crop((x0, y0, x0 + self._target_w, y0 + self._target_h))
            else:
                cropped_pil = pil_rgb.resize((self._target_w, self._target_h), Image.Resampling.BILINEAR)
            arr = np.asarray(cropped_pil, dtype=np.float32) / 255.0
            arr = (arr - self._mean) / self._std
            chw = np.transpose(arr, (2, 0, 1))
            nchw = np.ascontiguousarray(np.expand_dims(chw, axis=0), dtype=np.float32)
            return self.predict_logits(nchw)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt")
        with torch.inference_mode():
            outputs = self._model(**inputs)
        return outputs.logits.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def predict_logits(self, input_nchw: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(input_nchw, dtype=np.float32))
        with torch.inference_mode():
            if self._use_pt2_fallback:
                logits = self._pt2_module(tensor)
                return logits.detach().cpu().numpy().astype(np.float32).reshape(-1)
            outputs = self._model(pixel_values=tensor)
            return outputs.logits.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _load_class_labels(
    class_labels: Union[Path, str, Dict[str, Any], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if isinstance(class_labels, (str, Path)):
        raw = json.loads(Path(class_labels).read_text(encoding="utf-8"))
        entries = raw["classes"] if isinstance(raw, dict) else raw
    elif isinstance(class_labels, dict):
        entries = class_labels["classes"]
    elif isinstance(class_labels, list):
        entries = class_labels
    else:
        raise TypeError(f"Unsupported class_labels type: {type(class_labels).__name__}")

    if not entries:
        raise ValueError("class_labels manifest must not be empty.")
    sorted_entries = sorted(entries, key=lambda c: int(c["index"]))
    return sorted_entries


class VisionPipeline:
    """
    End-to-end Pipeline A orchestrator:
      1. OpenCV image quality gate (`assess_image_quality`)
      2. ImageNet NCHW preprocessing (`preprocess_for_model`)
      3. Pluggable backend inference (`VisionBackend`)
      4. Uncertainty / low-confidence (< 0.65) safety gating
      5. Deterministic severity estimation (`estimate_severity`)
      6. Strict `VisionFinding` schema emission
    """

    def __init__(
        self,
        backend: VisionBackend,
        class_labels: Optional[Union[Path, str, Dict[str, Any], List[Dict[str, Any]]]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_blur_score: float = DEFAULT_MIN_BLUR_SCORE,
        min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
        max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
    ) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0.0, 1.0], got {confidence_threshold}"
            )
        self.backend = backend
        effective_labels = (
            class_labels
            if class_labels is not None
            else getattr(backend, "class_labels", None)
        )
        if effective_labels is None:
            raise ValueError("class_labels must be provided when backend has no class_labels.")
        self.classes = _load_class_labels(effective_labels)
        self.confidence_threshold = float(confidence_threshold)
        self.min_blur_score = float(min_blur_score)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)

    def predict(
        self,
        image_bgr: np.ndarray,
        pest_boxes: Optional[Sequence[Sequence[float]]] = None,
    ) -> VisionFinding:
        """Executes Pipeline A on a BGR image and returns a strictly validated `VisionFinding`."""
        finding, _, _ = self.predict_with_details(image_bgr, pest_boxes=pest_boxes)
        return finding

    @staticmethod
    def _extract_leaf_roi(
        image_bgr: np.ndarray,
        leaf_bbox: Sequence[int],
        pad_scale: float = 1.2,
    ) -> np.ndarray:
        """Extracts a square padded ROI centered around an off-center detected leaf."""
        h_img, w_img = image_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in leaf_bbox]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        side = int(round(max(bw, bh) * pad_scale))
        side = min(max(side, 64), max(h_img, w_img))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx1 = max(0, min(w_img - min(side, w_img), cx - side // 2))
        ry1 = max(0, min(h_img - min(side, h_img), cy - side // 2))
        rx2 = min(w_img, rx1 + side)
        ry2 = min(h_img, ry1 + side)
        if (rx2 - rx1) < 32 or (ry2 - ry1) < 32:
            return image_bgr
        return image_bgr[ry1:ry2, rx1:rx2]

    def predict_with_details(
        self,
        image_bgr: np.ndarray,
        pest_boxes: Optional[Sequence[Sequence[float]]] = None,
        crop_hint: Optional[str] = None,
        top_k: int = 3,
    ) -> Tuple[VisionFinding, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes Pipeline A and returns `(VisionFinding, top_k_predictions, diagnostics)`.

        Enforces:
          1. Image sharpness and exposure quality gating
          2. Non-leaf / insufficient-foliage rejection
          3. Off-center leaf ROI framing assist when center crop misses foliage
          4. Crop-hint compatibility check
          5. Multi-signal uncertainty gating (threshold + margin + normalized entropy)
        """
        t0 = time.perf_counter()

        quality = assess_image_quality(
            image_bgr,
            min_blur_score=self.min_blur_score,
            min_brightness=self.min_brightness,
            max_brightness=self.max_brightness,
        )

        # Gate 1: Reject blurry / underexposed / overexposed frames before classification
        if not quality.image_quality_ok:
            elapsed_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
            return (
                VisionFinding(
                    crop="unknown",
                    condition_name="unclear_image",
                    category=VisionCategory.UNKNOWN,
                    confidence=0.0,
                    severity_level=SeverityLevel.UNDETERMINED,
                    infected_area_pct=0.0,
                    image_quality_ok=False,
                    blur_score=quality.blur_score,
                    inference_latency_ms=round(elapsed_ms, 4),
                    runtime_backend=self.backend.backend_name,
                ),
                [],
                {
                    "diagnosis_status": "poor_image_quality",
                    "uncertainty_reason": "poor_image_quality",
                    "quality_issues": list(quality.issues),
                    "severity_note": "Undetermined due to poor image quality.",
                },
            )

        # Gate 1b: Non-leaf / no-plant foliage check (avoids forcing 38 disease labels on desks/walls/text)
        foliar_stats = assess_leaf_foliar_coverage(image_bgr)
        if foliar_stats["foliar_green_yellow_pct"] < 0.8 and foliar_stats["leaf_area_pct"] < 1.5:
            elapsed_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
            return (
                VisionFinding(
                    crop="unknown",
                    condition_name="non_leaf_or_no_plant",
                    category=VisionCategory.UNKNOWN,
                    confidence=0.0,
                    severity_level=SeverityLevel.UNDETERMINED,
                    infected_area_pct=0.0,
                    image_quality_ok=True,
                    blur_score=quality.blur_score,
                    inference_latency_ms=round(elapsed_ms, 4),
                    runtime_backend=self.backend.backend_name,
                ),
                [],
                {
                    "diagnosis_status": "non_leaf_image",
                    "uncertainty_reason": "non_leaf_or_no_plant",
                    "foliar_green_yellow_pct": foliar_stats["foliar_green_yellow_pct"],
                    "leaf_coverage_pct": foliar_stats["leaf_area_pct"],
                    "framing_mode": "rejected_non_leaf",
                    "severity_note": "No plant foliage detected in image.",
                },
            )

        # Off-center field leaf framing assist: preserve canonical center-crop whenever center has foliage
        framing_mode = "standard_center_crop"
        eval_bgr = image_bgr
        if foliar_stats["center_leaf_pct"] < 10.0 and foliar_stats["leaf_bbox"] is not None:
            candidate_roi = self._extract_leaf_roi(image_bgr, foliar_stats["leaf_bbox"])
            if candidate_roi.shape[:2] != image_bgr.shape[:2]:
                eval_bgr = candidate_roi
                framing_mode = "auto_leaf_roi_crop"

        # Preprocess & Run Edge Backend
        if hasattr(self.backend, "predict_bgr_logits"):
            logits = self.backend.predict_bgr_logits(eval_bgr)
        else:
            input_nchw = preprocess_for_model(eval_bgr, size=(224, 224))
            logits = self.backend.predict_logits(input_nchw)
        if logits.shape[0] != len(self.classes):
            raise ValueError(
                f"Model output logits dimension ({logits.shape[0]}) does not match "
                f"class_labels count ({len(self.classes)})."
            )

        probs = _stable_softmax(logits)
        ranked_indices = [int(i) for i in np.argsort(-probs)]
        top_predictions: List[Dict[str, Any]] = [
            {
                "crop": self.classes[idx]["crop"],
                "condition_name": self.classes[idx]["condition_name"],
                "category": self.classes[idx]["category"],
                "confidence": round(float(np.clip(probs[idx], 0.0, 1.0)), 4),
                "label": self.classes[idx].get("hf_label") or self.classes[idx]["label"],
            }
            for idx in ranked_indices[: max(1, int(top_k))]
        ]

        top1_prob = float(np.clip(probs[ranked_indices[0]], 0.0, 1.0))
        top2_prob = float(np.clip(probs[ranked_indices[1]], 0.0, 1.0)) if len(ranked_indices) > 1 else 0.0
        prob_margin = round(max(0.0, top1_prob - top2_prob), 4)
        norm_entropy = round(
            float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))) / np.log(max(2, len(probs)))),
            4,
        )

        # Honest unsupported-crop check when farmer specifies crop_hint
        norm_hint = (crop_hint or "").strip().lower()
        supported_crops = {
            str(c.get("crop", "")).lower()
            for c in self.classes
            if c.get("crop") not in {"general", "unknown"}
        }
        base_diag: Dict[str, Any] = {
            "foliar_green_yellow_pct": foliar_stats["foliar_green_yellow_pct"],
            "leaf_coverage_pct": foliar_stats["leaf_area_pct"],
            "framing_mode": framing_mode,
            "top1_top2_margin": prob_margin,
            "normalized_entropy": norm_entropy,
            "severity_note": "Visual HSV/CIE Lab color-segmentation estimate only; not clinically validated agronomic severity.",
        }

        if norm_hint and norm_hint not in {"auto", "unknown", ""} and norm_hint not in supported_crops:
            elapsed_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
            base_diag.update(
                {
                    "diagnosis_status": "unsupported_crop",
                    "uncertainty_reason": "unsupported_crop",
                }
            )
            return (
                VisionFinding(
                    crop=norm_hint,
                    condition_name=f"Unsupported crop in vision model ({norm_hint})",
                    category=VisionCategory.UNKNOWN,
                    confidence=0.0,
                    severity_level=SeverityLevel.UNDETERMINED,
                    infected_area_pct=0.0,
                    image_quality_ok=True,
                    blur_score=quality.blur_score,
                    inference_latency_ms=round(elapsed_ms, 4),
                    runtime_backend=self.backend.backend_name,
                ),
                top_predictions,
                base_diag,
            )

        top_idx = ranked_indices[0]
        confidence = top1_prob
        label_entry = self.classes[top_idx]

        # Crop mismatch guard when user explicitly selected a supported crop
        if norm_hint and norm_hint in supported_crops and str(label_entry.get("crop", "")).lower() != norm_hint:
            elapsed_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
            base_diag.update(
                {
                    "diagnosis_status": "crop_mismatch_uncertain",
                    "uncertainty_reason": f"Selected '{norm_hint}', model predicted '{label_entry.get('crop')}'",
                }
            )
            return (
                VisionFinding(
                    crop=norm_hint,
                    condition_name=f"Crop mismatch: visual match resembles {label_entry['crop']} ({label_entry['condition_name']})",
                    category=VisionCategory.UNKNOWN,
                    confidence=round(confidence, 6),
                    severity_level=SeverityLevel.UNDETERMINED,
                    infected_area_pct=0.0,
                    image_quality_ok=True,
                    blur_score=quality.blur_score,
                    inference_latency_ms=round(elapsed_ms, 4),
                    runtime_backend=self.backend.backend_name,
                ),
                top_predictions,
                base_diag,
            )

        # Gate 2: Multi-signal uncertainty routing (probability + margin + Shannon entropy)
        is_ambiguous = (prob_margin < 0.15) or (norm_entropy > 0.55)
        if confidence < self.confidence_threshold or is_ambiguous:
            severity = estimate_severity(
                image_bgr=eval_bgr,
                category=VisionCategory.UNKNOWN,
                image_quality_ok=True,
            )
            elapsed_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
            candidate_crop = (
                "unknown" if label_entry["crop"] == "general" else label_entry["crop"]
            )
            reason = (
                "ambiguous_multiple_conditions"
                if (is_ambiguous and confidence >= 0.40)
                else "low_confidence"
            )
            base_diag.update(
                {
                    "diagnosis_status": reason,
                    "uncertainty_reason": reason,
                }
            )
            return (
                VisionFinding(
                    crop=candidate_crop,
                    condition_name=label_entry["condition_name"],
                    category=VisionCategory.UNKNOWN,
                    confidence=round(confidence, 6),
                    severity_level=severity.severity_level,
                    infected_area_pct=severity.infected_area_pct,
                    image_quality_ok=True,
                    blur_score=quality.blur_score,
                    inference_latency_ms=round(elapsed_ms, 4),
                    runtime_backend=self.backend.backend_name,
                ),
                top_predictions,
                base_diag,
            )

        # Gate 3: Confident classification + deterministic severity estimation
        category = VisionCategory(label_entry["category"])
        severity = estimate_severity(
            image_bgr=eval_bgr,
            category=category,
            image_quality_ok=True,
            pest_boxes=pest_boxes,
        )
        elapsed_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
        diag_status = (
            "high_confidence_supported"
            if (category == VisionCategory.HEALTHY or label_entry.get("kb_card_id") is not None)
            else "unsupported_disease_unmapped_kb"
        )
        base_diag.update(
            {
                "diagnosis_status": diag_status,
                "uncertainty_reason": None,
                "severity_method": severity.estimation_method,
            }
        )

        return (
            VisionFinding(
                crop=label_entry["crop"],
                condition_name=label_entry["condition_name"],
                category=category,
                confidence=round(confidence, 6),
                severity_level=severity.severity_level,
                infected_area_pct=severity.infected_area_pct,
                image_quality_ok=True,
                blur_score=quality.blur_score,
                inference_latency_ms=round(elapsed_ms, 4),
                runtime_backend=self.backend.backend_name,
            ),
            top_predictions,
            base_diag,
        )
