"""
Phase 2 Verification Suite: Pipeline A Offline Vision & Edge Export.

Tests:
  1. OpenCV image quality gating (sharp vs blurry, dark, overexposed).
  2. ImageNet NCHW float32 preprocessing (`preprocess_for_model`).
  3. Deterministic severity estimation on synthetic healthy (0%), low (<15%),
     moderate (~25%), and severe (~55%) diseased leaves, plus pest boxes/contours.
  4. MobileNetV3 model construction from `agri_kb.json`, PlantVillage/IP102
     ImageFolder class mapping, `.pt2` ExportedProgram round-trip, and
     conditional/dependency-safe ONNX, ExecuTorch, and HailoRT backends.
  5. `VisionPipeline` confidence (< 0.65) & image-quality gates producing
     strict `VisionFinding` JSON contracts with real `inference_latency_ms`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.schemas.contracts import (  # noqa: E402
    SeverityLevel,
    VisionCategory,
    VisionFinding,
)
from ps1_offline_assistant.vision.inference import (  # noqa: E402
    ExecuTorchVisionBackend,
    HailoRTVisionBackend,
    MissingBackendDependencyError,
    OnnxVisionBackend,
    PyTorchVisionBackend,
    VisionBackend,
    VisionPipeline,
)
from ps1_offline_assistant.vision.preprocess import (  # noqa: E402
    assess_image_quality,
    preprocess_for_model,
)
from ps1_offline_assistant.vision.severity import estimate_severity  # noqa: E402
from ps1_offline_assistant.vision.train_and_export import (  # noqa: E402
    DEFAULT_KB_PATH,
    EdgeExportDependencyError,
    build_class_labels_from_kb,
    build_vision_model,
    export_to_executorch,
    export_to_onnx,
    export_to_torch_exported,
    map_imagefolder_to_kb_classes,
    save_class_labels_manifest,
)


def _make_synthetic_leaf(lesion_fraction: float = 0.0, size: int = 320) -> np.ndarray:
    """
    Generates a deterministic BGR image containing a crisp green leaf ellipse
    (with darker green veins for sharp edge gradient) on a neutral light-gray
    background, optionally painted with brown necrotic lesion pixels covering
    `lesion_fraction` of the leaf area.
    """
    img = np.full((size, size, 3), (215, 215, 215), dtype=np.uint8)
    center = (size // 2, size // 2)
    axes = (int(size * 0.34), int(size * 0.22))

    leaf_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(leaf_mask, center, axes, 0, 0, 360, 255, thickness=-1)

    # Healthy green leaf interior + crisp midrib/veins
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
        total_leaf_px = len(ys)
        target_lesion_px = int(round(total_leaf_px * lesion_fraction))
        # Paint contiguous left-to-right lesion band across the leaf for exact area control
        order = np.argsort(xs + 0.15 * ys)
        lesion_idx = order[:target_lesion_px]
        img[ys[lesion_idx], xs[lesion_idx]] = (30, 68, 128)  # BGR brown necrotic tone

    return img


class _FixedLogitsBackend(VisionBackend):
    """Deterministic in-memory logits backend for testing confidence & severity gates."""

    def __init__(self, logits: np.ndarray, name: str = "deterministic_test_backend") -> None:
        self._logits = np.asarray(logits, dtype=np.float32)
        self._name = name

    @property
    def backend_name(self) -> str:
        return self._name

    def predict_logits(self, input_nchw: np.ndarray) -> np.ndarray:
        if input_nchw.shape != (1, 3, 224, 224):
            raise AssertionError(f"Expected (1, 3, 224, 224), got {input_nchw.shape}")
        return self._logits


class TestImageQualityAndPreprocess(unittest.TestCase):
    def test_sharp_vs_blurry_dark_overexposed(self) -> None:
        sharp_leaf = _make_synthetic_leaf(lesion_fraction=0.0)
        sharp_q = assess_image_quality(sharp_leaf)
        self.assertTrue(sharp_q.image_quality_ok, f"Unexpected issues: {sharp_q.issues}")
        self.assertGreater(sharp_q.blur_score, 50.0)

        blurry = cv2.GaussianBlur(sharp_leaf, (35, 35), 12.0)
        blur_q = assess_image_quality(blurry)
        self.assertFalse(blur_q.image_quality_ok)
        self.assertLess(blur_q.blur_score, 50.0)

        dark = np.full((224, 224, 3), 12, dtype=np.uint8)
        dark_q = assess_image_quality(dark)
        self.assertFalse(dark_q.image_quality_ok)

        overexposed = np.full((224, 224, 3), 245, dtype=np.uint8)
        over_q = assess_image_quality(overexposed)
        self.assertFalse(over_q.image_quality_ok)

    def test_preprocess_for_model_nchw_float32(self) -> None:
        img = _make_synthetic_leaf(lesion_fraction=0.25)
        tensor = preprocess_for_model(img, size=(224, 224))
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(tensor)))


class TestDeterministicSeverityEstimator(unittest.TestCase):
    def test_healthy_moderate_and_severe_synthetic_leaves(self) -> None:
        healthy_img = _make_synthetic_leaf(lesion_fraction=0.0)
        healthy_sev = estimate_severity(healthy_img, "healthy", image_quality_ok=True)
        self.assertEqual(healthy_sev.severity_level, SeverityLevel.LOW)
        self.assertEqual(healthy_sev.infected_area_pct, 0.0)

        # Low severity (< 15%)
        low_img = _make_synthetic_leaf(lesion_fraction=0.08)
        low_sev = estimate_severity(low_img, "disease", image_quality_ok=True)
        self.assertEqual(low_sev.severity_level, SeverityLevel.LOW)
        self.assertGreater(low_sev.infected_area_pct, 3.0)
        self.assertLess(low_sev.infected_area_pct, 15.0)

        # Moderate severity (~25%)
        mod_img = _make_synthetic_leaf(lesion_fraction=0.25)
        mod_sev = estimate_severity(mod_img, "disease", image_quality_ok=True)
        self.assertEqual(mod_sev.severity_level, SeverityLevel.MODERATE)
        self.assertGreaterEqual(mod_sev.infected_area_pct, 18.0)
        self.assertLessEqual(mod_sev.infected_area_pct, 32.0)

        # Severe severity (~55%)
        sev_img = _make_synthetic_leaf(lesion_fraction=0.55)
        sev_res = estimate_severity(sev_img, "disease", image_quality_ok=True)
        self.assertEqual(sev_res.severity_level, SeverityLevel.SEVERE)
        self.assertGreater(sev_res.infected_area_pct, 45.0)

    def test_gated_and_pest_severity_modes(self) -> None:
        img = _make_synthetic_leaf(lesion_fraction=0.25)
        bad_q = estimate_severity(img, "disease", image_quality_ok=False)
        self.assertEqual(bad_q.severity_level, SeverityLevel.UNDETERMINED)
        self.assertEqual(bad_q.infected_area_pct, 0.0)

        unk = estimate_severity(img, "unknown", image_quality_ok=True)
        self.assertEqual(unk.severity_level, SeverityLevel.UNDETERMINED)
        self.assertEqual(unk.infected_area_pct, 0.0)

        # Pest bounding-box mode (4 pests -> moderate)
        pest_mod = estimate_severity(
            _make_synthetic_leaf(0.0),
            "pest",
            image_quality_ok=True,
            pest_boxes=[[10, 10, 30, 30], [40, 40, 60, 60], [70, 70, 90, 90], [100, 100, 120, 120]],
        )
        self.assertEqual(pest_mod.severity_level, SeverityLevel.MODERATE)


class TestModelExportAndVisionPipeline(unittest.TestCase):
    def test_kb_manifest_and_imagefolder_mapping(self) -> None:
        classes = build_class_labels_from_kb(DEFAULT_KB_PATH)
        self.assertEqual(len(classes), 18)  # 17 disease/pest cards + 1 healthy class
        self.assertEqual(classes[-1]["label"], "healthy")

        mapping = map_imagefolder_to_kb_classes(
            [
                "Tomato___Early_blight",
                "Potato___Late_blight",
                "Tomato___healthy",
                "Unknown_Class_XYZ",
            ]
        )
        self.assertIsNotNone(mapping["Tomato___Early_blight"])
        self.assertIsNotNone(mapping["Potato___Late_blight"])
        self.assertEqual(mapping["Tomato___healthy"], 17)
        self.assertIsNone(mapping["Unknown_Class_XYZ"])

    def test_mobilenetv3_export_and_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            manifest = save_class_labels_manifest(out_dir / "class_labels.json")
            model = build_vision_model(
                num_classes=manifest["num_classes"],
                arch="mobilenet_v3_small",
            )

            # Native PyTorch ExportedProgram (.pt2) export + round-trip inference
            pt2_path = export_to_torch_exported(model, out_dir / "mobilenet_v3_small.pt2")
            self.assertTrue(pt2_path.is_file())
            self.assertGreater(pt2_path.stat().st_size, 0)

            backend = PyTorchVisionBackend(pt2_path)
            pipeline = VisionPipeline(backend=backend, class_labels=out_dir / "class_labels.json")

            finding = pipeline.predict(_make_synthetic_leaf(lesion_fraction=0.25))
            self.assertIsInstance(finding, VisionFinding)
            self.assertGreater(finding.inference_latency_ms, 0.0)
            self.assertEqual(finding.runtime_backend, "pytorch_export_cpu")
            # Round-trip strict JSON check
            parsed = VisionFinding.model_validate_json(finding.model_dump_json())
            self.assertEqual(parsed, finding)

            # Conditional ONNX export check (real export if installed; strict dependency error if absent)
            onnx_out = out_dir / "mobilenet_v3_small.onnx"
            if importlib.util.find_spec("onnx") is not None:
                export_to_onnx(model, onnx_out)
                self.assertTrue(onnx_out.is_file())
                if importlib.util.find_spec("onnxruntime") is not None:
                    ort_backend = OnnxVisionBackend(onnx_out)
                    ort_pipeline = VisionPipeline(ort_backend, manifest)
                    ort_finding = ort_pipeline.predict(_make_synthetic_leaf(0.25))
                    self.assertEqual(ort_finding.runtime_backend, "onnxruntime_cpu")
                    self.assertGreater(ort_finding.inference_latency_ms, 0.0)
            else:
                with self.assertRaises(EdgeExportDependencyError):
                    export_to_onnx(model, onnx_out)
                self.assertFalse(onnx_out.exists())

            # Conditional ExecuTorch export check (real .pte if installed; strict error after torch.export if absent)
            pte_out = out_dir / "mobilenet_v3_small.pte"
            if importlib.util.find_spec("executorch") is not None:
                export_to_executorch(model, pte_out)
                self.assertTrue(pte_out.is_file())
                et_backend = ExecuTorchVisionBackend(pte_out)
                et_pipeline = VisionPipeline(et_backend, manifest)
                et_finding = et_pipeline.predict(_make_synthetic_leaf(0.25))
                self.assertEqual(et_finding.runtime_backend, "executorch_xnnpack_cpu")
            else:
                with self.assertRaises(EdgeExportDependencyError):
                    export_to_executorch(model, pte_out)
                self.assertFalse(pte_out.exists())

    def test_backend_missing_dependency_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_file = Path(tmpdir) / "dummy.bin"
            dummy_file.write_bytes(b"placeholder")

            if importlib.util.find_spec("hailo_platform") is None:
                with self.assertRaises(MissingBackendDependencyError):
                    HailoRTVisionBackend(dummy_file)
            if importlib.util.find_spec("onnxruntime") is None:
                with self.assertRaises(MissingBackendDependencyError):
                    OnnxVisionBackend(dummy_file)
            if importlib.util.find_spec("executorch") is None:
                with self.assertRaises(MissingBackendDependencyError):
                    ExecuTorchVisionBackend(dummy_file)

    def test_quality_and_low_confidence_gates(self) -> None:
        classes = build_class_labels_from_kb(DEFAULT_KB_PATH)
        num_classes = len(classes)
        rice_blast_idx = next(c["index"] for c in classes if c["label"] == "rice_blast")

        # 1. High-confidence disease logits -> confident finding + ~25% moderate severity
        high_logits = np.full(num_classes, -5.0, dtype=np.float32)
        high_logits[rice_blast_idx] = 5.0
        pipeline_hi = VisionPipeline(_FixedLogitsBackend(high_logits), classes)

        sharp_mod_leaf = _make_synthetic_leaf(lesion_fraction=0.25)
        finding_hi = pipeline_hi.predict(sharp_mod_leaf)
        self.assertTrue(finding_hi.image_quality_ok)
        self.assertEqual(finding_hi.crop, "rice")
        self.assertEqual(finding_hi.condition_name, "Rice Blast")
        self.assertEqual(finding_hi.category, VisionCategory.DISEASE)
        self.assertGreater(finding_hi.confidence, 0.95)
        self.assertEqual(finding_hi.severity_level, SeverityLevel.MODERATE)
        self.assertGreaterEqual(finding_hi.infected_area_pct, 18.0)

        # 2. Blurry image -> short-circuits before model inference
        blurry_leaf = cv2.GaussianBlur(sharp_mod_leaf, (35, 35), 12.0)
        finding_blur = pipeline_hi.predict(blurry_leaf)
        self.assertFalse(finding_blur.image_quality_ok)
        self.assertEqual(finding_blur.category, VisionCategory.UNKNOWN)
        self.assertEqual(finding_blur.condition_name, "unclear_image")
        self.assertEqual(finding_blur.severity_level, SeverityLevel.UNDETERMINED)
        self.assertEqual(finding_blur.infected_area_pct, 0.0)

        # 3. Low-confidence (< 0.65) logits -> routed to category="unknown" with raw confidence preserved
        # With 18 classes: logit[target]=2.2 and others=0.0 -> softmax prob ~ 9.02 / (9.02 + 17) ~ 0.347 < 0.65
        low_logits = np.zeros(num_classes, dtype=np.float32)
        low_logits[rice_blast_idx] = 2.2
        pipeline_lo = VisionPipeline(_FixedLogitsBackend(low_logits), classes)

        finding_lo = pipeline_lo.predict(sharp_mod_leaf)
        self.assertTrue(finding_lo.image_quality_ok)
        self.assertEqual(finding_lo.category, VisionCategory.UNKNOWN)
        self.assertLess(finding_lo.confidence, 0.65)
        self.assertGreater(finding_lo.confidence, 0.25)
        self.assertEqual(finding_lo.severity_level, SeverityLevel.UNDETERMINED)
        self.assertEqual(finding_lo.infected_area_pct, 0.0)

        # 4. Non-leaf textured brown surface (wood/soil texture without plant foliage) -> rejected safely
        non_leaf = np.full((256, 256, 3), (45, 75, 130), dtype=np.uint8)
        non_leaf[::8, ::8] = (25, 50, 100)
        finding_nl, nl_preds, nl_diag = pipeline_hi.predict_with_details(non_leaf)
        self.assertEqual(finding_nl.category, VisionCategory.UNKNOWN)
        self.assertEqual(finding_nl.condition_name, "non_leaf_or_no_plant")
        self.assertEqual(nl_preds, [])
        self.assertEqual(nl_diag["diagnosis_status"], "non_leaf_image")

        # 5. Ambiguous close top-2 logits -> rejected as ambiguous_multiple_conditions
        amb_logits = np.full(num_classes, -5.0, dtype=np.float32)
        amb_logits[0] = 4.0
        amb_logits[1] = 3.9
        amb_pipe = VisionPipeline(_FixedLogitsBackend(amb_logits), classes)
        finding_amb, _, amb_diag = amb_pipe.predict_with_details(sharp_mod_leaf)
        self.assertEqual(finding_amb.category, VisionCategory.UNKNOWN)
        self.assertEqual(amb_diag["diagnosis_status"], "ambiguous_multiple_conditions")

        # 6. Detached brown soil patch does not inflate healthy leaf severity
        leaf_on_soil = _make_synthetic_leaf(lesion_fraction=0.0, size=320)
        leaf_on_soil[10:45, 10:80] = (30, 68, 128)  # detached brown patch in corner
        soil_sev = estimate_severity(leaf_on_soil, "disease", image_quality_ok=True)
        self.assertLess(soil_sev.infected_area_pct, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
