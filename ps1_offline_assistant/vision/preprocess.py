"""
Image Quality Gate & OpenCV Preprocessing for FieldWise PS1 (Pipeline A).

Provides deterministic, offline image checks before vision inference:
  - Laplacian variance blur check (`blur_score`)
  - Mean grayscale exposure check (underexposure / overexposure rejection)
  - Standard ImageNet NCHW float32 preprocessing for MobileNetV3 edge models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_MIN_BLUR_SCORE = 50.0
DEFAULT_MIN_BRIGHTNESS = 35.0
DEFAULT_MAX_BRIGHTNESS = 225.0


@dataclass(frozen=True)
class ImageQualityReport:
    """Result of OpenCV image quality assessment."""

    image_quality_ok: bool
    blur_score: float
    mean_brightness: float
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_quality_ok": self.image_quality_ok,
            "blur_score": self.blur_score,
            "mean_brightness": self.mean_brightness,
            "issues": list(self.issues),
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def _validate_image_array(image_bgr: np.ndarray) -> np.ndarray:
    """Validates that the input is a non-empty uint8/numeric BGR image array."""
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray image, got {type(image_bgr).__name__}.")
    if image_bgr.size == 0 or image_bgr.ndim not in (2, 3):
        raise ValueError(
            f"Invalid image shape {image_bgr.shape}; expected non-empty 2D or 3D BGR array."
        )
    if image_bgr.ndim == 3 and image_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel BGR image, got {image_bgr.shape[2]} channels."
        )
    return image_bgr


def assess_image_quality(
    image_bgr: np.ndarray,
    min_blur_score: float = DEFAULT_MIN_BLUR_SCORE,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
) -> ImageQualityReport:
    """
    Computes Laplacian variance (`blur_score`) and mean brightness on a BGR image.

    Flags `image_quality_ok = False` when severely blurred, too dark, or overexposed.
    """
    img = _validate_image_array(image_bgr)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.dtype != np.uint8:
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        gray_u8 = gray

    # High-res smartphone photos (>512px) can have lower full-frame Laplacian
    # variance despite sharp subject focus; normalize to max 512px edge.
    h_img, w_img = gray_u8.shape[:2]
    max_edge = max(h_img, w_img)
    if max_edge > 512:
        scale = 512.0 / float(max_edge)
        eval_gray = cv2.resize(
            gray_u8,
            (max(1, int(round(w_img * scale))), max(1, int(round(h_img * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        eval_gray = gray_u8

    laplacian = cv2.Laplacian(eval_gray, cv2.CV_64F)
    blur_score = float(laplacian.var())
    mean_brightness = float(np.mean(gray_u8))

    issues: List[str] = []
    if blur_score < min_blur_score:
        issues.append(
            f"Image too blurry (Laplacian variance {blur_score:.2f} < {min_blur_score:.2f})."
        )
    if mean_brightness < min_brightness:
        issues.append(
            f"Image too dark / underexposed (mean brightness {mean_brightness:.2f} < {min_brightness:.2f})."
        )
    if mean_brightness > max_brightness:
        issues.append(
            f"Image overexposed / washed out (mean brightness {mean_brightness:.2f} > {max_brightness:.2f})."
        )

    return ImageQualityReport(
        image_quality_ok=len(issues) == 0,
        blur_score=round(blur_score, 4),
        mean_brightness=round(mean_brightness, 4),
        issues=issues,
    )


def preprocess_for_model(
    image_bgr: np.ndarray,
    size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Preprocesses a BGR image for MobileNetV3 edge inference:
      1. BGR -> RGB
      2. Resize to `size` (width, height)
      3. Scale to [0, 1] and normalize with ImageNet mean/std
      4. Convert HWC -> NCHW float32 contiguous array `(1, 3, H, W)`
    """
    img = _validate_image_array(image_bgr)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    target_w, target_h = int(size[0]), int(size[1])
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target size {size}; width and height must be > 0.")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    src_h, src_w = rgb.shape[:2]
    interp = cv2.INTER_AREA if (src_w > target_w or src_h > target_h) else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=interp)

    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    nchw = np.ascontiguousarray(np.expand_dims(chw, axis=0), dtype=np.float32)
    return nchw
