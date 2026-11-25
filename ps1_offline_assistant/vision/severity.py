"""
Deterministic OpenCV Severity Estimator for FieldWise PS1 (Pipeline A).

Estimates foliar infected area percentage using HSV + CIE Lab color-space
segmentation for `disease` findings and bounding-box/contour visual proxies for
`pest` findings, enforcing safe handling on low-quality, healthy, and unknown
classifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ps1_offline_assistant.schemas.contracts import SeverityLevel, VisionCategory


@dataclass(frozen=True)
class SeverityEstimate:
    """Structured output from deterministic severity estimation."""

    severity_level: SeverityLevel
    infected_area_pct: float
    leaf_pixel_count: int
    lesion_pixel_count: int
    estimation_method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity_level": self.severity_level.value,
            "infected_area_pct": self.infected_area_pct,
            "leaf_pixel_count": self.leaf_pixel_count,
            "lesion_pixel_count": self.lesion_pixel_count,
            "estimation_method": self.estimation_method,
        }

    def __getitem__(self, key: str) -> Any:
        val = getattr(self, key)
        if key == "severity_level" and isinstance(val, SeverityLevel):
            return val.value
        return val


def _normalize_category(category: Union[VisionCategory, str]) -> VisionCategory:
    if isinstance(category, VisionCategory):
        return category
    return VisionCategory(str(category).strip().lower())


def _map_pct_to_severity(infected_area_pct: float) -> SeverityLevel:
    """
    Maps infected area percentage to severity brackets:
      - low:      < 15%
      - moderate: 15% to 40%
      - severe:   > 40%
    """
    if infected_area_pct < 15.0:
        return SeverityLevel.LOW
    if infected_area_pct <= 40.0:
        return SeverityLevel.MODERATE
    return SeverityLevel.SEVERE


def _segment_leaf_and_lesions(
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses HSV + CIE Lab masks to isolate leaf foreground from neutral background
    and separate healthy green tissue from chlorotic/necrotic lesions.

    Returns `(leaf_mask, healthy_mask, lesion_mask)` as boolean arrays.
    """
    if image_bgr.dtype != np.uint8:
        img_u8 = np.clip(image_bgr, 0, 255).astype(np.uint8)
    else:
        img_u8 = image_bgr

    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Healthy chlorophyll green foliage:
    # OpenCV HSV hue ~ 34-95 (green-cyan range), saturated, non-dark, Lab a* greenish (<128)
    healthy_green = (
        (h >= 34)
        & (h <= 95)
        & (s >= 35)
        & (v >= 28)
        & (a_ch < 128)
    )

    # Chlorotic yellow/pale-yellow lesions: Hue ~ 18-33 with clear chroma
    chlorotic_yellow = (
        (h >= 18)
        & (h <= 33)
        & (s >= 45)
        & (v >= 45)
    )

    # Necrotic brown/rust/reddish-tan lesions:
    # Hue in [0, 18) or [165, 180], saturated, or Lab warm-red/brown shift (a* >= 128, b* >= 132)
    necrotic_brown = (
        (((h < 18) | (h >= 165)) & (s >= 35) & (v >= 20) & (v <= 220))
        | ((a_ch >= 130) & (b_ch >= 132) & (s >= 30) & (l_ch >= 20) & (l_ch <= 215))
    )

    # Combine initial foliar chroma mask and close small gaps to retain dark necrotic interiors
    chroma_leaf = healthy_green | chlorotic_yellow | necrotic_brown
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_u8 = cv2.morphologyEx(
        chroma_leaf.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel
    )

    # Fill enclosed interior holes inside leaf contours (captures black necrotic lesion cores)
    contours, _ = cv2.findContours(
        closed_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled_u8 = np.zeros_like(closed_u8)
    foliar_core = healthy_green | chlorotic_yellow
    for cnt in contours:
        if cv2.contourArea(cnt) < 25.0:
            continue
        comp_mask = np.zeros_like(closed_u8)
        cv2.drawContours(comp_mask, [cnt], -1, color=255, thickness=cv2.FILLED)
        comp_bool = comp_mask > 0
        comp_px = int(np.count_nonzero(comp_bool))
        core_px = int(np.count_nonzero(comp_bool & foliar_core))
        # Reject isolated brown background soil/wood contours that have no green/chlorotic leaf tissue
        if comp_px > 0 and (core_px >= 10 or (core_px / comp_px) >= 0.02):
            filled_u8[comp_bool] = 255
    leaf_mask = filled_u8 > 0

    # Dark necrotic pixels lying strictly inside the closed leaf silhouette
    dark_necrotic_interior = leaf_mask & (~healthy_green) & (v < 90) & (l_ch < 95)

    lesion_mask = leaf_mask & (
        (chlorotic_yellow | necrotic_brown | dark_necrotic_interior) & (~healthy_green)
    )
    healthy_mask = leaf_mask & healthy_green & (~lesion_mask)
    return leaf_mask, healthy_mask, lesion_mask


def assess_leaf_foliar_coverage(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Measures foliar green/chlorotic presence and leaf region bounding box
    for non-leaf rejection and off-center framing diagnostics.
    """
    if image_bgr.dtype != np.uint8:
        img_u8 = np.clip(image_bgr, 0, 255).astype(np.uint8)
    else:
        img_u8 = image_bgr

    h_img, w_img = img_u8.shape[:2]
    total_px = max(1, h_img * w_img)

    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    _, a_ch, _ = cv2.split(lab)

    healthy_green = (h >= 34) & (h <= 95) & (s >= 35) & (v >= 28) & (a_ch < 128)
    chlorotic_yellow = (h >= 18) & (h <= 33) & (s >= 45) & (v >= 45)
    foliar_gy_px = int(np.count_nonzero(healthy_green | chlorotic_yellow))
    foliar_green_yellow_pct = round(100.0 * foliar_gy_px / total_px, 3)

    leaf_mask, _, _ = _segment_leaf_and_lesions(img_u8)
    leaf_px = int(np.count_nonzero(leaf_mask))
    leaf_area_pct = round(100.0 * leaf_px / total_px, 3)

    short = min(h_img, w_img)
    y0 = max(0, (h_img - short) // 2)
    x0 = max(0, (w_img - short) // 2)
    center_crop_mask = leaf_mask[y0 : y0 + short, x0 : x0 + short]
    center_area = max(1, short * short)
    center_leaf_pct = round(100.0 * int(np.count_nonzero(center_crop_mask)) / center_area, 3)

    ys, xs = np.where(leaf_mask)
    leaf_bbox: Optional[Tuple[int, int, int, int]] = None
    if len(xs) > 0 and len(ys) > 0:
        leaf_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    return {
        "foliar_green_yellow_pct": foliar_green_yellow_pct,
        "leaf_area_pct": leaf_area_pct,
        "center_leaf_pct": center_leaf_pct,
        "leaf_bbox": leaf_bbox,
    }


def _estimate_pest_severity(
    image_bgr: np.ndarray,
    pest_boxes: Optional[Sequence[Sequence[float]]] = None,
) -> SeverityEstimate:
    """
    Pest severity estimator combining:
      1. Explicit detector bounding-box count & coverage (when pest boxes are supplied):
           - 1-2 pests & <15% area -> low
           - 3-6 pests or 15%-40% -> moderate
           - >=7 pests or >40%     -> severe
      2. OpenCV fallback proxy (when running single-label classification without boxes):
           - Estimates foliar feeding/damage area via HSV/Lab lesion masking combined
             with distinct dark/high-contrast contour blobs inside the foreground leaf.
    """
    leaf_mask, _, lesion_mask = _segment_leaf_and_lesions(image_bgr)
    leaf_pixels = int(np.count_nonzero(leaf_mask))
    lesion_pixels = int(np.count_nonzero(lesion_mask))
    damage_pct = (
        min(100.0, max(0.0, 100.0 * lesion_pixels / leaf_pixels))
        if leaf_pixels > 0
        else 0.0
    )

    if pest_boxes is not None:
        h_img, w_img = image_bgr.shape[:2]
        img_area = max(1, h_img * w_img)
        box_area_sum = 0.0
        for box in pest_boxes:
            if len(box) >= 4:
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                box_area_sum += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        box_pct = min(100.0, (box_area_sum / img_area) * 100.0)
        count = len(pest_boxes)
        effective_pct = round(max(damage_pct, box_pct), 2)

        if count >= 7 or effective_pct > 40.0:
            sev = SeverityLevel.SEVERE
        elif count >= 3 or effective_pct >= 15.0:
            sev = SeverityLevel.MODERATE
        elif count >= 1 or effective_pct > 0.0:
            sev = SeverityLevel.LOW
        else:
            sev = SeverityLevel.UNDETERMINED

        return SeverityEstimate(
            severity_level=sev,
            infected_area_pct=effective_pct,
            leaf_pixel_count=leaf_pixels,
            lesion_pixel_count=lesion_pixels,
            estimation_method="pest_bounding_box_and_foliar_damage",
        )

    # OpenCV visual fallback proxy when bounding boxes are unavailable (classification-only):
    # Count discrete medium-to-large anomalous contours within/against leaf region.
    if leaf_pixels == 0:
        return SeverityEstimate(
            severity_level=SeverityLevel.UNDETERMINED,
            infected_area_pct=0.0,
            leaf_pixel_count=0,
            lesion_pixel_count=0,
            estimation_method="pest_no_leaf_foreground_detected",
        )

    lesion_u8 = lesion_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(lesion_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_blob_area = max(12.0, 0.0005 * leaf_pixels)
    significant_blobs = [c for c in contours if cv2.contourArea(c) >= min_blob_area]

    # Map either damage percentage or discrete anomalous blob count to severity
    if damage_pct > 40.0 or len(significant_blobs) >= 7:
        sev = SeverityLevel.SEVERE
    elif damage_pct >= 15.0 or len(significant_blobs) >= 3:
        sev = SeverityLevel.MODERATE
    else:
        sev = SeverityLevel.LOW

    return SeverityEstimate(
        severity_level=sev,
        infected_area_pct=round(damage_pct, 2),
        leaf_pixel_count=leaf_pixels,
        lesion_pixel_count=lesion_pixels,
        estimation_method="pest_opencv_foliar_damage_estimate",
    )


def estimate_severity(
    image_bgr: np.ndarray,
    category: Union[VisionCategory, str],
    image_quality_ok: bool,
    pest_boxes: Optional[Sequence[Sequence[float]]] = None,
) -> SeverityEstimate:
    """
    Deterministic severity estimator for Pipeline A.

    Rules:
      - `image_quality_ok == False` -> `severity_level="undetermined"`, `infected_area_pct=0.0`
      - `category == "unknown"`    -> `severity_level="undetermined"`, `infected_area_pct=0.0`
      - `category == "healthy"`    -> `severity_level="low"`, `infected_area_pct=0.0`
      - `category == "disease"`    -> HSV/Lab foliar lesion segmentation:
                                      low (<15%), moderate (15-40%), severe (>40%)
      - `category == "pest"`       -> bounding-box/contour count or foliar damage fallback
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ValueError("Expected non-empty numpy.ndarray BGR image.")

    cat = _normalize_category(category)

    if not image_quality_ok or cat == VisionCategory.UNKNOWN:
        return SeverityEstimate(
            severity_level=SeverityLevel.UNDETERMINED,
            infected_area_pct=0.0,
            leaf_pixel_count=0,
            lesion_pixel_count=0,
            estimation_method="gated_undetermined",
        )

    if cat == VisionCategory.HEALTHY:
        return SeverityEstimate(
            severity_level=SeverityLevel.LOW,
            infected_area_pct=0.0,
            leaf_pixel_count=0,
            lesion_pixel_count=0,
            estimation_method="healthy_zero_lesion",
        )

    if cat == VisionCategory.PEST:
        return _estimate_pest_severity(image_bgr, pest_boxes=pest_boxes)

    # Disease branch: HSV + Lab leaf foreground & lesion ratio
    leaf_mask, _, lesion_mask = _segment_leaf_and_lesions(image_bgr)
    leaf_pixels = int(np.count_nonzero(leaf_mask))
    lesion_pixels = int(np.count_nonzero(lesion_mask))

    if leaf_pixels == 0:
        return SeverityEstimate(
            severity_level=SeverityLevel.UNDETERMINED,
            infected_area_pct=0.0,
            leaf_pixel_count=0,
            lesion_pixel_count=0,
            estimation_method="disease_no_leaf_foreground_detected",
        )

    infected_area_pct = round(
        min(100.0, max(0.0, (100.0 * lesion_pixels) / leaf_pixels)), 2
    )
    severity_level = _map_pct_to_severity(infected_area_pct)

    return SeverityEstimate(
        severity_level=severity_level,
        infected_area_pct=infected_area_pct,
        leaf_pixel_count=leaf_pixels,
        lesion_pixel_count=lesion_pixels,
        estimation_method="disease_hsv_lab_segmentation",
    )
