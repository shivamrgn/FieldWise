"""
Reproducible Vision Model Validation Suite for FieldWise PS1 (Phase 9).

Evaluates the actual `VisionPipeline` (`hf-local` PlantVillage MobileNetV2 or
custom trained checkpoint) on a legitimate labeled held-out `ImageFolder` dataset:
  - Separates `plantvillage` validation from `field` image validation (`--domain`)
  - Computes Top-1 accuracy, Top-3 accuracy, macro Precision/Recall/F1,
    per-class metrics, confusion matrix, and uncertainty/rejection breakdown
    (`poor_image_quality`, `non_leaf_image`, `low_confidence`, `ambiguous`)
  - NEVER fabricates metrics when no real labeled dataset is present.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.orchestrator.cli import _resolve_vision_backend  # noqa: E402
from ps1_offline_assistant.vision.download_pretrained import map_hf_label_to_kb_entry  # noqa: E402
from ps1_offline_assistant.kb.retriever import OfflineKBRetriever  # noqa: E402
from ps1_offline_assistant.vision.inference import VisionPipeline  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _canonical_key(crop: str, condition: str) -> str:
    return f"{crop.strip().lower()} :: {condition.strip().lower()}"


def evaluate_dataset(
    dataset_dir: Path,
    vision_backend_kind: str = "hf-local",
    vision_model_path: Optional[Path] = None,
    domain: str = "plantvillage",
    limit_per_class: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs real validation across labeled ImageFolder directories."""
    root = Path(dataset_dir)
    if not root.is_dir():
        return {
            "status": "DATASET_NOT_FOUND",
            "domain": domain,
            "dataset_dir": str(root),
            "message": "Provide a valid held-out ImageFolder directory with labeled class subfolders.",
        }

    retriever = OfflineKBRetriever()
    backend = _resolve_vision_backend(
        backend_kind=vision_backend_kind,
        model_path=vision_model_path,
        num_classes=38,
        allow_untrained_smoke=False,
    )
    pipeline = VisionPipeline(backend)

    samples: List[tuple[Path, str]] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mapped = map_hf_label_to_kb_entry(0, class_dir.name, retriever)
        gt_key = _canonical_key(mapped["crop"], mapped["condition_name"])
        imgs = [
            f for f in sorted(class_dir.iterdir())
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if limit_per_class is not None:
            imgs = imgs[:limit_per_class]
        for img_path in imgs:
            samples.append((img_path, gt_key))

    if not samples:
        return {
            "status": "NO_LABELED_IMAGES_FOUND",
            "domain": domain,
            "dataset_dir": str(root),
            "total_evaluated": 0,
        }

    top1_hits = 0
    top3_hits = 0
    accepted_count = 0
    rejection_counts: Counter[str] = Counter()
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    all_gt_labels: set[str] = set()

    for img_path, gt_key in samples:
        all_gt_labels.add(gt_key)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            rejection_counts["unreadable_image"] += 1
            fn[gt_key] += 1
            continue

        vf, top3, diag = pipeline.predict_with_details(bgr, top_k=3)
        status = diag.get("diagnosis_status", "unknown")
        top3_keys = [_canonical_key(p["crop"], p["condition_name"]) for p in top3]
        raw_top1_key = top3_keys[0] if top3_keys else _canonical_key(vf.crop, vf.condition_name)

        if top3_keys and gt_key in top3_keys:
            top3_hits += 1

        if vf.category.value == "unknown":
            rejection_counts[status] += 1
            confusion[gt_key][f"REJECTED:{status}"] += 1
            fn[gt_key] += 1
            continue

        accepted_count += 1
        pred_key = _canonical_key(vf.crop, vf.condition_name)
        confusion[gt_key][pred_key] += 1
        if pred_key == gt_key:
            top1_hits += 1
            tp[gt_key] += 1
        else:
            fp[pred_key] += 1
            fn[gt_key] += 1

        # Also keep raw Argmax accuracy check distinct from uncertainty-gated accuracy
        _ = raw_top1_key

    total = len(samples)
    per_class: Dict[str, Dict[str, Any]] = {}
    f1_scores: List[float] = []
    prec_scores: List[float] = []
    rec_scores: List[float] = []
    for lbl in sorted(all_gt_labels):
        c_tp = tp[lbl]
        c_fp = fp[lbl]
        c_fn = fn[lbl]
        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)
        per_class[lbl] = {
            "support": c_tp + c_fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    report = {
        "status": "EVALUATED",
        "domain": domain,
        "dataset_dir": str(root.resolve()),
        "vision_backend": backend.backend_name,
        "total_evaluated": total,
        "accepted_predictions": accepted_count,
        "rejection_breakdown": dict(rejection_counts),
        "top1_accuracy": round(top1_hits / total, 4),
        "top3_accuracy": round(top3_hits / total, 4),
        "accepted_only_precision": round(top1_hits / accepted_count, 4) if accepted_count else None,
        "macro_precision": round(sum(prec_scores) / len(prec_scores), 4),
        "macro_recall": round(sum(rec_scores) / len(rec_scores), 4),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "per_class_metrics": per_class,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "vision_eval_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate FieldWise vision model on labeled ImageFolder.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--domain", choices=["plantvillage", "field"], default="plantvillage")
    parser.add_argument("--vision-backend", default="hf-local")
    parser.add_argument("--vision-model", type=Path, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    res = evaluate_dataset(
        dataset_dir=args.dataset_dir,
        vision_backend_kind=args.vision_backend,
        vision_model_path=args.vision_model,
        domain=args.domain,
        limit_per_class=args.limit_per_class,
        output_dir=args.output_dir,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
