"""
Compact Vision Model Training & Edge Export for FieldWise PS1 (Pipeline A).

Supports:
  - Offline MobileNetV3 (small/large) initialization aligned strictly with
    `ps1_offline_assistant/data/agri_kb.json` (`disease_pest_cards` + `healthy`).
  - Standard ImageFolder dataset mapping for PlantVillage / Cassava / IP102.
  - Static-shape (1, 3, 224, 224) export paths:
      1. `export_to_executorch`: `torch.export.export` + `to_edge_transform_and_lower`
         with `XnnpackPartitioner` -> `.pte` for Arm CPU.
      2. `export_to_onnx`: static-shape ONNX export for local verification and
         Hailo Dataflow Compiler `.hef` generation (Raspberry Pi 5 + AI HAT+).
      3. `export_to_torch_exported`: native PyTorch `.pt2` `ExportedProgram`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.kb.retriever import OfflineKBRetriever  # noqa: E402


DEFAULT_KB_PATH = _PS1_ROOT / "data" / "agri_kb.json"
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)


class EdgeExportDependencyError(RuntimeError):
    """Raised when an optional edge export runtime (ExecuTorch or ONNX) is unavailable."""


def _is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def build_class_labels_from_kb(kb_path: Path = DEFAULT_KB_PATH) -> List[Dict[str, Any]]:
    """
    Derives the canonical classifier label manifest strictly from `agri_kb.json`
    (`disease_pest_cards` sorted deterministically by ID + `healthy` class).
    """
    kb_file = Path(kb_path)
    if not kb_file.is_file():
        raise FileNotFoundError(f"Offline KB file not found: {kb_file}")

    raw_kb = json.loads(kb_file.read_text(encoding="utf-8"))
    cards = sorted(raw_kb.get("disease_pest_cards", []), key=lambda c: c["id"])
    if not cards:
        raise ValueError(f"No disease_pest_cards found in {kb_file}")

    classes: List[Dict[str, Any]] = []
    for idx, card in enumerate(cards):
        classes.append(
            {
                "index": idx,
                "label": card["id"],
                "kb_card_id": card["id"],
                "crop": card["crop"],
                "condition_name": card["condition"],
                "category": card["type"],
            }
        )

    classes.append(
        {
            "index": len(classes),
            "label": "healthy",
            "kb_card_id": None,
            "crop": "general",
            "condition_name": "Healthy",
            "category": "healthy",
        }
    )
    return classes


def save_class_labels_manifest(
    output_path: Path,
    kb_path: Path = DEFAULT_KB_PATH,
) -> Dict[str, Any]:
    """Writes `class_labels.json` alongside exported edge artifacts."""
    classes = build_class_labels_from_kb(kb_path=kb_path)
    raw_kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    manifest = {
        "kb_schema_version": raw_kb.get("kb_schema_version", "1.0.0"),
        "num_classes": len(classes),
        "input_shape": list(DEFAULT_INPUT_SHAPE),
        "classes": classes,
        "index_to_label": {str(c["index"]): c["label"] for c in classes},
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def map_imagefolder_to_kb_classes(
    folder_names: List[str],
    kb_path: Path = DEFAULT_KB_PATH,
) -> Dict[str, Optional[int]]:
    """
    Maps raw PlantVillage / Cassava / IP102 ImageFolder class directory names
    to canonical `class_labels.json` indices using `OfflineKBRetriever`.
    Unmapped folders map to `None`.
    """
    retriever = OfflineKBRetriever(kb_path=kb_path)
    manifest_classes = build_class_labels_from_kb(kb_path=kb_path)
    label_to_idx = {c["label"]: c["index"] for c in manifest_classes}

    mapping: Dict[str, Optional[int]] = {}
    for folder in folder_names:
        norm = folder.strip().lower()
        if "healthy" in norm:
            mapping[folder] = label_to_idx["healthy"]
            continue

        # Attempt direct card lookup (supports PlantVillage `Crop___Condition` & IP102 names)
        crop_hint = folder.split("___")[0] if "___" in folder else None
        card = retriever.get_disease_pest_card(folder, crop=crop_hint)
        if card is None and crop_hint is not None:
            card = retriever.get_disease_pest_card(folder, crop=None)
        mapping[folder] = label_to_idx[card.id] if card is not None else None
    return mapping


def build_vision_model(
    num_classes: int,
    arch: str = "mobilenet_v3_small",
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """
    Constructs a compact MobileNetV3 classifier strictly offline (`weights=None`
    unless a local `weights_path` checkpoint is provided).
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
    else:
        raise ValueError(
            f"Unsupported arch '{arch}'. Expected 'mobilenet_v3_small' or 'mobilenet_v3_large'."
        )

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if weights_path is not None:
        ckpt = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)

    model.eval()
    return model


def _build_training_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            imagenet_norm,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            imagenet_norm,
        ]
    )
    return train_tf, eval_tf


def train_vision_classifier(
    train_dir: Path,
    output_dir: Path,
    val_dir: Optional[Path] = None,
    kb_path: Path = DEFAULT_KB_PATH,
    arch: str = "mobilenet_v3_small",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Trains MobileNetV3 offline on local ImageFolder layouts (PlantVillage / Cassava / IP102).
    Logs actual observed epoch loss/accuracy without fabricating benchmark metrics.
    """
    train_path = Path(train_dir)
    if not train_path.is_dir():
        raise FileNotFoundError(f"Training ImageFolder directory not found: {train_path}")

    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = save_class_labels_manifest(out_dir / "class_labels.json", kb_path=kb_path)
    train_tf, eval_tf = _build_training_transforms(224)
    train_ds = datasets.ImageFolder(str(train_path), transform=train_tf)
    folder_map = map_imagefolder_to_kb_classes(train_ds.classes, kb_path=kb_path)

    # Remap dataset targets into canonical KB class indices where matched
    remapped_samples = []
    skipped_unmapped = 0
    for img_path, raw_cls in train_ds.samples:
        folder_name = train_ds.classes[raw_cls]
        kb_idx = folder_map.get(folder_name)
        if kb_idx is None:
            skipped_unmapped += 1
            continue
        remapped_samples.append((img_path, kb_idx))

    if not remapped_samples:
        raise ValueError(
            "No ImageFolder class folders matched canonical KB classes in agri_kb.json."
        )

    train_ds.samples = remapped_samples
    train_ds.targets = [t for _, t in remapped_samples]
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )

    val_loader = None
    if val_dir is not None and Path(val_dir).is_dir():
        val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tf)
        val_map = map_imagefolder_to_kb_classes(val_ds.classes, kb_path=kb_path)
        val_samples = [
            (p, val_map[val_ds.classes[c]])
            for p, c in val_ds.samples
            if val_map.get(val_ds.classes[c]) is not None
        ]
        if val_samples:
            val_ds.samples = val_samples
            val_ds.targets = [t for _, t in val_samples]
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            )

    model = build_vision_model(num_classes=manifest["num_classes"], arch=arch)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    best_val_acc = -1.0
    best_ckpt = out_dir / "best_mobilenet_v3.pt"

    for epoch in range(1, epochs + 1):
        t_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * targets.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))

        scheduler.step()
        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_acc: Optional[float] = None
        if val_loader is not None:
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.inference_mode():
                for v_in, v_tgt in val_loader:
                    v_in, v_tgt = v_in.to(device), v_tgt.to(device)
                    v_preds = model(v_in).argmax(dim=1)
                    v_correct += int((v_preds == v_tgt).sum().item())
                    v_total += int(v_tgt.size(0))
            val_acc = v_correct / max(1, v_total)

        score = val_acc if val_acc is not None else train_acc
        if score >= best_val_acc:
            best_val_acc = score
            torch.save(model.state_dict(), best_ckpt)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_acc": round(val_acc, 6) if val_acc is not None else None,
                "elapsed_sec": round(time.perf_counter() - t_start, 3),
            }
        )

    return {
        "checkpoint_path": str(best_ckpt),
        "class_labels_path": str(out_dir / "class_labels.json"),
        "num_classes": manifest["num_classes"],
        "skipped_unmapped_samples": skipped_unmapped,
        "history": history,
    }


def export_to_torch_exported(
    model: nn.Module,
    output_path: Path,
    example_input: Optional[torch.Tensor] = None,
) -> Path:
    """
    Traces static `(1, 3, 224, 224)` model using `torch.export.export` and saves
    a native PyTorch `.pt2` ExportedProgram completely offline.
    """
    model.eval()
    sample = (
        example_input
        if example_input is not None
        else torch.randn(*DEFAULT_INPUT_SHAPE, dtype=torch.float32)
    )
    exported = torch.export.export(model, (sample,))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, out)
    return out


def export_to_executorch(
    model: nn.Module,
    output_path: Path,
    example_input: Optional[torch.Tensor] = None,
    quantize_int8: bool = True,
) -> Path:
    """
    Exports a PyTorch MobileNetV3 model to ExecuTorch `.pte` for Arm CPU:
      `torch.export.export` -> optional XNNPACK PT2E quantization ->
      `to_edge_transform_and_lower(..., partitioner=[XnnpackPartitioner()])` -> `.pte`.
    """
    model.eval()
    sample = (
        example_input
        if example_input is not None
        else torch.randn(*DEFAULT_INPUT_SHAPE, dtype=torch.float32)
    )

    # Stage 1: Always validate static graph capture via torch.export.export
    exported_program = torch.export.export(model, (sample,))

    # Stage 2: Lower via ExecuTorch + XnnpackPartitioner
    if not _is_module_available("executorch"):
        raise EdgeExportDependencyError(
            "torch.export.export succeeded, but 'executorch' is not installed in this "
            "environment. Install executorch to run to_edge_transform_and_lower with "
            "XnnpackPartitioner and emit .pte artifacts."
        )

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    program_to_lower = exported_program

    if quantize_int8 and _is_module_available("executorch.backends.xnnpack.quantizer"):
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )
            from torchao.quantization.pt2e.quantize_pt2e import (
                convert_pt2e,
                prepare_pt2e,
            )

            quantizer = XNNPACKQuantizer()
            quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
            prepared = prepare_pt2e(exported_program.module(), quantizer)
            prepared(sample)
            converted = convert_pt2e(prepared)
            program_to_lower = torch.export.export(converted, (sample,))
        except ImportError:
            # Proceed with FP32 XNNPACK lowering if optional torchao quantizer is absent
            program_to_lower = exported_program

    edge_program = to_edge_transform_and_lower(
        program_to_lower,
        partitioner=[XnnpackPartitioner()],
    )
    executorch_program = edge_program.to_executorch()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        executorch_program.write_to_file(handle)
    return out


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    example_input: Optional[torch.Tensor] = None,
    opset_version: int = 17,
) -> Path:
    """
    Exports a static-shape `(1, 3, 224, 224)` ONNX model suitable for local
    onnxruntime validation and Hailo Dataflow Compiler `.hef` compilation.
    """
    has_onnx = _is_module_available("onnx")
    has_onnxscript = _is_module_available("onnxscript")
    if not has_onnx:
        raise EdgeExportDependencyError(
            "The 'onnx' package is not installed in this environment. Install 'onnx' "
            "(and 'onnxscript' for PyTorch dynamo ONNX export) to emit static .onnx models."
        )

    model.eval()
    sample = (
        example_input
        if example_input is not None
        else torch.randn(*DEFAULT_INPUT_SHAPE, dtype=torch.float32)
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs: Dict[str, Any] = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    if not has_onnxscript:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, (sample,), str(out), **export_kwargs)

    import onnx  # noqa: WPS433

    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export compact MobileNetV3 for FieldWise PS1."
    )
    parser.add_argument(
        "--kb-path", type=Path, default=DEFAULT_KB_PATH, help="Path to agri_kb.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PS1_ROOT / "vision" / "artifacts",
        help="Directory for exported models and class_labels.json",
    )
    parser.add_argument(
        "--arch",
        choices=["mobilenet_v3_small", "mobilenet_v3_large"],
        default="mobilenet_v3_small",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Optional local ImageFolder training dataset directory",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="Optional local ImageFolder validation dataset directory",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional existing local .pt checkpoint to export",
    )
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--export-executorch", action="store_true")
    parser.add_argument("--export-pt2", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = save_class_labels_manifest(out_dir / "class_labels.json", kb_path=args.kb_path)
    weights_path = args.weights_path

    if args.train_dir is not None:
        summary = train_vision_classifier(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_dir=out_dir,
            kb_path=args.kb_path,
            arch=args.arch,
            epochs=args.epochs,
        )
        weights_path = Path(summary["checkpoint_path"])
        print("Training completed:", json.dumps(summary, indent=2))

    model = build_vision_model(
        num_classes=manifest["num_classes"],
        arch=args.arch,
        weights_path=weights_path,
    )

    if args.export_pt2:
        pt2_path = export_to_torch_exported(model, out_dir / f"{args.arch}.pt2")
        print(f"Exported PyTorch ExportedProgram -> {pt2_path}")
    if args.export_onnx:
        onnx_path = export_to_onnx(model, out_dir / f"{args.arch}.onnx")
        print(f"Exported static ONNX -> {onnx_path}")
    if args.export_executorch:
        pte_path = export_to_executorch(model, out_dir / f"{args.arch}.pte")
        print(f"Exported ExecuTorch PTE -> {pte_path}")


if __name__ == "__main__":
    main()
