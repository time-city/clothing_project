#!/usr/bin/env python3
"""Evaluate the clothing classifier on a labeled image folder.

Expected dataset layout:

data_dir/
  ao_khoac/
    img1.jpg
    img2.png
  ao_so_mi/
  ao_thun/
  ...

Usage examples:
  python test_model.py --data-dir ./data
  python test_model.py --data-dir ./data --model-type mobilenet
  python test_model.py --data-dir ./data --batch-size 32 --json-out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

DISPLAY_CLASS_NAMES = {
    "ao_khoac": "Áo khoác",
    "ao_so_mi": "Áo sơ mi",
    "ao_thun": "Áo thun",
    "quan_jean": "Quần jean",
    "quan_short": "Quần short",
    "quan_tay": "Quần tây",
    "sweater_hoodie": "Áo len / Hoodie",
    "vay": "Váy",
}


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ImageFolderDataset(Dataset):
    def __init__(self, data_dir: Path, class_names: list[str], transform=None, image_preprocessor=None):
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        self.image_preprocessor = image_preprocessor
        self.samples: list[tuple[Path, int]] = []
        self.missing_classes: list[str] = []
        self.extra_classes: list[str] = []

        folder_names = {item.name for item in data_dir.iterdir() if item.is_dir()}
        expected_names = set(class_names)

        self.missing_classes = sorted(expected_names - folder_names)
        self.extra_classes = sorted(folder_names - expected_names)

        for class_index, class_name in enumerate(class_names):
            class_dir = data_dir / class_name
            if not class_dir.is_dir():
                continue

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, class_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path)
        if self.image_preprocessor is not None:
            image = self.image_preprocessor(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(image_path)


def load_model_and_classes(model_type: str):
    model_type = model_type.lower()
    if model_type == "mobilenet":
        from mobilenet_model.model import CLASS_NAMES, load_model, prepare_image_for_inference

        model_path = Path(__file__).parent / "mobilenet_model" / "clothing_model.pth"
        pretty_name = "MobileNet"
    else:
        from ai_model.model import CLASS_NAMES, load_model, prepare_image_for_inference

        model_path = Path(__file__).parent / "ai_model" / "best_model.pth"
        pretty_name = "ResNet50"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(model_path), device=device)
    return model, CLASS_NAMES, model_path, pretty_name, device, prepare_image_for_inference


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def display_class_name(class_name: str) -> str:
    return DISPLAY_CLASS_NAMES.get(class_name, class_name)


def display_class_names(class_names: list[str]) -> list[str]:
    return [display_class_name(name) for name in class_names]


def evaluate(model, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    per_class_top1 = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            top1_predictions = torch.argmax(probabilities, dim=1)
            topk = min(5, probabilities.shape[1])
            top5_predictions = torch.topk(probabilities, k=topk, dim=1).indices

            batch_total = labels.size(0)
            total += batch_total
            top1_correct += (top1_predictions == labels).sum().item()

            top5_match = (top5_predictions == labels.unsqueeze(1)).any(dim=1)
            top5_correct += top5_match.sum().item()

            for true_label, predicted_label in zip(labels.cpu().tolist(), top1_predictions.cpu().tolist()):
                confusion[true_label, predicted_label] += 1
                per_class_total[true_label] += 1
                if true_label == predicted_label:
                    per_class_top1[true_label] += 1

    return {
        "total": total,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "confusion": confusion,
        "per_class_total": per_class_total,
        "per_class_top1": per_class_top1,
    }


def print_confusion_matrix(confusion: np.ndarray, class_names: list[str]):
    width = max(10, max(len(name) for name in class_names))
    header = "true\\pred".ljust(width)
    for name in class_names:
        header += f"{name[:10].rjust(12)}"
    print("\nConfusion matrix (top-1):")
    print(header)
    for idx, name in enumerate(class_names):
        row = name.ljust(width)
        for value in confusion[idx]:
            row += f"{int(value):12d}"
        print(row)


def print_summary(
    *,
    model_name: str,
    model_path: Path,
    data_dir: Path,
    class_names: list[str],
    metrics: dict,
    dataset,
):
    total = metrics["total"]
    if total == 0:
        raise RuntimeError(f"No images found under {data_dir}")

    top1 = metrics["top1_correct"] / total
    top5 = metrics["top5_correct"] / total

    print("\n" + "=" * 72)
    print(f"Model: {model_name}")
    print(f"Checkpoint: {model_path}")
    print(f"Data dir: {data_dir}")
    print(f"Images evaluated: {total}")
    print(f"Classes: {len(class_names)}")
    print("=" * 72)
    print(f"Top-1 accuracy: {format_percent(top1)}")
    print(f"Top-5 accuracy: {format_percent(top5)}")

    print("\nClass labels:")
    print("  " + ", ".join(display_class_names(class_names)))

    if dataset.missing_classes:
        print("\nMissing class folders:")
        for name in dataset.missing_classes:
            print(f"  - {name}")

    if dataset.extra_classes:
        print("\nExtra folders ignored:")
        for name in dataset.extra_classes:
            print(f"  - {name}")

    print("\nPer-class accuracy:")
    for idx, name in enumerate(class_names):
        total_for_class = int(metrics["per_class_total"][idx])
        correct_for_class = int(metrics["per_class_top1"][idx])
        acc = (correct_for_class / total_for_class) if total_for_class else 0.0
        print(f"  {display_class_name(name):16s} {correct_for_class:4d}/{total_for_class:<4d}  {format_percent(acc)}")

    print_confusion_matrix(metrics["confusion"], display_class_names(class_names))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the clothing classifier on labeled images.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Folder containing one subfolder per class.",
    )
    parser.add_argument(
        "--model-type",
        choices=("resnet", "mobilenet"),
        default="resnet",
        help="Which model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to save the evaluation summary as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    model, class_names, model_path, model_name, device, image_preprocessor = load_model_and_classes(args.model_type)

    dataset = ImageFolderDataset(
        data_dir,
        class_names,
        transform=build_transform(),
        image_preprocessor=image_preprocessor,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No images found in {data_dir}. Expected folders like: {', '.join(class_names[:3])}, ..."
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, device=device, num_classes=len(class_names))

    print_summary(
        model_name=model_name,
        model_path=model_path,
        data_dir=data_dir,
        class_names=class_names,
        metrics=metrics,
        dataset=dataset,
    )

    if args.json_out:
        summary = {
            "model_name": model_name,
            "model_path": str(model_path),
            "data_dir": str(data_dir),
            "device": str(device),
            "total_images": int(metrics["total"]),
            "top1_accuracy": float(metrics["top1_correct"] / metrics["total"]),
            "top5_accuracy": float(metrics["top5_correct"] / metrics["total"]),
            "per_class": {},
        }

        for idx, name in enumerate(class_names):
            total_for_class = int(metrics["per_class_total"][idx])
            correct_for_class = int(metrics["per_class_top1"][idx])
            summary["per_class"][name] = {
                "total": total_for_class,
                "correct": correct_for_class,
                "accuracy": float(correct_for_class / total_for_class) if total_for_class else None,
            }

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON saved to: {args.json_out}")


if __name__ == "__main__":
    main()