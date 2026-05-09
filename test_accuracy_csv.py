#!/usr/bin/env python3
"""
Evaluate model accuracy using test.csv with Cloudinary URLs
"""
import csv
import argparse
import json
from pathlib import Path
from io import BytesIO
import sys

import numpy as np
import requests
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_model.model import load_model, predict
from mobilenet_model.model import load_model as load_mobilenet


CLASS_NAMES = [
    "ao_khoac", "ao_so_mi", "ao_thun", "quan_jean",
    "quan_short", "quan_tay", "sweater_hoodie", "vay"
]

CLASS_DISPLAY_NAMES = {
    "ao_khoac": "Áo khoác",
    "ao_so_mi": "Áo sơ mi",
    "ao_thun": "Áo thun",
    "quan_jean": "Quần jean",
    "quan_short": "Quần short",
    "quan_tay": "Quần tây",
    "sweater_hoodie": "Sweater/Hoodie",
    "vay": "Váy"
}


def fetch_image_from_url(url, timeout=10):
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


def extract_class_from_public_id(public_id):
    """Extract class name from public_id (e.g., 'ao_khoac/asos_000701' -> 'ao_khoac')"""
    return public_id.split('/')[0]


def evaluate_model(csv_path, model_path, model_type='resnet', device='cpu', batch_size=1, json_out=None):
    """
    Evaluate model on test.csv data
    
    Args:
        csv_path: Path to test.csv
        model_path: Path to model checkpoint
        model_type: 'resnet' or 'mobilenet'
        device: 'cpu' or 'cuda'
        batch_size: Batch size (for future batching support)
        json_out: Optional path to save results as JSON
    """
    
    print(f"Loading model from {model_path}...")
    if model_type == 'mobilenet':
        model = load_mobilenet(model_path, device=device)
    else:
        model = load_model(model_path, device=device)
    
    print(f"Reading {csv_path}...")
    true_labels = []
    pred_labels = []
    pred_scores = []
    failed_images = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total = len(rows)
    print(f"Evaluating {total} images...")
    
    for idx, row in enumerate(rows, 1):
        url = row['secure_url']
        public_id = row['public_id']
        true_class = extract_class_from_public_id(public_id)
        
        # Fetch image
        img = fetch_image_from_url(url)
        if img is None:
            failed_images.append((url, true_class))
            continue
        
        # Run inference
        try:
            classes, scores, _ = predict(model, img, device=device, return_top_k=1)
            pred_class = classes[0]
            pred_score = float(scores[0])
            
            true_labels.append(true_class)
            pred_labels.append(pred_class)
            pred_scores.append(pred_score)
            
            if idx % 20 == 0:
                print(f"  [{idx}/{total}] Processed", end='\r')
        except Exception as e:
            print(f"\nError processing {url}: {e}")
            failed_images.append((url, true_class))
            continue
    
    print(f"\n\nProcessed: {len(true_labels)}/{total} images ({len(failed_images)} failed)")
    
    # Calculate metrics
    if len(true_labels) == 0:
        print("No successful predictions!")
        return
    
    top1_accuracy = accuracy_score(true_labels, pred_labels)
    
    # Per-class metrics
    class_metrics = {}
    for cls in CLASS_NAMES:
        cls_mask = np.array(true_labels) == cls
        if np.any(cls_mask):
            cls_true = np.array(true_labels)[cls_mask]
            cls_pred = np.array(pred_labels)[cls_mask]
            
            prec = precision_score(cls_true, cls_pred, zero_division=0, average='weighted')
            rec = recall_score(cls_true, cls_pred, zero_division=0, average='weighted')
            f1 = f1_score(cls_true, cls_pred, zero_division=0, average='weighted')
            
            count = np.sum(cls_mask)
            correct = np.sum(cls_true == cls_pred)
            acc = correct / count if count > 0 else 0
            
            class_metrics[cls] = {
                'count': int(count),
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'correct': int(correct)
            }
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=CLASS_NAMES)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Top-1 Accuracy: {top1_accuracy:.4f} ({int(top1_accuracy * len(true_labels))}/{len(true_labels)})")
    print(f"Average Confidence: {np.mean(pred_scores):.4f}")
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Count':>6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 66)
    for cls in CLASS_NAMES:
        if cls in class_metrics:
            m = class_metrics[cls]
            display_name = CLASS_DISPLAY_NAMES.get(cls, cls)
            print(f"{display_name:<20} {m['count']:>6} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    
    print("\nConfusion Matrix:")
    print("Predicted →", end=" ")
    for cls in CLASS_NAMES:
        print(f"{cls[:4]:>5}", end="")
    print()
    for i, true_cls in enumerate(CLASS_NAMES):
        print(f"{true_cls[:4]:<10}", end="")
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i][j]:>5}", end="")
        print()
    
    if len(failed_images) > 0:
        print(f"\n⚠️  Failed images ({len(failed_images)}):")
        for url, cls in failed_images[:5]:
            print(f"  - {cls}: {url[:60]}...")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more")
    
    # Save to JSON if requested
    if json_out:
        results = {
            'total_images': total,
            'successful': len(true_labels),
            'failed': len(failed_images),
            'top1_accuracy': top1_accuracy,
            'average_confidence': float(np.mean(pred_scores)),
            'per_class_metrics': class_metrics,
            'confusion_matrix': cm.tolist()
        }
        with open(json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {json_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on test.csv')
    parser.add_argument('--csv', type=str, default='test.csv', help='Path to test.csv')
    parser.add_argument('--model', type=str, default='ai_model/best_model.pth', help='Path to model')
    parser.add_argument('--model-type', type=str, default='resnet', choices=['resnet', 'mobilenet'], help='Model type')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--json-out', type=str, default=None, help='Save results as JSON')
    
    args = parser.parse_args()
    
    evaluate_model(
        csv_path=args.csv,
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
        json_out=args.json_out
    )
