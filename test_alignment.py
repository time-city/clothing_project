#!/usr/bin/env python3
"""
Test alignment: So sánh logic Colab (gold standard) với webapp logic
để tìm ra chỗ diverge khi dự đoán khác nhau.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# ==========================================
# A. COLAB GOLD STANDARD LOGIC
# ==========================================

CLASS_NAMES_DICT = {
    0: 'ao_khoac',
    1: 'ao_so_mi',
    2: 'ao_thun',
    3: 'quan_jean',
    4: 'quan_short',
    5: 'quan_tay',
    6: 'sweater_hoodie',
    7: 'vay'
}

class ColabClothingClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


def colab_preprocess_image(image_input):
    """COLAB GOLD STANDARD: Image preprocessing"""
    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input
    
    # Xử lý alpha channel
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img_processed = bg
    else:
        img_processed = img.convert('RGB')
    
    return img_processed


def colab_predict(model, image_input, device='cpu'):
    """COLAB GOLD STANDARD: Prediction logic"""
    img = colab_preprocess_image(image_input)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs_percent = F.softmax(outputs, dim=1)[0] * 100
    
    top_prob_percent, top_idx = torch.max(probs_percent, 0)
    pred_class = CLASS_NAMES_DICT[top_idx.item()]
    confidence = top_prob_percent.item() / 100.0  # Convert back to [0, 1]
    
    return pred_class, confidence, top_idx.item()


# ==========================================
# B. WEBAPP LOGIC
# ==========================================

from ai_model.model import (
    prepare_image_for_inference as webapp_prepare_image,
    predict as webapp_predict,
    CLASS_NAMES as webapp_class_names,
    CLASS_NAME_MAP as webapp_class_map
)


# ==========================================
# C. TEST & ALIGNMENT CHECK
# ==========================================

def test_image_alignment(image_path, model_colab, model_webapp, device='cpu'):
    """Test một ảnh duy nhất để xem logic Colab vs Webapp"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # COLAB prediction
    colab_class, colab_conf, colab_idx = colab_predict(model_colab, image_path, device)
    
    # WEBAPP prediction
    webapp_classes, webapp_scores, webapp_indices = webapp_predict(
        model_webapp, image_path, device=device, return_top_k=1
    )
    webapp_class = webapp_classes[0]
    webapp_conf = float(webapp_scores[0])
    webapp_idx = int(webapp_indices[0])
    
    print(f"\nCOLAB (Gold Standard):")
    print(f"  Class: {colab_class:20s} (idx={colab_idx})")
    print(f"  Confidence: {colab_conf:.6f}")
    print(f"  Confidence %: {colab_conf*100:.2f}%")
    
    print(f"\nWEBAPP (Current):")
    print(f"  Class: {webapp_class:20s} (idx={webapp_idx})")
    print(f"  Confidence: {webapp_conf:.6f}")
    print(f"  Confidence %: {webapp_conf*100:.2f}%")
    
    # Compare
    match = (colab_class == webapp_class)
    conf_diff = abs(colab_conf - webapp_conf)
    
    print(f"\n📊 COMPARISON:")
    print(f"  Class Match: {'✅ YES' if match else '❌ NO'}")
    print(f"  Confidence Difference: {conf_diff:.6f}")
    
    if not match:
        print(f"  ⚠️  DIVERGENCE: Expected {colab_class}, got {webapp_class}")
    
    return {
        'image': os.path.basename(image_path),
        'match': match,
        'colab_class': colab_class,
        'webapp_class': webapp_class,
        'colab_conf': colab_conf,
        'webapp_conf': webapp_conf,
        'conf_diff': conf_diff
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}\n")
    
    # Load models
    model_path = Path('ai_model/best_model.pth')
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📥 Loading model from {model_path}...")
    
    # COLAB model
    model_colab = ColabClothingClassifier(num_classes=8)
    model_colab.load_state_dict(torch.load(str(model_path), map_location=device))
    model_colab.to(device)
    model_colab.eval()
    print("  ✅ Colab model loaded")
    
    # WEBAPP model
    from ai_model.model import load_model as webapp_load
    model_webapp = webapp_load(str(model_path), device=device)
    print("  ✅ Webapp model loaded")
    
    # Find test images
    test_dirs = [
        Path('data'),
        Path('test_images'),
        Path('web_app/uploads'),
    ]
    
    test_images = []
    for test_dir in test_dirs:
        if test_dir.exists():
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.PNG', '*.JPG'):
                test_images.extend(test_dir.glob(f'**/{ext}'))
    
    test_images = list(set(test_images))[:10]  # Limit to 10 images
    
    if not test_images:
        print("\n⚠️  No test images found. Please place images in:")
        print("  - data/")
        print("  - test_images/")
        print("  - web_app/uploads/")
        return
    
    print(f"\n📂 Found {len(test_images)} test images\n")
    
    # Test each image
    results = []
    for img_path in test_images:
        try:
            result = test_image_alignment(str(img_path), model_colab, model_webapp, device)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
    
    # Summary
    print(f"\n\n{'='*70}")
    print("ALIGNMENT SUMMARY")
    print(f"{'='*70}\n")
    
    matches = sum(1 for r in results if r['match'])
    total = len(results)
    
    print(f"Matching Predictions: {matches}/{total} ({100*matches/total:.1f}%)")
    
    if matches < total:
        print(f"\n❌ DIVERGENCES Found:")
        for r in results:
            if not r['match']:
                print(f"  {r['image']}: Colab={r['colab_class']}, Webapp={r['webapp_class']} (diff={r['conf_diff']:.6f})")
    else:
        print(f"\n✅ ALL PREDICTIONS ALIGNED - COLAB = WEBAPP")
    
    # Stats
    avg_conf_diff = np.mean([r['conf_diff'] for r in results])
    print(f"\nAverage confidence difference: {avg_conf_diff:.6f}")


if __name__ == '__main__':
    main()
