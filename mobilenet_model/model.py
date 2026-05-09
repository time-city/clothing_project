"""
MobileNet Model Module - Lightweight Clothing Classification
Optimized for fast inference and mobile/edge devices

Advantages:
- 10x smaller than ResNet
- 5x faster inference
- Works great on CPU & mobile
- ~80% accuracy (vs ResNet ~85%)

Disadvantages:
- Slightly lower accuracy
- Less powerful for complex patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ====== CLASS NAMES ======
CLASS_NAME_MAP = {
    0: 'ao_khoac',
    1: 'ao_so_mi',
    2: 'ao_thun',
    3: 'quan_jean',
    4: 'quan_short',
    5: 'quan_tay',
    6: 'sweater_hoodie',
    7: 'vay',
}

CLASS_NAMES = [CLASS_NAME_MAP[index] for index in range(len(CLASS_NAME_MAP))]

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def prepare_image_for_inference(image_input):
    """Load and normalize an image for inference with white alpha compositing."""
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input)
    else:
        image = image_input

    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert('RGBA')
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    else:
        image = image.convert('RGB')

    return image

# ====== MODEL ARCHITECTURE ======
class MobileNetClassifier(nn.Module):
    """
    MobileNetV3 Model cho phân loại quần áo
    
    Lightweight architecture:
    - MobileNetV3 Small: 2.5MB
    - MobileNetV3 Large: 5.4MB
    - Fast inference: 10-50ms per image
    
    Good for:
    - Mobile apps (iOS/Android)
    - Edge devices (Raspberry Pi)
    - Low-latency requirements
    - Resource-constrained environments
    """
    
    def __init__(self, num_classes=8, variant='large'):
        """
        Args:
            num_classes: Số lượng class (mặc định 8)
            variant: 'small' (2.5MB) hoặc 'large' (5.4MB)
        """
        super(MobileNetClassifier, self).__init__()
        
        # Load MobileNetV3 pre-trained
        if variant == 'small':
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:  # large
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Replace final FC layer
        # MobileNetV3 có classifier thay vì fc
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)


# ====== LOAD MODEL ======
def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu', variant='large'):
    """
    Load trained MobileNet model weights
    
    Args:
        model_path (str): Path to trained model (.pth file)
        device (str): 'cuda' hoặc 'cpu'
        variant (str): 'small' hoặc 'large'
    
    Returns:
        model: Loaded model ready for inference
    
    Example:
        model = load_model('mobilenet_model/clothing_model.pth', variant='large')
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    model = MobileNetClassifier(num_classes=len(CLASS_NAMES), variant=variant)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"✅ MobileNet model loaded from: {model_path}")
    print(f"   Device: {device}")
    print(f"   Variant: {variant}")
    print(f"   Model size: {model_size_mb:.1f}MB")
    print(f"   Classes: {len(CLASS_NAMES)}")
    
    return model


# ====== PREDICTION ======
def predict(model, image_input, device='cuda' if torch.cuda.is_available() else 'cpu', 
            return_top_k=5):
    """
    Dự đoán class của ảnh (MobileNet)
    
    Args:
        model: Trained model
        image_input: PIL Image hoặc path to image
        device: 'cuda' hoặc 'cpu'
        return_top_k: Trả về top K predictions
    
    Returns:
        top_classes: List các class names (top K)
        top_scores: List các confidence scores (top K)
        top_indices: List các class indices (top K)
    
    Example:
        classes, scores, indices = predict(model, 'photo.jpg')
        print(f"Top prediction: {classes[0]} ({scores[0]:.2%})")
    """
    image = prepare_image_for_inference(image_input)
    image_tensor = inference_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities_percent = F.softmax(outputs, dim=1)[0] * 100
        top_scores_percent, indices = torch.topk(probabilities_percent, return_top_k)

    top_scores = (top_scores_percent / 100.0).cpu().numpy()
    indices = indices.cpu().numpy()

    top_classes = [CLASS_NAMES[int(index)] for index in indices]

    return top_classes, top_scores, indices


# ====== BATCH PREDICTION ======
def predict_batch(model, images_list, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Dự đoán batch ảnh (MobileNet)
    
    Args:
        model: Trained model
        images_list: List PIL Images
        device: Device
    
    Returns:
        predictions: List của (class_name, score) tuples
    """
    images_tensor = torch.stack([
        inference_transform(prepare_image_for_inference(img))
        for img in images_list
    ]).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(images_tensor)
        probabilities_percent = F.softmax(outputs, dim=1) * 100
        scores_percent, indices = torch.max(probabilities_percent, dim=1)
    
    predictions = []
    for score, idx in zip(scores_percent.cpu().numpy(), indices.cpu().numpy()):
        predictions.append((CLASS_NAMES[int(idx)], float(score) / 100.0))
    
    return predictions


# ====== TEST ======
if __name__ == "__main__":
    print("MobileNet Clothing Classifier Model")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Total classes: {len(CLASS_NAMES)}")
    print("\nBenefits:")
    print("✅ 10x smaller than ResNet")
    print("✅ 5x faster inference")
    print("✅ Works on CPU/Mobile")
    print("✅ ~80% accuracy")
