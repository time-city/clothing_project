"""
ResNet50 Model Module - Clothing Classification
Best accuracy with good speed tradeoff

Advantages:
- High accuracy (~85%)
- Powerful for complex patterns
- Well-established architecture
- Transfer learning friendly

Disadvantages:
- Slower than MobileNet (5-10x)
- Larger model size (~100MB)
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
class ClothingClassifier(nn.Module):
    """
    ResNet50 Model cho phân loại quần áo
    
    High-performance architecture:
    - ResNet50: ~100MB (full weights)
    - Inference: 50-100ms per image
    - Accuracy: ~85%
    
    Good for:
    - Desktop/server applications
    - High accuracy requirements
    - Complex clothing patterns
    - Production quality
    """
    
    def __init__(self, num_classes=8, pretrained=True):
        super(ClothingClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace final layer for 8 clothing classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)


# ====== MODEL LOADING ======
def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load trained ResNet50 model
    
    Args:
        model_path: Path to saved model weights (.pth file)
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        model: Loaded model ready for inference
    
    Example:
        model = load_model('clothing_model.pth', device='cuda')
    """
    model = ClothingClassifier(num_classes=8)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats and legacy key layouts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # If checkpoint uses 'backbone.' prefix, convert to 'resnet.' for this model
        if any(key.startswith('backbone.') for key in state_dict.keys()):
            # Convert backbone.* -> resnet.*
            resnet_state_dict = {
                key.replace('backbone.', 'resnet.'): value 
                for key, value in state_dict.items() 
                if key.startswith('backbone.')
            }
        else:
            # Older checkpoints were saved from plain torchvision ResNet50,
            # add 'resnet.' prefix
            resnet_state_dict = {f'resnet.{key}': value for key, value in state_dict.items()}
        
        model.load_state_dict(resnet_state_dict, strict=False)
    
    # Move to device and eval mode
    model = model.to(device)
    model.eval()
    
    return model


# ====== PREDICTION ======
def predict(model, image_input, device='cuda' if torch.cuda.is_available() else 'cpu', return_top_k=5):
    """
    Dự đoán class của ảnh (ResNet)
    
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
def predict_batch(model, image_list, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Dự đoán nhiều ảnh cùng lúc (batch)
    
    Args:
        model: Trained model
        image_list: List of PIL Images hoặc paths
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        predictions: List of (top_class, confidence) tuples
    
    Example:
        images = ['photo1.jpg', 'photo2.jpg']
        results = predict_batch(model, images)
        for img, pred in zip(images, results):
            print(f"{img}: {pred[0]} ({pred[1]:.2%})")
    """
    predictions = []

    for image_input in image_list:
        top_classes, top_scores, _ = predict(model, image_input, device=device, return_top_k=1)
        predictions.append((top_classes[0], float(top_scores[0])))

    return predictions


if __name__ == '__main__':
    print("🎽 ResNet50 Clothing Classifier")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"Total classes: {len(CLASS_NAMES)}")
