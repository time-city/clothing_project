"""
app.py - Flask Web App cho phân loại quần áo

Đơn giản, dễ dùng, tích hợp Cloudinary.
Chạy: python app.py hoặc gunicorn -w 2 -b 0.0.0.0:5000 app:app
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import os
from io import BytesIO
import tempfile
from dotenv import load_dotenv

# Load environment variables từ .env file
load_dotenv()

# Import model từ ai_model folder
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'ai_model'))
try:
    from train import ClothingClassifier
except ImportError:
    print("WARNING: Could not import ClothingClassifier from train.py")
    ClothingClassifier = None

# Import Cloudinary helper
try:
    from cloudinary_helper import create_cloudinary_helper
    CLOUDINARY_AVAILABLE = True
except ImportError:
    print("WARNING: Cloudinary not available")
    CLOUDINARY_AVAILABLE = False

# ==================== Initialize Flask ====================
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Class names
CLASS_NAMES = [
    "Áo khoác",     # 0
    "Áo sơ mi",     # 1
    "Áo thun",      # 2
    "Quần Jean",    # 3
    "Quần short",   # 4
    "Quần tây",     # 5
    "Váy",          # 6
    "Váy dài"       # 7
]

# Model paths
MODEL_PATH = Path(__file__).parent.parent / 'ai_model' / 'best_model.pth'

# Global model
classifier = None
image_transform = None
cloudinary_helper = None


def load_model():
    """Load model từ checkpoint"""
    global classifier, image_transform
    
    try:
        if not MODEL_PATH.exists():
            print(f"Model not found at {MODEL_PATH}")
            return False
        
        print(f"Loading model from {MODEL_PATH}...")
        classifier = ClothingClassifier(num_classes=8)
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            classifier.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            classifier.model.load_state_dict(checkpoint)
        
        classifier.model.to(device)
        classifier.model.eval()
        
        # Image transforms
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


def init_cloudinary():
    """Initialize Cloudinary helper"""
    global cloudinary_helper
    
    if not CLOUDINARY_AVAILABLE:
        print("⚠️  Cloudinary not installed. Install with: pip install cloudinary")
        return False
    
    try:
        # Load từ environment variables (từ .env file)
        cloud_name = os.getenv('cloud_name')
        api_key = os.getenv('api_key')
        api_secret = os.getenv('api_secret')
        
        if not all([cloud_name, api_key, api_secret]):
            print("⚠️  Cloudinary credentials not found in .env file")
            print("   Create .env file with: cloud_name, api_key, api_secret")
            return False
        
        cloudinary_helper = create_cloudinary_helper(cloud_name, api_key, api_secret)
        print("✅ Cloudinary initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing Cloudinary: {str(e)}")
        return False


# ==================== Routes ====================

@app.route('/')
def index():
    """Serve main HTML page"""
    model_loaded = classifier is not None
    return render_template('index.html', 
                         model_loaded=model_loaded,
                         class_names=CLASS_NAMES)


@app.route('/api/status', methods=['GET'])
def api_status():
    """Get API status and info"""
    return jsonify({
        'status': 'ok',
        'model_loaded': classifier is not None,
        'device': str(device),
        'num_classes': len(CLASS_NAMES),
        'classes': CLASS_NAMES
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Inference API - POST image, GET predictions"""
    try:
        # Check model
        if classifier is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Ensure best_model.pth exists in ai_model folder.'
            }), 400
        
        # Get image
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided in request'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Load image
        try:
            image = Image.open(BytesIO(file.read())).convert('RGB')
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            }), 400
        
        # Upload to Cloudinary (optional)
        cloudinary_url = None
        if cloudinary_helper is not None:
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    image.save(tmp.name)
                    
                    # Upload to Cloudinary
                    result = cloudinary_helper.upload_image(tmp.name)
                    if result['success']:
                        cloudinary_url = result['url']
                    
                    # Clean up
                    os.unlink(tmp.name)
            except Exception as e:
                print(f"Cloudinary upload error: {str(e)}")
        
        # Predict
        with torch.no_grad():
            image_tensor = image_transform(image).unsqueeze(0).to(device)
            output = classifier.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Format results
        probs = probabilities[0].cpu().numpy()
        predicted_idx = predicted.item()
        confidence_value = float(confidence.item())
        
        result = {
            'success': True,
            'prediction': {
                'class_index': predicted_idx,
                'class_name': CLASS_NAMES[predicted_idx],
                'confidence': confidence_value,
                'confidence_percent': confidence_value * 100
            },
            'all_probabilities': {
                CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probs)
            }
        }
        
        # Add Cloudinary URL if uploaded
        if cloudinary_url:
            result['image_url'] = cloudinary_url
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


# ==================== Error Handlers ====================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size: 16MB'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ==================== Main ====================

if __name__ == '__main__':
    print("Starting Flask server...")
    
    # Try to load model
    model_loaded = load_model()
    if model_loaded:
        print("✅ Model loaded successfully")
    else:
        print("⚠️  Model not loaded - will run in demo mode")
    
    # Try to initialize Cloudinary
    cloudinary_ok = init_cloudinary()
    if cloudinary_ok:
        print("✅ Cloudinary ready for image uploads")
    else:
        print("⚠️  Cloudinary not configured - using local upload only")
    
    # Run app (use port 8000 to avoid conflicts)
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,
        use_reloader=False
    )
