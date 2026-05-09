"""
app.py - Flask Web App cho phân loại quần áo (Multi-Model Support)

Hoàn toàn tự động - chỉ cần train xong model, copy vào ai_model/ hoặc mobilenet_model/ và chạy!
Chạy: python app.py hoặc MODEL_TYPE=mobilenet python app.py
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import json
from pathlib import Path
import os
from io import BytesIO
import tempfile
from dotenv import load_dotenv

# Load environment variables từ .env file
load_dotenv()

# Import models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Model type (default: resnet, options: resnet, mobilenet)
MODEL_TYPE = os.getenv('MODEL_TYPE', 'resnet').lower()
print(f"🎽 Selected model type: {MODEL_TYPE}")

# Import appropriate model
if MODEL_TYPE == 'mobilenet':
    try:
        from mobilenet_model.model import MobileNetClassifier as Classifier
        from mobilenet_model.model import load_model, predict, CLASS_NAMES, prepare_image_for_inference
        MODEL_PATH = Path(__file__).parent.parent / 'mobilenet_model' / 'clothing_model.pth'
        MODEL_VARIANT = 'MobileNet'
    except ImportError as e:
        print(f"ERROR: Could not import MobileNet model: {e}")
        MODEL_TYPE = 'resnet'
        from ai_model.model import ClothingClassifier as Classifier
        from ai_model.model import load_model, predict, CLASS_NAMES, prepare_image_for_inference
        MODEL_PATH = Path(__file__).parent.parent / 'ai_model' / 'best_model.pth'
        MODEL_VARIANT = 'ResNet50'
else:
    try:
        from ai_model.model import ClothingClassifier as Classifier
        from ai_model.model import load_model, predict, CLASS_NAMES, prepare_image_for_inference
        MODEL_PATH = Path(__file__).parent.parent / 'ai_model' / 'best_model.pth'
        MODEL_VARIANT = 'ResNet50'
    except ImportError as e:
        print(f"ERROR: Could not import ResNet model: {e}")
        print("Make sure ai_model/model.py exists!")
        Classifier = None
        CLASS_NAMES = []
        MODEL_PATH = None
        MODEL_VARIANT = 'Unknown'

# Import Cloudinary helper
try:
    from web_app.cloudinary_helper import create_cloudinary_helper
    CLOUDINARY_AVAILABLE = True
except ImportError:
    print("⚠️  Cloudinary not installed or helper not found")
    CLOUDINARY_AVAILABLE = False

# ==================== Initialize Flask ====================
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")
print(f"📊 Model path: {MODEL_PATH}")

# Global model
classifier = None
cloudinary_helper = None

CLASS_DISPLAY_NAMES = {
    'ao_khoac': 'Áo khoác',
    'ao_so_mi': 'Áo sơ mi',
    'ao_thun': 'Áo thun',
    'quan_jean': 'Quần jean',
    'quan_short': 'Quần short',
    'quan_tay': 'Quần tây',
    'sweater_hoodie': ' Hoodie',
    'vay': 'Váy',
}


def display_class_name(class_name: str) -> str:
    return CLASS_DISPLAY_NAMES.get(class_name, class_name)


def display_class_names(class_names: list[str]) -> list[str]:
    return [display_class_name(name) for name in class_names]


def load_trained_model():
    """Load trained model từ saved weights"""
    global classifier
    
    try:
        if MODEL_PATH is None or not MODEL_PATH.exists():
            print(f"❌ Model not found at {MODEL_PATH}")
            print(f"   Please download trained model from Colab & place it at {MODEL_PATH}")
            return False
        
        print(f"📥 Loading {MODEL_VARIANT} model from {MODEL_PATH}...")
        classifier = load_model(str(MODEL_PATH), device=device)
        
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
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
                         class_names=display_class_names(CLASS_NAMES),
                         model_type=MODEL_TYPE,
                         model_variant=MODEL_VARIANT)


@app.route('/api/status', methods=['GET'])
def api_status():
    """Get API status and info"""
    return jsonify({
        'status': 'ok',
        'model_loaded': classifier is not None,
        'model_type': MODEL_TYPE,
        'model_variant': MODEL_VARIANT,
        'device': str(device),
        'num_classes': len(CLASS_NAMES),
        'classes': display_class_names(CLASS_NAMES),
        'available_models': ['resnet', 'mobilenet']
    })


@app.route('/api/models', methods=['GET'])
def api_models():
    """Get available models and their status"""
    resnet_exists = (Path(__file__).parent.parent / 'ai_model' / 'best_model.pth').exists()
    mobilenet_exists = (Path(__file__).parent.parent / 'mobilenet_model' / 'clothing_model.pth').exists()
    
    return jsonify({
        'available': {
            'resnet': resnet_exists,
            'mobilenet': mobilenet_exists
        },
        'current': {
            'type': MODEL_TYPE,
            'variant': MODEL_VARIANT,
            'loaded': classifier is not None,
            'classes': display_class_names(CLASS_NAMES),
        }
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Inference API - POST image, GET predictions"""
    try:
        # Check model
        if classifier is None:
            return jsonify({
                'success': False,
                'error': f'Model not loaded. Ensure the selected checkpoint exists in the {MODEL_VARIANT.lower()} folder.'
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
            image = Image.open(BytesIO(file.read()))
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
                    prepare_image_for_inference(image).save(tmp.name)
                    
                    # Upload to Cloudinary
                    result = cloudinary_helper.upload_image(tmp.name)
                    if result['success']:
                        cloudinary_url = result['url']
                    
                    # Clean up
                    os.unlink(tmp.name)
            except Exception as e:
                print(f"Cloudinary upload error: {str(e)}")
        
        # Predict using model
        top_classes, top_scores, top_indices = predict(classifier, image, device=device, return_top_k=1)
        top_classes_display = [display_class_name(name) for name in top_classes]
        
        # Format results
        result = {
            'success': True,
            'model_used': MODEL_VARIANT,
            'prediction': {
                'class_index': int(top_indices[0]),
                'class_name': top_classes_display[0],
                'confidence': float(top_scores[0]),
                'confidence_percent': float(top_scores[0]) * 100
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
    print("\n" + "🎽"*30)
    print("CLOTHING CLASSIFIER - WEB APP (Multi-Model)")
    print("🎽"*30)
    
    # Try to load model
    model_loaded = load_trained_model()
    if model_loaded:
        print(f"✅ {MODEL_VARIANT} model ready for predictions")
    else:
        print("⚠️  Model not loaded - will run in demo mode")
        print(f"   Expected path: {MODEL_PATH}")
    
    # Try to initialize Cloudinary
    cloudinary_ok = init_cloudinary()
    if cloudinary_ok:
        print("✅ Cloudinary ready for image uploads")
    else:
        print("⚠️  Cloudinary not configured - using local upload only")
    
    # Print info
    print(f"\n🌐 Starting Flask server...")
    print(f"   URL: http://localhost:8000")
    print(f"   Model: {MODEL_VARIANT} ({MODEL_TYPE})")
    print(f"   Device: {device}")
    print(f"   Classes: {len(CLASS_NAMES)}")
    print(f"\n📝 API Endpoints:")
    print(f"   GET  / - Web interface")
    print(f"   GET  /api/status - Server status & info")
    print(f"   GET  /api/models - Available models")
    print(f"   POST /api/predict - Get prediction")
    print(f"\n💡 To use MobileNet instead of ResNet:")
    print(f"   MODEL_TYPE=mobilenet python app.py")
    print("\n" + "🎽"*30 + "\n")
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,
        use_reloader=False
    )
