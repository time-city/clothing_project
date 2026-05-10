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

# Run BOTH models (resnet + mobilenet) together (no model selection)

# Import modules
import ai_model.model as resnet_module
import mobilenet_model.model as mobilenet_module

# Paths to checkpoints
RESNET_PATH = Path(__file__).parent.parent / 'ai_model' / 'best_model.pth'
MOBILENET_PATH = Path(__file__).parent.parent / 'mobilenet_model' / 'clothing_model.pth'

# Unified class names
CLASS_NAMES = resnet_module.CLASS_NAMES

MODEL_VARIANTS = {
    'resnet': {
        'module': resnet_module,
        'path': RESNET_PATH,
        'variant': 'ResNet50',
    },
    'mobilenet': {
        'module': mobilenet_module,
        'path': MOBILENET_PATH,
        'variant': 'MobileNet',
    },
}


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
print(f"📊 Checkpoints: resnet={RESNET_PATH} | mobilenet={MOBILENET_PATH}")


# Global models
resnet_classifier = None
mobilenet_classifier = None

# status dict to report errors to frontend
model_status = {
    'resnet': {'loaded': False, 'error': None, 'variant': MODEL_VARIANTS['resnet']['variant']},
    'mobilenet': {'loaded': False, 'error': None, 'variant': MODEL_VARIANTS['mobilenet']['variant']},
}

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


def load_one_model(model_key: str):
    """Load one model and save its status for frontend error reporting."""
    global resnet_classifier, mobilenet_classifier

    meta = MODEL_VARIANTS[model_key]
    module = meta['module']
    model_path = meta['path']

    try:
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"📥 Loading {meta['variant']} model from {model_path}...")
        model = module.load_model(str(model_path), device=device)

        if model_key == 'resnet':
            resnet_classifier = model
        else:
            mobilenet_classifier = model

        model_status[model_key]['loaded'] = True
        model_status[model_key]['error'] = None
        print("✅ Model loaded successfully!")
        return True

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Error loading {meta['variant']}: {str(e)}")
        print(tb)

        model_status[model_key]['loaded'] = False
        model_status[model_key]['error'] = f"{str(e)}\n{tb}"
        return False


def load_trained_models():
    ok_res = load_one_model('resnet')
    ok_mob = load_one_model('mobilenet')
    return ok_res or ok_mob



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
    return render_template(
        'index.html',
        class_names=display_class_names(CLASS_NAMES),
        model_status=model_status,
    )



@app.route('/api/status', methods=['GET'])
def api_status():
    """Get API status and info for both models."""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'num_classes': len(CLASS_NAMES),
        'classes': display_class_names(CLASS_NAMES),
        'available_models': {
            'resnet': bool(RESNET_PATH.exists()),
            'mobilenet': bool(MOBILENET_PATH.exists()),
        },
        'model_status': model_status,
    })



@app.route('/api/models', methods=['GET'])
def api_models():
    """Get available models and their status (for frontend)."""
    return jsonify({
        'device': str(device),
        'classes': display_class_names(CLASS_NAMES),
        'model_status': model_status,
        'available': {
            'resnet': bool(RESNET_PATH.exists()),
            'mobilenet': bool(MOBILENET_PATH.exists()),
        },
    })



@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Inference API - run BOTH models on the same uploaded image."""
    try:
        # Get image
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided in request'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Load image
        try:
            image = Image.open(BytesIO(file.read()))
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid image file: {str(e)}'}), 400

        # Upload to Cloudinary (optional) once
        cloudinary_url = None
        if cloudinary_helper is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    resnet_module.prepare_image_for_inference(image).save(tmp.name)
                    result = cloudinary_helper.upload_image(tmp.name)
                    if result.get('success'):
                        cloudinary_url = result.get('url')
                    os.unlink(tmp.name)
            except Exception as e:
                print(f"Cloudinary upload error: {str(e)}")

        import time
        predictions = {
            'resnet': None,
            'mobilenet': None,
        }
        timings = {
            'resnet_ms': None,
            'mobilenet_ms': None,
        }

        # ResNet predict
        if resnet_classifier is not None:
            try:
                start = time.perf_counter()
                top_classes, top_scores, top_indices = resnet_module.predict(
                    resnet_classifier, image, device=device, return_top_k=1
                )
                end = time.perf_counter()
                timings['resnet_ms'] = (end - start) * 1000.0
                predictions['resnet'] = {
                    'class_index': int(top_indices[0]),
                    'class_name': display_class_name(top_classes[0]),
                    'confidence': float(top_scores[0]),
                    'confidence_percent': float(top_scores[0]) * 100,
                }
            except Exception as e:
                predictions['resnet'] = None
                model_status['resnet']['loaded'] = False
                model_status['resnet']['error'] = f"Predict error: {str(e)}"

        # MobileNet predict
        if mobilenet_classifier is not None:
            try:
                start = time.perf_counter()
                top_classes, top_scores, top_indices = mobilenet_module.predict(
                    mobilenet_classifier, image, device=device, return_top_k=1
                )
                end = time.perf_counter()
                timings['mobilenet_ms'] = (end - start) * 1000.0
                predictions['mobilenet'] = {
                    'class_index': int(top_indices[0]),
                    'class_name': display_class_name(top_classes[0]),
                    'confidence': float(top_scores[0]),
                    'confidence_percent': float(top_scores[0]) * 100,
                }
            except Exception as e:
                predictions['mobilenet'] = None
                model_status['mobilenet']['loaded'] = False
                model_status['mobilenet']['error'] = f"Predict error: {str(e)}"


        if predictions['resnet'] is None and predictions['mobilenet'] is None:
            return jsonify({'success': False, 'error': 'No model could predict.', 'model_status': model_status}), 400

        result = {
            'success': True,
            'predictions': predictions,
            'model_status': model_status,
            'timings_ms': timings,
        }

        if cloudinary_url:
            result['image_url'] = cloudinary_url

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500



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
    
    # Try to load both models
    models_ok = load_trained_models()
    if models_ok:
        print("✅ At least one model ready for predictions")
    else:
        print("⚠️  Neither model loaded - predictions will fail")

    
    # Try to initialize Cloudinary
    cloudinary_ok = init_cloudinary()
    if cloudinary_ok:
        print("✅ Cloudinary ready for image uploads")
    else:
        print("⚠️  Cloudinary not configured - using local upload only")
    
    # Print info
    print(f"\n🌐 Starting Flask server...")
    print(f"   URL: http://localhost:8000")
    print(f"   Models: ResNet50 + MobileNet")

    print(f"   Device: {device}")
    print(f"   Classes: {len(CLASS_NAMES)}")
    print(f"\n📝 API Endpoints:")
    print(f"   GET  / - Web interface")
    print(f"   GET  /api/status - Server status & info")
    print(f"   GET  /api/models - Available models")
    print(f"   POST /api/predict - Get prediction")
    print(f"\n💡 To use MobileNet instead of ResNet:")
    print("   (Hệ thống chạy cả ResNet50 + MobileNet để so sánh)")
    print("\n" + "🎽"*30 + "\n")
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,
        use_reloader=False
    )
