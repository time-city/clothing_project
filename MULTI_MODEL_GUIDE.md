# 🎽 Multi-Model Setup Guide

## Overview

Your clothing classifier now supports **two models**:
- **ResNet50**: High accuracy (~85%), larger model (~95MB)
- **MobileNet**: Fast inference (~80% accuracy), lightweight (~5MB)

## ⚡ Quick Start

### Step 1: Train Models on Google Colab

**Train ResNet50:**
```bash
# In Colab, run colab_train.py
# Model saves to: ai_model/resnet50-11ad3fa6.pth
```

**Train MobileNet (optional):**
```bash
# Modify colab_train.py to use MobileNetV3 instead of ResNet50
# Model saves to: mobilenet_model/clothing_model.pth
```

### Step 2: Download Models Locally

After training, download the `.pth` files:
- `ai_model/resnet50-11ad3fa6.pth` → ResNet50 weights
- `mobilenet_model/clothing_model.pth` → MobileNet weights

### Step 3: Run Web App

**Using ResNet50 (default):**
```bash
python web_app/app.py
```

**Using MobileNet:**
```bash
MODEL_TYPE=mobilenet python web_app/app.py
```

**Both Available (user can choose):**
- If both models exist, users can switch via dropdown in the web UI

## 📋 API Endpoints

### `/api/status` - Server Status
```json
{
  "model_loaded": true,
  "model_type": "resnet",
  "model_variant": "ResNet50",
  "device": "cuda",
  "classes": 8,
  "available_models": ["resnet", "mobilenet"]
}
```

### `/api/models` - Available Models
```json
{
  "available": {
    "resnet": true,
    "mobilenet": false
  },
  "current": {
    "type": "resnet",
    "variant": "ResNet50",
    "loaded": true
  }
}
```

### `/api/predict` - Get Prediction
**Request:**
```bash
POST /api/predict
Content-Type: multipart/form-data
image: <file>
```

**Response:**
```json
{
  "success": true,
  "model_used": "ResNet50",
  "prediction": {
    "class_name": "ao_thun",
    "confidence": 0.92,
    "confidence_percent": 92.0
  },
  "top_5": [
    {"rank": 1, "class_name": "ao_thun", "confidence": 0.92},
    {"rank": 2, "class_name": "ao_khoac", "confidence": 0.05},
    ...
  ]
}
```

## 🔄 Model Comparison

| Feature | ResNet50 | MobileNet |
|---------|----------|-----------|
| **Accuracy** | ~85% | ~80% |
| **Model Size** | ~95MB | ~5MB |
| **Inference Speed** | 100-200ms | 10-50ms |
| **GPU Memory** | 500MB+ | 100MB+ |
| **Best For** | High accuracy critical | Speed, mobile, edge |

## 🎛️ Web UI Features

1. **Model Selector Dropdown**
   - Choose between ResNet50 and MobileNet
   - Automatically disables unavailable models
   - Selection saved in browser localStorage

2. **Model Indicator**
   - Shows which model made the prediction
   - Displays model availability status
   - Device and classes information

3. **Top-5 Predictions**
   - Shows confidence for each class
   - Visual rank badges (#1, #2, etc.)
   - Bar chart visualization

## 🚀 Performance Tips

### ResNet50
- Use when accuracy is critical
- Works best with GPU (CUDA)
- Good for complex/ambiguous images

### MobileNet
- Use for real-time applications
- Runs well on CPU
- Fast enough for mobile apps
- Good for clear, distinct clothing items

## 📁 Project Structure

```
clothing_project/
├── ai_model/
│   ├── model.py           # ResNet50 model
│   ├── resnet50-11ad3fa6.pth # ResNet50 weights
│   └── __init__.py
│
├── mobilenet_model/
│   ├── model.py           # MobileNet model
│   ├── clothing_model.pth # MobileNet weights (optional)
│   └── __init__.py
│
├── web_app/
│   ├── app.py             # Flask app with multi-model support
│   ├── templates/
│   │   └── index.html     # UI with model selector
│   └── static/
│       ├── css/style.css  # Styling
│       └── js/app.js      # JavaScript logic
│
└── colab_train.py         # Colab training script
```

## 🔧 Environment Variables

```bash
# .env file
MODEL_TYPE=resnet          # default: resnet, options: resnet, mobilenet
cloud_name=your_cloud      # Cloudinary (optional)
api_key=your_key           # Cloudinary (optional)
api_secret=your_secret     # Cloudinary (optional)
```

## ✅ Checklist

- [ ] Train ResNet50 on Colab
- [ ] Download `ai_model/resnet50-11ad3fa6.pth`
- [ ] (Optional) Train MobileNet on Colab
- [ ] (Optional) Download `mobilenet_model/clothing_model.pth`
- [ ] Run `python web_app/app.py`
- [ ] Test with sample images
- [ ] Choose model via dropdown
- [ ] Check predictions

## 🐛 Troubleshooting

**"Model not found" error:**
- Ensure you've downloaded the `.pth` file from Colab
- Check file is in correct folder: `ai_model/` or `mobilenet_model/`
- Verify filename is `clothing_model.pth`

**Model selector disabled:**
- Model file doesn't exist at expected path
- Check `/api/models` endpoint to see available models

**Slow predictions with ResNet50:**
- Using CPU instead of GPU
- Check `deviceInfo` shows CUDA
- For CPU-only, use MobileNet instead

**Predictions inconsistent between models:**
- Models trained on slightly different data splits
- Normal behavior - models have different architectures
- Choose based on your accuracy/speed preference

## 📚 References

- **ResNet**: Deep Residual Learning (He et al., 2015)
- **MobileNet**: Efficient Convolutional Neural Networks (Howard et al., 2017)
- **PyTorch**: [pytorch.org](https://pytorch.org)
- **TorchVision**: Pre-trained models for computer vision

## 💡 Next Steps

1. **Fine-tune models** with your own dataset
2. **Deploy** to cloud (Heroku, AWS, Azure)
3. **Containerize** with Docker
4. **Monitor** predictions and user feedback
5. **Retrain** periodically with new data

---

🎽 **Happy classifying!** Choose the model that best fits your needs.
