# 🎽 Clothing Classifier - Complete End-to-End Setup

## ⚡ 3-Step Quick Start

### Step 1: Train (on Colab)
```bash
# Copy colab_train.py code to Colab
# Script auto-detects GPU & trains
# Download resnet50-11ad3fa6.pth from the model source
```

### Step 2: Setup (Local Machine)
```bash
cd clothing_project
chmod +x quick_setup.sh
./quick_setup.sh
```

### Step 3: Run (Local Machine)
```bash
cd web_app
source venv/bin/activate
python app.py
```

**Open browser**: http://localhost:8000 🎉

---

## 📚 Full Documentation

- **Detailed Setup**: See `DEPLOYMENT_GUIDE.md`
- **Training Guide**: See `COLAB_TRAINING_GUIDE.md`
- **API Reference**: See `COLAB_QUICK_REFERENCE.md`
- **Auto-Config Info**: See `AUTO_CONFIG_README.md`

---

## 🗂️ What's Included

### Training
- ✅ `colab_train.py` - Auto-optimized training script
- ✅ Auto GPU detection (T4/V100/A100)
- ✅ Download ALL images from GitHub
- ✅ Save model to Google Drive

### Model
- ✅ `ai_model/model.py` - ResNet50 classifier
- ✅ Easy-to-use API
- ✅ Support for inference & batch prediction

### Web App
- ✅ `web_app/app.py` - Flask server
- ✅ REST API for predictions
- ✅ Clean web interface
- ✅ Cloudinary integration (optional)

---

## 🚀 Usage Examples

### Python API
```python
from ai_model.model import load_model, predict

# Load model
model = load_model('ai_model/resnet50-11ad3fa6.pth')

# Predict
classes, scores, indices = predict(model, 'photo.jpg')
print(f"Top prediction: {classes[0]} ({scores[0]:.0%})")
```

### REST API
```bash
curl -X POST \
  -F "image=@shirt.jpg" \
  http://localhost:8000/api/predict
```

### Web Interface
- Open http://localhost:8000
- Upload image
- See prediction instantly

---

## 📊 Classes Supported

1. `ao_khoac` - Áo khoác (Jacket)
2. `ao_so_mi` - Áo sơ mi (Shirt)
3. `ao_thun` - Áo thun (T-shirt)
4. `quan_jean` - Quần jeans (Jeans)
5. `quan_short` - Quần short (Shorts)
6. `quan_tay` - Quần tây (Trousers)
7. `sweater_hoodie` - Sweater/Hoodie
8. `vay` - Váy (Dress)

---

## 🎯 Expected Performance

| Config | Images | Train Time (T4) | Accuracy |
|--------|--------|-----------------|----------|
| Standard | 500 | 30 min | ~78% |
| Full | 1000+ | 60 min | ~82% |
| All data | Full | 90 min | ~85%+ |

---

## 🔧 Troubleshooting

### Model Not Loading
```
Check: ai_model/resnet50-11ad3fa6.pth exists
Ensure: Exact filename "resnet50-11ad3fa6.pth"
```

### Port Already in Use
```
Kill: lsof -ti:8000 | xargs kill -9
Or change port in app.py
```

### GPU Issues
```
Automatic fallback to CPU
Check: torch.cuda.is_available()
```

---

## 📦 Requirements

- Python 3.8+
- GPU (optional, CPU works too)
- 4GB RAM minimum
- 500MB disk space

---

## 🚀 Next Steps

1. Train model on Colab
2. Download to local machine
3. Run setup script
4. Start web app
5. Upload images & get predictions
6. Deploy to production (Heroku/AWS/Docker)

---

**Ready? Start with `colab_train.py` on Google Colab!** 🎉
