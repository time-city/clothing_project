# 🚀 Complete Setup Guide - From Training to Deployment

## 📋 Overview

```
1. Train on Colab (colab_train.py)
   ↓
2. Download model (resnet50-11ad3fa6.pth)
   ↓
3. Place in ai_model/ folder
   ↓
4. Run web app (python app.py)
   ↓
5. Start predicting! 🎉
```

---

## 🎯 Step 1: Training on Google Colab

### 1.1 Open Colab
- Go to https://colab.research.google.com/
- Create new notebook

### 1.2 Install Dependencies (Cell 1)
```python
!pip install torch torchvision pandas tqdm pillow requests matplotlib numpy
```

### 1.3 Mount Google Drive (Cell 2)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 1.4 Copy Training Script (Cell 3)
- Copy entire code from `colab_train.py`
- Paste into Colab cell
- **Run** (it will auto-detect GPU & download all images)

### 1.5 Wait for Training
- Script automatically:
  - Detects GPU type (T4/V100/A100)
  - Downloads ALL images from GitHub
  - Trains with optimal config
  - Saves model: `/content/drive/My Drive/resnet50-11ad3fa6.pth`
  - Saves graphs: `/content/drive/My Drive/training_history.png`

**Estimated time:**
- T4: 60-90 minutes
- V100: 45-60 minutes  
- A100: 30-40 minutes

---

## 💾 Step 2: Download Model

### 2.1 From Google Drive
1. Open Google Drive: https://myaccount.google.com/
2. Find `resnet50-11ad3fa6.pth`
3. Download to your computer

### 2.2 Place in Project
```bash
# Copy downloaded file to:
cp ~/Downloads/resnet50-11ad3fa6.pth /path/to/clothing_project/ai_model/
```

**Important**: File name must be exactly `resnet50-11ad3fa6.pth`

---

## 🖥️ Step 3: Setup Web App Locally

### 3.1 Install Dependencies
```bash
cd clothing_project/web_app
pip install -r requirements_web.txt
```

Or install manually:
```bash
pip install flask torch torchvision pillow python-dotenv cloudinary
```

### 3.2 Check Model Location
```bash
ls -la ../ai_model/resnet50-11ad3fa6.pth
# Should output: .../resnet50-11ad3fa6.pth
```

### 3.3 (Optional) Setup Cloudinary for Cloud Uploads
1. Create `.env` file in `web_app/` folder:
```
cloud_name=YOUR_CLOUD_NAME
api_key=YOUR_API_KEY
api_secret=YOUR_API_SECRET
```

2. Get credentials from: https://cloudinary.com/

---

## 🚀 Step 4: Run Web App

### 4.1 Start Server
```bash
cd clothing_project/web_app
python app.py
```

### 4.2 Open Browser
- Go to: http://localhost:8000
- You should see the web interface

### 4.3 Test Prediction
1. Upload an image (jeans, shirt, etc.)
2. Click "Predict"
3. See results in real-time!

---

## 📡 API Usage

### Get Status
```bash
curl http://localhost:8000/api/status
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "num_classes": 8,
  "classes": [
    "ao_khoac",
    "ao_so_mi",
    "ao_thun",
    "quan_jean",
    "quan_short",
    "quan_tay",
    "sweater_hoodie",
    "vay"
  ]
}
```

### Make Prediction
```bash
curl -X POST \
  -F "image=@photo.jpg" \
  http://localhost:8000/api/predict
```

Response:
```json
{
  "success": true,
  "prediction": {
    "class_index": 3,
    "class_name": "quan_jean",
    "confidence": 0.9842,
    "confidence_percent": 98.42
  },
  "top_5": [
    {
      "rank": 1,
      "class_name": "quan_jean",
      "confidence": 0.9842,
      "confidence_percent": 98.42
    },
    {
      "rank": 2,
      "class_name": "quan_short",
      "confidence": 0.0124,
      "confidence_percent": 1.24
    }
  ]
}
```

---

## 🐛 Troubleshooting

### ❌ Model Not Found Error
**Error**: `Model not found at .../ai_model/clothing_model.pth`

**Solution**:
1. Check file exists: `ls ai_model/clothing_model.pth`
2. File name must be exactly: `clothing_model.pth`
3. Download from Google Drive if missing

### ❌ Import Error: Could not import model
**Error**: `Could not import model: No module named 'ai_model'`

**Solution**:
1. Check `ai_model/` folder exists
2. Check `ai_model/__init__.py` exists
3. Check `ai_model/model.py` exists
4. Reinstall: `pip install -e .`

### ❌ CUDA Out of Memory
**Error**: `CUDA out of memory`

**Solution**:
1. Restart Flask: `Ctrl+C` then `python app.py`
2. Use CPU instead (auto-fallback if no GPU)

### ❌ Port 8000 Already in Use
**Error**: `Address already in use`

**Solution**:
1. Kill process: `lsof -ti:8000 | xargs kill -9`
2. Or change port in `app.py`

### ⚠️ Cloudinary Not Working
**Error**: `Cloudinary credentials not found`

**Solution**:
- Optional! Works fine without it
- Set up `.env` if you want cloud upload
- Local upload always works

---

## 📦 Project Structure

```
clothing_project/
├── colab_train.py                 # ← Use for training on Colab
│
├── ai_model/                      # ← Model folder
│   ├── __init__.py
│   ├── model.py                   # ← Model code (auto-loaded)
│   └── clothing_model.pth         # ← Download from Colab here!
│
└── web_app/
    ├── app.py                     # ← Run this (python app.py)
    ├── requirements_web.txt
    ├── .env                       # ← Optional Cloudinary config
    ├── templates/
    │   └── index.html
    └── static/
        ├── css/
        └── js/
```

---

## 🚀 Deployment to Production

### Option 1: Heroku
```bash
# Install Heroku CLI
# Create Procfile
echo "web: gunicorn -w 2 -b 0.0.0.0:$PORT app:app" > Procfile

# Deploy
heroku create clothing-app
heroku config:set MODEL_PATH=ai_model/clothing_model.pth
git push heroku main
```

### Option 2: AWS
```bash
# Use EC2 + GitHub Actions for auto-deploy
# Upload model to S3
# Download in EC2 on startup
```

### Option 3: Docker
```bash
# Build image
docker build -t clothing-app .

# Run
docker run -p 8000:8000 clothing-app
```

---

## ✅ Checklist

- [ ] Trained model on Colab
- [ ] Downloaded `clothing_model.pth`
- [ ] Placed in `ai_model/` folder
- [ ] Installed web_app requirements
- [ ] Started Flask app (`python app.py`)
- [ ] Opened http://localhost:8000
- [ ] Tested prediction with image
- [ ] (Optional) Set up Cloudinary

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `colab_train.py` | Training script for Colab |
| `ai_model/model.py` | Model architecture & inference |
| `ai_model/clothing_model.pth` | Trained weights (download from Colab) |
| `web_app/app.py` | Flask web server |
| `web_app/templates/index.html` | Web interface |

---

## 🎉 Done!

Your clothing classifier is now:
- ✅ Trained on 8,000+ images
- ✅ Running locally on http://localhost:8000
- ✅ Ready for predictions
- ✅ Deployable to production

**Enjoy! 👔👗👖**
