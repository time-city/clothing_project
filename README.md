# 👗 Clothing Classification Project

Dự án hoàn chỉnh phân loại quần áo (8 lớp) sử dụng **ResNet50** Transfer Learning.

## 📋 Tổng Quan Dự Án

Dự án được chia làm **2 phần chính**:

### 1️⃣ **AI Model** (`/ai_model`)
- **Mô hình**: ResNet50 + Transfer Learning
- **Framework**: PyTorch
- **Chức năng**: Huấn luyện model phân loại quần áo
- **File chính**:
  - `dataset.py` - Xử lý dữ liệu
  - `train.py` - Huấn luyện model
  - `AI_README.md` - Hướng dẫn chi tiết

### 2️⃣ **Web App** (`/web_app`)
- **Framework**: Flask + HTML/CSS/JS
- **Giao diện**: Đơn giản, sạch sẽ, không gradient
- **Chức năng**: Upload ảnh, dự đoán, xem kết quả
- **File chính**:
  - `app.py` - Flask server
  - `templates/index.html` - HTML page
  - `static/css/style.css` - Styling
  - `static/js/app.js` - Frontend logic
  - `SOFTWARE_README.md` - Hướng dẫn chi tiết

## 🎯 8 Loại Quần Áo

1. **👕 Áo Thun** (T-shirt)
2. **👔 Áo Sơ Mi** (Shirt)
3. **🧥 Áo Khoác** (Coat/Jacket)
4. **👖 Quần Tây** (Trousers)
5. **👖 Quần Jean** (Jeans)
6. **🧶 Áo Len** (Sweater)
7. **👗 Váy** (Dress)
8. **🩳 Quần Short** (Shorts)

## 🚀 Quick Start

### Bước 1: Chuẩn Bị Dataset

```bash
# Tạo cấu trúc folder
mkdir -p data/{áo\ thun,áo\ sơ\ mi,áo\ khoác,quần\ tây,quần\ jean,áo\ len,váy,quần\ short}

# Download ảnh vào từng folder (500+ ảnh cho mỗi class)
```

### Bước 2: Huấn Luyện Model (Local hoặc Google Colab)

#### **Local Setup:**
```bash
cd ai_model

# Cài thư viện
pip install -r requirements_ai.txt

# Huấn luyện
python train.py --dataset ../data --epochs 20 --batch_size 32

# Kiểm tra output
ls -la
# Phải có: best_model.pth
```

#### **Google Colab Setup (Khuyên nghị):**
```python
# 1. Mở: https://colab.research.google.com
# 2. Upload project hoặc clone từ GitHub:
!git clone https://github.com/yourname/clothing_project.git
%cd clothing_project/ai_model

# 3. Cài dependencies
!pip install -r requirements_ai.txt

# 4. Tải dataset (ví dụ từ local hoặc URL)
# Tạo folder data trong Colab

# 5. Huấn luyện (tự động dùng GPU)
!python train.py --dataset /content/clothing_project/data --epochs 20

# 6. Model lưu tại /content/clothing_project/ai_model/best_model.pth
```

### Bước 3: Chạy Web App

#### **Local:**
```bash
cd web_app

# Cài thư viện
pip install -r requirements_web.txt

# Chạy Flask app
python app.py

# Truy cập: http://localhost:5000
```

#### **Google Colab:**
```python
%cd /content/clothing_project/web_app

# Cài dependencies
!pip install -r requirements_web.txt

# Chạy với ngrok (public URL)
!pip install flask-ngrok pyngrok

# Sửa app.py thêm:
# from flask_ngrok import run_with_ngrok
# run_with_ngrok(app)

# Chạy
!python app.py
```

## 📁 Cấu Trúc Thư Mục

```
clothing_project/
│
├── ai_model/                    # Phần AI Model
│   ├── dataset.py              # Xử lý dataset
│   ├── train.py                # Huấn luyện model
│   ├── AI_README.md            # Hướng dẫn AI
│   ├── requirements_ai.txt     # Dependencies cho AI
│   └── best_model.pth          # Model checkpoint (sau training)
│
├── web_app/                     # Phần Web App
│   ├── app.py                  # Flask server
│   ├── SOFTWARE_README.md      # Hướng dẫn Web App
│   ├── requirements_web.txt    # Dependencies cho Web
│   ├── templates/
│   │   └── index.html          # HTML page
│   └── static/
│       ├── css/
│       │   └── style.css       # Styling (clean, no gradients)
│       └── js/
│           └── app.js          # Frontend logic
│
├── data/                        # Dataset folder (tạo sau)
│   ├── áo thun/
│   ├── áo sơ mi/
│   ├── áo khoác/
│   ├── quần tây/
│   ├── quần jean/
│   ├── áo len/
│   ├── váy/
│   └── quần short/
│
└── README.md                    # File này
```

## ⚙️ Yêu Cầu Hệ Thống

### Hardware
- **CPU**: Intel i5 trở lên
- **GPU**: NVIDIA GeForce GTX 1050 (tùy chọn, để tăng tốc)
- **RAM**: Tối thiểu 8GB (16GB khuyên nghị)
- **Storage**: 10GB (cho model + dataset)

### Software
- **Python**: 3.8+
- **pip**: 21.0+
- **Git** (tùy chọn)

## 📚 Hướng Dẫn Chi Tiết

### Phần AI Model
Xem file `ai_model/AI_README.md` để có hướng dẫn chi tiết về:
- Chuẩn bị dataset
- Cài đặt PyTorch
- Huấn luyện model
- Hiểu các metrics
- Tips cải thiện accuracy

### Phần Web App
Xem file `web_app/SOFTWARE_README.md` để có hướng dẫn chi tiết về:
- Cài đặt Streamlit
- Tích hợp Cloudinary
- Sử dụng dashboard
- Deploy lên server
- Troubleshooting

## 🔑 Yêu Cầu Bên Ngoài

### Google Colab (Khuyên Nghị)
- **Miễn phí**: Không cần setup
- **GPU**: Tesla T4/P100 tốc độ cao
- **Storage**: 15GB persistent storage
- **Link**: https://colab.research.google.com/

### Cloudinary (Optional)
1. Đăng ký: https://cloudinary.com/
2. **Không bắt buộc** cho Flask version
3. **Gói miễn phí** bao gồm:
   - ✅ 25GB storage
   - ✅ 25GB bandwidth/tháng

## 📊 Kỳ Vọng Hiệu Năng

Với dataset 500+ ảnh cho mỗi class:
- **Accuracy**: 88-94%
- **Training time**: 30-60 phút (CPU) / 5-15 phút (GPU)
- **Inference time**: 50-100ms per image
- **Model size**: ~100MB

## 💡 Thông Tin Thêm

### Transfer Learning
Dự án sử dụng ResNet50 pre-trained từ ImageNet:
- ✅ Giảm thời gian huấn luyện 50%
- ✅ Cải thiện accuracy 10-20%
- ✅ Cần ít dữ liệu hơn (1000 ảnh đủ)

### Flask + HTML/CSS/JS
Giao diện web được thiết kế:
- **Đơn giản**: Không quá phức tạp
- **Sạch sẽ**: White/light gray background
- **Dễ dùng**: Upload drag-and-drop
- **Không gradient**: Chỉ dùng màu solid
- **Responsive**: Hoạt động tốt trên mobile

### Google Colab Optimization
Model training được tối ưu cho Colab:
- **Batch size**: 16 (tránh OOM trên T4)
- **Path detection**: Auto detect `/content/data`
- **Checkpoint**: Tự động lưu best_model.pth

## 🔧 Troubleshooting Nhanh

| Lỗi | Nguyên Nhân | Cách Khắc Phục |
|-----|-----------|----------------|
| Model not found | Chưa huấn luyện | Chạy `python train.py` trong `ai_model` |
| Out of memory (Colab) | Batch size quá lớn | Default 16 đã được tối ưu cho T4 |
| Flask connection refused | Port 5000 bị dùng | Chạy trên port khác: `app.run(port=8000)` |
| Ảnh upload thất bại | File quá lớn/format sai | Check: <16MB, JPG/PNG/BMP |
| Accuracy thấp | Dataset ít/xấu | Collect thêm ảnh, data augmentation |
| Colab timeout | Training quá lâu | Split dataset, train partial epochs |

## 📚 Tài Liệu Liên Quan

- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Chart.js](https://www.chartjs.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)
- [Google Colab](https://colab.research.google.com/)

## 👥 Các Tác Giả

Dự án được xây dựng hoàn toàn bằng **tiếng Việt** với chú thích đầy đủ.

## 📄 License

Dự án này là miễn phí, có thể tự do sử dụng và chỉnh sửa.

---

**Phiên bản**: 1.0  
**Cập nhật lần cuối**: 2024  
**Status**: ✅ Ready to use
