# 🎯 Quick Start Guide - Hướng Dẫn Nhanh

## ⚡ 5 Bước Khởi Động Nhanh

### 1️⃣ Clone/Download Dự Án
```bash
cd /path/to/clothing_project
```

### 2️⃣ Chuẩn Bị Dataset (Tùy Chọn)

#### Cách A: Dùng ảnh có sẵn
Skip step này, dùng file test

#### Cách B: Tự tải ảnh
```bash
# Tạo cấu trúc
mkdir -p data/{áo\ thun,áo\ sơ\ mi,áo\ khoác,quần\ tây,quần\ jean,áo\ len,váy,quần\ short}

# Download ảnh từ:
# - Google Image Search
# - Bing Image Search
# - Kaggle datasets

# Đặt vào từng folder (500+ ảnh cho mỗi class)
```

### 3️⃣ Huấn Luyện Model

```bash
cd ai_model

# Cài thư viện
pip install -r requirements_ai.txt

# Chạy training
python train.py --dataset ../data --epochs 20

# ⏳ Chờ ~30-60 phút (hoặc 5-15 phút nếu có GPU)
# Output: checkpoints/best_model.pth
```

### 4️⃣ Cấu Hình Cloudinary

```bash
cd ../web_app

# 1. Đăng ký: https://cloudinary.com/
# 2. Lấy Cloud Name, API Key, API Secret
# 3. Tạo file .streamlit/secrets.toml:

cat > .streamlit/secrets.toml << EOF
CLOUDINARY_CLOUD_NAME = "your_cloud_name"
CLOUDINARY_API_KEY = "your_api_key"
CLOUDINARY_API_SECRET = "your_api_secret"
EOF

# 4. Lưu file!
```

### 5️⃣ Chạy Web App

```bash
# Chạy từ thư mục gốc của project
bash run.sh

# 🌐 Truy cập: http://localhost:8000
```

## ✅ Kiểm Tra Lại

Nếu thành công:
- ✅ Terminal hiển thị: "Starting Flask server..."
- ✅ Browser mở tại http://localhost:8000
- ✅ Giao diện hiển thị title "👗 Clothing Classification Dashboard"

## 🆘 Gặp Vấn Đề?

| Vấn đề | Giải pháp |
|------|----------|
| **Cài đặt thư viện lâu** | Dùng `pip cache purge && pip install -r requirements.txt` |
| **GPU không nhận diện** | Chạy `python -c "import torch; print(torch.cuda.is_available())"` |
| **Cloudinary error** | Kiểm tra credentials, đảm bảo copy đúng vào secrets.toml |
| **Port 8000 đã dùng** | Dừng tiến trình đang chiếm port hoặc sửa `app.py`/`run.sh` để dùng port khác |

## 📖 Xem Thêm

- 🧠 Chi tiết về AI: `ai_model/AI_README.md`
- 🌐 Chi tiết về Web: `web_app/SOFTWARE_README.md`
- 📋 Tổng quan: `README.md`

---

**Thời gian ước tính:**
- Setup: 5 phút
- Training: 30-60 phút
- **Tổng cộng: < 2 giờ!** ⚡
