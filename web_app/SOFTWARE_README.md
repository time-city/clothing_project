# 🌐 Hướng Dẫn Web App - Flask + HTML/CSS/JS

## 📋 Giới Thiệu

Phần Web App của dự án là một **Flask Web Application** đơn giản, nhẹ nhàng cho phép người dùng:
- ✅ Upload ảnh quần áo qua giao diện web sạch đẹp
- ✅ Dự đoán loại quần áo sử dụng ResNet50
- ✅ Xem kết quả dự đoán và xác suất theo từng loại
- ✅ Hiển thị biểu đồ trực quan (Chart.js)

**Không cần Streamlit hay Cloudinary** - giao diện HTML/CSS/JS đơn giản, dễ deploy!

## 🛠️ Cài Đặt Môi Trường

### 1. Điều Kiện Tiên Quyết

```bash
# Đảm bảo Python 3.8+ đã cài
python --version

# Cài pip (nếu chưa có)
python -m pip install --upgrade pip
```

### 2. Tạo Virtual Environment (Khuyên nghị)

```bash
# Tạo
python -m venv venv

# Kích hoạt (macOS/Linux)
source venv/bin/activate

# Kích hoạt (Windows)
venv\Scripts\activate
```

### 3. Cài Đặt Thư Viện

```bash
cd web_app
pip install -r requirements_web.txt
```

**Các thư viện chính:**
- `Flask`: Web framework
- `torch`, `torchvision`: PyTorch models
- `Pillow`: Xử lý ảnh
- `cloudinary`: Upload ảnh lên cloud

## 🔐 Cấu Hình Cloudinary (Optional)

### Bước 1: Tạo Tài Khoản Cloudinary

1. Truy cập: https://cloudinary.com/
2. Click **Sign Up** (đăng ký miễn phí)
3. Hoàn thành xác minh email

### Bước 2: Lấy Credentials

1. Sau khi đăng nhập, vào **Dashboard**
2. Tìm phần **Account Details** ở trên cùng
3. Bạn sẽ thấy:
   - **Cloud Name**: `xxxxxx`
   - **API Key**: `123456789`
   - **API Secret**: `abcdefghijklmnop`

### Bước 3: Cấu Hình Credentials (Chọn 1 trong 3 cách)

#### **Cách 1: Environment Variables (Khuyên nghị)**

```bash
# macOS/Linux
export CLOUDINARY_CLOUD_NAME="your_cloud_name"
export CLOUDINARY_API_KEY="your_api_key"
export CLOUDINARY_API_SECRET="your_api_secret"

# Sau đó chạy app
python app.py
```

```bash
# Windows (Command Prompt)
set CLOUDINARY_CLOUD_NAME=your_cloud_name
set CLOUDINARY_API_KEY=your_api_key
set CLOUDINARY_API_SECRET=your_api_secret

# Sau đó chạy app
python app.py
```

#### **Cách 2: File .env**

Tạo file `web_app/.env`:
```
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

Cài `python-dotenv`:
```bash
pip install python-dotenv
```

App sẽ tự động load từ `.env`

#### **Cách 3: Sửa config file**

Xem file `config/.env.example` để hướng dẫn chi tiết

### Bước 4: Test Cloudinary

```bash
# Chạy app
python app.py

# Upload ảnh thông qua web UI
# Ảnh sẽ được upload lên Cloudinary
# Check Cloudinary Dashboard > Media Library để xem
```

---

## 📦 Chuẩn Bị Model


Trước khi chạy web app, bạn cần có model đã huấn luyện:

```bash
# 1. Trở lại thư mục ai_model
cd ../ai_model

# 2. Chuẩn bị dataset (xem AI_README.md)

# 3. Huấn luyện model
python train.py --dataset ./data --epochs 20

# 4. Kiểm tra output
ls -la
# Phải có file: best_model.pth
```

## 🚀 Chạy Web App

### Cách 1: Chạy Bình Thường

```bash
cd web_app
python app.py
```

Flask sẽ tự động chạy tại: **http://localhost:5000**

### Cách 2: Chạy Với Port Custom

```bash
python app.py
# Sau đó bạn có thể sửa trong app.py:
# app.run(port=8000)
```

### Cách 3: Chạy Với Gunicorn (Production)

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

### Cách 4: Chạy Trên Google Colab

```python
# Trong Colab cell:
!cd /content/drive/MyDrive/clothing_project/web_app && pip install -r requirements_web.txt
!cd /content/drive/MyDrive/clothing_project/web_app && python app.py
```

## 📱 Sử Dụng Web App

### Giao Diện

**Header (Đầu trang):**
- 👗 Tiêu đề: "Phân Loại Quần Áo"
- Mô tả: "Tải ảnh quần áo để AI phân loại..."

**Upload Section:**
- Kéo & thả ảnh hoặc click để chọn file
- Hỗ trợ JPG, PNG, BMP (tối đa 16MB)

**Main Content:**
- 🖼️ Hiển thị ảnh preview
- 🔍 Button "Dự Đoán"
- 📊 Kết quả và biểu đồ

### Quy Trình Sử Dụng

1. **Upload ảnh**: Kéo ảnh vào hoặc click chọn file
2. **Xem preview**: Ảnh sẽ hiển thị với thông tin file
3. **Dự đoán**: Click nút "🔍 Dự Đoán"
4. **Xem kết quả**: Kết quả và biểu đồ sẽ hiển thị tức thì

## 📊 Hiểu Kết Quả

### Kết Quả Chính
- **Tên loại quần áo**: Dự đoán chính
- **Độ tự tin**: Xác suất dự đoán (0-100%)

### Xác Suất Theo Loại
- Thanh tiến trình cho mỗi loại quần áo
- Hiển thị % xác suất

### Biểu Đồ
- Bar chart toàn bộ 8 loại quần áo
- Xác suất được sắp xếp từ cao đến thấp

## 🎨 Thiết Kế Giao Diện

Giao diện được thiết kế:
- ✅ **Đơn giản**: Không quá phức tạp
- ✅ **Sạch sẽ**: White/light gray background
- ✅ **Dễ dùng**: Các button rõ ràng
- ✅ **Không gradient**: Chỉ dùng màu solid
- ✅ **Responsive**: Hoạt động tốt trên mobile

## 🔧 Tùy Chỉnh Giao Diện

### Thay Đổi Màu Sắc
Chỉnh sửa trong `static/css/style.css`:

```css
/* Thay đổi màu primary */
.btn-primary {
    background-color: #3498db;  /* 👈 THAY ĐỔI TẠI ĐÂY */
}
```

### Thêm Class Mới
Nếu muốn support thêm loại quần áo:
1. Sửa `CLASS_NAMES` trong `app.py`
2. Sửa `num_classes` khi train model
3. Huấn luyện lại model

### Tùy Chỉnh Font Chữ
Trong `static/css/style.css`:

```css
body {
    font-family: 'Segoe UI', Roboto, ... ;  /* 👈 THAY ĐỔI TẠI ĐÂY */
}
```

## 💡 Tips & Tricks

### 1. Ảnh Upload Tốt Nhất
- ✅ Ảnh có nền sạch
- ✅ Quần áo chiếm >50% ảnh
- ✅ Ánh sáng tốt
- ✅ Độ phân giải >= 224x224

### 2. Tăng Tốc Độ Dự Đoán
- Sử dụng GPU (nếu có)
- Giảm kích thước ảnh
- Dùng quantized model

### 3. Xử Lý Lỗi

**Ảnh không upload:**
1. Kiểm tra kích thước file
2. Kiểm tra format (JPG, PNG, BMP)

**Dự đoán chậm:**
1. Kiểm tra kết nối internet
2. Kiểm tra server Flask
3. Thử restart app

## 📤 Deploy Lên Server

### Deploy Lên Heroku

```bash
# Tạo requirements.txt
echo "Flask==2.3.2
torch==2.0.1
torchvision==0.15.2
gunicorn==20.1.0" > requirements.txt

# Tạo Procfile
echo "web: gunicorn -w 2 -b 0.0.0.0:\$PORT web_app.app:app" > Procfile

# Deploy
git push heroku main
```

### Deploy Lên AWS/GCP/Azure

```bash
# Sử dụng gunicorn cho production
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Deploy Trên Google Colab

```python
# Trong Colab, cài dependencies
!pip install flask torch torchvision pillow

# Clone project hoặc upload files
!cd /content && git clone https://github.com/yourname/clothing_project.git

# Chạy Flask với ngrok để access từ browser
!pip install flask-ngrok
# Sau đó import flask_ngrok trong app.py
```

## ❓ Troubleshooting

### Lỗi 1: "Cannot import module 'flask'"
```
Cách khắc phục:
1. Kiểm tra virtual environment
2. Chạy: pip install -r requirements_web.txt
3. Kiểm tra: python -c "import flask; print(flask.__version__)"
```

### Lỗi 2: "Model not found at ..."
```
Cách khắc phục:
1. Kiểm tra file best_model.pth tồn tại trong ai_model/
2. Chạy training.py trước: cd ../ai_model && python train.py
3. Sao chép best_model.pth đúng vị trí
```

### Lỗi 3: "Address already in use"
```
Cách khắc phục:
1. Tìm process đang dùng port 5000: lsof -i :5000
2. Kill process: kill -9 <PID>
3. Hoặc chạy trên port khác: app.run(port=8000)
```

### Lỗi 4: "Ảnh upload thất bại"
```
Cách khắc phục:
1. Kiểm tra kích thước file (tối đa 16MB)
2. Kiểm tra format file
3. Kiểm tra thư mục uploads tồn tại
```

## 📚 Tài Liệu Liên Quan

- **Flask**: https://flask.palletsprojects.com/
- **PyTorch**: https://pytorch.org/docs/stable/index.html
- **Chart.js**: https://www.chartjs.org/docs/latest/

## 📧 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra đầu ra lỗi trong terminal
2. Xem Troubleshooting phía trên
3. Kiểm tra file requirements có đầy đủ không

---

**Viết bởi**: Web App Team  
**Phiên bản**: 2.0 (Flask)  
**Ngày cập nhật**: 2024
