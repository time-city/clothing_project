# 🧠 Hướng Dẫn AI Model - Phân Loại Quần Áo

## 📋 Giới Thiệu

Phần AI của dự án sử dụng **ResNet50** (Transfer Learning) để phân loại 8 loại quần áo:
1. **Áo thun** (T-shirt)
2. **Áo sơ mi** (Shirt)
3. **Áo khoác** (Coat/Jacket)
4. **Quần tây** (Trousers)
5. **Quần jean** (Jeans)
6. **Áo len** (Sweater)
7. **Váy** (Dress)
8. **Quần short** (Shorts)

## 🛠️ Cài Đặt Môi Trường

### 1. Cài đặt Python 3.8+
Đảm bảo bạn có Python 3.8 hoặc cao hơn:
```bash
python --version
```

### 2. Tạo Virtual Environment (Khuyên nghị)
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (macOS/Linux)
source venv/bin/activate

# Kích hoạt (Windows)
venv\Scripts\activate
```

### 3. Cài đặt Thư Viện
```bash
pip install -r requirements_ai.txt
```

**Các thư viện chính:**
- `torch`: PyTorch framework
- `torchvision`: Pre-trained models
- `Pillow`: Xử lý ảnh
- `matplotlib`: Vẽ biểu đồ
- `numpy`: Tính toán số học

### 4. GPU Support (Tùy chọn)
Nếu bạn có GPU NVIDIA, cài đặt CUDA version của PyTorch:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hoặc CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Kiểm tra GPU:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
```

## 📁 Chuẩn Bị Dataset

### Cấu Trúc Thư Mục

Tạo thư mục `data` với cấu trúc như sau:
```
data/
├── áo thun/
│   ├── áo_thun_1.jpg
│   ├── áo_thun_2.jpg
│   └── ...
├── áo sơ mi/
│   ├── áo_sơ_mi_1.jpg
│   ├── áo_sơ_mi_2.jpg
│   └── ...
├── áo khoác/
├── quần tây/
├── quần jean/
├── áo len/
├── váy/
└── quần short/
```

### Yêu Cầu Ảnh
- **Định dạng**: JPG, PNG, BMP, GIF
- **Kích thước**: Bất kỳ (sẽ được chuẩn hóa thành 224x224)
- **Số lượng**: Tối thiểu 50-100 ảnh cho mỗi class (càng nhiều càng tốt)
- **Chất lượng**: Ảnh rõ, không bị mờ

### Thu Thập Dataset

**Các nguồn khuyên nghị:**
1. **Bing Image Search** / **Google Image Search**: Tải hàng loạt ảnh
2. **Fashion Datasets**: ImageNet, Fashion-MNIST
3. **Kaggle**: Các fashion datasets công khai

**Download từ Kaggle:**
```bash
# Cần cài Kaggle API
pip install kaggle

# Configure credentials (đặt kaggle.json vào ~/.kaggle/)
# Download dataset
kaggle datasets download -d [dataset-id]
```

## 🚀 Huấn Luyện Mô Hình

### Chạy Training

#### Cách 1: Mặc định
```bash
python train.py
```
- Dataset folder: `./data`
- Epochs: 20
- Batch size: 32
- Learning rate: 0.001
- Output: `./checkpoints`

#### Cách 2: Tùy chỉnh Parameters
```bash
python train.py \
    --dataset ./data \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0001 \
    --output ./checkpoints
```

#### Cách 3: Sử dụng Python Script
```python
from train import train

model, history = train(
    dataset_dir='./data',
    num_epochs=30,
    batch_size=32,
    learning_rate=0.001,
    output_dir='./checkpoints',
    freeze_backbone=True
)
```

### Output của Training

Sau khi training hoàn thành, bạn sẽ nhận được:

```
checkpoints/
├── best_model.pth          # Model tốt nhất (validation accuracy cao nhất)
├── final_model.pth         # Model ở epoch cuối cùng
├── class_mapping.json      # Ánh xạ class name <-> index
├── training_history.json   # Lịch sử Loss/Accuracy
└── training_history.png    # Biểu đồ Loss & Accuracy
```

## 📊 Hiểu Biểu Đồ Training

File `training_history.png` hiển thị:

**Trái - Training & Validation Loss:**
- Nên giảm dần theo thời gian
- Nếu validation loss tăng lên: overfitting
- Cách khắc phục: Tăng regularization, dropout, data augmentation

**Phải - Training & Validation Accuracy:**
- Nên tăng dần theo thời gian
- Accuracy cao hơn 85% là tốt
- Nếu val accuracy không tăng: learning rate quá thấp

## 🔧 Transfer Learning Giải Thích

ResNet50 pre-trained trên ImageNet có thể nhận diện rất nhiều features của ảnh. Chúng tôi:

1. **Freeze backbone**: Giữ nguyên các layer conv (đã huấn luyện từ ImageNet)
2. **Replace FC layer**: Thay đổi layer fully connected cuối cùng cho 8 classes
3. **Fine-tune FC layer**: Chỉ huấn luyện lại layer FC mới

**Lợi ích:**
- ✓ Cần ít dữ liệu hơn (1000+ ảnh đủ)
- ✓ Huấn luyện nhanh hơn (~30 phút)
- ✓ Accuracy cao hơn (85-95%)

## 💡 Tips Cải Thiện Mô Hình

### 1. Tăng Dữ Liệu
- Collect thêm ảnh, đặc biệt cho class nào có ít ảnh
- Mục tiêu: 500+ ảnh cho mỗi class

### 2. Data Augmentation
Dataset.py đã có augmentation:
- Random rotation, flip, color jitter
- Có thể điều chỉnh trong `get_transforms()`

### 3. Hyperparameters
```python
# Thử các learning rates khác nhau
lr = [0.001, 0.0001, 0.00001]

# Thử batch sizes khác
batch_size = [16, 32, 64]

# Thử unfreeze backbone (fine-tuning)
# Bỏ freeze_backbone=False trong train()
```

### 4. Class Imbalance
Nếu một class có ít ảnh hơn:
- Dùng `class_weight` trong loss
- Oversample class ít
- Undersample class nhiều

## 🧪 Test Model (Inference)

Tạo file `test_inference.py`:
```python
import torch
from torchvision import transforms
from PIL import Image
from train import ClothingClassifier
import json

# Load model
model = ClothingClassifier(num_classes=8)
model.load_state_dict(torch.load('./checkpoints/best_model.pth'))
model.eval()

# Load class mapping
with open('./checkpoints/class_mapping.json', 'r', encoding='utf-8') as f:
    class_mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Inference trên 1 ảnh
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
    
    predicted_class = idx_to_class[predicted_idx]
    return predicted_class, confidence, probabilities[0].tolist()

# Test
class_name, confidence, all_probs = predict('./test_image.jpg')
print(f"Dự đoán: {class_name}")
print(f"Độ tự tin: {confidence*100:.2f}%")
```

Chạy:
```bash
python test_inference.py
```

## 📈 Benchmark (Hiệu Năng Kỳ Vọng)

Với dataset 500+ ảnh cho mỗi class:
- **Accuracy**: 88-94%
- **Training time**: 30-60 phút (CPU) / 5-15 phút (GPU)
- **Inference time**: 50-100ms per image

## ❓ Troubleshooting

### Lỗi: "CUDA out of memory"
```python
# Giảm batch size
batch_size = 8  # thay vì 32

# Hoặc force sử dụng CPU
device = torch.device('cpu')
```

### Lỗi: "No images found"
```
❌ Kiểm tra cấu trúc thư mục data/
✓ Mỗi thư mục con phải chứa ảnh
✓ Tên file phải có extension (.jpg, .png, ...)
```

### Model không hội tụ (accuracy không tăng)
```python
# Tăng learning rate
learning_rate = 0.01

# Hoặc unfreeze backbone
model = ClothingClassifier(freeze_backbone=False)

# Hoặc tăng data
# + Collect thêm ảnh
# + Tăng epochs
```

## 📚 Tài Liệu Thêm

- PyTorch: https://pytorch.org/
- ResNet: https://arxiv.org/abs/1512.03385
- Transfer Learning: https://cs231n.github.io/transfer-learning/

---

**Viết bởi**: AI Model Team  
**Phiên bản**: 1.0  
**Ngày cập nhật**: 2024
