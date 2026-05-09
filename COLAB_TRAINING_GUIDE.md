# 🎽 Hướng Dẫn Training ResNet50 trên Google Colab

## 📋 Mục Lục
1. [Chuẩn Bị](#chuẩn-bị)
2. [Hướng Dẫn Chi Tiết](#hướng-dẫn-chi-tiết)
3. [Các Thông Số Cấu Hình](#các-thông-số-cấu-hình)
4. [Giải Quyết Vấn Đề](#giải-quyết-vấn-đề)

---

## 🎯 Chuẩn Bị

### Yêu Cầu
- ✅ Google Account (miễn phí)
- ✅ Google Drive (để save model)
- ✅ RAM trên Colab (12GB miễn phí, 36GB nếu Pro)
- ✅ GPU (Tesla T4 miễn phí, V100/A100 nếu Pro)

### Data
- ✅ Dữ liệu từ: https://github.com/duyaivy/CRAWL_DATA/tree/main/cloudinary_links
- ✅ Format: CSV files (ao_khoac.csv, quan_jean.csv, v.v.)
- ✅ Cột URL: `secure_url`

---

## 📝 Hướng Dẫn Chi Tiết

### Bước 1: Tạo Colab Notebook
1. Mở https://colab.research.google.com/
2. Chọn `File` → `New notebook`
3. Đặt tên: "Clothing_ResNet50_Training"

### Bước 2: Kiểm Tra GPU
Trong **Cell đầu tiên**, chạy:
```python
!nvidia-smi
```
**Kết quả mong muốn**: Sẽ hiện GPU (Tesla T4, V100, A100, v.v.)

### Bước 3: Cài Đặt Dependencies
Chạy ở **Cell thứ 2**:
```python
!pip install torch torchvision pandas tqdm pillow requests matplotlib numpy
```

### Bước 4: Mount Google Drive
Chạy ở **Cell thứ 3**:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Nó sẽ yêu cầu ủy quyền - click link và cho phép.

### Bước 5: Clone Repository (Optional)
Nếu muốn sử dụng script từ repo:
```python
!git clone https://github.com/time-city/clothing_project.git
%cd clothing_project
```

Hoặc copy code trực tiếp vào Cell (xem bên dưới).

### Bước 6: Chạy Training Script
**Cell chính** - Copy toàn bộ code từ `colab_train.py` vào một Cell mới, rồi chạy:

```python
# [PASTE ENTIRE colab_train.py CODE HERE]
```

Hoặc nếu đã clone repo:
```python
exec(open('colab_train.py').read())
```

### Bước 7: Trả Lời Câu Hỏi Cấu Hình
Script sẽ hỏi bạn các thông số:

```
Images per class? (1-6, default=3): 3
Epochs to train? (1-4, default=2): 2
Batch size? (1-3, default=2): 2
Learning rate? (1-3, default=2): 2
Freeze backbone? (1-2, default=1): 1
```

### Bước 8: Đợi Training
- ⏱️ Thời gian phụ thuộc vào số ảnh & epochs
- 📊 Script sẽ hiển thị: Loss, Train Acc, Val Acc
- 💾 Model tốt nhất được lưu tự động

### Bước 9: Download Model
Sau training:
1. Mở Google Drive
2. Tìm file `clothing_model.pth`
3. Download về máy
4. Copy vào thư mục `ai_model/` của project

---

## 🎛️ Các Thông Số Cấu Hình

### 1️⃣ Images Per Class
| Option | Số ảnh | Thời gian | GPU Memory | Chất Lượng |
|--------|--------|-----------|-----------|-----------|
| 1      | 100    | 5-10 min  | 2GB       | ⭐        |
| 2      | 300    | 15-30 min | 4GB       | ⭐⭐      |
| 3      | 500    | 30-60 min | 6GB       | ⭐⭐⭐    |
| 4      | 1000   | 60-90 min | 8GB+      | ⭐⭐⭐⭐  |
| 5      | 1500   | 90-120 min| 10GB+     | ⭐⭐⭐⭐⭐ |
| 6      | 2000   | 120+ min  | 12GB+     | ⭐⭐⭐⭐⭐ |

**Khuyến nghị:**
- Đầu tiên: Chọn **1 (100)** để test
- Chuẩn: Chọn **3 (500)** - balanced
- Production: Chọn **4-6** (1000-2000) - chất lượng cao

### 2️⃣ Epochs (Vòng Lặp Training)
| Option | Epochs | Thời gian | Accuracy |
|--------|--------|-----------|----------|
| 1      | 5      | Nhanh     | ⭐⭐     |
| 2      | 10     | Chuẩn     | ⭐⭐⭐   |
| 3      | 20     | Chậm      | ⭐⭐⭐⭐ |
| 4      | 30     | Rất chậm  | ⭐⭐⭐⭐⭐ |

**Khuyến nghị:**
- Test nhanh: **1 (5 epochs)**
- Sản xuất: **2 hoặc 3** (10-20 epochs)

### 3️⃣ Batch Size
| Option | Size | Memory | Tốc Độ | Chất Lượng |
|--------|------|--------|--------|-----------|
| 1      | 16   | 3GB    | Chậm   | Cao       |
| 2      | 32   | 5GB    | Trung  | Trung     |
| 3      | 64   | 8GB    | Nhanh  | Thấp      |

**Khuyến nghị:**
- Colab miễn phí: **2 (32)** - cân bằng tốt
- GPU mạnh: **3 (64)** - nhanh hơn

### 4️⃣ Learning Rate
| Option | Rate   | Độ Nhạy | Ổn Định |
|--------|--------|---------|---------|
| 1      | 0.0001 | Chậm    | Rất ổn  |
| 2      | 0.001  | Vừa    | Ổn      |
| 3      | 0.01   | Nhanh  | Bất ổn  |

**Khuyến nghị:**
- Transfer Learning: **2 (0.001)**
- Fine-tuning: **1 (0.0001)**

### 5️⃣ Freeze Backbone
| Option | Chế Độ | Tốc Độ | Memory | Accuracy |
|--------|--------|--------|--------|----------|
| 1      | Frozen | Nhanh  | Thấp   | Ổn       |
| 2      | Unfrozen | Chậm | Cao    | Cao      |

**Khuyến nghị:**
- Đầu tiên: **1 (Frozen)** - nhanh & ít memory
- Fine-tuning: **2 (Unfrozen)** - chất lượng tốt hơn

---

## ⚙️ Cấu Hình Được Khuyến Nghị

### 🚀 Quick Test (5-10 phút)
```
Images: 100
Epochs: 5
Batch size: 16
Learning rate: 0.001
Freeze: Yes
```
**Dùng khi**: Muốn test script nhanh

### 🎓 Standard Training (30-45 phút)
```
Images: 300
Epochs: 10
Batch size: 32
Learning rate: 0.001
Freeze: Yes
```
**Dùng khi**: Có thời gian vừa phải, GPU T4

### 🏆 Production Quality (60-90 phút)
```
Images: 500
Epochs: 20
Batch size: 32
Learning rate: 0.0001
Freeze: No
```
**Dùng khi**: Muốn model chất lượng, Colab Pro

### 💎 Maximum Quality (120-180 phút)
```
Images: 1000
Epochs: 30
Batch size: 32
Learning rate: 0.0001
Freeze: No
```
**Dùng khi**: Dùng V100/A100, muốn accuracy cao nhất

### 👑 Ultra Quality (180+ phút)
```
Images: 1500-2000
Epochs: 30-40
Batch size: 64
Learning rate: 0.00005
Freeze: No (hoặc Partially)
```
**Dùng khi**: GPU A100, dự đoán production real-world

---

## 📊 Output của Script

### Folder Structure
```
clothing_data/
├── ao_khoac/
│   ├── ao_khoac_00000.jpg
│   ├── ao_khoac_00001.jpg
│   └── ...
├── quan_jean/
│   └── ...
└── ...
```

### Model Outputs
- **clothing_model.pth** - Trained model (lưu ở Google Drive)
- **training_history.png** - Graph Loss & Accuracy

### Console Output Ví Dụ
```
============================================================
🎽  CLOTHING CLASSIFIER - RESNET50 TRAINING
Google Colab Edition
============================================================

🖥️  Device: cuda
   GPU: Tesla T4

============================================================
  ⚙️  CONFIGURATION
============================================================

📊 How many images per class?
  1. 100 (Quick test)
  2. 300 (Medium)
  3. 500 (Large)
  4. 1000 (Full training)
  5. 1500 (Extra large)
  6. 2000 (Maximum)
Select (1-6, default=3): 4

⏱️  How many epochs to train?
  1. 5 (Quick)
  2. 10 (Standard)
  3. 20 (Better)
  4. 30 (Best)
Select (1-4, default=2): 3

🔋 Batch size?
  1. 16 (Low memory)
  2. 32 (Standard)
  3. 64 (High memory)
Select (1-3, default=2): 2

🎯 Learning rate?
  1. 0.0001 (Slow, stable)
  2. 0.001 (Standard)
  3. 0.01 (Fast, unstable)
Select (1-3, default=2): 2

❄️  Freeze backbone (Transfer Learning)?
  1. Yes (Faster, less memory)
  2. No (Slower, better accuracy)
Select (1-2, default=1): 2

============================================================
📋 YOUR CONFIGURATION
============================================================
  Images per class: 1000
  Epochs: 20
  Batch size: 32
  Learning rate: 0.001
  Freeze backbone: False
============================================================

============================================================
STEP 1: LOADING DATA
============================================================
📥 LOAD DATA FROM GITHUB
Loading ao_khoac... ✓ Loaded 1200 images
Loading ao_so_mi... ✓ Loaded 950 images
Loading ao_thun... ✓ Loaded 2100 images
Loading quan_jean... ✓ Loaded 1800 images
Loading quan_short... ✓ Loaded 880 images
Loading quan_tay... ✓ Loaded 1050 images
Loading sweater_hoodie... ✓ Loaded 650 images
Loading vay... ✓ Loaded 1200 images

============================================================
STEP 2: DOWNLOADING IMAGES
============================================================
⬇️  DOWNLOADING IMAGES
  ao_khoac            : 1000 success, 15 failed
  ao_so_mi            :  950 success,  8 failed
  ao_thun             : 1000 success, 12 failed
  quan_jean           : 1000 success, 20 failed
  quan_short          :  880 success, 10 failed
  quan_tay            : 1000 success,  5 failed
  sweater_hoodie      :  650 success,  8 failed
  vay                 : 1000 success, 14 failed

✅ Total downloaded: 7480 images

============================================================
STEP 3: PREPARING DATASET
============================================================
📁 Dataset loaded: 7480 images
📂 Classes: ['ao_khoac', 'ao_so_mi', 'ao_thun', 'quan_jean', 'quan_short', 'quan_tay', 'sweater_hoodie', 'vay']
  Train: 5984 images
  Val: 1496 images

============================================================
STEP 4: CREATING MODEL
============================================================
✓ ResNet50 model created (8 classes)
  Total parameters: 23,512,392
  Trainable parameters: 23,512,392

============================================================
STEP 5: TRAINING MODEL
============================================================
Epoch 1/20 [TRAIN]: 100%|████████| 187/187 [02:15<00:00,  1.38it/s]
Loss: 2.0934 | Train Acc: 35.12% | Val Acc: 42.34%
  ✅ Model saved! (Best Acc: 42.34%)

Epoch 2/20 [TRAIN]: 100%|████████| 187/187 [02:14<00:00,  1.39it/s]
Loss: 1.6234 | Train Acc: 52.45% | Val Acc: 58.90%
  ✅ Model saved! (Best Acc: 58.90%)

Epoch 3/20 [TRAIN]: 100%|████████| 187/187 [02:13<00:00,  1.41it/s]
Loss: 1.2456 | Train Acc: 65.78% | Val Acc: 68.45%
  ✅ Model saved! (Best Acc: 68.45%)

[... epochs 4-19 ...]

Epoch 20/20 [TRAIN]: 100%|████████| 187/187 [02:15<00:00,  1.37it/s]
Loss: 0.4321 | Train Acc: 88.92% | Val Acc: 84.56%
  ✅ Model saved! (Best Acc: 84.56%)

============================================================
STEP 6: SAVING RESULTS
============================================================
✅ Model saved to: /content/drive/My Drive/clothing_model.pth
   Download it from Google Drive!
📊 Training history saved to: /content/drive/My Drive/training_history.png

============================================================
✨ TRAINING COMPLETED!
============================================================
```

---

## 🔧 Giải Quyết Vấn Đề

### ❌ Error: "Out of Memory" (OOM)
**Nguyên nhân**: GPU memory không đủ

**Giải pháp:**
1. Giảm batch_size: `32` → `16`
2. Giảm images_per_class: `1000` → `500`
3. Giảm epochs: `20` → `10`
4. Restart runtime: `Runtime` → `Restart runtime`
5. Nếu vẫn lỗi, chuyển sang Colab Pro (V100/A100)

**Code kiểm tra:**
```python
import torch
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"GPU Name: {torch.cuda.get_device_name()}")
```

### ❌ Error: "Network Timeout" khi download ảnh
**Nguyên nhân**: Cloudinary server bận hoặc mạng yếu

**Giải pháp:**
1. Chạy lại script - nó sẽ retry những ảnh failed
2. Tăng timeout trong code: `timeout=10` → `timeout=30`
3. Giảm images_per_class tạm thời
4. Kiểm tra internet: `!ping -c 1 google.com`

### ❌ Error: "Failed to mount Google Drive"
**Nguyên nhân**: Permission hoặc auth token expired

**Giải pháp:**
```python
# Chạy lại cell này
from google.colab import drive
drive.mount('/content/drive')
```
Nó sẽ yêu cầu ủy quyền - click link và copy auth code

### ❌ Error: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
**Nguyên nhân**: Số class không match

**Giải pháp:**
- Đảm bảo dữ liệu đã được load đúng
- Kiểm tra: `print(len(dataset.classes))`

### ⚠️ Model Accuracy Quá Thấp (< 60%)
**Nguyên nhân**: Model chưa training đủ

**Giải pháp:**
1. Tăng epochs: `10` → `20-30`
2. Tăng images_per_class: `300` → `500-1000`
3. Giảm learning rate: `0.001` → `0.0001`
4. Unfreeze backbone: chọn option `2`
5. Tăng data augmentation

**Kiểm tra tiến độ:**
- Training acc tăng không → model quá khó
- Training acc tăng, val acc không → overfitting

### ⚠️ Training Quá Chậm (> 10 min/epoch)
**Nguyên nhân**: Batch size quá nhỏ, num_workers quá cao

**Giải pháp:**
1. Tăng batch_size: `16` → `32` hoặc `64`
2. Giảm epochs: `30` → `20`
3. Giảm images_per_class: `1000` → `500`
4. Kiểm tra GPU: `!nvidia-smi`

### ⚠️ GPU Not Detected
**Nguyên nhân**: Runtime không bật GPU

**Giải pháp:**
1. Bật GPU: `Runtime` → `Change runtime type` → `GPU` → Save
2. Kiểm tra: Chạy `!nvidia-smi`
3. Nếu vẫn không có, restart Colab

### ⚠️ Model Accuracy Cao nhưng Slow
**Nguyên nhân**: Learning rate quá thấp

**Giải pháp:**
1. Tăng learning rate: `0.0001` → `0.001`
2. Hoặc tăng batch_size: `16` → `32`

### ⚠️ Validation Loss Tăng (Overfitting)
**Nguyên nhân**: Model memorize training data

**Giải pháp:**
1. Tăng data augmentation (xem Tips & Tricks)
2. Giảm model complexity (freeze backbone)
3. Tăng dropout (nếu modify model)
4. Sử dụng early stopping

---

## 💡 Tips & Tricks

### 1. Optimize Learning - Learning Rate Scheduler
Để learning rate giảm dần theo epochs:
```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Thêm vào cuối mỗi epoch trong train_model:
scheduler.step()
```

### 2. Early Stopping
Dừng training nếu validation accuracy không cải thiện:
```python
patience = 5
best_acc = 0
patience_counter = 0

if val_acc > best_acc:
    best_acc = val_acc
    patience_counter = 0
    torch.save(model.state_dict(), save_path)
else:
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### 3. Save Checkpoints Mỗi Epoch
```python
# Thêm vào cuối mỗi epoch
checkpoint_path = f'/content/drive/My Drive/checkpoint_epoch_{epoch+1}.pth'
torch.save(model.state_dict(), checkpoint_path)
```

### 4. Advanced Data Augmentation
```python
train_transform = transforms.Compose([
    transforms.RandomRotation(30),  # Xoay 0-30 độ
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Dịch
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Zoom
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.GaussianBlur(kernel_size=3),  # Blur
    transforms.RandomPerspective(distortion_scale=0.2),  # Perspective
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 5. Visualize Predictions
```python
def visualize_predictions(model, test_loader, num_images=9):
    model.eval()
    images, labels = next(iter(test_loader))
    outputs = model(images[:num_images])
    _, predicted = torch.max(outputs, 1)
    
    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().transpose(1, 2, 0))
        plt.title(f"True: {labels[i]}, Pred: {predicted[i]}")
        plt.axis('off')
    plt.show()
```

### 6. Save Model Summary
```python
from torchsummary import summary

# Thêm sau khi create model
print(summary(model, input_size=(3, 224, 224)))

# Hoặc đơn giản:
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")
```

### 7. Monitor GPU Usage
```python
# Chạy trong cell riêng
import GPUtil
import time

while True:
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU {gpu.id}: {gpu.load*100:.1f}% ({gpu.memoryUsed}/{gpu.memoryTotal} MB)")
    time.sleep(5)
```

### 8. Test Model sau Training
```python
# Đánh giá trên test set
model.eval()
test_loss = 0
test_acc = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_acc += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_acc = 100 * test_acc / len(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
```

---

## 📈 Performance Benchmarks

### Training Time by Configuration (Colab Free T4)
| Images | Epochs | Batch | Time/Epoch | Total Time |
|--------|--------|-------|-----------|-----------|
| 100    | 5      | 16    | ~15 sec   | ~2 min    |
| 300    | 10     | 32    | ~45 sec   | ~8 min    |
| 500    | 20     | 32    | ~90 sec   | ~30 min   |
| 1000   | 20     | 32    | ~180 sec  | ~60 min   |
| 1500   | 30     | 32    | ~270 sec  | ~135 min  |
| 2000   | 30     | 32    | ~360 sec  | ~180 min  |

### Estimated Accuracy by Config
| Images | Freeze | Epochs | Accuracy |
|--------|--------|--------|----------|
| 100    | Yes    | 5      | ~65%     |
| 300    | Yes    | 10     | ~72%     |
| 500    | No     | 20     | ~78%     |
| 1000   | No     | 20     | ~82%     |
| 1500   | No     | 30     | ~85%     |
| 2000   | No     | 30     | ~87%+    |

---

## 🚀 Advanced: Custom Training Loop

Nếu muốn modify training process:
```python
# Tạo file train_custom.py
def custom_train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(train_loader)
```

---

## ⚡ Quick Start (30 giây)

1. **Mở Colab**: https://colab.research.google.com/
2. **Cell 1** - Cài dependencies:
```python
!pip install torch torchvision pandas tqdm pillow requests matplotlib numpy
```
3. **Cell 2** - Mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
4. **Cell 3** - Copy toàn bộ `colab_train.py` vào và chạy
5. **Trả lời 5 câu hỏi** cấu hình
6. **Đợi training** xong (từ 5 min đến 3 hours tùy config)
7. **Download model** từ Google Drive

✅ **Xong!** Model ready to use.

---

## 📞 Support

Nếu có vấn đề:
1. Kiểm tra Console Output (tìm error message)
2. Xem phần "Giải Quyết Vấn Đề" trên
3. Restart Runtime và chạy lại
4. Liên hệ với team development

---

## 📚 Resources

- [PyTorch Docs](https://pytorch.org/docs/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Google Colab Guide](https://colab.research.google.com/notebooks/welcome.ipynb)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

**Happy Training! 🎉**
