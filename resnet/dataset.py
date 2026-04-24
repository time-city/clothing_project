"""
dataset.py
Module xử lý dữ liệu cho mô hình phân loại quần áo.

Chức năng:
- Load ảnh từ các folder class khác nhau
- Chuẩn hóa kích thước ảnh thành 224x224
- Chia dữ liệu thành train/val sets
- Tạo DataLoader cho PyTorch
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


class ClothingDataset(Dataset):
    """
    Custom Dataset class cho dữ liệu phân loại quần áo.
    
    Attributes:
        img_dir (str): Đường dẫn folder chứa dữ liệu
        img_list (list): Danh sách đường dẫn ảnh
        labels (list): Danh sách nhãn tương ứng
        class_to_idx (dict): Ánh xạ tên class -> index
        transform (transforms.Compose): Các transform áp dụng lên ảnh
    """
    
    def __init__(self, img_dir, transform=None):
        """
        Khởi tạo Dataset.
        
        Args:
            img_dir (str): Đường dẫn folder dữ liệu (cấu trúc: img_dir/class_name/image.jpg)
            transform (transforms.Compose): Các augmentation transforms
        """
        self.img_dir = img_dir
        self.img_list = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Load tất cả ảnh từ các thư mục con (mỗi thư mục = 1 class)
        class_idx = 0
        for class_name in sorted(os.listdir(img_dir)):
            class_path = os.path.join(img_dir, class_name)
            
            # Bỏ qua nếu không phải folder
            if not os.path.isdir(class_path):
                continue
            
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            
            # Scan tất cả file ảnh trong thư mục
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    img_path = os.path.join(class_path, img_name)
                    self.img_list.append(img_path)
                    self.labels.append(class_idx)
            
            class_idx += 1
        
        print(f"✓ Đã load {len(self.img_list)} ảnh từ {len(self.class_to_idx)} class")
        print(f"  Classes: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        """Trả về số lượng ảnh trong dataset."""
        return len(self.img_list)
    
    def __getitem__(self, idx):
        """
        Lấy 1 mẫu từ dataset.
        
        Args:
            idx (int): Index của ảnh
            
        Returns:
            tuple: (ảnh đã transform, nhãn)
        """
        img_path = self.img_list[idx]
        label = self.labels[idx]
        
        # Load ảnh
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ Lỗi load ảnh {img_path}: {e}")
            # Trả về ảnh đen nếu lỗi
            image = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """
    Trả về các transform cho training và validation.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Transform cho training: có augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),           # Xoay ngẫu nhiên 15 độ
        transforms.ColorJitter(0.2, 0.2, 0.2),   # Thay đổi brightness, contrast, saturation
        transforms.RandomHorizontalFlip(),        # Lật ngang ngẫu nhiên
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Dịch chuyển ngẫu nhiên
        transforms.Resize((256, 256)),            # Resize thành 256x256 trước
        transforms.CenterCrop((224, 224)),        # Crop tâm thành 224x224
        transforms.ToTensor(),                    # Chuyển thành tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],         # ImageNet mean
            std=[0.229, 0.224, 0.225]           # ImageNet std
        )
    ])
    
    # Transform cho validation: không augmentation
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def create_data_loaders(dataset_dir, batch_size=32, num_workers=2, test_split=0.2):
    """
    Tạo DataLoaders cho training và validation.
    
    Args:
        dataset_dir (str): Đường dẫn folder dataset
        batch_size (int): Kích thước batch
        num_workers (int): Số worker cho data loading
        test_split (float): Tỷ lệ split train/val (mặc định 80/20)
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_transform, val_transform = get_transforms()
    
    # Load dataset
    dataset = ClothingDataset(dataset_dir, transform=train_transform)
    
    # Chia train/val
    dataset_size = len(dataset)
    train_size = int(dataset_size * (1 - test_split))
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    # Thay đổi transform cho val_dataset
    val_dataset.dataset.transform = val_transform
    
    # Tạo DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ Đã tạo DataLoaders:")
    print(f"  - Training samples: {train_size}")
    print(f"  - Validation samples: {val_size}")
    print(f"  - Batch size: {batch_size}")
    
    return train_loader, val_loader, dataset.class_to_idx, dataset.idx_to_class


if __name__ == "__main__":
    """
    Test code để kiểm tra dataset loading.
    
    Cách chạy:
        python dataset.py
    """
    # Đường dẫn test (thay đổi tùy theo cấu trúc của bạn)
    dataset_dir = "./data"  # Folder chứa các subfolder: áo thun, áo sơ mi, ...
    
    if os.path.exists(dataset_dir):
        train_loader, val_loader, class_to_idx, idx_to_class = create_data_loaders(
            dataset_dir,
            batch_size=16
        )
        
        # In thông tin
        print(f"\n✓ Class mapping: {class_to_idx}")
        
        # Kiểm tra 1 batch
        print("\nKiểm tra 1 batch từ training set:")
        for images, labels in train_loader:
            print(f"  - Shape ảnh: {images.shape}")
            print(f"  - Shape labels: {labels.shape}")
            print(f"  - Sample labels: {labels[:5]}")
            break
    else:
        print(f"❌ Dataset folder '{dataset_dir}' không tồn tại.")
        print("Hãy tạo cấu trúc thư mục:")
        print("  ./data/")
        print("    ├── áo thun/")
        print("    ├── áo sơ mi/")
        print("    ├── áo khoác/")
        print("    ├── quần tây/")
        print("    ├── quần jean/")
        print("    ├── áo len/")
        print("    ├── váy/")
        print("    └── quần short/")
