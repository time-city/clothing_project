"""
train.py
Module huấn luyện mô hình ResNet50 cho phân loại quần áo.

Chức năng:
- Load mô hình ResNet50 pre-trained từ ImageNet
- Transfer Learning: freeze các layer sâu, huấn luyện lại layer cuối
- Theo dõi Loss và Accuracy
- Lưu checkpoint tốt nhất
- Vẽ biểu đồ training history
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from pathlib import Path
from dataset import create_data_loaders


class ClothingClassifier(nn.Module):
    """
    Mô hình ResNet50 cho phân loại quần áo.
    
    Sử dụng Transfer Learning:
    - Lấy ResNet50 pre-trained trên ImageNet
    - Freeze các layer conv
    - Replace fully connected layer cuối cho 8 classes
    """
    
    def __init__(self, num_classes=8, freeze_backbone=True):
        """
        Khởi tạo mô hình.
        
        Args:
            num_classes (int): Số lượng class (mặc định 8)
            freeze_backbone (bool): Có freeze backbone hay không
        """
        super(ClothingClassifier, self).__init__()
        
        # Load ResNet50 pre-trained
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze backbone nếu được yêu cầu
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Thay đổi fully connected layer
        in_features = self.backbone.fc.in_features  # 2048 cho ResNet50
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """
    Huấn luyện 1 epoch.
    
    Args:
        model: Mô hình PyTorch
        train_loader: DataLoader cho training set
        criterion: Loss function
        optimizer: Optimizer
        device: CPU hoặc GPU
        epoch: Epoch hiện tại
        total_epochs: Tổng số epoch
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Tính toán metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # In progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  [{epoch}/{total_epochs}] Batch {batch_idx + 1}/{len(train_loader)} - "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Kiểm chứng mô hình trên validation set.
    
    Args:
        model: Mô hình PyTorch
        val_loader: DataLoader cho validation set
        criterion: Loss function
        device: CPU hoặc GPU
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def plot_training_history(history, output_dir="./"):
    """
    Vẽ biểu đồ Loss và Accuracy.
    
    Args:
        history (dict): Dictionary chứa train_loss, val_loss, train_acc, val_acc
        output_dir (str): Thư mục lưu biểu đồ
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_history.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Biểu đồ đã lưu: {output_path}")
    plt.close()


def train(
    dataset_dir,
    num_epochs=20,
    batch_size=32,
    learning_rate=0.001,
    output_dir="./checkpoints",
    freeze_backbone=True,
    device=None
):
    """
    Hàm chính để huấn luyện mô hình.
    
    Args:
        dataset_dir (str): Đường dẫn folder dataset
        num_epochs (int): Số epoch huấn luyện
        batch_size (int): Kích thước batch
        learning_rate (float): Learning rate
        output_dir (str): Thư mục lưu checkpoint
        freeze_backbone (bool): Có freeze backbone hay không
        device: CPU hoặc GPU (auto-detect nếu None)
    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Sử dụng device: {device}")
    
    # Tạo output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📁 Đang load dataset...")
    train_loader, val_loader, class_to_idx, idx_to_class = create_data_loaders(
        dataset_dir,
        batch_size=batch_size
    )
    
    # Khởi tạo mô hình
    print("\n🧠 Khởi tạo mô hình ResNet50...")
    model = ClothingClassifier(num_classes=len(class_to_idx), freeze_backbone=freeze_backbone)
    model.to(device)
    
    # Loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training loop
    print(f"\n🚀 Bắt đầu huấn luyện ({num_epochs} epochs)...\n")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = Path(output_dir) / 'best_model.pth'
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        print("🔄 Training...")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # Validation
        print("✔️ Validating...")
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Lưu history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # In kết quả
        print(f"\nKết quả Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Lưu best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✨ Mô hình tốt nhất được lưu (Acc: {val_acc:.4f})")
    
    # Tính thời gian huấn luyện
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"\n{'='*60}")
    print(f"✓ Huấn luyện hoàn thành!")
    print(f"⏱️ Thời gian: {hours}h {minutes}m {seconds}s")
    print(f"{'='*60}")
    
    # Lưu final model
    final_model_path = Path(output_dir) / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Final model lưu tại: {final_model_path}")
    print(f"✓ Best model lưu tại: {best_model_path}")
    
    # Lưu class mapping
    class_mapping_path = Path(output_dir) / 'class_mapping.json'
    with open(class_mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {int(k): v for k, v in idx_to_class.items()}
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Class mapping lưu tại: {class_mapping_path}")
    
    # Vẽ biểu đồ
    print("\n📊 Vẽ biểu đồ...")
    plot_training_history(history, output_dir)
    
    # Lưu history
    history_path = Path(output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history lưu tại: {history_path}")
    
    return model, history


if __name__ == "__main__":
    """
    Cách chạy:
        python train.py --dataset ./data --epochs 20 --batch_size 32
    
    Hoặc với Google Colab:
        !python train.py --dataset /content/data --epochs 20 --batch_size 16
    """
    
    import argparse
    import sys
    
    # Nếu không có argument, dùng default (cho Colab)
    if len(sys.argv) == 1:
        # Default cho Colab
        dataset_path = '/content/data' if Path('/content/data').exists() else './data'
        train(
            dataset_dir=dataset_path,
            num_epochs=20,
            batch_size=16,  # Nhỏ hơn cho Colab GPU
            learning_rate=0.001,
            output_dir='./checkpoints',
            freeze_backbone=True
        )
    else:
        parser = argparse.ArgumentParser(description='Huấn luyện mô hình phân loại quần áo')
        parser.add_argument('--dataset', type=str, default='./data',
                           help='Đường dẫn folder dataset')
        parser.add_argument('--epochs', type=int, default=20,
                           help='Số epoch huấn luyện')
        parser.add_argument('--batch_size', type=int, default=16,
                           help='Batch size (16 cho Colab GPU, 32 cho local)')
        parser.add_argument('--lr', type=float, default=0.001,
                           help='Learning rate')
        parser.add_argument('--output', type=str, default='./checkpoints',
                           help='Thư mục lưu checkpoint')
        parser.add_argument('--freeze', type=bool, default=True,
                           help='Có freeze backbone hay không')
        
        args = parser.parse_args()
        
        # Kiểm tra dataset
        if not Path(args.dataset).exists():
            print(f"❌ Dataset folder '{args.dataset}' không tồn tại!")
            print("Hãy tạo cấu trúc thư mục như trong AI_README.md")
        else:
            train(
                dataset_dir=args.dataset,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                output_dir=args.output,
                freeze_backbone=args.freeze
            )
