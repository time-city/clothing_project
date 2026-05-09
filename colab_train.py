"""
Colab Training Script - ResNet50 for Clothing Classification
Tải data từ GitHub CSV, xử lý, train và save model về Google Drive
"""

# ====== PHẦN 1: CÀI ĐẶT VÀ IMPORT ======
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torchvision.models as models
from PIL import Image
import requests
from io import BytesIO, StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

# Mount Google Drive (chạy ở Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    COLAB_MODE = True
except:
    COLAB_MODE = False
    print("Not running on Colab")

# ====== PHẦN 2: LOAD DATA TỪ GITHUB ======
def load_data_from_github():
    """
    Load URLs từ GitHub cloudinary_links (CSV format)
    """
    print("\n" + "="*60)
    print("📥 LOAD DATA FROM GITHUB")
    print("="*60)
    
    github_url = "https://raw.githubusercontent.com/duyaivy/CRAWL_DATA/main/cloudinary_links"
    
    # Các file CSV chứa links
    clothing_classes = {
        'ao_khoac': 'ao_khoac.csv',
        'ao_so_mi': 'ao_so_mi.csv',
        'ao_thun': 'ao_thun.csv',
        'quan_jean': 'quan_jean.csv',
        'quan_short': 'quan_short.csv',
        'quan_tay': 'quan_tay.csv',
        'sweater_hoodie': 'sweater_hoodie.csv',
        'vay': 'vay.csv'
    }
    
    data_dict = {}
    
    for class_name, csv_file in clothing_classes.items():
        try:
            url = f"{github_url}/{csv_file}"
            print(f"Loading {class_name}...", end=" ")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                # Extract URLs từ cột 'secure_url'
                if 'secure_url' in df.columns:
                    links = df['secure_url'].tolist()
                    data_dict[class_name] = links
                    print(f"✓ Loaded {len(links)} images")
                else:
                    print(f"✗ 'secure_url' column not found")
            else:
                print(f"✗ Failed (Status: {response.status_code})")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not data_dict:
        print("\n⚠️  No data loaded! Check GitHub links.")
    
    return data_dict


# ====== PHẦN 3: DOWNLOAD ẢNH TỪ CLOUDINARY ======
def download_images(data_dict, output_dir="./clothing_data", max_per_class=500, img_size=224):
    """
    Download ảnh từ Cloudinary URLs
    
    Args:
        data_dict: Dictionary with class names and URLs
        output_dir: Output directory path
        max_per_class: Max images per class
        img_size: Image resize size (224x224 for ResNet)
    """
    print("\n" + "="*60)
    print("⬇️  DOWNLOADING IMAGES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    total_downloaded = 0
    
    for class_name, links in data_dict.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        success_count = 0
        failed_count = 0
        
        pbar = tqdm(links[:max_per_class], desc=f"{class_name:20s}", unit="img")
        for idx, link in enumerate(pbar):
            try:
                if pd.isna(link) or not link:
                    continue
                    
                response = requests.get(str(link), timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    # Convert RGBA to RGB if needed
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to standard size
                    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    
                    # Save
                    img_path = os.path.join(class_dir, f"{class_name}_{success_count:05d}.jpg")
                    img.save(img_path, quality=85)
                    success_count += 1
                    
                    if success_count >= max_per_class:
                        break
                else:
                    failed_count += 1
                        
            except Exception as e:
                failed_count += 1
        
        total_downloaded += success_count
        print(f"  ✓ {class_name:20s}: {success_count:3d} success, {failed_count:3d} failed")
    
    print(f"\n✅ Total downloaded: {total_downloaded} images")
    return output_dir


# ====== PHẦN 4: CUSTOM DATASET ======
class ClothingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = {}
        
        # Load class names
        class_idx = 0
        for class_name in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                self.classes[class_idx] = class_name
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
                class_idx += 1
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return None, None
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ====== PHẦN 5: MODEL ======
class ClothingClassifier(nn.Module):
    def __init__(self, num_classes=6, freeze_backbone=True):
        super(ClothingClassifier, self).__init__()
        
        # Load ResNet50 pre-trained
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


# ====== PHẦN 6: HÀM TRAINING ======
def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001, device='cuda', save_path='./best_model.pth'):
    """
    Huấn luyện mô hình
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate for optimizer
        device: Device (cuda or cpu)
        save_path: Path to save best model
    """
    criterion = nn.CrossEntropyLoss()
    
    # Only train FC layer if backbone is frozen, otherwise train all
    if model.backbone.fc.weight.requires_grad:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.backbone.fc.parameters(), lr=learning_rate)
    
    best_acc = 0
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", unit="batch")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]", unit="batch")
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)
        
        print(f"  Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Model saved! (Best Acc: {best_acc:.2f}%)")
    
    return train_losses, val_accs


# ====== PHẦN 7: MAIN EXECUTION ======
def print_header(title):
    """In ra header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def get_user_config():
    """
    Tự động chọn cấu hình tối ưu dựa trên GPU type
    Download FULL ảnh từ GitHub (không giới hạn)
    """
    print_header("⚙️  AUTO-DETECTING OPTIMAL CONFIG")
    
    # Detect GPU type
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {gpu_name}")
        print(f"📊 VRAM: {gpu_memory:.1f} GB")
    else:
        gpu_name = "CPU"
        gpu_memory = 0
        print(f"🖥️  Device: CPU (Slow! Use GPU if possible)")
    
    # Auto-select config based on GPU
    # Using all available images (no limit!)
    if "A100" in gpu_name:
        # Ultra powerful GPU
        config_dict = {
            'max_per_class': 999999,  # No limit - get all images
            'epochs': 40,
            'batch_size': 128,
            'learning_rate': 0.00005,
            'freeze_backbone': False,
            'gpu_tier': 'A100'
        }
        print("\n✨ A100 Detected! Using MAXIMUM Quality Config")
        print("📥 Downloading ALL images from GitHub (no limit)...")
        
    elif "V100" in gpu_name or "A10" in gpu_name:
        # Very good GPU
        config_dict = {
            'max_per_class': 999999,  # No limit - get all images
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.0001,
            'freeze_backbone': False,
            'gpu_tier': 'V100'
        }
        print("\n🚀 V100/A10 Detected! Using Ultra Quality Config")
        print("📥 Downloading ALL images from GitHub (no limit)...")
        
    elif "T4" in gpu_name or gpu_memory < 8:
        # Standard Colab GPU
        config_dict = {
            'max_per_class': 999999,  # No limit - get all images
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'freeze_backbone': False,
            'gpu_tier': 'T4'
        }
        print("\n⚡ T4 Detected! Using Production Quality Config")
        print("📥 Downloading ALL images from GitHub (no limit)...")
        
    elif "P100" in gpu_name or gpu_memory >= 16:
        # Good GPU
        config_dict = {
            'max_per_class': 999999,  # No limit - get all images
            'epochs': 25,
            'batch_size': 64,
            'learning_rate': 0.0001,
            'freeze_backbone': False,
            'gpu_tier': 'P100'
        }
        print("\n🎓 P100 Detected! Using High Quality Config")
        print("📥 Downloading ALL images from GitHub (no limit)...")
        
    else:
        # Unknown/CPU - use conservative config
        config_dict = {
            'max_per_class': 999999,  # No limit - get all images
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'freeze_backbone': True,
            'gpu_tier': 'Unknown'
        }
        print(f"\n⚠️  Unknown GPU Detected! Using Conservative Config")
        print("📥 Downloading ALL images from GitHub (no limit)...")
    
    print_header("📋 AUTO-SELECTED CONFIGURATION")
    print(f"  GPU Tier: {config_dict['gpu_tier']}")
    print(f"  Images per class: ALL (999999 max)")
    print(f"  Epochs: {config_dict['epochs']}")
    print(f"  Batch size: {config_dict['batch_size']}")
    print(f"  Learning rate: {config_dict['learning_rate']}")
    print(f"  Freeze backbone: {config_dict['freeze_backbone']}")
    print("="*60)
    print("✅ Starting training with optimal settings...\n")
    
    return config_dict


if __name__ == "__main__":
    print("\n" + "🎽 "*20)
    print("CLOTHING CLASSIFIER - RESNET50 TRAINING")
    print("Google Colab Edition")
    print("🎽 "*20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
    
    # Get user configuration
    config = get_user_config()
    
    # Step 1: Load data URLs
    print_header("STEP 1: LOADING DATA")
    data_dict = load_data_from_github()
    
    if not data_dict:
        print("❌ No data loaded. Exiting...")
        exit(1)
    
    # Step 2: Download images
    print_header("STEP 2: DOWNLOADING IMAGES")
    data_dir = download_images(
        data_dict, 
        output_dir="./clothing_data",
        max_per_class=config['max_per_class'],
        img_size=224
    )
    
    # Step 3: Prepare dataset
    print_header("STEP 3: PREPARING DATASET")
    
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ClothingDataset(data_dir, transform=train_transform)
    print(f"📁 Dataset loaded: {len(dataset)} images")
    print(f"📂 Classes: {list(dataset.classes.values())}")
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    
    # Step 4: Create model
    print_header("STEP 4: CREATING MODEL")
    num_classes = len(dataset.classes)
    model = ClothingClassifier(
        num_classes=num_classes, 
        freeze_backbone=config['freeze_backbone']
    )
    model = model.to(device)
    print(f"✓ ResNet50 model created ({num_classes} classes)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Step 5: Train
    print_header("STEP 5: TRAINING MODEL")
    save_path = '/content/drive/My Drive/resnet50-11ad3fa6.pth' if COLAB_MODE else './best_model.pth'
    os.makedirs(os.path.dirname(save_path) if COLAB_MODE else '.', exist_ok=True)
    
    train_losses, val_accs = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        device=device, 
        save_path=save_path
    )
    
    # Step 6: Save results
    print_header("STEP 6: SAVING RESULTS")
    if COLAB_MODE:
        print(f"✅ Model saved to: /content/drive/My Drive/resnet50-11ad3fa6.pth")
        print(f"   Download it from Google Drive!")
    else:
        print(f"✅ Model saved to: {save_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    if COLAB_MODE:
        plt.savefig('/content/drive/My Drive/training_history.png', dpi=100)
        print(f"📊 Training history saved to: /content/drive/My Drive/training_history.png")
    else:
        plt.savefig('./training_history.png', dpi=100)
        print(f"📊 Training history saved to: ./training_history.png")
    plt.show()
    
    print("\n" + "="*60)
    print("✨ TRAINING COMPLETED!")
    print("="*60)
