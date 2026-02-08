# -*- coding: utf-8 -*-
"""
Encephalitis Detection - Model Training Script
Fixed version for local execution (Windows/Mac/Linux)

This script:
1. Downloads brain MRI dataset
2. Trains DenseNet121 model
3. Saves trained model as .pth file
"""

import os
import tarfile
import shutil
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import urllib.request

print("="*60)
print("🧠 Encephalitis Detection - Model Training")
print("="*60)

# Configuration
DATASET_URL = "https://figshare.com/ndownloader/files/28399209"
DATASET_FILE = "brain_mri_dataset.tar.gz"
DATASET_PATH = "./NINS_Dataset"
MODEL_SAVE_PATH = "densenet121_pretrained_mri.pth"

# Classes to keep
classes_to_keep = [
    'Encephalomalacia with gliotic change',
    'NMOSD ADEM',
    'Ischemic change demyelinating plaque',
    'Demyelinating lesions',
    'White Matter Disease',
    'Microvascular ischemic change',
    'Cerebral abscess',
    'Brain Infection',
    'Stroke (Demyelination)',
    'Brain Atrophy',
    'Brain Infection with abscess',
    'Postoperative encephalomalacia',
    'Focal pachymeningitis',
    'Leukoencephalopathy with subcortical cysts',
    'Obstructive Hydrocephalus',
    'Cerebral Hemorrhage',
    'Normal'
]

# Step 1: Download dataset
print("\n📥 Step 1: Downloading dataset...")
print(f"   URL: {DATASET_URL}")
print(f"   Size: ~2.5 GB (this may take 5-10 minutes)")

if not os.path.exists(DATASET_FILE):
    try:
        def download_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r   Progress: {percent}%", end='')
        
        urllib.request.urlretrieve(DATASET_URL, DATASET_FILE, download_progress)
        print("\n   ✅ Download complete!")
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        print("\n   Alternative: Download manually from:")
        print(f"   {DATASET_URL}")
        print(f"   Save as: {DATASET_FILE}")
        exit(1)
else:
    print(f"   ✅ Dataset file already exists: {DATASET_FILE}")

# Step 2: Extract dataset
print("\n📂 Step 2: Extracting dataset...")
if not os.path.exists(DATASET_PATH):
    try:
        with tarfile.open(DATASET_FILE) as tar:
            tar.extractall(path=DATASET_PATH)
        print(f"   ✅ Extracted to: {DATASET_PATH}")
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        exit(1)
else:
    print(f"   ✅ Dataset already extracted: {DATASET_PATH}")

# Step 3: Filter dataset to keep only specified classes
print("\n🔍 Step 3: Filtering dataset to 17 classes...")
removed_count = 0
for class_folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, class_folder)
    if os.path.isdir(folder_path) and class_folder not in classes_to_keep:
        shutil.rmtree(folder_path)
        removed_count += 1

print(f"   ✅ Removed {removed_count} unwanted classes")
print(f"   ✅ Keeping {len(classes_to_keep)} classes")

# Step 4: Setup training
print("\n⚙️  Step 4: Setting up training...")

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  No GPU detected - training will be slower (2-3 hours)")

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(10, shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
print("\n📊 Loading dataset...")
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=train_transform)
num_classes = len(full_dataset.classes)
print(f"   Total images: {len(full_dataset)}")
print(f"   Number of classes: {num_classes}")

# Split into train/validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

print(f"   Training images: {train_size}")
print(f"   Validation images: {val_size}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 5: Create model
print("\n🧠 Step 5: Creating DenseNet121 model...")
model = models.densenet121(pretrained=True)

# Freeze feature extraction layers
for param in model.parameters():
    param.requires_grad = False

# Modify classifier for our task
in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Only train the classifier
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)
print(f"   ✅ Model created with {num_classes} output classes")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=1e-4, weight_decay=1e-5)

# Step 6: Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5):
    print("\n🚀 Step 6: Training model...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Early stopping patience: {patience}")
    print("="*60)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx+1}/{len(train_loader)}", end='\r')

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"   Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"   Val Loss:   {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f}")
        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc.item())

        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"   ✅ New best model! Val Acc: {best_acc:.4f}")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*60)
    print(f"✅ Training Complete!")
    print(f"   Best Validation Accuracy: {best_acc:.4f}")
    print("="*60)
    
    model.load_state_dict(best_model_wts)

    # Plot results
    print("\n📊 Generating training plots...")
    plot_training_results(train_acc_history, val_acc_history, 
                         train_loss_history, val_loss_history)

    return model

def plot_training_results(train_acc, val_acc, train_loss, val_loss):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(train_acc, label='Training Accuracy', marker='o')
    ax1.plot(val_acc, label='Validation Accuracy', marker='s')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(train_loss, label='Training Loss', marker='o')
    ax2.plot(val_loss, label='Validation Loss', marker='s')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training vs Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("   ✅ Plots saved to: training_results.png")
    
    try:
        plt.show()
    except:
        print("   (Display not available in this environment)")

# Step 7: Start training
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Step 8: Save model
print(f"\n💾 Step 7: Saving model...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
file_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)  # Convert to MB
print(f"   ✅ Model saved to: {MODEL_SAVE_PATH}")
print(f"   File size: {file_size:.1f} MB")

# Final summary
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"Model file: {MODEL_SAVE_PATH}")
print(f"Training plots: training_results.png")
print("\nNext steps:")
print("1. Move model file to models/ folder:")
print(f"   mv {MODEL_SAVE_PATH} ../models/")
print("2. Start the backend server:")
print("   cd ../backend && python app.py")
print("3. Start the frontend:")
print("   cd ../frontend && npm start")
print("="*60)
