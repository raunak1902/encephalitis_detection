import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# =====================================================
# 1️⃣ DATASET PATH (LOCAL WINDOWS PATH)
# =====================================================
dataset_path = r"C:\Users\rauna\Downloads\encephalitis_detection\training\NINS_Dataset"

# =====================================================
# 2️⃣ KEEP ONLY REQUIRED CLASSES
# =====================================================
classes_to_keep = [
    'Encephalomalacia with gliotic change',
    'NMOSD  ADEM',
    'Ischemic change  demyelinating plaque',
    'demyelinating lesions',
    'White Matter Disease',
    'Microvascular ischemic change',
    'focal pachymeningitis',
    'Leukoencephalopathy with subcortical cysts',
    'Obstructive Hydrocephalus',
    'Cerebral Hemorrhage',
    'Normal'
]

print("Cleaning dataset...")

for class_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(folder_path) and class_folder not in classes_to_keep:
        print("Removing:", class_folder)
        import shutil
        shutil.rmtree(folder_path)

# =====================================================
# 3️⃣ DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# 4️⃣ TRANSFORMS (AUGMENTATION)
# =====================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(0.2,0.2,0.2,0.2),
    transforms.RandomAffine(10, shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =====================================================
# 5️⃣ LOAD DATASET
# =====================================================
full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)

print("Classes:", full_dataset.classes)
print("Total Images:", len(full_dataset))

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16)

num_classes = len(full_dataset.classes)

# =====================================================
# 6️⃣ LOAD PRETRAINED DENSENET
# =====================================================
print("Loading DenseNet121...")

model = models.densenet121(weights="DEFAULT")

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

# =====================================================
# 7️⃣ TRAIN FUNCTION
# =====================================================
def train_model(model, epochs=10):
    best_acc = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()

        running_loss = 0
        running_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, 1)
            running_loss += loss.item()
            running_correct += torch.sum(preds == labels)

        train_acc = running_correct.double() / len(train_dataset)
        print("Train Acc:", train_acc.item())

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs,1)
                correct += torch.sum(preds == labels)

        val_acc = correct.double() / len(val_dataset)
        print("Val Acc:", val_acc.item())

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    print("\nBest Validation Accuracy:", best_acc.item())
    return model

# =====================================================
# 8️⃣ TRAIN MODEL
# =====================================================
model = train_model(model, epochs=10)

# =====================================================
# 9️⃣ SAVE MODEL
# =====================================================
torch.save(model.state_dict(), "densenet_mri_model.pth")
print("Model saved as densenet_mri_model.pth")
