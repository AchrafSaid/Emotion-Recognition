import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

# =========================
# CHECK GPU
# =========================

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor()

])

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()

])

# =========================
# LOAD DATASET
# =========================

train_dataset = datasets.ImageFolder(

    root=r'C:\Users\omara\OneDrive\Desktop\ML\Emotion-Recognition\Omar\DATASET\Train',

    transform=train_transform

)

test_dataset = datasets.ImageFolder(

    root=r'C:\Users\omara\OneDrive\Desktop\ML\Emotion-Recognition\Omar\DATASET\Test',

    transform=test_transform

)

# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=True

)

test_loader = DataLoader(

    test_dataset,

    batch_size=16,

    shuffle=False

)

# =========================
# LOAD PRETRAINED CONVNEXT
# =========================

model = models.convnext_tiny(

    weights=models.ConvNeXt_Tiny_Weights.DEFAULT

)

# =========================
# CHANGE OUTPUT LAYER
# =========================

model.classifier[2] = nn.Linear(

    model.classifier[2].in_features,

    7

)

# =========================
# MOVE MODEL TO GPU
# =========================

model = model.to(device)

# =========================
# LOSS FUNCTION
# =========================

criterion = nn.CrossEntropyLoss()

# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.0001

)

# =========================
# TRAINING LOOP
# =========================

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    print(f"\nEpoch [{epoch+1}/{epochs}]")

    print(f"Loss: {running_loss / len(train_loader):.4f}")

    print(f"Train Accuracy: {train_acc:.2f}%")

# =========================
# TESTING
# =========================

model.eval()

correct = 0

total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total

print(f"\nTest Accuracy: {test_acc:.2f}%")