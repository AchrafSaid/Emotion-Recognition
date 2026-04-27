from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Download the dataset from Kaggle and place it in the same directory as this script:https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?resource=download
# 1. Define preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# 2. Load dataset
train_data = datasets.ImageFolder(
    root="train",
    transform=transform
)

test_data = datasets.ImageFolder(
    root="test",
    transform=transform
)

# 3. Create loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 4. Print info
print("Train size:", len(train_data))
print("Test size:", len(test_data))
print("Classes:", train_data.classes)