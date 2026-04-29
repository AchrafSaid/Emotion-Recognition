from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Download the dataset from Kaggle and place it in the same directory as this script:https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?resource=download
# 1. Define preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), #new added (num_output_channels=1)
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),   #new, flip the image horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) #new
])

# 2. Load dataset
train_data = datasets.ImageFolder(
    root="C:\\Users\\nadee\\Downloads\\Emotion-Recognition\\archive\\train",  #change based on your path
    transform=transform
)

test_data = datasets.ImageFolder(
    root="C:\\Users\\nadee\\Downloads\\Emotion-Recognition\\archive\\test",  #change based on your path
    transform=transform
)

#new: split train_data into train and validation sets
from torch.utils.data import random_split

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size

train_data, val_data = random_split(train_data, [train_size, val_size])
#------------------------



# 3. Create loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2) #new changed loaders, batch_size=32 → efficient GPU usage shuffle=True → prevents memorization num_workers=2 → faster training
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)  #
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2) #

# 4. Print info
print("Train size:", len(train_data))
print("Validation size:", len(val_data))
print("Test size:", len(test_data))
print("Classes:", train_data.dataset.classes)