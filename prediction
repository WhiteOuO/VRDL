import torch
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
import torch.nn as nn
import sys
# Model parameters

sys.stdout.reconfigure(encoding='utf-8')
num_classes = 100  # Number of classes
model_path = "best_resnet_model.pth"  # change to your model name here

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
test_transform = transforms.Compose([
    transforms.Resize((320, 320)),  # Resize test images to 288x288
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_dataset = ImageFolder(root='./data/train', transform=transform)
# Custom dataset for test images (no labels, just filenames and images)
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_filenames = sorted(os.listdir(test_dir))  # Sort for consistency
        self.transform = transform
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.test_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, os.path.splitext(self.image_filenames[idx])[0]  # Remove file extension

# Use the updated test_transform for test dataset
test_dataset = TestDataset(test_dir='./data/test', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  Load pretrained SEResNeXt-50 from timm library
# model = timm.create_model('resnetaa50d.sw_in12k_ft_in1k', pretrained=True)
# model = timm.create_model('resnetaa101d.sw_in12k_ft_in1k', pretrained=True)
model = timm.create_model('seresnet152d.ra2_in1k', pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, num_classes)
)

state_dict = torch.load(model_path, map_location=device, weights_only=True)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

model = model.to(device)
model.eval()

# Predict and store results
predictions = []
image_filenames = []

with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for fn, pred in zip(filenames, predicted.cpu().numpy()):
            pred_label = train_dataset.classes[pred]
            predictions.append(pred_label)
            image_filenames.append(fn)


# Save predictions to CSV
prediction_df = pd.DataFrame({
    "image_name": image_filenames,
    "pred_label": predictions
})
prediction_df.to_csv("prediction.csv", index=False)

print("Predictions saved to prediction.csv")
