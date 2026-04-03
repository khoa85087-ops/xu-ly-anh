import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import random
from torchvision import transforms

# ===== MODEL =====
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===== LOAD MODEL =====
model = CNN()
model.load_state_dict(torch.load("cnn_cifar10.pth", map_location="cpu"))
model.eval()

# ===== CLASS =====
classes = ['airplane','automobile','bird','cat',
           'deer','dog','frog','horse','ship','truck']

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ===== RANDOM ẢNH =====
folder = "cifar100"
files = os.listdir(folder)

img_name = random.choice(files)
img_path = os.path.join(folder, img_name)

image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)

# ===== PREDICT =====
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Ảnh:", img_name)
print("Dự đoán:", classes[predicted.item()])

input("Nhấn Enter để thoát...")





