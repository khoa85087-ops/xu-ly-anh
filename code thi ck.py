import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # tích chập 1 từ 3 kệnh in tạo ra 32 kênh out ,không làm thay đổi kích thước n=(n-3+2*1)/1 + 1 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # tích chập 2 từ 32 kênh in tạo ra 64 kênh out 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # tích chập 3 từ 64 kênh in tạo ra 128 kênh out 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # từ 2048 cuột thành 512 cột 
        self.fc1 = nn.Linear(2048, 512)
        # từ 512 cột thành 10 cột 
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # tích chập 1 : 32*32*3 -> 32*32*32
        x = self.conv1(x)
        # qua hàm relu : vẫn là 32*32*32 
        x = F.relu(x)
        # dùng của sổ lọc 2*2 cho pooling :32*32*32 -> 16*16*32
        x = F.max_pool2d(x, 2) 
        # qua hàm tích chập và qua hàm relu : 16*16*32 ->16*16*64->16*16*64
        x = F.relu(self.conv2(x))
        # dùng chưa sổ lọc 2*2 cho pooling: 16*16*64->8*8*64
        x = F.max_pool2d(x, 2)
         # qua hàm tích chập và qua hàm relu : 8*8*64->8*8*128->8*8*128 
        x = F.relu(self.conv3(x))
        # dùng chưa sổ lọc 2*2 cho pooling:8*8*128 -> 4*4*128 
        x = F.max_pool2d(x, 2)
        # chuyển thành vecto cột : 2048* 1 
        x = x.view(-1, 2048)
        # qua fc1 2048*1->512*1 
        x = F.relu(self.fc1(x))
        #qua fc2 còn 512*1 ->10*1  
        x = self.fc2(x)

        return x

# TIỀN XỬ LÝ : [0;-255] -> [0;-1]->[-1 ;1] ĐỂ INPUT GẦN 0 -> DỄ TRAIN 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# MẤY CÁI DƯỚI LÀ ĐỐI TƯƠNG=CƠ BẢN LÀ BẢN NÂNG CẤP CỦA STRUCT 
#LƯU 50K ẢNH TRAIN 
trainset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
#LẤY ẢNH THEO BATCH ĐỂ HỌC 
trainloader = DataLoader(
    trainset,
    batch_size=10,
    shuffle=True,
    num_workers=2
)
#LƯU 10K ẢNH KIỂM TRA 
testset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
#LẤY ẢNH THEO BATCH ĐỂ KIỂM TRA 
testloader = DataLoader(
    testset,
    batch_size=10,
    shuffle=False,
    num_workers=2
)
net = SimpleCNN()
# ===== LOSS + OPTIMIZER =====
avg_loss_history = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train 
for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(trainloader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    avg_loss_history.append(avg_loss)

    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

# vẽ đồ thị loss 
plt.figure()
plt.plot(avg_loss_history)
plt.title('Average Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# kiểm tra 
net.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.numpy())
        true_labels.extend(labels.numpy())

# đánh giá mô hình 
conf_matrix = confusion_matrix(true_labels, predictions)
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Confusion Matrix:\n{conf_matrix}')

# lưu mô hình 
from google.colab import drive
drive.mount('/content/drive')
path_on_drive = '/content/drive/My Drive/cifar_net_khoa.pth'
torch.save(net.state_dict(), path_on_drive)
