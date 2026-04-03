
#THƯ VIỆN  CIFAR 10 
from torch.utils.data import DataLoader
#GỌI THƯ VIỆN TIỀN XỬ LÝ VÀ CHIA ẢNH 
from torchvision import datasets, transforms

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
