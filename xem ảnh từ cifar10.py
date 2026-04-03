import matplotlib.pyplot as plt

classes = trainset.classes
#chọn lớp ảnh: airplane / automobile / bird / cat / deer / dog / frog / horse / ship / truck
target_class = 'dog'
target_idx = classes.index(target_class)

n = 111   # lấy ảnh thứ n (bạn đổi số này)

count = 0

for img, label in trainset:
    if label == target_idx:
        if count == n:
            img = img * 0.5 + 0.5
            img = img.permute(1, 2, 0)

            plt.imshow(img)
            plt.title(classes[label])
            plt.axis('off')
            break
        count += 1
