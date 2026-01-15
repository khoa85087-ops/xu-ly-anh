import cv2
import numpy as np
import matplotlib.pyplot as plt
# đọc ảnh 
img=cv2.imread('image.jpg')
# chuyển từ BRG -->RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# chuyển từ RGB -->ẢNH XÁM
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# hiển thị ảnh màu
plt.imshow(img_rgb)
plt.axis("off")
plt.show()


# lọc trung bình  
img_blu=cv2.blur(img_rgb,(3,3))
# hiển thị sau lọc 
plt.imshow(img_blu)
plt.axis("off")
plt.show()
