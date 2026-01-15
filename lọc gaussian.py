import cv2
import numpy as np
import matplotlib.pyplot as plt
# đọc ảnh
img=cv2.imread('image.jpg')
# chuyển từ BRG -->RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# chuyển từ RGB -->ẢNH XÁM
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# Hiển thị ảnh màu
plt.imshow(img_rgb)
plt.axis("off")
plt.show()



# lọc gaussian
img_gaussian=cv2.GaussianBlur(img_rgb,(5,5),0)
# hiển thị sau lọc
plt.imshow(img_gaussian)
plt.axis("off")
plt.show()
