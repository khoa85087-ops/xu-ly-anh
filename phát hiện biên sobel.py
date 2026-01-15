
import cv2
import numpy as np
import matplotlib.pyplot as plt
# đọc ảnh
img=cv2.imread('image_2.jpg')
# chuyển từ BRG -->RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# chuyển từ RGB -->ẢNH XÁM
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# Plot sẽ tự tô màu nên cần  cmap sang gray
plt.imshow(img_gray, cmap='gray')
plt.axis("off")
plt.show()


# phát hiện biên sobel
sobel_x=cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)
sobel_y=cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3)
sobel_com=cv2.magnitude(sobel_x,sobel_y)
# hiển thị sau lọc
plt.imshow(sobel_com,cmap='gray')
plt.axis("off")
plt.show()
