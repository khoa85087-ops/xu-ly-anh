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



# phân ngưỡng thích nghi  
img_phan_nguong=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# hiển thị sau lọc 
plt.imshow(img_phan_nguong,cmap='gray')
plt.axis("off")
plt.show()
