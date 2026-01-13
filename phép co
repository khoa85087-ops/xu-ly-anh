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


# tạo ô lọc 
kernel=np.ones((3,3),np.uint8)
#phep co
img_co=cv2.erode(img_gray,kernel,iterations=1)
# hiển thị sau lọc 
plt.imshow(img_co,cmap='gray')
plt.axis("off")
plt.show()
