import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

#read input image
img = cv2.imread('weed.jpg')
#cv2.imshow('Natural Image', img)
#from colored to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray Image', gray)
# histogram for gray image
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Plot the histogram
plt.hist(gray.ravel(), 256, [0, 256])
#plt.show()

# change brightness of the gray image using gamma correction 
gamma = np.random.uniform(0.5, 2.0)


###################### 
######on the gray scale image
# using look-up table
start_time = time.time()
look_up_table = np.power(np.arange(0, 256)/255.0, gamma)*255.0
look_up_table = look_up_table.astype(np.uint8)
gamma_corrected_img_lut = cv2.LUT(gray, look_up_table)
print(f"Using look-up table method  {time.time() - start_time:.5f} seconds")

#without look-up table
start_time = time.time()
gamma_corrected_img = np.power(gray/255.0, gamma)
gamma_corrected_img = np.uint8(gamma_corrected_img*255)
print(f"Without look-up table method  {time.time() - start_time:.5f} seconds")

#execution time of both methods
cv2.imshow('with look-up table', gamma_corrected_img_lut)
cv2.imshow('without look-up table', gamma_corrected_img)

#histogram for both gamma methods 
hist_gamma_corrected = cv2.calcHist([gamma_corrected_img],[0],None,[256],[0,256])
cv2.imshow('Histogram (Gamma Corrected Image)', hist_gamma_corrected)
hist1 = cv2.calcHist([gamma_corrected_img_lut], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([gamma_corrected_img], [0], None, [256], [0, 256])

plt.plot(hist1)
plt.show()
plt.plot(hist2)
plt.show()

#####################3#
#### on the original image
# using look-up table
start_time = time.time()
look_up_table = np.power(np.arange(0, 256)/255.0, gamma)*255.0
look_up_table = look_up_table.astype(np.uint8)
gamma_corrected_img_lut = cv2.LUT(img, look_up_table)
print(f"Using look-up table method  {time.time() - start_time:.5f} seconds")

#without look-up table
start_time = time.time()
gamma_corrected_img = np.power(img/255.0, gamma)
gamma_corrected_img = np.uint8(gamma_corrected_img*255)
print(f"Without look-up table method  {time.time() - start_time:.5f} seconds")

#execution time of both methods
cv2.imshow('with look-up table', gamma_corrected_img_lut)
cv2.imshow('without look-up table', gamma_corrected_img)

#histogram for both gamma methods 
hist_gamma_corrected = cv2.calcHist([gamma_corrected_img],[0],None,[256],[0,256])
cv2.imshow('Histogram (Gamma Corrected Image)', hist_gamma_corrected)
hist1 = cv2.calcHist([gamma_corrected_img_lut], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([gamma_corrected_img], [0], None, [256], [0, 256])

plt.plot(hist1)
plt.show()
plt.plot(hist2)
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
