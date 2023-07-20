import cv2
from matplotlib.mlab import window_none
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

img = cv2.imread('images\FCB.png')
cv2.imshow('Coloured Image',img)

#2-converte into grey level image

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grey Scale Image',gray_image)
print('Image Dimensions :', img.shape)


#5-
 
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2
  
  
# Define parameters.
r1 = 70
s1 = 30
r2 = 120
s2 = 180

  
# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize(pixelVal)
  
# Apply contrast stretching.
contrast_stretched = pixelVal_vec(gray_image, r1, s1, r2, s2)
  
# Save edited image.
cv2.imwrite('contrast_stretch.jpg', contrast_stretched)
cv2.imshow('Intensity Transformation',gray_image)

#Piecewise-Linear Transformation Functions 
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2
  

  
# Define parameters.
r1 = 70
s1 = 0
r2 = 140
s2 = 255
  
# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize(pixelVal)
  
# Apply contrast stretching.
contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)
  
# Save edited image.
cv2.imwrite('contrast_stretch.jpg', contrast_stretched)


#############################

hist,bins = np.histogram(gray_image.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(gray_image.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

equ = cv2.equalizeHist(gray_image)
cv2.imshow('equ.png',equ)

  
################
#3- histogram for grey  image
dst = cv2.calcHist(img, [0], None, [256], [0,256])
dst = cv2.calcHist(gray_image, [0], None, [256], [0,256])

plt.hist(gray_image.ravel(),256,[0,256])
plt.title('Histogram for gray scale image')
plt.show()


#4-Print the necessary features/values that can be useful to identify the shape of the histogram
plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=dst, bins=8, color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Normal Distribution Histogram',fontsize=15)
plt.show()

np.random.seed(10**7)
mu = 121 
sigma = 21
x = mu + sigma * np.random.randn(1000)
   
num_bins = 100
   
n, bins, patches = plt.hist(x, num_bins, 
                            density = 1, 
                            color ='green',
                            alpha = 0.7)
   
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
  
plt.plot(bins, y, '--', color ='black')
  
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
  
plt.title('matplotlib.pyplot.hist() function Example\n\n',
          fontweight ="bold")
  
plt.show()


