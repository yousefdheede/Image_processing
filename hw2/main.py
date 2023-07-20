
from cv2 import HOUGH_GRADIENT
import numpy as np
import cv2 
import pandas as pd 
import matplotlib.pyplot as plt
x=y=z=0
#read image
img = cv2.imread('i3.jpg',cv2.IMREAD_GRAYSCALE)

_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

contours, hierarchy= cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
font = cv2.FONT_HERSHEY_DUPLEX
for cnt in contours:
    approx =cv2.approxPolyDP(cnt,0.001*cv2.arcLength(cnt,True),True)
    cv2.drawContours(img,[cnt],0,(0),2)
    print(len(approx))
    x=approx.ravel()[0]
    y=approx.ravel()[-1]
    
    
    if len(approx) ==106:
        cv2.putText(img,"curve",(x,y),font,1,(0))
    elif len(approx) ==32:
        cv2.putText(img,"face",(x,y),font,1,(0))
    elif len(approx) ==40:
        cv2.putText(img,"nose",(x,y),font,1,(0))
    elif len(approx) ==31:
        cv2.putText(img,"eye ",(x,y),font,1,(0))
    elif len(approx) ==74:
        cv2.putText(img,"circle",(x,y),font,1,(0))
   
    elif len(approx) ==94:
        cv2.putText(img,"Triangle",(x,y),font,1,(0))
    elif len(approx) ==39:
        cv2.putText(img,"Rectangle",(x,y),font,1,(0))
    elif len(approx) ==58:
        cv2.putText(img,"line",(x,y),font,1,(0))
    elif len(approx) ==119:
        cv2.putText(img,"mouth",(x,y),font,1,(0))
        
        
        
        
cv2.imshow('shapes', img) 
cv2.waitKey(0)
cv2.destroyAllWindows()