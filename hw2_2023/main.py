import cv2
import numpy as np


cap = cv2.VideoCapture(0)


lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_green = np.array([40, 100, 100])
upper_green = np.array([70, 255, 255])
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])


prev_gray = None
pinch_distance = None
stored_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #  frame to HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold 
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in each mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour for each color
    contour_red = max(contours_red, key=cv2.contourArea, default=None)
    contour_green = max(contours_green, key=cv2.contourArea, default=None)
    contour_blue = max(contours_blue, key=cv2.contourArea, default=None)

    
    if contour_red is not None:
        x, y, w, h = cv2.boundingRect(contour_red)
        cv2.line(frame, (x + w // 2, 0), (x + w // 2, frame.shape[0]), (0, 0, 255), thickness=2)

   

    if contour_blue is not None:
        hull = cv2.convexHull(contour_blue)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], 0, 255, -1)

        stored_image = cv2.bitwise_and(frame, frame, mask=mask)

    if stored_image is not None:
        cv2.imshow('Stored Image', stored_image)

    cv2.imshow('Video Stream', frame)

    key = cv2.waitKey(1)
    if key == ord('w'):
        break


# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
