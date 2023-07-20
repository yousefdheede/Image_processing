import cv2
import numpy as np

def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) >= 2:
            rect1 = cv2.boundingRect(contours[0])
            rect2 = cv2.boundingRect(contours[1])
            center1 = (int(rect1[0] + rect1[2]/2), int(rect1[1] + rect1[3]/2))
            center2 = (int(rect2[0] + rect2[2]/2), int(rect2[1] + rect2[3]/2))
            distance = calculate_distance(center1, center2)

            if distance < 100:
                frame = cv2.resize(frame, None, fx=2.1, fy=2.1)
            elif distance > 150:
                frame = cv2.resize(frame, None, fx=0.8, fy=0.8)

        cv2.imshow('green', frame)

        if cv2.waitKey(1) & 0xFF == ord('w'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
