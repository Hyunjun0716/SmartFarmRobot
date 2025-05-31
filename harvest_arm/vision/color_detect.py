import cv2 as cv
import numpy as np

# wbcam or image file
webcam_vaild = False
image_path = "SmartFarmRobot\\harvest_arm\\vision\\sample.jpeg"  # Change to your image path if USE_WEBCAM is False

if webcam_vaild:
    cap = cv.VideoCapture(0)
else:
    frame = cv.imread(image_path)

while True:
    if webcam_vaild:
        ret, frame = cap.read()
        if not ret:
            break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # red color detection
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)

    # red color detection (second range for red)
    # OpenCV uses two ranges for red color detection due to the circular nature of the HSV color space
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv.bitwise_or(mask1, mask2)

    # green color detection
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    green_mask = cv.inRange(hsv, lower_green, upper_green)

    red_result = cv.bitwise_and(frame, frame, mask=red_mask)
    green_result = cv.bitwise_and(frame, frame, mask=green_mask)

    combined = cv.addWeighted(red_result, 1, green_result, 1, 0)

    cv.imshow("Ripe (Red) & Unripe (Green) Detection", combined)

    if webcam_vaild:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv.waitKey(0)
        break

if webcam_vaild:
    cap.release()
cv.destroyAllWindows()
