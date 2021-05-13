import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    # _, frame = cap.read()
    frame = cv.imread("colortest.png")
    frame = cv.flip(frame, 1)
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Set range for red color and define mask
    red_lower = np.array([136, 150, 100], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv.inRange(hsvFrame, red_lower, red_upper)

    kernal = np.ones((5, 5), "uint8")

    red_mask = cv.dilate(red_mask, kernal)
    res_red = cv.bitwise_and(hsvFrame, hsvFrame, mask=red_mask)

    contours, _ = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    areas = [cv.contourArea(i) for i in contours]
    if len(areas) > 0:
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 200:
                cv.drawContours(frame, cnt, -1, (255, 0, 255), 5)
                peri = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

    cv.imshow('result', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
