import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # frame = cv.imread("HSV_color_space_HS.png")
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    red_lower = np.array([25, 150, 100], np.uint8)
    red_upper = np.array([35, 255, 255], np.uint8)
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

                # calculate moments for each contour
                M = cv.moments(cnt)

                # calculate x,y coordinate of center
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # draw the contour and center of the shape on the image
                cv.circle(frame, (cX, cY), 5, (255, 255, 255), -1)

    cv.imshow('result', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()