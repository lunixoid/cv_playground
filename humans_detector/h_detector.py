# coding=utf-8
__author__ = 'romankalashnikov'

import numpy as np
import cv2

def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

cap = cv2.VideoCapture("/Users/romankalashnikov/Documents/Projects/reporoll/freelansim/humans_detector/sample.mp4")
# cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
while ret == False:
    ret, frame1 = cap.read()

ret, frame = cap.read()[1]
t_minus = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

while 1:
    diff = diffImg(t_minus, t, t_plus)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, [0, 255, 0], 3)

    cv2.imshow("Show diff", diff)
    cv2.imshow("Original", frame)

    # Read next image
    t_minus = t
    t = t_plus
    ret, frame = cap.read()# [1]
    t_plus = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyWindow("Show diff")
        break

cap.release()
cv2.destroyAllWindows()
