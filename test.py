import cv2
import numpy as np
import copy
from guided_filter.core.filter import GuidedFilter
import os
import sklearn
import skfuzzy as fuzz
import matplotlib
from skfuzzy import control as ctrl
import math

INPUT_IMAGE_PATH = "D:\Study Material\Digital Image Processing\Backlit-image-enhancement\input images\\15.jpg"
img = cv2.imread(INPUT_IMAGE_PATH)
I = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
L = cv2.bilateralFilter(I,6,25,25)
R = I/L

cv2.imwrite("L.jpg",L)

cv2.imwrite("I.jpg",I)
I2 = R*L
cv2.imwrite("I2.jpg",I2)
mm = 0
for x in range(R.shape[0]):
    for y in range(R.shape[1]):
        if L[x,y]==0:
            continue
        if mm < R[x,y]:
            mm = R[x,y]
print(mm)
for x in range(R.shape[0]):
    for y in range(R.shape[1]):
        R[x,y] = R[x,y]/mm
R = R*255
print(R)
cv2.imwrite("R.jpg",R)
cv2.imshow("R",R)
cv2.waitKey(0)

print(I2)