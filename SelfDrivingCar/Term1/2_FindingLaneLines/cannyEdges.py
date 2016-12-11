#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread('exit_ramp.jpg')
plt.imshow(image)

import cv2  #bringing in OpenCV libraries
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #grayscale conversion
plt.imshow(gray, cmap='gray')