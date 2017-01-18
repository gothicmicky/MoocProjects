import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os


### step 1
### Compute the camera calibration matrix and distortion coefficients 
### given a set of chessboard images.


nx = 9 # of inner corners in x
ny = 6 # of inner corners in y

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/*.jpg')

def 
# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)
		# Draw and display the corners
		cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
		directory = 'output_images/1_corners_found/'
		if not os.path.exists(directory):
			os.makedirs(directory)
		write_name = directory+'corners_found'+str(idx)+'.jpg'
		cv2.imwrite(write_name, img)
		#cv2.imshow('img', img)
		cv2.waitKey(500)

cv2.destroyAllWindows()






#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image) 