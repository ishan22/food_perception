import cv2
import numpy as np
from numpy.linalg import multi_dot
from matplotlib import pyplot as plt
from time import time

# NOTES
# 1) USE MOMENTS OF CONTOURS TO FIND THE CENTROID OF THE FORK 
# THEN USE THE CENTROID AND FIND THE DISTANCE TO THE TIPS OF THE FORK
# AND THEN MAKE A DECISION FROM THAT DISTANCE

# 2) USE THE SLIDING IMAGE WINDOW (TEMPLATE MATCHING) TO FIND THE 
# FORK IN THE IMAGE AND THEN FIND THE CENTER OF THAT BOUNDING BOX
# AND USE THAT DISTANCE FROM THAT CENTER TO THE CENTROID OF THE FORK

# 3) USE TEMPLATE MATCHING AND FIND THE AREA OF THE BOUDING BOX TO 
# MAKE A DECISION

# 4) LOOK AT THE DEPTH IMAGES AND MULTIPLY PIXELS VALUES BY 500 TO GET
# A BETTER IMAGE AND THEN BOUND TO THE PIXEL AREA OF THE BOUNDING BOX
# AND THEN USE A THRESHOLDING TO FIND THE MAJORITY VOTE


def  nothing(x):
	pass
 #not_skewered/sample_0000
 # skewered/sample_0008
img = cv2.imread('food_collection_data/tilted_vertical_skewer_isolated/data_collection/cauliflower-angle-90-trial-6/color/image_4.png', 0)
template = cv2.imread('template_big.png', 0)
w, h = template.shape[::-1]

depth = cv2.imread('food_collection_data/tilted_vertical_skewer_isolated/data_collection/cauliflower-angle-90-trial-6/depth/image_4.png', 0)

t = time()
depth[...] = depth[...] * 500 #fastest method of product ~ 0.0007 seconds
print time() - t


# img = cv2.convertScaleAbs(img, alpha=0, beta=0)

blur = cv2.GaussianBlur(img, (7, 7), 0)
blur_edges = cv2.Canny(blur, 0, 140)


# erosion_size = 7
# erosion_type = cv2.MORPH_RECT
# element = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
depth = cv2.erode(depth, (5, 5))
# image, contours, hierarchy = cv2.findContours(blur_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cx = []
# cy = []

# # DRAW CONTOURS AND CENTROIDS

# for shape in contours:
# 	area = cv2.contourArea(shape)
# 	M = cv2.moments(shape)
# 	hull = cv2.convexHull(shape)
# 	if area > 10:
# 		cx.append(int(M['m10']/area))
# 		cy.append(int(M['m01']/area))
# 		cX = int(M['m10']/area)
# 		cY = int(M['m01']/area)
# 		cv2.drawContours(img, [shape], -1, (255, 255, 255), 3)
# 		cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)

# END CONTOURS AND CENTROIDS

# TEMPLATE MATCHING

res = cv2.matchTemplate(blur_edges, template, cv2.TM_CCORR_NORMED)
cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1);

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img, (top_left[0], top_left[1] - 15), bottom_right, 255, 2)
cv2.rectangle(depth, top_left, bottom_right, 255, 2)
# END TEMPLATE MATCHING
window_bottom = (top_left[0] + 78, int(top_left[1] + 0.35*h))
cv2.rectangle(img, (top_left[0]+15, top_left[1]-12), window_bottom, 255, 4);

depth_cp = depth[top_left[1]-12:int(top_left[1]+(0.35*h)), top_left[0]+15:top_left[0]+78].copy()
fork_crop = img[top_left[1]:int(top_left[1]+(0.35*h)), top_left[0]+15:top_left[0]+78].copy()


# MAJORITY BLACK = ITEM SUCCESSFULLY ACQUIRED
# MAJORITY WHITE = ITEM ISN'T SUCCESSFUL
nonzero = cv2.countNonZero(depth_cp)
total = depth_cp.shape[0] * depth_cp.shape[1]
zero = total - nonzero
ratio = zero * 100 / float(total)
error = 1

print ratio

if ratio >= 50 - error:
	print "YES"
else:
	print "NO"

cv2.imshow("Original", img)
# cv2.imshow("Canny", edges)
cv2.imshow("blur_Canny", blur_edges)
cv2.imshow("depth", depth)
cv2.imshow("depth_fork", depth_cp)
cv2.imshow("img_fork", fork_crop)
# cv2.imshow("contours", image)

cv2.waitKey(0)