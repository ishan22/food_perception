import cv2
import numpy as np
from numpy.linalg import multi_dot
from matplotlib import pyplot as plt
from time import time

def  nothing(x):
	pass
 #not_skewered/sample_0000
 # skewered/sample_0008
img = cv2.imread('food_collection_data/tilted_angled_lettuce/data_collection/broccoli-angle-90-trial-7/color/image_4.png', 0)
template = cv2.imread('template_mid.png', 0)
w, h = template.shape[::-1]
depth = cv2.imread('food_collection_data/tilted_angled_lettuce/data_collection/broccoli-angle-90-trial-7/depth/image_4.png', 0)

t = time()
depth[...] = depth[...] * 500 #fastest method of product ~ 0.0007 seconds
print time() - t

y = 200
x = 300

img = img[y:y+280, x:x+250]
depth = depth[y:y+280, x:x+250]


img = cv2.convertScaleAbs(img, alpha=1.7, beta=-1.4)
template = cv2.convertScaleAbs(template, alpha=1.7, beta=-1.4)

blur = cv2.GaussianBlur(img, (5, 5), 0)
blur_edges = cv2.Canny(blur, 0, 140)


erosion_size = 3
erosion_type = cv2.MORPH_RECT	
element = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
depth = cv2.erode(depth, element)


# TEMPLATE MATCHING

res = cv2.matchTemplate(blur_edges, template, cv2.TM_CCORR)
cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1);

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (max_loc[0] + w, max_loc[1] + h)

cv2.rectangle(img, (top_left[0], top_left[1] - 15), bottom_right, 255, 2) # Draw rectangles of template matching
cv2.rectangle(depth, top_left, bottom_right, 255, 2)
# END TEMPLATE MATCHING
# window_bottom = (top_left[0] + 83, int(top_left[1] + 0.5*h))
# cv2.rectangle(img, (top_left[0]+12, top_left[1]+5), window_bottom, 255, 4);

depth_cp = depth[top_left[1]+10:int(top_left[1]+(0.30*h)), top_left[0]+5:top_left[0]+65].copy() # Crop to fork
fork_crop = img[top_left[1]+10:int(top_left[1]+(0.30*h)), top_left[0]+5:top_left[0]+65].copy()


# MAJORITY BLACK = ITEM SUCCESSFULLY ACQUIRED
# MAJORITY WHITE = ITEM ISN'T SUCCESSFUL
nonzero = cv2.countNonZero(depth_cp)
total = depth_cp.shape[0] * depth_cp.shape[1]
zero = total - nonzero
ratio = zero * 100 / float(total)
error = 1

if ratio >= 50 - error:
	print "YES"
else:
	print "NO"

cv2.imshow("Original", img)
cv2.imshow("blur_Canny", blur_edges)
cv2.imshow("depth", depth)
cv2.imshow("depth_fork", depth_cp)
cv2.imshow("img_fork", fork_crop)
cv2.imshow("template", template)

cv2.waitKey(0)