#!/usr/bin/env python

import cv2
import csv
import numpy as np

#import rospy
#from std_msgs.msg import String
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

template = cv2.imread('template_mid.png', 0)
w, h = template.shape[::-1]

# def publisher():
# 	pub = rospy.Publisher('chatter', String)
# 	rospy.init_node('food_detector', anonymous=True)

# 	rate = rospy.Rate(20) #20hz
# 	while not rospy.is_shutdown():
# 		result = process_image('../food_images/vertical_skewer_lettuce/data_collection/pepper-angle-90-trial-3/color/image_3.png',
# 			'../food_images/vertical_skewer_lettuce/data_collection/pepper-angle-0-trial-3/depth/image_3.png')
# 		rospy.loginfo(result)
# 		pub.publish(result)
# 		rate.sleep()




def process_image(rgb_image, depth_image):
	with open('image_db.csv', 'r+') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',')
		output_file = open('output.csv', 'w+')
		csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				csv_writer.writerow([row[2], "prediction"])
				line_count+=1
			else:
				print 'COLOR:' + row[0]

				img = cv2.imread(row[0], 0)
				depth = cv2.imread(row[1], 0)

				y = 200
				x = 300

				depth[...] = depth[...] * 500

				img = img[y:y+280, x:x+250]
				depth = depth[y:y+280, x:x+250]

				img = cv2.convertScaleAbs(img, alpha=1.7, beta=-1.4)
				# img = cv2.convertScaleAbs(img, alpha=2.7, beta=-1.0)

				blur = cv2.GaussianBlur(img, (7, 7), 0)
				blur_edges = cv2.Canny(blur, 0, 140)

				erosion_size = 0
				erosion_type = cv2.MORPH_RECT	
				element = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
				depth = cv2.erode(depth, element)
				# TEMPLATE MATCHING

				res = cv2.matchTemplate(blur_edges, template, cv2.TM_CCORR)
				cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1);

				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
				top_left = max_loc
				print max_val
				print top_left
				bottom_right = (max_loc[0] + w, max_loc[1] + h)

				# END TEMPLATE MATCHING

				depth_cp = depth[top_left[1]:int(top_left[1]+(0.25*h)), top_left[0]+5:top_left[0]+65].copy() # crop to fork

				# MAJORITY BLACK = ITEM SUCCESSFULLY ACQUIRED
				# MAJORITY WHITE = ITEM ISN'T SUCCESSFUL

				nonzero = cv2.countNonZero(depth_cp)
				total = depth_cp.shape[0] * depth_cp.shape[1]
				zero = total - nonzero
				ratio = zero * 100 / float(total)
				error = 1

				if ratio >= 50 - error:
					csv_writer.writerow([row[2], "success\n"])
				else:
					csv_writer.writerow([row[2], "fail\n"])

if __name__ == '__main__':
	process_image('../../food_images/vertical_skewer_lettuce/data_collection/pepper-angle-0-trial-3/depth/image_3.png', '../../food_images/vertical_skewer_lettuce/data_collection/pepper-angle-0-trial-3/depth/image_3.png')
