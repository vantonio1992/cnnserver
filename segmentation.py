import cv2
import os
import numpy as np
#image  = image file, size = desired size of subregion (32, in our case), path = where to place subregions (optional)


def segment(source, path, image, size):
	ctr = 0
	
	if not os.path.exists("%s/%s" % (source, path)):
		os.makedirs("%s/%s/Original" % (source, path))
		os.makedirs("%s/%s/Sobel_x" % (source, path))
		os.makedirs("%s/%s/Sobel_y" % (source, path))
		ctr = 1 #newly made folder
		
	img = cv2.imread("%s/%s.jpeg" %(source, image))
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	cv2.imwrite("%s/%s/%s_sobelx.jpeg" % (source, path, image), sobelx)
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	cv2.imwrite("%s/%s/%s_sobely.jpeg" % (source, path, image), sobely)

	length = img.shape[0]
	width = img.shape[1]

	rows = int(img.shape[0]/size)
	columns = int(img.shape[1]/size)
	subregions = {}
	subregions_x = {}
	subregions_y = {}
	for row in range(rows):
		image_list = []
		image_list_x = []
		image_list_y = []
		for column in range(columns):
			new_image = img[(row*size):((row+1)*size),(column*size):((column+1)*size)]
			new_image_x = sobelx[(row*size):((row+1)*size),(column*size):((column+1)*size)]
			new_image_y = sobely[(row*size):((row+1)*size),(column*size):((column+1)*size)]
			image_list.append(new_image)
			image_list_x.append(new_image_x)
			image_list_y.append(new_image_y)
			if ctr == 1:
				#where to place subregions
				cv2.imwrite("%s/%s/Original/sr(%d,%d).jpeg" % (source, path, row, column), new_image)
			 	cv2.imwrite("%s/%s/Sobel_x/sr(%d,%d).jpeg" % (source, path, row, column), new_image_x)
			 	cv2.imwrite("%s/%s/Sobel_y/sr(%d,%d).jpeg" % (source, path, row, column), new_image_y)
		subregions["row %d" % (row)] = image_list
		subregions_x["row %d" % (row)] = image_list_x
		subregions_y["row %d" % (row)] = image_list_y
	return {"subregions": subregions, "subregions_x": subregions_x, "subregions_y": subregions_y,
		"dimensions": [rows, columns, length, width]}
    
#print len(segment("SampleData", "train0", "train0.jpeg", 32)["dimensions"])

segment("Images", "sample0", "sample0", 32)