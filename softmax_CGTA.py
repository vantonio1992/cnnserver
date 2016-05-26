#standard packages

from PIL import Image

import csv
import numpy as np
import os
import tensorflow as tf
import timeit
import pickle

import datetime


#parameters, manual input
size = 32
files = 2

#gathering data
def feature(image):
    img = cv2.imread(image)
    
    length = img.shape[0]
    width = img.shape[1]
    classes = int(math.ceil(math.log(length*width+1,2)))
    class_width = int(math.ceil(255/float(classes)))
    
    blue_histogram=np.tile([0],classes)
    green_histogram=np.tile([0],classes)
    red_histogram=np.tile([0],classes)
    
    for row in range(length):
        for column in range(width):
            blue_value = img[row,column,0]
            green_value = img[row,column,1]
            red_value = img[row,column,2]
            
            blue_histogram[int(blue_value/float(class_width))] += 1
            green_histogram[int(green_value/float(class_width))] += 1
            red_histogram[int(red_value/float(class_width))] += 1

            
    #return [length, width]
    return np.append(blue_histogram, [green_histogram, red_histogram])

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




#read training data

for n in range(files):
    segments = segment("Training", "train%d" % (n), "train%d.jpeg" % (n), size)
    
    rows = segments["dimensions"][0]
    columns = segments["dimensions"][1]
    
    #forming y
    data = open("%s/train%d.txt" % (source,n), 'r')    
    y_file = []
    for line in data:
        sample = []
        for item in list(line):
            if item == "N":
                sample.append(0)
            elif item == "B":
                sample.append(1)
            elif item == "C":
                sample.append(2)
            else:
                continue
        y_file.append(np.array(sample))
        
    y_values["train%d" % (n)] = y_file
    
    data.close()

    #forming x
    x_values["train%d" % (n)] = []
    
    for i in range(rows):
        x_row = []
        for j in range(columns):
            temp = feature("%s/train%d/sr(%d,%d).jpeg" % (source, n, i, j))
            x_vec = np.append([1],temp)
            x_row.append(x_vec)
        x_values["train%d" % (n)].append(x_row)



#start of implementation
x = tf.placeholder(tf.float32, [None, 784])

#weights, bias incorporation

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#model implementation
y = tf.nn.softmax(tf.matmul(x, W) + b)


#Training start

#cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#learning rate = 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#initialize the variables
init = tf.initialize_all_variables()

#launch Session
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

 #model evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))