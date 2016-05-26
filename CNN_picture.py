import cv2
import math
import os
import random
import numpy as np

#for theta
theta_file = open("theta.txt", 'r')

theta = []

for line in theta_file:
    theta.append(float(line[:len(list(line))-1]))

theta_file.close
    
def feature(img):
    #edited from original
    
    length = img.shape[0]
    width = img.shape[1]
    classes = int(math.ceil(math.log(length*width+1,2)))
    class_width = int(math.ceil(255/float(classes)))
    
    blue_histogram=[0]*classes
    green_histogram=[0]*classes
    red_histogram=[0]*classes
    
    for row in range(length):
        for column in range(width):
            blue_value = img[row,column,0]
            green_value = img[row,column,1]
            red_value = img[row,column,2]
            
            blue_histogram[int(blue_value/float(class_width))] += 1
            green_histogram[int(green_value/float(class_width))] += 1
            red_histogram[int(red_value/float(class_width))] += 1

            
    return [blue_histogram, green_histogram, red_histogram]
    

#color source
yellow = cv2.imread("Colors/yellow.jpeg")
yellow_bgr = [1,0,255,255]
red = cv2.imread("Colors/red.jpeg")
red_bgr = [1,0,0,255]
green = cv2.imread("Colors/green.jpeg")
green_bgr = [1,0,255,0]


def cnnpicture(image, size):
    
    img = cv2.imread("Test/%s.jpeg" % (image))
    
    length = img.shape[0]
    width = img.shape[1]
    
    #colors int
    colors = {}
    
    for i in range(length):
        colors["row%d" % (i)] = [[0,0,0,0]]*width
        
    new_image = np.zeros((length,width, 3), np.uint8)
    new_image[:,:] = (0,0,0)

    for i in range(0,length-size+1):
        for j in range(0,width-size+1):
            x_list = feature(img[i:i+size,j:j+size])
            x = [1]
            for color in x_list:
                for number in color:
                    x.append(number)
            
            x_vec = [a*b for (a,b) in zip(theta,x)]
            x = sum(x_vec)
            y_est = 2 / (1 + math.exp(-x))
                
            #y_est = random.uniform(0,2)
            if y_est < 0.5:
                color_use = green_bgr    
            elif y_est >= 0.5 and y_est < 1.5:
                color_use = yellow_bgr
            else:
                color_use = red_bgr
             
            #register colors to the list
            for row in range(size):
                for column in range(size):
                    colors["row%d" % (i+row)][j+column] = [x+y for (x,y) in zip(colors["row%d" % (i+row)][j+column],color_use)]
                    
    for i in range(length):
        for j in range(width):
            ctr = colors["row%d" % (i)][j][0]
            new_image[i:i+1,j:j+1] = (int(colors["row%d" % (i)][j][1]/ctr), int(colors["row%d" % (i)][j][2]/ctr), int(colors["row%d" % (i)][j][3]/ctr))
            
    cv2.imwrite('Output/cnnoutput_%s.jpeg' % (image), new_image)
    
cnnpicture("sample4", 32)


