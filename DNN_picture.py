import cv2
import math
import os
import random
import numpy as np
import decimal

#for theta
theta_file = open("theta.txt", 'r')

theta_vec = []

for line in theta_file:
    theta_vec = np.append(theta_vec,[float(line[:len(list(line))-1])])

theta_file.close

#color source
yellow = cv2.imread("Colors/yellow.jpeg")
red = cv2.imread("Colors/red.jpeg")
green = cv2.imread("Colors/green.jpeg")


def segment(source, path, image, size):
    ctr = 0
    
    if not os.path.exists("%s/%s" % (source, path)):
        os.makedirs("%s/%s" % (source, path))
        ctr = 1 #newly made folder
        
    img = cv2.imread("%s/%s" %(source, image))
    length = img.shape[0]
    width = img.shape[1]

    rows = int(img.shape[0]/size)
    columns = int(img.shape[1]/size)
    subregions = {}
    for row in range(rows):
        image_list = []
        for column in range(columns):
            new_image = img[(row*size):((row+1)*size),(column*size):((column+1)*size)]
            image_list.append(new_image)
            if ctr == 1:
                cv2.imwrite("%s/%s/sr(%d,%d).jpeg" % (source, path, row, column), new_image) #where to place subregions
             
        subregions["row %d" % (row)] = image_list
    return {"subregions": subregions, "dimensions": [rows, columns, length, width]}
    
def feature(img):
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

        
def dnnpicture(image, border):
    green_ctr = 0
    red_ctr = 0
    yellow_ctr = 0
    
    size = 32
    img = cv2.imread("Testing/%s.jpeg" % (image))
    
    length = img.shape[0]
    width = img.shape[1]
    
    new_length = length + (length/size - 1) * border
    new_width = width + (width/size - 1) * border
    
    new_image = np.zeros((new_length,new_width, 3), np.uint8)
    new_image[:,:] = (0,0,0)
    
    border_t = 4
    
    border_image = np.zeros((length + (length/size - 1) * border_t,width + (width/size - 1) * border_t, 3), np.uint8)
    border_image[:,:] = (0,0,0)
    
    
    #with colors
    subregions = segment("Testing", image, "%s.jpeg" % (image), size)["subregions"]
    
    for i in range(len(subregions)):
        for j in range(len(subregions["row %d" % (i)])):
            
            border_image[(size+border_t)*i:(size+border_t)*i+size,(size+border_t)*j:(size+border_t)*j+size] = subregions["row %d" % (i)][j]
            
            temp = feature(subregions["row %d" % (i)][j])
            x_vec = np.append([1],temp)
            theta_x = theta_vec*x_vec
            theta_x_dot = sum(theta_x)
            print theta_x_dot
            y_est = float(2 / (1 + decimal.Decimal(-theta_x_dot).exp()))
            print y_est
            #y_est = random.uniform(0,2)
            if y_est < 0.5:
                new_image[(size+border)*i:(size+border)*i+size,(size+border)*j:(size+border)*j+size] = green

            elif y_est >= 0.5 and y_est < 1.5:
                new_image[(size+border)*i:(size+border)*i+size,(size+border)*j:(size+border)*j+size] = yellow
            else:
                new_image[(size+border)*i:(size+border)*i+size,(size+border)*j:(size+border)*j+size] = red        
    
    cv2.imwrite("Output/%s_colored_border%d.jpeg" % (image, border), new_image)
    cv2.imwrite("Output/%s_border%d.jpeg" % (image, border), border_image)
    

#image = raw_input("Image source: ")

#border = raw_input("Border size: ")

dnnpicture("verysmall", 4)