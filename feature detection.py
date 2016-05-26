import cv2
import numpy as np
import math

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
print feature("subregion.jpeg")