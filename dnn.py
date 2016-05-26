import cv2
import math
import os
import random
import numpy as np
import decimal
#segmentation start
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
#segmentation end

#feature detection start
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
            
            blue_histogram[int(math.floor(blue_value/float(class_width)))] += 1
            green_histogram[int(math.floor(green_value/float(class_width)))] += 1
            red_histogram[int(math.floor(red_value/float(class_width)))] += 1

            
    return np.append(blue_histogram, [green_histogram, red_histogram])
#feature detection end


#dnn start
#taking sum of errors
def error(errora_dict,files,rows):
    errora_sum = 0
    count = 0
    for n in range(files):
        for i in range(rows):
            errora_sum += sum(errora_dict["file%d" % (n)][i])
            count += len(errora_dict["file%d" % (n)][i])
    errora = errora_sum / count
    return errora

def dnn(source, files, size, tolerance, maxiter):
    alpha = 0.1
    errorb_dict = {}
    errorb = 0
    theta = {}
    theta_vec = []
    classes = {}
    class_list = ["N", "B", "C"]
    ctr = 0
    x_values = {}
    y_values = {}
    classes = math.ceil(math.log(size**2+1,2))
    
    #start reading from txt
    for n in range(files):
        dim = segment(source, "train%d" % (n), "train%d.jpeg" % (n), size)
        
        rows = dim[0]
        columns = dim[1]
        
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
    #return x_values["train0"][1][0]
    #return y_values["train0"][1]
    #end reading from txt

    #error handling
    theta_vec = []
    
    for i in range(int((classes*3+1))):
        theta_vec = np.append(theta_vec,[random.gauss(0,0.01)])
        
    #initial error computation
    errora_dict = {}

    for n in range(files):
        errora_file = []
        for i in range(rows):
            errora_row = []
            for j in range(columns):
                x_vec = x_values["train%d" % (n)][i][j]
                theta_x = theta_vec*x_vec
                theta_x_dot = sum(theta_x)
                y_est = float(2 / (1 + decimal.Decimal(-theta_x_dot).exp()))
                errora_row = np.append(errora_row,[(y_est - y_values["train%d" % (n)][i][j])**2 / 2])
            errora_file.append(errora_row)
        errora_dict["file%d" % (n)] = errora_file
    
    #place sum of errors here
    errora = error(errora_dict,files,rows)
    ctr = 0
    
    #sgd start
    while ctr <= maxiter:
        if errora + errorb < tolerance:
            break
            
        
        for n in range(files):        
            for i in range(rows):
                for j in range(columns):
                    x_vec = x_values["train%d" % (n)][i][j]
                    theta_x = theta_vec*x_vec
                    x = sum(theta_x)
                    y_est = float(2 / (1 + decimal.Decimal(-theta_x_dot).exp()))
                    y_true = y_values["train%d" % (n)][i][j]
                    

                    #sgd implementation
                    for k in range(len(theta_vec)):
                        theta_vec[k] -= alpha * (y_est - y_true) * x_vec[k]
                    
                    theta_x = theta_vec*x_vec
                    x = sum(theta_x)
                    y_est = float(2 / (1 + decimal.Decimal(-theta_x_dot).exp()))
                    errora_dict["file%d" % (n)][i][j] = (y_est - y_true)**2 / 2
                    errora = error(errora_dict,files,rows)
                    if errora + errorb < tolerance:
                        break
                
                if errora + errorb < tolerance:
                    break
            
            if errora + errorb < tolerance:
                break    
        else:        
            ctr += 1
    
    theta_file = open("theta.txt", 'w')

    for theta in theta_vec:
        theta_file.write(str(theta)+"\n")
        
    theta_file.close
#dnn end

print dnn("Training", 2, 32, 10**-12, 1000)

