#cd OneDrive/NAIST/Research/Dissertation/Codes_New

import numpy as np
import timeit
import random
import decimal
import timeit
import cv2
from matplotlib import pyplot as plt

#img = cv2.imread('Images/sample0.jpeg',0)
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#plt.subplot(1,2,2),plt.imshow(sobelx,cmap = 'gray')
#plt.title('Image with Sobel (x)'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

#plt.show()

#cv2.imwrite("sample_sobelx.jpeg", sobelx)


a = np.array([1,0,0])
b = a
c = np.array()
print np.append([c,a])