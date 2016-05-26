import cv2
import numpy as np
import math




img = cv2.imread("subregion.jpeg")
length = img.shape[0]
width = img.shape[1]
nclass = length*width
rgbvalue = np.ceil(np.asarray(img)/255*nclass)
rgbhist = np.empty((nclass,3))
for cc in range(nclass):
    rgbhist[cc,0] = np.sum(rgbvalue[:,:,0]==cc)
    rgbhist[cc,1] = np.sum(rgbvalue[:,:,1]==cc)
    rgbhist[cc,2] = np.sum(rgbvalue[:,:,2]==cc)

print len(rgbhist)