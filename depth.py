import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.misc

img = cv.imread('./dataset5/A/f/color_5_0009.png',0)
depth=cv.imread('./dataset5/A/f/depth_5_0009.png',0)
img=np.array(img)
depth=np.array(depth)
print(img.shape, depth.shape)
img_r=img.ravel()
depth_r=depth.ravel()
uniq=np.unique(depth_r)
print(uniq)
n=len(img_r)

for i in range(0,n):
	if(depth_r[i]==3 or depth_r[i]==2):
		img_r[i]=255
	else:
		img_r[i]=0
ind=1
ir=img_r.reshape(img.shape[0], img.shape[1])
scipy.misc.toimage(ir).save('./maps/'+str(ind)+'.png')


