import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.misc
import os


mini=[]
def loadImages():
	image_array = []
	shape_array = []
	label_array = []
	segment_array=[]
	base='./dataset5/A/'
	base2='./test/'
	directories=os.listdir(base)
	directories.sort()
	#print(directories)
	counter=0
	for folder in directories:
		for filename in os.listdir(base+folder):
			#print(filename)
			if 'depth' in filename:
				img = cv.imread(base+folder+'/'+filename,0)
				img = cv.resize(img, dsize=(73, 128), interpolation=cv.INTER_CUBIC)
				img=np.array(img)
				shape_array.append(img.shape)
				image_array.append(img.ravel())
				image=image_array[counter]
				n=len(image)
				for j in range(n):
					if(image[j]==2 or image[j]==3):
						image[j]=255
					else:

						image[j]=0
				print(image.shape)
				segment_array.append(image)
				ir=segment_array[counter].reshape(shape_array[counter][0], shape_array[counter][1])
				if not os.path.exists(base2+folder):
					os.makedirs(base2+folder)
				scipy.misc.toimage(ir).save(base2+folder+'/'+filename)
				label_array.append(folder)
				counter+=1
	return np.array(image_array), np.array(shape_array), np.array(label_array)


image_array,shape_array,label_array=loadImages()
print(image_array.shape)
print(label_array.shape)

print(shape_array)
a=0
b=0
minimumm=0
for i in shape_array:
	if(i[0]*i[1]>=minimumm):
		minimumm=i[0]*i[1]
		a=i[0]
		b=i[1]

print(minimumm)
print(a,b)

