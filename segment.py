import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.misc
import os
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def loadImages():
	image_array_ravel = []
	label_array_ravel = []
	base='test/'
	directories=os.listdir(base)
	directories.sort()
	#print(directories)
	counter=0
	for folder in directories:
		for filename in os.listdir(base+folder):
			if 'depth' in filename:
				img = cv.imread(base+folder+'/'+filename,0)
				# img = cv.resize(img, dsize=(73, 128), interpolation=cv.INTER_CUBIC)
				img=np.array(img)
				image_array_ravel.append(img.ravel())
				label_array_ravel.append(folder)
	return np.array(image_array_ravel),np.array(label_array_ravel)

image_array_ravel,label_array_ravel=loadImages()
print(image_array_ravel.shape)
print(label_array_ravel.shape)


x_train, x_test, y_train, y_test=train_test_split(image_array_ravel,label_array_ravel, test_size=0.1, random_state=2)

PCA = PCA(n_components=20)
pca_train = PCA.fit(x_train)
comp=pca_train.components_.T
pca_train=np.dot(x_train, comp)
pca_test=np.dot(x_test, comp)
classif = RandomForestClassifier()
classif.fit(pca_train, y_train)
print(classif.score(pca_test,y_test))





