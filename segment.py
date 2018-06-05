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
from skimage.transform import resize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# def loadImages():
# 	image_array_ravel = []
# 	label_array_ravel = []
# 	base='test/'
# 	directories=os.listdir(base)
# 	directories.sort()
# 	#print(directories)
# 	counter=0
# 	for folder in directories:
# 		for filename in os.listdir(base+folder):
# 			if 'depth' in filename:
# 				img = cv.imread(base+folder+'/'+filename,0)
# 				# img = cv.resize(img, dsize=(73, 128), interpolation=cv.INTER_CUBIC)
# 				img=np.array(img)
# 				image_array_ravel.append(img.ravel())
# 				label_array_ravel.append(folder)
# 	return np.array(image_array_ravel),np.array(label_array_ravel)


def loadTest(base):
	image_array_ravel = []
	label_array_ravel = []
	directories=os.listdir(base)
	directories.sort()
	for folder in directories:
		for filename in os.listdir(base+folder):
			if 'jpg' in filename:
				img=cv.imread(base+folder+'/'+filename, 0)
				img=resize(img, (73,128))
				img=np.array(img)
				image_array_ravel.append(img.ravel())
				label_array_ravel.append(filename[0])

	return image_array_ravel, label_array_ravel

image_array_ravel,label_array_ravel=loadTest('./submission/crop/')
# print(image_array_ravel.shape)
# print(label_array_ravel.shape)


x_train, x_test, y_train, y_test=train_test_split(image_array_ravel,label_array_ravel, test_size=0.1, random_state=2)

# print("Applying Without Transform")
# clf=RandomForestClassifier()
# clf.fit(x_train, y_train)
# print(clf.score(x_test,y_test))


# print("Applying PCA Transform")
# PCA = PCA(n_components=20)
# pca_train = PCA.fit(x_train)
# comp=pca_train.components_.T
# pca_train=np.dot(x_train, comp)
# pca_test=np.dot(x_test, comp)
# classif = RandomForestClassifier()
# classif.fit(pca_train, y_train)
# print(classif.score(pca_test,y_test))


# print("Applying LDA Transformation")
# lda=LDA().fit(x_train, y_train)
# x_train_transform=lda.transform(x_train)
# x_test_transform=lda.transform(x_test)
# classif=RandomForestClassifier()
# classif.fit(x_train_transform, y_train)
# print(classif.score(x_test_transform, y_test))



image_array_ravel,label_array_ravel=loadTest('./crop_skin/')

x_train, x_test, y_train, y_test=train_test_split(image_array_ravel,label_array_ravel, test_size=0.1, random_state=2)

print("Accuracy After Segmentation")

print("Applying Without Transform")
clf=RandomForestClassifier()
clf.fit(x_train, y_train)
print(clf.score(x_test,y_test))


print("Applying PCA Transform")
PCA = PCA(n_components=20)
pca_train = PCA.fit(x_train)
comp=pca_train.components_.T
pca_train=np.dot(x_train, comp)
pca_test=np.dot(x_test, comp)
classif = RandomForestClassifier()
classif.fit(pca_train, y_train)
print(classif.score(pca_test,y_test))

print("Applying LDA Transformation")
lda=LDA().fit(x_train, y_train)
x_train_transform=lda.transform(x_train)
x_test_transform=lda.transform(x_test)
classif=RandomForestClassifier()
classif.fit(x_train_transform, y_train)
print(classif.score(x_test_transform, y_test))



