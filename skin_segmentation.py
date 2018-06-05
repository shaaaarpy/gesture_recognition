#Skin Segmentation Data Set from https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation

import numpy as np
import cv2

from sklearn import tree
from sklearn.cross_validation import train_test_split
from skimage import io
import os


def TrainTree(data, labels):
    data= np.reshape(data,(data.shape[0],1,3))
    data= cv2.cvtColor(np.uint8(data), cv2.COLOR_BGR2HSV)
    data= np.reshape(data,(data.shape[0],3))

    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.10, random_state=23)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainData, trainLabels)
    return clf

def ApplyToImage(path,file_path,data,labels):
    clf= TrainTree(data, labels)

    img= io.imread(path)
    
    data= np.reshape(img,(img.shape[0]*img.shape[1],3))
    data= np.reshape(data,(data.shape[0],1,3))
    data= cv2.cvtColor(np.uint8(data), cv2.COLOR_BGR2HSV)
    data= np.reshape(data,(data.shape[0],3))
    
    predictedLabels= clf.predict(data)
    
    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
    cv2.imwrite(file_path,((-(imgLabels-1)+1)*255))
    



data = np.genfromtxt('./data/Skin_NonSkin.txt', dtype=np.int32)
labels= data[:,3]
data= data[:,0:3]


direc='./crop/'
directories=os.listdir(direc)
for c in directories:
    label=c
    images=os.listdir(direc+c)
    for im in images:
        if im.split('.')[-1]=='jpg':
            img=io.imread(direc+c+'/'+im)
            filepath='./crop/'+c+'/'+im
            file_path='./crop_skin/'+c+'/'+im
            ApplyToImage(filepath,True,file_path,data,labels)
