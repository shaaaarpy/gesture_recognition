import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PIL import Image
import scipy.misc

train_labels=[]
train_data=[]
test_labels=[]
test_data=[]


with open('./dataset/sign_mnist_train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        train_labels.append(int(row['label']))
        temp=[]
        col='pixel'
        for i in range(1,785):
        	temp.append(int(row[col+str(i)]))
        temp=np.array(temp)
        train_data.append(temp)

train_data=np.array(train_data)
train_labels=np.array(train_labels)
print(train_data.shape)


with open('./dataset/sign_mnist_test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_labels.append(int(row['label']))
        temp=[]
        col='pixel'
        for i in range(1,785):
        	temp.append(int(row[col+str(i)]))
        temp=np.array(temp)
        test_data.append(temp)

test_data=np.array(test_data)
test_labels=np.array(test_labels)
print(test_data.shape)

ind=0
for i in train_data:
    ind+=1
    ir=i.reshape(28,28)
    scipy.misc.toimage(ir, cmin=0, cmax=255).save('./image2/'+str(ind)+'.png')
    # scipy.misc.toimage('./images/'+str(ind)+'.png', ir)


PCA = PCA(n_components=20)
pca_train = PCA.fit(train_data)
comp=pca_train.components_.T
pca_train=np.dot(train_data, comp)
pca_test=np.dot(test_data, comp)
classif = RandomForestClassifier()
classif.fit(pca_train, train_labels)
print(classif.score(pca_test,test_labels))

