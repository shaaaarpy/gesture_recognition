import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
print(test_data.shape)

PCA = PCA(n_components=14)
pca_train = PCA.fit(train_data)
comp=pca_train.components_.T
pca_train=np.dot(train_data, comp)
pca_test=np.dot(test_data, comp)
classif = RandomForestClassifier()
classif.fit(pca_train, train_labels)
print(classif.score(pca_test,test_labels))

