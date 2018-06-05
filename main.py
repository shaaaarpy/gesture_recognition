import gesture_model
import pickle
import gzip
from skimage import io
from skimage.transform import resize
import os

def pickleModel(model, file_name):
	pickle.dump(model, gzip.open(file_name, 'wb'))


def loadClassifier(file):
	f = open(file, 'r+')
	model = pickle.load(f)
	f.close()
	return model


def pickleLoad(file_name):
	return pickle.load(gzip.open(file_name, 'rb'))

direc='../dataset3/Dataset/'

gs = gesture_model.image_segment()
userlist=['user_10','user_3','user_4','user_5','user_6','user_7','user_9']

user_tr = userlist
user_te = userlist[-1:]


# gs.fit(user_tr)

# pickleModel(gs, "sign_detector.pkl.gz")

# print "The GestureRecognizer is saved to disk"

# new_gr = pickleLoad("sign_detector.pkl.gz")
handDetector=loadClassifier('./handDetector.pkl')
new_gr=gesture_model.image_segment()
print("Load Complete")

directories=os.listdir(direc)
for c in directories:
	label=c
	images=os.listdir(direc+c)
	for im in images:
		if im.split('.')[-1]=='jpg':
			img=io.imread(direc+c+'/'+im)
			filepath='./crop/'+c+'/'+im
			new_gr.hand_segment(img, handDetector, filepath)
		
# img=io.imread('./Y2.jpg')
# new_gr.hand_segment(img, handDetector, './cropped.png')
# # new_gr=gs
# img=resize(img, (160,120))
# signDetector=loadClassifier('./signDetector.pkl')
# label_encoder=loadClassifier('./label_encoder.pkl')
