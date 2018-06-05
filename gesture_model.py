

from skimage import data, io, filters
import gzip

from skimage.feature import hog 
from skimage.transform import resize

import pickle
import pandas as pd
import time

import glob
from sklearn.naive_bayes import MultinomialNB,GaussianNB
import random
import csv
from os import listdir
import numpy as np
from sklearn.ensemble import  RandomForestClassifier
from skimage.color import rgb2gray

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from os.path import isfile, join
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import json
from skimage.transform import rescale
#https://medium.freecodecamp.org/weekend-projects-sign-language-and-static-gesture-recognition-using-scikit-learn-60813d600e79

def dumpclassifier(model,file_name):
	with open(file_name, 'wb') as f:
		pickle.dump(model, f)


direc='../dataset3/Dataset/'
direc_path='/home/mayank/Sem-6/SML/project/gesture_recognition/dataset3/Dataset/'


image_dim=[320,240]

handDetector=None
signDetector=None
label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])

def downloadfiles(filenames):
	dir_files = {}
	for x in filenames:
		print(x)
		dir_files[x]=io.imread(direc_path+x)
	return dir_files

def crop(img,x1,x2,y1,y2):
	crp=img[y1:y2,x1:x2]
	resize_x=128
	resize_y=128
	crp=resize(crp,((resize_x,resize_y)))
	return crp

def overlapping_area(detection_1, detection_2):
	
	y11 = detection_1[1]
	x22 = detection_2[0] + detection_2[3]
	x21 = detection_2[0]
	x11 = detection_1[0]
	y12 = detection_1[1] + detection_1[4]
	x12 = detection_1[0] + detection_1[3]
	
	y21 = detection_2[1]
	y22 = detection_2[1] + detection_2[4]
	temp1=min(x12, x22)
	temp2=max(x11, x21)
	temp3=temp1-temp2
	x_overlap = max(0,temp3)
	temp4=min(y12, y22)
	temp5=max(y11, y21)
	temp6=temp4-temp5
	y_overlap = max(0,temp6)

	overlap_area = x_overlap * y_overlap
	area_1 = detection_1[3] * detection_2[4]
	area_2 = detection_2[3] * detection_2[4]
	temp8=area_1+area_2
	total_area = temp8 - overlap_area
	temp7=overlap_area / float(total_area)
	return temp7
	
def handNonhand_imglist(boundbox,imgset):
	hand_images =[]
	non_hand_images =[]
	non_hand_labels=[]
	hand_labels=[]
	overlap_thresh=0.5

	for name_img in boundbox.image:
		curr_image = boundbox[boundbox['image']==name_img].values[0]
		x1 = curr_image[1]
		y1 = curr_image[2]
		x2 = curr_image[3]
		y2 = curr_image[4]
		side_len = curr_image[5]
		hand_bool=curr_image[6]
		least_hand=0

		boundary_1 = [x1,y1,0,side_len,side_len]
		
		cropped_image_rgb=crop(imgset[name_img],x1,x2,y1,y2)
		
		cropped_image_grey=rgb2gray(cropped_image_rgb)

		cropped_image_hog=hog(cropped_image_grey)
		hand_images.append(cropped_image_hog)
		hand_labels.append(1)
		while least_hand <= 1:
			x1_new = random.randint(0,image_dim[0]-side_len)
			y1_new = random.randint(0,image_dim[1]-side_len)
			x2_new = x1_new+side_len
			y2_new=y1_new+side_len
			crp = crop(imgset[name_img],x1_new,x2_new,y1_new,y2_new)
			crp_grey=rgb2gray(crp)
			crop_hog_vec = hog(crp_grey)
			boundary_2 = [x1_new,y1_new, 0, side_len, side_len]

			z = overlapping_area(boundary_1,boundary_2)
			if least_hand <= 1 and z <= overlap_thresh:
				non_hand_images.append(crop_hog_vec)
				non_hand_labels.append(0)
				least_hand += 1
			if least_hand== 1:
				break
	hand_labels.extend(non_hand_labels)
	hand_images.extend(non_hand_images)
	return hand_images,hand_labels



def do_hardNegativeMining(cached_window,boundbox, imgset, model, step_x, step_y):
	print("HNM doing")
	lis = []
	labels=[]
	false_positives = 0
	true_positives=0
	for name_img in boundbox.image:
		true_positives+=1
		tupl = boundbox[boundbox['image']==name_img].values[0]
		x1 = tupl[1]
		y1 = tupl[2]
		x2 = tupl[3]
		y2 = tupl[4]
		side = tupl[5]
		boundary_1 = [x1,y1,0,side,side]
		for x in range(0,image_dim[0]-side,step_x):
			for y in range(0,image_dim[1]-side,step_y):
				boundary_2 = [x,y,0,side,side]
				z = overlapping_area(boundary_1,boundary_2)
				prediction = model.predict([cached_window[str(name_img)+str(x)+str(y)]])[0]
				true_positives+=1
				if prediction == 1 and z<=0.5:
					lis.append(cached_window[str(name_img)+str(x)+str(y)])
					labels.append(0)
					true_positives-=1
					false_positives += 1

	return lis,labels, false_positives


def caching(imgset, boundbox, step_x, step_y):
	list_dic_of_hogs = []
	dic = {}
	i = 0
	for img in boundbox.image:
		tupl = boundbox[boundbox['image']==img].values[0]
		x1 = tupl[1]
		y1 = tupl[2]
		side = tupl[5]
		i += 1 
		imaage = imgset[img]
		for x in range(0,320-side,step_x):
			for y in range(0,240-side,step_y):
				cropped_image=crop(imaage,x,x+side,y,y+side)
				cropped_image_grey=rgb2gray(cropped_image)
				cropped_image_hog=hog(cropped_image_grey)
				dic[str(img+str(x)+str(y))]=cropped_image_hog
	return dic 

def improve_Classifier_using_HNM(hog_list, label_list, boundbox, imgset, threshold=20, max_iterations=15): 
	print ("Hard Negative Mining")
	no_of_false_positives = 1000000000
	i = 1
	step_x = image_dim[0]/10
	step_y = image_dim[1]/10

	classi  = MultinomialNB()
	cached_wind = caching(imgset, boundbox, step_x, step_y)

	while (i>0):
		
		model = classi.partial_fit(hog_list, label_list, classes = [0,1])

		ret = do_hardNegativeMining(cached_wind,boundbox, imgset, model, step_x=step_x, step_y=step_y)
		no_of_false_positives = ret[2]
		label_list = ret[1]
		hog_list = ret[0]
		
		if no_of_false_positives == 0 or no_of_false_positives<=threshold or i>max_iterations:
			return model
		
		i += 1




def image_pyramid_step(model, img, scale):
	y_border = rescaled_img.shape[0]
	detected_box = []
	side = 128
	rescaled_img = rescale(img, scale)
	x_border = rescaled_img.shape[1]
	max_confidence_seen = -1
 
	for x in range(0,x_border-side,32):
		for y in range(0,y_border-side,24):
			cropped_img = crop(rescaled_img,x,x+side,y,y+side)
			cropped_img_grey=rgb2gray(cropped_img)
			cropped_img_hog=hog(cropped_img_grey)

			confidence = model.predict_proba([cropped_img_hog])

			if confidence[0][1] > max_confidence_seen:
				detected_box = [x, y, confidence[0][1], scale]
				max_confidence_seen = confidence[0][1]

	return detected_box



def non_max_suppression_fast(boxes, overlapThresh):
	print "Perfmoring NMS:"

	flag=0
	if len(boxes) == 0:
		flag=1

	if flag==1:
		return []
	

	boxes = boxes.astype("float")
	
	pick = []

	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	s = boxes[:,4]
	temp1=(x2 - x1 + 1)
	temp2=(y2 - y1 + 1)
	temp3=temp1*temp2
	area = temp3
	idxs = np.argsort(s)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		tempa=x1[idxs[:last]]
		tempb=y1[idxs[:last]]
		tempc=x2[idxs[:last]]
		tempd= y2[idxs[:last]]
		xx1 = np.maximum(x1[i], tempa)
		yy1 = np.maximum(y1[i],tempb )
		xx2 = np.minimum(x2[i],tempc )
		yy2 = np.minimum(y2[i],tempd)
		tempe=xx2 - xx1 + 1
		tempf=yy2 - yy1 + 1
		width = np.maximum(0, tempe)
		height = np.maximum(0, tempf)

		overlap = (widtheight * h) / area[idxs[:last]]


		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	tempe=boxes[pick].astype("int")
	return tempe

class image_segment(object):

	def fit(self, user_folder):

		print "Fit starts"
		boundbox=pd.DataFrame()
		my_list=[]
		for user in user_folder:
			my_list.append(pd.read_csv(direc+user+'/'+user+'_loc.csv',index_col=None,header=0))
		boundbox = pd.concat(my_list)
		boundbox['side']=boundbox['bottom_right_x']-boundbox['top_left_x']
		boundbox['hand']=1

		imageset = downloadfiles(boundbox.image.unique())

		hog_list,label_list=handNonhand_imglist(boundbox,imageset)
		print("Hog Loaded")

		train_data = []
		train_label = []

		for user in user_folder:
			user_images = glob.glob(direc_path+user+'/*.jpg')

			boundingbox = pd.read_csv(direc_path+user+'/'+user+'_loc.csv')
			
			for rows in boundingbox.iterrows():
				cropped_img = crop(imageset[rows[1]['image']], rows[1]['top_left_x'], rows[1]['bottom_right_x'], rows[1]['top_left_y'], rows[1]['bottom_right_y'])
				cropped_image_grey=rgb2gray(cropped_img)
				cropped_image_hog=hog(cropped_image_grey)
				train_data.append(cropped_image_hog.tolist())
				train_label.append(rows[1]['image'].split('/')[1][0])

		train_label=label_encoder.fit_transform(train_label)

		print("Multiclass data loaded")

		handDetector = improve_Classifier_using_HNM(hog_list, label_list, boundbox, imageset, threshold=20, max_iterations=15)
		print("HNM Performed")

		dumpclassifier(handDetector, 'handDetector.pkl')

	def hand_segment(self,image, handDetector, filepath):
		scale_factors=[1]
		detected_box=[]
		scaled_sides=[]
		ind=0
		max_suppression=[]
		suppression_thresh=0.4
		for f in scale_factors:
			detected_box.append(image_pyramid_step(handDetector,image,f))
			# print("x:" ,image_pyramid_step(handDetector, image, f)[0])
			detected_box[ind][0]/=f
			detected_box[ind][1]/=f
			scaled_sides.append(128/f)
			lis_tup=[detected_box[ind][0], detected_box[ind][1], detected_box[ind][0]+scaled_sides[ind],detected_box[ind][1]+scaled_sides[ind],detected_box[ind][2]]
			max_suppression.append(lis_tup)
			ind+=1

		max_suppression=np.array(max_suppression)
		final_images=non_max_suppression_fast(max_suppression, suppression_thresh)
		best_segment=final_images[0]
		x1_b=best_segment[0]
		y1_b=best_segment[1]
		x2_b=best_segment[2]
		y2_b=best_segment[3]
		side_b=x2_b-x1_b
		position=[x1_b, y1_b, x2_b, y2_b]

		cropped_image_rgb=crop(image,x1_b,x2_b,y1_b,y2_b)
		io.imsave(filepath,cropped_image_rgb)
		# cropped_image_grey=rgb2gray(cropped_image_rgb)
		# cropped_image_hog=hog(cropped_image_grey)

		# prediction=signDetector.predict_proba([cropped_image_hog])[0]
		# return position,final_prediction

















