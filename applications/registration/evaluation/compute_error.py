import os
import tifffile
import cv2
import numpy as np
import argparse
import pandas as pd
from RansacAffineModel import RansacAffineModel
from PIL import Image
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def convert_tform2matrix(tform):
	#print (tform)
	tform = tform.split()
	
	M = [[0, 0, 0 ],[0, 0, 0]]
	M[0][0] = float(tform[0])
	M[1][0] = float(tform[1])
	M[0][1] = float(tform[2])
	M[1][1] = float(tform[3])
	M[0][2] = float(tform[4])
	M[1][2] = float(tform[5])
	
	M = np.array(M)
	return M

def transformAffine(fp,A):
		
		fp = np.vstack((fp,np.ones(fp.shape[1])))
		#print(fp.shape)
		#print(A.shape)
		newfp = np.dot(A,fp)
		return newfp.T

def compute_error(tform_ground, tform_predict, img_size):
	
	x,y = np.meshgrid(range(1500),range(1500))
	P = (np.vstack((x.flatten(), y.flatten())))
	pts_ground = transformAffine(P,convert_tform2matrix(tform_ground))
	pts_predict = transformAffine(P,convert_tform2matrix(tform_predict))


	E = np.sqrt(sum((pts_ground.T-pts_predict.T))**2)

	meanE = np.mean(E)
	return meanE


def get_percent(img, bordersz):
	
	imgsize = img.shape[0]*img.shape[1]	
	img = img[bordersz: img.shape[0] - bordersz, bordersz: img.shape[1] - bordersz]
	
	ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)



	connectivity = 4  
	output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
	num_labels = output[0]
	labels = output[1]
	stats = output[2]
	


	for i in range(0,num_labels):
		if stats[i,4] <= 400:
			for p in range(stats[i,0],stats[i,0]+stats[i,2]):
				for q in range(stats[i,1],stats[i,1]+stats[i,3]):
					thresh[p,q] = 0

	return 100*np.count_nonzero(thresh)/imgsize 

def compute_all(opts):
	mylist = pd.read_csv(opts.csvfile)
	ind =0
	for index,row in mylist.iterrows():
		#print(index)
		tform_ground = row['ground_truth_forrest']
		tform_predicted = row['ground_truth_sharmi']
		imgfile = "/allen/aics/modeling/cheko/projects/pytorch_fnet/results/2d/registration_2/" + row['path_prediction_unpropped']
		f32 = Image.open(imgfile)
		img = np.asarray(f32.convert('I;16'))
		img = cv2.convertScaleAbs(img)
		#kernel = np.ones((7,7), np.uint8)
		#img_erosion = cv2.erode(img, kernel, iterations=1)
		img = cv2.equalizeHist(img)
		#cv2.imwrite("data/registration/equalized.jpg",img)
		#matplotlib.pyplot.hist(img.flatten())
		#matplotlib.pyplot.savefig("data/registration/histogram.jpg",dpi=500)
		percent = get_percent(img,20) 
		#if percent >= 1.0:
		E = compute_error(tform_ground,tform_predicted,1500)
		if E > 20:
			print("THIS IS THE CASSE: ")
			print(index,E)
			#print (index,E,percent)
		mylist.loc[index,'pixel_error'] = E
		mylist.loc[index,'content_percentage'] = percent
		ind = ind+1
		#else:
		#	mylist.loc[index,'pixel_error'] = None
		print(index)		
		#if index ==100:
		#		break
		
	mylist.to_csv(opts.outputcsvfile)
	print (ind)
	
if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--csvfile', default='mycsv.csv', help='CSV FILE to process')
        parser.add_argument('--outputcsvfile', default='mycsvout.csv', help='CSV FILE to output')
        opts = parser.parse_args()
        print(opts)  
        compute_all(opts)


	
