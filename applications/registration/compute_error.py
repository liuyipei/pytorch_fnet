import os
import tifffile
import cv2
import numpy as np
import argparse
import pandas as pd
from RansacAffineModel import RansacAffineModel
from PIL import Image
from scipy import signal


def convert_tform2matrix(tform):
	print (tform)
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
		print(fp.shape)
		print(A.shape)
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

def compute_all(opts):
	mylist = pd.read_csv(opts.csvfile)
	for index,row in mylist.iterrows():

		tform_ground = row['tform']
		tform_predicted = row['predicted_tform']
		E = compute_error(tform_ground,tform_predicted,150)
		print (E)
		mylist.loc[index,'pixel_error'] = E
		
	mylist.to_csv(opts.outputcsvfile)
	
	
if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--csvfile', default='mycsv.csv', help='CSV FILE to process')
        parser.add_argument('--outputcsvfile', default='mycsvout.csv', help='CSV FILE to output')
        opts = parser.parse_args()
        print(opts)  
        compute_all(opts)


	
