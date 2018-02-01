import os
import tifffile
import cv2
import numpy as np
import argparse
import pandas as pd
from RansacAffineModel import RansacAffineModel
from PIL import Image
from scipy import signal

def ransac(src_pts, dest_pts):
	print("test")
	
def prune_matches(matches):
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodpts ])
		
def template_match_pyramid(img1, img2, scales, initdelta):
	mindeg = 0
	maxdeg = 360
	delta = initdelta
	for i in range (0,scales):

		print ("This is i: %d"%i)
		degree,maxloc,A = template_match(img1, img2, mindeg, maxdeg, delta)
		mindeg = degree - delta
		maxdeg = degree + delta		
		delta = delta*0.1
		
	return degree,maxloc,A

def template_match(img1, img2, mindeg, maxdeg, delta):
	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

	cur_val = -100000000000
	degree = mindeg
	maxloc = [0,0]
	method = eval(methods[1])
	
	alldegrees = [mindeg]
	curdeg = mindeg

	print("creating all degrees: %f %f %f"%(mindeg, maxdeg, delta))

	while curdeg+delta <= maxdeg:
		alldegrees.append(curdeg+delta)	
		curdeg = curdeg + delta
		print(alldegrees)
	print (alldegrees)

	for deg in alldegrees:
	#for deg in range(mindeg,maxdeg,delta):
		rows,cols = img1.shape
		A = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
		dst = cv2.warpAffine(img1,A,(cols,rows))
		
		res = cv2.matchTemplate(img2,dst,method)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		if max_val > cur_val:
			print ("DEGREE: %d, MAX VAL: %f "%(deg, max_val))		
			print (max_loc)
			cur_val = max_val
			maxloc = max_loc
			degree = deg
	return degree,maxloc, A

def compute_SIFT(img1,img2):

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create(sigma=3.0,edgeThreshold=20)
	
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# Apply ratio test
	good = []
	goodpts = []
	for m,n in matches:
	    if m.distance < 0.9*n.distance:
                good.append([m])
                goodpts.append(m)

	if outImg !="":
		#visualize matches
		img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
		cv2.imwrite(outImg,img3)



	if len(good)>3:
	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodpts ])
	    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in goodpts ])
	    M = cv2.estimateRigidTransform(src_pts, dst_pts,True)
	    model = RansacAffineModel()
	    M = RansacAffineModel.A_from_ransac(src_pts.T,dst_pts.T,model)[0]
	    

	else:
	    print("Not enough matches are found !")
	    M = None


	return M

def compute_registration(filename, template, outImg = ""):


	f32 = Image.open(template)
	img1 = np.asarray(f32.convert('I;16'))
	img1 = cv2.convertScaleAbs(img1)
	img1 = cv2.equalizeHist(img1)	
	img1 = cv2.resize(img1, (0,0), fx=0.1, fy=0.1)
	img2 = cv2.imread(filename,0) 
	img2 = cv2.equalizeHist(img2)
		
	
	if 1 ==1:
		degree,maxloc,A = template_match_pyramid(img1, img2, 3,5)
		print (degree)
		print (maxloc)
		print(A)
		M = A*0.1
		M [0][2] = A[0][2] + maxloc[0]
		M [1][2] = A[1][2] + maxloc[1]
	
	return M


def register_all(opts):
	mylist = pd.read_csv(opts.csvfile)
	i = 0
	for index,row in mylist.iterrows():
		if index >-1:
			filename = row['mbp_file'][5:]	
			template = opts.datadirectory + "/" + row['path_prediction_unpropped']
			print(filename)
			print (template)
			outimage = template.replace("prediction_unpropped.tiff","matches.tiff")
			print (outimage)
			#exit(0)
			M = compute_registration(filename,template,outimage)
			if M is None:
				renderM = '0.0 0.0 0.0 0.0 0.0 0.0'
			else:
				renderM = '%f %f %f %f %f %f'%(M[0][0], M[1][0], M[0][1], M[1][1], M[0,2], M[1][2])
			mylist.loc[index,'predicted_tform'] = renderM
			print(renderM)
			print(row['tform'])
			i = i + 1
			#if i>1:
			#	break
	mylist.to_csv(opts.outputcsvfile)
	
	
if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--csvfile', default='mycsv.csv', help='CSV FILE to process')
        parser.add_argument('--datadirectory', default='mydir', help='Data directory where results live')
        parser.add_argument('--outputcsvfile', default='mycsvout.csv', help='CSV FILE to output')
        opts = parser.parse_args()
        print(opts)  
        register_all(opts)


	
