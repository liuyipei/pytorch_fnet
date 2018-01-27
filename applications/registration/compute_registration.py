import os
import tifffile
import cv2
import numpy as np
import argparse
import pandas as pd

def compute_registration(queryImg, templateImg, outImg = ""):


	img1 = cv2.imread(template,0)          # queryImage
	img2 = cv2.imread(filename,0) # trainImage

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()


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
	    if m.distance < 0.5*n.distance:
                good.append([m])
                goodpts.append(m)

	if outImg !="":
		#visualize matches
		img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
		cv2.imwrite(out,img3)



	#calculate affine transform
	if len(good)>3:
	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodpts ])
	    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in goodpts ])
	    M = cv2.estimateRigidTransform(src_pts, dst_pts,True)
	    

	else:
	    print("Not enough matches are found !")
	    M = None

	return M

def parse_csv(csvfile):
	csvlist = pd.read_csv(path_losses_csv)
	return csvlist

def register_all(opts):
	mylist = parse_csv(opts.csvfile)
	for line in mylist:
		#need to go over the csv thing with group !!!
		filename = line['LM_img']		
		template = line['predicted_img']
		M = compute_registration(filename,template,"")
		print(M)
	
if __name__ == '__main__':
	#parser = argparse.ArgumentParser()
        #parser.add_argument('--csvfile', type=Str, default=0, help='CSV FILE to process')
	#opts = parser.parse_args()
        #register_all(opts)

	filename = "data/registration/template.tif"
	template = "data/registration/tile.tif"
	out = "data/registration/out.jpg"
	M = compute_registration(filename,template,out)
	print(M)
	
