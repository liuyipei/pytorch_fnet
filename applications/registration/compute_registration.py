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


def convert_matrix2tform(M):
	tform = "%f %f %f %f %f %f"%(M[0][0], M[1][0], M[0][1], M[1][1], M[0][2], M[1][2])
	return tform

def transformAffine(fp,A):
		
		fp = np.vstack((fp,np.ones(fp.shape[1])))
		print(fp.shape)
		print(A.shape)
		newfp = np.dot(A,fp)
		return newfp.T



def compute_m_from_optflow(flowx,flowy):
	
	threshx = np.abs(flowx.flatten())  
	threshy = np.abs(flowy.flatten()) 
	threshx = [i for i,v in enumerate(threshx) if (v > 0)]
	threshy = [i for i,v in enumerate(threshy) if (v > 0)]	
	thresh = np.union1d(threshx,threshy)


	imgsz = flowx.shape[0]
	x,y = np.meshgrid(range(imgsz),range(imgsz))
	
	x = x.flatten()
	y = y.flatten()
	fx = flowx.flatten()
	fy = flowy.flatten()

	x = np.asarray([x[i] for i in thresh])
	y = np.asarray([y[i] for i in thresh])
	fx = np.asarray([fx[i] for i in thresh])
	fy = np.asarray([fy[i] for i in thresh])


	P = (np.vstack((x, y)))
	P = P.astype(np.float32)
	
	Q = (np.vstack((x + fx, y + fy)))
	Q = Q.astype(np.float32)


	M = cv2.estimateRigidTransform(P.T,Q.T,False)

	return M


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


	#img1 = cv2.GaussianBlur(img1,(5,5),0)
	#img2 = cv2.GaussianBlur(img2,(5,5),0)

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

def compute_SIFT(img1,img2,outImg,signal):

	if signal is not None:
		f32 = Image.open(signal)
		sig = np.asarray(f32.convert('I;16'))
		sig = cv2.convertScaleAbs(sig)
		print(sig.shape)
		#sig = sig[2:1502,2:1502]
		cv2.imwrite("data/registration/SIGNAL.jpg",sig)

	cv2.imwrite("data/registration/IMG1.jpg",img1)
	cv2.imwrite("data/registration/IMG2.jpg",img2)
	

	sc = 0.1
	img1 = cv2.resize(img1, (0,0), fx=sc, fy=sc)
	img2 = cv2.resize(img2, (0,0), fx=sc, fy=sc)

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create(sigma=1.6,nOctaveLayers=7)
	
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)


	print(len(matches))


	# Apply ratio test
	good = []
	goodpts = []
	for m,n in matches:
	    if m.distance < 0.6*n.distance:
                good.append([m])
                goodpts.append(m)

	if outImg !="":
		#visualize matches
		img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
		cv2.imwrite(outImg,img3)

	print(len(goodpts))

	if len(goodpts)>3:
	    src_pts = np.float32([ kp1[m.queryIdx].pt  for m in goodpts ])
	    dst_pts = np.float32([ kp2[m.trainIdx].pt  for m in goodpts ])
	    M = cv2.estimateRigidTransform(src_pts, dst_pts,True)
	    model = RansacAffineModel()
	    M = RansacAffineModel.A_from_ransac(src_pts.T,dst_pts.T,model)[0]
	    

	else:
	    print("Not enough matches are found !")
	    M = None


	return M


def adjust_tform_scale(tform,scale):
	M = convert_tform2matrix(tform)
	M[0][2] = M[0][2]*scale
	M[1][2] = M[1][2]*scale
	tform = convert_matrix2tform(M)
	return tform


def compute_error(tform_ground, tform_predict):
	
	x,y = np.meshgrid(range(1500),range(1500))
	P = (np.vstack((x.flatten(), y.flatten())))
	pts_ground = transformAffine(P,convert_tform2matrix(tform_ground))
	pts_predict = transformAffine(P,convert_tform2matrix(tform_predict))


	E = np.sqrt(sum((pts_ground.T-pts_predict.T))**2)

	meanE = np.mean(E)
	return meanE

def compute_registration(filename, template, outImg = "", signal = None):


	f32 = Image.open(template)
	img11 = np.asarray(f32.convert('I;16'))
	img11 = cv2.convertScaleAbs(img11)
	img11 = cv2.equalizeHist(img11)	
	img1 = cv2.resize(img11, (0,0), fx=0.1, fy=0.1)
	img2 = cv2.imread(filename,0) 
	img2 = cv2.equalizeHist(img2)


	#img1 = cv2.GaussianBlur(img1,(5,5),0)
	#img2 = cv2.GaussianBlur(img2,(5,5),0)
	
	#template match
	degree,maxloc,A = template_match_pyramid(img1, img2, 6,10)
	print (degree)
	print (maxloc)
	print(A)
	M = A*0.1
	M [0][2] = A[0][2] + maxloc[0]
	M [1][2] = A[1][2] + maxloc[1]

	


	#calculate image
	#rows,cols = img11.shape
	#iM = cv2.invertAffineTransform(M)
	#dst = cv2.warpAffine(img2,iM,(cols,rows))
	#cv2.imwrite("data/registration/warped.jpg",dst)
	
	#improve with SIFT
	#sz = 150
	#img2crop = img2
	#img2crop = img2[int(round(M [1][2])) - sz: int(round(M [1][2]))+sz, int(round(M [0][2])) - sz: int(round(M [0][2]))+sz]
	#M_sift = compute_SIFT(img1,img2crop,"data/registration/matches.jpg")
	#M_sift = compute_SIFT(img11,dst,"data/registration/matches.jpg",signal)
	#print("This is M and M Sift")
	#print(M)	
	#print (M_sift)
	

	#improve with subpixel
	rows,cols = img11.shape
	iM = cv2.invertAffineTransform(M)
	dst = cv2.warpAffine(img2,iM,(cols,rows))
	cv2.imwrite("data/registration/testwarp.jpg",dst)
	flow = cv2.calcOpticalFlowFarneback(dst, img11,None, 0.5, 1, 10, 5, 5, 5.0, 0)
	cv2.imwrite('data/registration/flowimage0.png',flow[:,:,0])
	cv2.imwrite('data/registration/flowimage1.png',flow[:,:,1])
	

	M2 = compute_m_from_optflow(flow[:,:,0],flow[:,:,1])
	



	#M5 = compute_SIFT(img11,dst,"data/registration/matches.jpg")



	print("This is M: ")
	print(M)
	print("This is M2: ")
	print(M2)

	#print(M.shape)

	M = np.concatenate((M, [[0,0,1]]), axis=0)
	M2 = np.concatenate((M2, [[0,0,1]]), axis=0)
	combined = np.matmul(M, M2)


	print ("This is combined: ")
	print (combined)

	#print (combined[:2])

	#M3 = combined[:2]
	#iM = cv2.invertAffineTransform(M3)
	#dst = cv2.warpAffine(img2,iM,(cols,rows))
	#cv2.imwrite("data/registration/testwarp1.jpg",dst)

	return combined[:2]
	#return M[:2]


	#return M

def register_all(opts):
	mylist = pd.read_csv(opts.csvfile)
	i = 0
	for index,row in mylist.iterrows():
		if index == 1:
			filename = row['mbp_file'][5:]	
			signal =  row['path_signal']
			print (signal)
		
			template = opts.datadirectory + "/" + row['path_prediction_unpropped']
			print(filename)
			print (template)
			outimage = template.replace("prediction_unpropped.tiff","matches.tiff")
			print (outimage)
			#exit(0)
			M = compute_registration(filename,template,outimage,signal)
			if M is None:
				renderM = '0.0 0.0 0.0 0.0 0.0 0.0'
			else:
				renderM = '%f %f %f %f %f %f'%(M[0][0], M[1][0], M[0][1], M[1][1], M[0,2], M[1][2])
			mylist.loc[index,'predicted_tform'] = renderM
			print("This is render M ")
			print(renderM)
			ground_forrest = adjust_tform_scale(row['ground_truth_forrest'],0.1)
			ground_sharmi = adjust_tform_scale(row['ground_truth_sharmi'],0.1)
			print(ground_forrest)
			print(compute_error(ground_forrest, renderM))
			print(ground_sharmi)
			print(compute_error(ground_sharmi, renderM))



			#i = i + 1
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


	
