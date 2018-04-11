import os
import tifffile
import cv2
import numpy as np
import argparse
import pandas as pd
from RansacAffineModel import RansacAffineModel
from PIL import Image
from scipy import signal
import random

		
def get_percent(imgfile, bordersz):


	f32 = Image.open(imgfile)
	img = np.asarray(f32.convert('I;16'))
	img = cv2.convertScaleAbs(img)
	img = cv2.equalizeHist(img)	

	imgsize = img.shape[0]*img.shape[1]	
	img = img[bordersz: img.shape[0] - bordersz, bordersz: img.shape[1] - bordersz]
	
	ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


	#remove 20 x 20 size blobs

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


def compute_registration(filename, template, outImg = "", signal = None):


	f32 = Image.open(template)
	img11 = np.asarray(f32.convert('I;16'))
	img11 = cv2.convertScaleAbs(img11)
	img11 = cv2.equalizeHist(img11)	
	img1 = cv2.resize(img11, (0,0), fx=0.1, fy=0.1)
	img2 = cv2.imread(filename,0) 
	img2 = cv2.equalizeHist(img2)


	img1 = cv2.GaussianBlur(img1,(5,5),0)
	img2 = cv2.GaussianBlur(img2,(5,5),0)

	percent = get_percent(img1,20) 


	
	#template match
	degree,maxloc,A = template_match_pyramid(img1, img2, 3,5)
	print (degree)
	print (maxloc)
	print(A)
	M = A*0.1
	M [0][2] = A[0][2] + maxloc[0]
	M [1][2] = A[1][2] + maxloc[1]

	


	#calculate image
	rows,cols = img11.shape
	iM = cv2.invertAffineTransform(M)
	dst = cv2.warpAffine(img2,iM,(cols,rows))
	
	#improve with SIFT
	#sz = 150
	#img2crop = img2
	#img2crop = img2[int(round(M [1][2])) - sz: int(round(M [1][2]))+sz, int(round(M [0][2])) - sz: int(round(M [0][2]))+sz]
	#M_sift = compute_SIFT(img1,img2crop,"data/registration/matches.jpg")
	M_sift = compute_SIFT(img11,dst,"data/registration/matches.jpg",signal)
	print("This is M and M Sift")
	print(M)	
	print (M_sift)
	

	#improve with subpixel
	#rows,cols = img11.shape
	#iM = cv2.invertAffineTransform(M)
	#dst = cv2.warpAffine(img2,iM,(cols,rows))
	#cv2.imwrite("data/registration/testwarp.jpg",dst)
	#flow = cv2.calcOpticalFlowFarneback(dst, img11,None, 0.5, 1, 10, 5, 5, 5.0, 0)
	#cv2.imwrite('data/registration/flowimage0.png',flow[:,:,0])
	#cv2.imwrite('data/registration/flowimage1.png',flow[:,:,1])
	

	#M2 = compute_m_from_optflow(flow[:,:,0],flow[:,:,1])
	



	#M5 = compute_SIFT(img11,dst,"data/registration/matches.jpg")



	#print("This is M: ")
	#print(M)
	#print("This is M2: ")
	#print(M2)

	#print(M.shape)

	M = np.concatenate((M, [[0,0,1]]), axis=0)
	M2 = np.concatenate((M_sift, [[0,0,1]]), axis=0)
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

def create_projects(opts):

	for first_ind in range(0,15):

		fi= first_ind * 100
		if not os.path.isdir(opts.outputdirectory):
			os.makedirs(opts.outputdirectory)

		mylist = pd.read_csv(opts.csvfile)
		i = 0
		for index,row in mylist.iterrows():
			if index > fi:
				mbpfile = row['mbp_file'][5:]	
				signal =  row['path_signal']
				prediction =  opts.datadirectory + "/" + row['path_prediction_unpropped']
				tform = row['tform']

				tok = tform.split()

				M = np.matrix([[float(tok[0]), float(tok[1])],[float(tok[2]), float(tok[3])]])
				theta = np.radians(random.randint(-10,10))
				c, s = np.cos(theta), np.sin(theta)
				R = np.matrix([[c, -s], [s, c]])
				S = np.matrix([[10, 0], [0, 10]])

				print( M)
				print (R)
				print (R*M)
				M = S*R*M

				#b1 = float(tok[0]) + random.randint(-100,100)*0.0001
				#b2 = float(tok[1]) + random.randint(-100,100)*0.0001
				#b3 = float(tok[2]) + random.randint(-100,100)*0.0001
				#b4 = float(tok[3]) + random.randint(-100,100)*0.0001
				tx = float(tok[4]) +  random.randint(-5,5)
				ty = float(tok[5]) + random.randint(-5,5)
			
				#tr_string = "%s, %s, %s, %s, %f, %f"%(tok[0], tok[1], tok[2], tok[3], tx, ty)
				tr_string = "%f, %f, %f, %f, %f, %f"%(M[0,0], M[0,1], M[1,0], M[1,1], tx*10, ty*10)


				percent = get_percent(prediction, 20)

				if percent > 1.0:
					dirname = os.path.join(opts.outputdirectory,'%02d'%index)
					if not os.path.isdir(dirname):
						os.makedirs(dirname)
					filename = os.path.join(dirname, 'project.xml')				
					newcontent = []
					with open("data/registration/template.xml") as f:
		    				content = f.readlines()
					with open(filename, 'a') as f:

						for l in content:


							if 'file_path="/allen'in l:
								l = 'file_path="%s"'%signal
							if 'file_path="/nas3' in l:
								l = 'file_path="%s"'%mbpfile
							if 'transform="matrix(' in l:
								if not 'transform="matrix(10.0,0.0,0.0,10.0,0.0,0.0)"' in l:
									l = 'transform="matrix(%s)"'%tr_string
								#else:
								#	l = 'transform="matrix(10.0, 0.0, 0.0, 10.0, 0.0, 0.0)"'
							newcontent.append(l)
							#print (l)
							f.write(l )
										
					f.close()
					i = i + 1
					print (i)
				if i>5:
					break
	
	
	
if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--csvfile', default='mycsv.csv', help='CSV FILE to process')
        parser.add_argument('--datadirectory', default='dir', help='datadir')
        parser.add_argument('--outputdirectory',default='outputdir', help = 'output directory ')
        opts = parser.parse_args()
        print(opts)  
        create_projects(opts)


	
