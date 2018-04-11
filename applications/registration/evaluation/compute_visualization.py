import numpy as np
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def get_percent(img, bordersz):
	
	imgsize = img.shape[0]*img.shape[1]
	img = img[bordersz: img.shape[0] - bordersz, bordersz: img.shape[1] - bordersz]
	ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	return 100*np.count_nonzero(thresh)/imgsize 


def compute_all(opts):
	exclude = [3, 11, 19,20,22,27,29,30,31,37,44,46,47,12,25,39,48,43]
	E = []
	I = []
	C = []
	percent = []
	mylist = pd.read_csv(opts.csvfile)
	for index,row in mylist.iterrows():
		#print(index)
		#imgfile = "/allen/aics/modeling/cheko/projects/pytorch_fnet/results/2d/registration_2/" + row['path_prediction_unpropped']
		#f32 = Image.open(imgfile)
		#img = np.asarray(f32.convert('I;16'))
		#img = cv2.convertScaleAbs(img)
		#img = cv2.equalizeHist(img)	
		#percent.append( get_percent(img,20) )

	
		error = row['pixel_error']
		percent = row['content_percentage']
		#if index not in exclude:
		#if (percent[index] >= 1.0) & (error < 20):
		if not np.isnan(error) :
			if error < 20:
				E.append(error)		
				I.append(index)
				C.append(percent)
		if error>20:
			print (index)

		

	
	ypos = range(0,len(E))
	print(len(E))
	
	hist, bin_edges = np.histogram(E,bins=25)
	matplotlib.pyplot.bar(bin_edges[:-1],hist)
	matplotlib.pyplot.ylabel('Number of Images')
	matplotlib.pyplot.xlabel('Average Pixel Error')
	matplotlib.pyplot.ylim(0,23)
	matplotlib.pyplot.xlim(0,6)
	matplotlib.pyplot.savefig("%s_hist.eps"%opts.outputfileprefix,dpi=600)

	#matplotlib.pyplot.bar(I, E, align='center', alpha=0.5)
	#matplotlib.pyplot.xticks(ypos, I,fontsize=6)


	#matplotlib.pyplot.plot(E,C,'ro')
	#matplotlib.pyplot.ylabel('Content Percent')
	#matplotlib.pyplot.xlabel('Average Pixel Error')
	#matplotlib.pyplot.savefig("%s_similarity_function.eps"%opts.outputfileprefix,dpi=600)
	
	print("Mean = %f"%np.mean(E))
	print("Median = %f"%np.median(E))
	print("Std Dev = %f"%np.std(E))
	print(len(E))
	print(type(percent))	
	#print("Excluded: ")
	#for ex in exclude:
	#	print (ex, percent[ex])
	#print("Not Excluded: ")
	#for i in range(0,len(percent)):
	#	if i not in exclude:
	#		print(i,percent[i])

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--csvfile', default='mycsv.csv', help='CSV FILE to process')
        parser.add_argument('--outputfileprefix', default='myfig.jpg', help='output file with visualization')
        opts = parser.parse_args()
        print(opts)  
        compute_all(opts)


	
