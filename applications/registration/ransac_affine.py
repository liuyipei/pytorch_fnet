import os

class RansacAffineModel(object):
	def __init__(self):
		print("init")
	def fit(self,data):
		data = data.T
		fp = data[:3,4]
		tp = data[3:,4]
		return cv2.estimateRigidTransform(fp, tp,True)

	def get_error(self,data,A):	
		data = data.T
		fp = data[:3]
		tp = data[3:]
		fp_transformed = dot(A,fp)
		return sqrt(sum((tp-fp_transformed))**2,axis=0)
	def A_from_ransac(fp,tp,model,maxiter=1000,match_threshold=10):
		data = vstack((fp,tp))
		A,ransac_data = ransac.ransac(data.T,model,4,maxiter,match_treshold,10,return_all=True)
		return A,ransac_data['inliers']
