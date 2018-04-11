import numpy as np 
import cv2
import scipy as sp
import itertools
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.misc
import cv2.ximgproc

class removeSpecular:
	def __init__(self,filename):
		self.image=self.readImage(filename)
		# self.image=scipy.misc.imresize(self.image,0.125)
		self.ROI=np.ones(self.image.shape[0:2],dtype=bool)
	def launch(self):
		self.computeChromaticities()
		self.removeFromRoiZeroValuedPixels()
		self.computeSphericalCoordinates()
		# self.plotRGBSpace()
	def readImage(self,filename,resize=1.0):
		"""read image and convert to float32"""
		image=cv2.imread(filename)
		# image=scipy.misc.imresize(image,0.25)
		# image=scipy.misc.imresize(image,resize)
		image=image.astype('float32')/255
		return image
	def computeChromaticities(self):
		RGBSum=self.image[...,0]+self.image[...,1]+self.image[...,2]
		self.B=self.image[...,0]/RGBSum;self.G=self.image[...,1]/RGBSum;self.R=self.image[...,2]/RGBSum
		# self.R/=RGB;self.G/=RGB;self.B/=RGB
		# self.sumChomaticity=
	def removeFromRoiZeroValuedPixels(self):
		sumChromaticity=self.R+self.B+self.G
		self.ROI=np.logical_and(self.ROI,np.logical_not(np.isnan(sumChromaticity)))

	def computeSphericalCoordinates(self):
		gammaG=0.33333334
		gammaR=0.33333334
		gammaB=0.33333334
		self.theta=np.arctan((algo.G-gammaG)/(algo.R-gammaR))
		self.phi=np.arctan((algo.B-gammaB)/((algo.G-gammaG)**2+(algo.R-gammaR)**2))
		self.thetaP=np.hstack((algo.theta.reshape(-1,1),algo.phi.reshape(-1,1)))
	def computeChromaticityDistanceOptimized(self):
		self.distanceChromaticities=np.zeros((self.thetaP.shape[0],self.thetaP.shape[0]))
		# for i in range(0,self.distanceChromaticities.shape[0]):
		
			# for j in range(0,self.distanceChromaticities.shape[1]):
		# iter
		iterDistanceChromaticities=np.nditer(self.distanceChromaticities,op_flags=['writeonly'])
		iterFirst=np.nditer(self.thetaP)
		iterFirst.next()
		while(not(iterDistanceChromaticities.finished)):
			while(not(iterFirst.finished)):
				array1=np.hstack((iterFirst.next(),iterFirst.next()))
				iterSecond=np.nditer(self.thetaP)
				while(not(iterSecond.finished)):
					array2=np.hstack((iterSecond.next(),iterSecond.next()))
					iterDistanceChromaticities[0]=cv2.norm(array1-array2,cv2.NORM_L1)
					iterDistanceChromaticities.next()
	def computeChromaticityDistance(self):
		self.distanceChromaticities=np.zeros((self.thetaP.shape[0],self.thetaP.shape[0]))
		for i in range(0,self.distanceChromaticities.shape[0]):
			arrayi=self.thetaP[i]
			# print(i)
			for j in range(i+1,self.distanceChromaticities.shape[1]):
				arrayj=self.thetaP[j]
				self.distanceChromaticities[i,j]=cv2.norm(arrayi-arrayj,cv2.NORM_L1)
	def computeL1Distance(self,p,q):
		return cv2.norm(p-q,cv2.NORM_L1) 
	def initClustering(self):
		self.ROICluster=np.zeros(self.thetaP.shape[0],dtype='uint8')
		unlabelledDotIndexes=(np.arange(self.ROICluster.shape[0]))
		self.mn=np.zeros((1,2));self.mn[0,:]=np.array((np.inf,np.inf)) #the cluster of index 0 is of center +inf,+inf, and is added for homogeneity of indexes
		T=np.pi/3

		#add the first element and its value as the mean
		self.N=1# number of clusters
		self.ROICluster[0]=self.N
		self.mn=np.vstack((self.mn,algo.thetaP[0,:]))
		
		##a list whose element is equal to True if it is an index of a labelled element
		a=self.ROICluster!=0
		
		##a list whose element is equal to True if it is an index of an element associated with a NaN value
		b=(np.transpose(np.isnan(self.thetaP))[0]+np.transpose(np.isnan(self.thetaP))[1])

		##construction of the unlabelled indexes
		unlabelledDotIndexes=unlabelledDotIndexes[np.logical_or(a,b)]
		
		# iterate throught the unlabelled pixels of self.thetaP
		for index in unlabelledDotIndexes:
			# for clusterMean in self.mn:
			distanceToClusters=np.array([self.computeL1Distance(self.thetaP[index,:],clusterMean) for clusterMean in self.mn])
			indexOfClosestCluster=np.argmin(distanceToClusters)
			if (distanceToClusters[indexOfClosestCluster]<T):
				# self.ROICluster[index]=indexOfClosestCluster
				##here we update the mean of he corresponding clust
				nbOfElementsOfCorrespondingCluster=(self.ROICluster==indexOfClosestCluster).sum()
				self.mn[indexOfClosestCluster,:]=nbOfElementsOfCorrespondingCluster*self.mn[indexOfClosestCluster,:]+algo.thetaP[index,:]
				self.mn[indexOfClosestCluster,:]/=(nbOfElementsOfCorrespondingCluster+1)
				self.ROICluster[index]=indexOfClosestCluster
			else:
				self.N+=1
				self.mn=np.vstack((self.mn,algo.thetaP[index,:]))
	def printImagesWithEsc(self,*images):
		"""in a unique window,show at the left and at the right, the image before and after the filtering"""
		# concatenatedImages=np.hstack((self.image,self.diffuseImage))
		titleWindow="a Window"
		# tupleImages=images
		# concatenatedImages=np.hstack(tupleImages)
		# for i in range(0,len(images))
		# cv2.namedWindow(titleWindow,cv2.WINDOW_AUTOSIZE)
		# cv2.resizeWindow(titleWindow, 0.5)
		for i in range(0,len(images)):
			cv2.namedWindow(titleWindow,cv2.WINDOW_NORMAL)
			cv2.imshow(titleWindow,images[i])
			while(True):
				k=cv2.waitKey()
				if k==27:
					cv2.destroyWindow(titleWindow)
					break


	def plotRGBSpace(self):
		ROI=self.ROI.reshape(-1)
		R=self.R.reshape(-1)
		G=self.G.reshape(-1)
		B=self.B.reshape(-1)
		
		R=R[~np.logical_not(ROI)]
		G=G[~np.logical_not(ROI)]
		B=B[~np.logical_not(ROI)]

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(R,G,B, c='r', marker= 'o')
		ax.set_xlabel('R')
		ax.set_ylabel('G')
		ax.set_zlabel('B')
		plt.show()
		# sumChromaticity=algo.R+algo.B+algo.G
		# algo.R=algo.R[~np.isnan(sumChomaticity)]
		# algo.G=algo.G[~np.isnan(sumChomaticity)]
		# algo.B=algo.B[~np.isnan(sumChomaticity)]
if __name__ == "__main__":
	algo=removeSpecular(sys.argv[1])
	algo.launch()
