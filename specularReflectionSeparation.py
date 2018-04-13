import numpy as np 
import cv2
import scipy as sp
import itertools
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.misc
import cv2.ximgproc
import rand_cmap 
import cv2.ml
import subprocess
import os
import re
import pdb
import shadowRemoval

class removeSpecular:
	def __init__(self,filename,resize=1.0):
		self.image=self.readImage(filename,resize)
		# self.image=scipy.misc.imresize(self.image,0.125)
		self.ROI=np.ones(self.image.shape[0:2],dtype=bool)
	def launch(self):
		self.computeChromaticities()
		self.removeFromRoiZeroValuedPixels()
		self.computeSphericalCoordinates()
		self.initClustering()
		# algo.updateClustering()
		self.computeMatingCoefficient()
		self.computeDiffusePart()
		self.computeSpecularPart()
		# self.plotRGBSpace()
	def readImage(self,filename,resize=1.0):
		"""read image and convert to float32"""
		image=cv2.imread(filename)
		# self.estimateGamma(filename)
		image=scipy.misc.imresize(image,resize)
		# image=scipy.misc.imresize(image,resize)
		image=image.astype('float32')/255.0
		return image

	def removeShadow(self):
		algo.image=shadowRemoval.removeShadow(algo.image)
		
	def estimateGamma(self,filename):
		filenameOutput=os.path.splitext(filename)[0]+"Whitened"+os.path.splitext(filename)[1]
		outputShell=subprocess.check_output(['./iic/iic',filename,filenameOutput])
		outputShell=re.split(',|:',outputShell)
		self.gammaR=float(outputShell[1])
		self.gammaG=float(outputShell[2])
		self.gammaB=float(outputShell[3])

	def computeChromaticities(self):
		self.RGBSum=self.image[...,0]+self.image[...,1]+self.image[...,2]
		self.B=self.image[...,0]/self.RGBSum;self.G=self.image[...,1]/self.RGBSum;self.R=self.image[...,2]/self.RGBSum
		# self.R/=RGB;self.G/=RGB;self.B/=RGB
		# self.sumChomaticity=
	def removeFromRoiZeroValuedPixels(self):
		sumChromaticity=self.R+self.B+self.G
		self.ROI=np.logical_and(self.ROI,np.logical_not(np.isnan(sumChromaticity)))

	
	def computeSphericalCoordinates(self):
		self.gammaG=0.33333334
		self.gammaR=0.33333334
		self.gammaB=0.33333334
		self.theta=np.arctan((algo.G-self.gammaG)/(algo.R-self.gammaR))
		self.phi=np.arctan((algo.B-self.gammaB)/cv2.sqrt((algo.G-self.gammaG)**2+(algo.R-self.gammaR)**2))
		self.radius=cv2.sqrt((algo.B-self.gammaB)**2+(algo.G-self.gammaG)**2+(algo.R-self.gammaR)**2)

		self.deltaP=np.hstack((algo.theta.reshape(-1,1),algo.phi.reshape(-1,1)))

	def computeChromaticityDistanceOptimized(self):
		self.distanceChromaticities=np.zeros((self.deltaP.shape[0],self.deltaP.shape[0]))
		# for i in range(0,self.distanceChromaticities.shape[0]):
		
			# for j in range(0,self.distanceChromaticities.shape[1]):
		# iter
		iterDistanceChromaticities=np.nditer(self.distanceChromaticities,op_flags=['writeonly'])
		iterFirst=np.nditer(self.deltaP)
		iterFirst.next()
		while(not(iterDistanceChromaticities.finished)):
			while(not(iterFirst.finished)):
				array1=np.hstack((iterFirst.next(),iterFirst.next()))
				iterSecond=np.nditer(self.deltaP)
				while(not(iterSecond.finished)):
					array2=np.hstack((iterSecond.next(),iterSecond.next()))
					iterDistanceChromaticities[0]=cv2.norm(array1-array2,cv2.NORM_L1)
					iterDistanceChromaticities.next()
	def computeChromaticityDistance(self):
		self.distanceChromaticities=np.zeros((self.deltaP.shape[0],self.deltaP.shape[0]))
		for i in range(0,self.distanceChromaticities.shape[0]):
			arrayi=self.deltaP[i]
			# print(i)
			for j in range(i+1,self.distanceChromaticities.shape[1]):
				arrayj=self.deltaP[j]
				self.distanceChromaticities[i,j]=cv2.norm(arrayi-arrayj,cv2.NORM_L1)
	def computeL1Distance(self,p,q):
		return cv2.norm(p-q,cv2.NORM_L1) 

	def initClustering(self,T=np.pi/3):
		self.ROICluster=np.zeros(self.deltaP.shape[0],dtype='uint8')# the value of each element of self.ROICluster represents the cluster to whom belongs the pixel with the associated index
		unlabelledDotIndexes=(np.arange(self.ROICluster.shape[0]))
		self.mn=np.zeros((1,2));self.mn[0,:]=np.array((np.inf,np.inf)) #the cluster of index 0 is of center +inf,+inf, and is added for homogeneity of indexes, it represents unclassified elements at the current iteration or outliers
		# T=np.pi/6

		
		##a list whose element is equal to True if it is an index of an element associated with a NaN value
		a=(np.transpose(np.isnan(self.deltaP))[0]+np.transpose(np.isnan(self.deltaP))[1])

		unlabelledDotIndexes=unlabelledDotIndexes[~a]


		#add the first element and its value as the mean
		self.N=1# number of clusters
		firstRemainningUnlabelledElement=unlabelledDotIndexes[0]
		self.ROICluster[firstRemainningUnlabelledElement]=self.N # the others stays at 0, it means they are outliers
		self.mn=np.vstack((self.mn,algo.deltaP[firstRemainningUnlabelledElement,:]))

		##a list whose element is equal to True if it is an index of a labelled element
		# b=(self.ROICluster!=0)

		##construction of the unlabelled indexes
		unlabelledDotIndexes=unlabelledDotIndexes[1:]
		
		# iterate throught the unlabelled pixels of self.deltaP
		for index in unlabelledDotIndexes:
			# for clusterMean in self.mn:
			print(float(index)/float(unlabelledDotIndexes.max()))
			distanceToClusters=np.array([self.computeL1Distance(self.deltaP[index,:],clusterMean) for clusterMean in self.mn])
			indexOfClosestCluster=np.argmin(distanceToClusters)
			if (distanceToClusters[indexOfClosestCluster]<T):
				# self.ROICluster[index]=indexOfClosestCluster
				##here we update the mean of the corresponding cluster
				nbOfElementsOfCorrespondingCluster=(self.ROICluster==indexOfClosestCluster).sum()
				newMean=nbOfElementsOfCorrespondingCluster*self.mn[indexOfClosestCluster,:]+algo.deltaP[index,:]
				self.mn[indexOfClosestCluster,:]=newMean/(nbOfElementsOfCorrespondingCluster+1)
				self.ROICluster[index]=indexOfClosestCluster
			else:
				self.N+=1
				self.mn=np.vstack((self.mn,algo.deltaP[index,:]))
				self.ROICluster[index]=self.N
	def updateClustering(self):
		print('updating clustering by knn')
		train=cv2.ml.TrainData_create(algo.deltaP, cv2.ml.ROW_SAMPLE,algo.ROICluster.astype('float32'))
		knn = cv2.ml.KNearest_create()
		knn.train(train)
		algo.ROICluster=knn.predict(algo.deltaP)[1].astype('uint8')

	def computeMatingCoefficient(self):
		radii=self.radius.reshape(-1)
		rmax=np.array([radii[self.ROICluster==el].max() for el in range(1,self.mn.shape[0])])
		rmax=np.hstack((np.NaN,rmax))
		self.alpha=np.zeros(radii.shape,dtype='float32')
		for el in range(1,self.mn.shape[0]):
			self.alpha[self.ROICluster==el]=radii[self.ROICluster==el]/rmax[el]
		# self.alpha.reshape(self.image[0:2].shape)
		self.alpha=self.alpha.reshape(self.image.shape[0:2])
		self.alpha=cv2.medianBlur(self.alpha,5)
	# def reshapeAll(self):

	def computeDiffusePart(self):
		self.diffuse=np.zeros(self.image.shape)
		self.diffuse[...,0]=(self.B-(1-self.alpha)*self.gammaB)*self.RGBSum
		self.diffuse[...,1]=(self.G-(1-self.alpha)*self.gammaG)*self.RGBSum
		self.diffuse[...,2]=(self.R-(1-self.alpha)*self.gammaR)*self.RGBSum
				# self.B=self.image[...,0]/RGBSum;self.G=self.image[...,1]/RGBSum;self.R=self.image[...,2]/RGBSum
# 	
	def computeSpecularPart(self):
		self.specular=self.image-self.diffuse
	def printImagesWithEsc(self,*images):
		"""in a unique window,show at the left and at the right, the image before and after the filtering"""
		# concatenatedImages=np.hstack((self.image,self.diffuseImage))
		titleWindow="a Window"
		# for 
		# tupleImages=images
		concatenatedImages=np.hstack((images))
		cv2.namedWindow(titleWindow,cv2.WINDOW_NORMAL)
		cv2.imshow(titleWindow,concatenatedImages)
		k=cv2.waitKey()
		if k==27:
			cv2.destroyWindow(titleWindow)
	def normalizeImage(self,image):
		# if (len(image.shape)>2):
		# 	nbChannel=1
		# else
		# channel=image.shape[2]
		result=np.zeros(image.shape)
		if (len(image.shape)==3):
			channel=image.shape[2]
			for k in range(0,channel):
				imagek=image[...,k]
				minValue=imagek.min()
				maxValue=imagek.max()
				result[...,k]=(imagek-minValue)/(maxValue-minValue)
			return result
		elif (len(image.shape)==2):
			return (image-image.min())/(image.max()-image.min())
		# min1=np.array((algo.image[...,0].min(),algo.image[...,1].min(),algo.image[...,2].min()))
		# max1=np.array((algo.image[...,0].max(),algo.image[...,1].max(),algo.image[...,2].max()))
		# result=np.zeros()
		# return (image-min1)/(max1-min1)
		# return normalizedImage
	def printImagesWithEscWithNormalization(self,*images):
		"""in a unique window,show at the left and at the right, the image before and after the filtering"""
		# concatenatedImages=np.hstack((self.image,self.diffuseImage))
		images=[self.normalizeImage(el) for el in list(images)]
		titleWindow="a Window"
		# for 
		# tupleImages=images
		concatenatedImages=np.hstack((images))
		cv2.namedWindow(titleWindow,cv2.WINDOW_NORMAL)
		cv2.imshow(titleWindow,concatenatedImages)
		k=cv2.waitKey()
		if k==27:
			cv2.destroyWindow(titleWindow)
	def printImageBeforeAndAfter(self):
		"""in a unique window,show at the left and at the right, the image before and after the filtering"""
		concatenatedImages=np.hstack((self.image,self.diffuse))
		cv2.namedWindow("imageBeforeAndAfterSpecularRemoval",cv2.WINDOW_NORMAL)
		cv2.imshow("imageBeforeAndAfterSpecularRemoval",concatenatedImages)
		while(True):
			k=cv2.waitKey()
			if k==27:
				cv2.destroyWindow("imageBeforeAndAfterSpecularRemoval")
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
		ax.scatter(self.gammaR,self.gammaG,self.gammaB,c='k', marker= 'o',s=200)
		ax.set_xlabel('R')
		ax.set_ylabel('G')
		ax.set_zlabel('B')
		plt.show()

	# def get_cmap(n, name='hsv'):
	# 	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
	# 	RGB color; the keyword argument name must be a standard mpl colormap name.'''
	# 	return plt.cm.get_cmap(name, n)

	def plotRGBSpaceWithDiscriminationOfClusters(self,clusters=None):
		ROI=self.ROI.reshape(-1)
		R=self.R.reshape(-1)
		G=self.G.reshape(-1)
		B=self.B.reshape(-1)
		
		# R=R[~np.logical_not(ROI)]
		# G=G[~np.logical_not(ROI)]
		# B=B[~np.logical_not(ROI)]

		nbClusters=self.mn.shape[0]

		fig = plt.figure()
		# new_cmap = rand_cmap.rand_cmap(100, type='bright', first_color_black=True, last_color_black=False, verbose=True)
		ax = fig.add_subplot(111, projection='3d')
		from itertools import cycle
		cycol = cycle('bgrcm')
		# ax.scatter(X,Y, c=label, cmap=new_cmap, vmin=0, vmax=num_labels)
		# for (int i in range(1,self.mn.shape[0])):
		if clusters==None:
			for i in range(1,nbClusters):
				ax.scatter(R[self.ROICluster==i],G[self.ROICluster==i],B[self.ROICluster==i], c=next(cycol))
		else:
			for i in clusters:
				ax.scatter(R[self.ROICluster==i],G[self.ROICluster==i],B[self.ROICluster==i], c=next(cycol))

		ax.set_xlabel('R')
		ax.set_ylabel('G')
		ax.set_zlabel('B')

		ax.scatter(self.gammaR,self.gammaG,self.gammaB,c='k', marker= 'o',s=200)
		plt.show()

if __name__ == "__main__":
	if(len(sys.argv)>2):
		algo=removeSpecular(sys.argv[1],float(sys.argv[2]))
	else:
		algo=removeSpecular(sys.argv[1])
	
	algo.launch()
	# algo.initClustering()
	# algo.updateClustering()
	# algo.computeMatingCoefficient()
	# algo.computeDiffusePart()
	# algo.computeSpecularPart()
	# algo.printImageBeforeAndAfter()
	# algo.plotRGBSpaceWithDiscriminationOfClusters()
