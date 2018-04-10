import numpy as np 
import cv2
import scipy as sp
import itertools
import sys
import matplotlib.pyplot as plt
import scipy.misc
import cv2.ximgproc

class removeSpecular:
	def __init__(self,filename,resize=1.0):
		self.image=self.readImage(filename,resize)
		self.sigmaColor=0.06
		self.sigmaSpace=5
		self.radius=3
		# self.epsilon=0.000000000000001
		self.epsilon=0.0
		#compute sigmaMax and lambdaMax, beginning of the algorithm 1 of the paper
		self.initSigmaMax()
		self.computeLambdaMax()
		

		self.filteredSigmaMax=np.zeros(self.lambdaMax.shape).astype('float32')
		self.diffuseImage=np.zeros(self.image.shape).astype('float32')

	def launchAlgorithm(self):

		#the rest of the algorithm 1, it's the most computatively expensive part
		self.iterateMultipleUntilConditionBroken()
		#the computation of the diffusion image
		self.computeDiffusePart()
		# we show then the initial image, and the final iamge
		self.printImageBeforeAndAfter()
	def readImage(self,filename,resize=1.0):
		"""read image and convert to float32"""
		image=cv2.imread(filename);
		image=scipy.misc.imresize(image,resize)
		image=image.astype('float32')/255
		return image
	def initSigmaMax(self):
		image0=self.image[...,0]	
		image1=self.image[...,1]	
		image2=self.image[...,2]
		# epsilon=0.000000000000001
		self.sigmaMax=cv2.max(cv2.max(image0,image1),image2)
		# epsilon=0.00000001# just to avoid invalid value on division
		sumImage=image0+image1+image2
		self.sigmaMax=self.sigmaMax/(sumImage+self.epsilon)
		self.sigmaMaxInit=self.sigmaMax
		# sigmaMax=np.nan_to_num(sigmaMax)
		# sigmaMax=cv2.divide(sigmaMax,sumImage)
		# return self.sigmaMax
	def computeSigmaMin(self):
		image0=self.image[...,0]	
		image1=self.image[...,1]	
		image2=self.image[...,2]
		# epsilon=0.000000000000001
		self.sigmaMin=cv2.min(cv2.min(image0,image1),image2)
		sumImage=image0+image1+image2
		self.sigmaMin=self.sigmaMin/(sumImage+self.epsilon)
		# sigmaMin=np.nan_to_num(sigmaMin)
		# sigmaMin=cv2.divide(sigmaMin,sumImage)
		# return sigmaMin

	def computeLambdaMax(self):
		# epsilon=0.000000000000001
		self.computeSigmaMin()
		self.lambdaMax=(self.sigmaMax-self.sigmaMin)/(1-3*self.sigmaMin)

	# def convertToUint8(self,image):
		# return (image*255).astype('uint8')
		# image=scipy.misc.imresize(image,resize)
		# image=image.astype('float32')/255
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

###non useful  functions for the main algorithm 

	def computeSum(self):
		image0=self.image[...,0]	
		image1=self.image[...,1]	
		image2=self.image[...,2]
		sumImage=image0+image1+image2
		return sumImage
	def computeSigma(self):
		image0=self.image[...,0]	
		image1=self.image[...,1]	
		image2=self.image[...,2]
		sigma=np.zeros(self.image.shape)
		#sigmaMin=cv2.min(cv2.min(image0,image1),image2)
		sumImage=image0+image1+image2
		sigma[...,0]=image0/sumImage
		sigma[...,1]=image1/sumImage
		sigma[...,2]=image2/sumImage
		return sigma
	def computeLambdac(self):
		sigma=self.computeSigma()
		self.computeSigmaMin()
		lambdac=np.zeros(sigma.shape)
		epsilon=0.000000000000001
		lambdac[...,0]=(sigma[...,0]-self.sigmaMin)/(1-3*self.sigmaMin+epsilon)
		lambdac[...,1]=(sigma[...,1]-self.sigmaMin)/(1-3*self.sigmaMin+epsilon)
		lambdac[...,2]=(sigma[...,2]-self.sigmaMin)/(1-3*self.sigmaMin+epsilon)
		return lambdac
	def computePseudoCodedDiffuse(self):
		lambdac=self.computeLambdac()
		for i in range(0,lambdac.shape[0]):
			for j in range(0,lambdac.shape[1]):
				lambdacij=lambdac[i,j]
				argmaxLambdac=lambdacij.argmax()
				sumOfResiliant=1-lambdacij[argmaxLambdac]
				lambdacij=lambdacij/sumOfResiliant*0.5
				lambdacij[argmaxLambdac]=0.5
				# lambdac[i,j]=lambdacij/lambdacij.sum()
		return lambdac
				# sigmaMax
		# itLambdac=nditer(lambdac)

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
	
	def computeGaussKernel(self,p,q,sigma):
		"""compute exp(-||p-q||^{2}/2*sigma^{2}) """
		norm=cv2.norm(p-q,cv2.NORM_L2)
		return sp.exp(-norm**2/(2*sigma**2))

	def computeGaussKernelOnPixelPosition(self):
		"""compute for a square array of size 2*radius+1, the gaussian kernel  every pixel with respect to the central pixel, N.B: this array is independant of the position of the pixel"""
		result=np.zeros((2*self.radius+1,2*self.radius+1)).astype('float32')
		center=np.array((self.radius,self.radius))
		for i in range(0,result.shape[0]):
			for j in range(0,result.shape[1]):
				ij=np.array((i,j))
				# result[i,j]=self.computeGaussKernel(center,ij,self.sigmaSpace)
				resultij=cv2.norm(ij-center,cv2.NORM_L2)
				result[i,j]=sp.exp(-resultij**2/(2*self.sigmaSpace**2))
		return result
	
	def computeGaussKernelOnPixelIntensities(self,array):
		"""compute for 'array' of size (2*self.radius+1,2*self.radius+1), the gaussian kernel  between the intensity of each pixel and the central pixel"""
		result=np.zeros((2*self.radius+1,2*self.radius+1)).astype('float32')
		centerIntensity=array[self.radius,self.radius]
		for i in range(0,array.shape[0]):
			for j in range(0,array.shape[1]):
				# result[i,j]=self.computeGaussKernel(centerIntensity,array[i,j],self.sigmaColor)
				resultij=array[i,j]-centerIntensity
				result[i,j]=sp.exp(-resultij**2/(2*self.sigmaColor**2))
		return result
	
	def computeGaussKernelOnLambdaIntensitiesAroundPixel(self,currentPixel):
		"""compute for the subArray of size (2*self.radius+1,2*self.radius+1) of lambdaMax, centered on currentPixel, the gaussian kernel  between the intensity of each pixel and the central pixel"""
		i=currentPixel[0];j=currentPixel[1]
		lambdaMaxLocal=self.lambdaMax[i-self.radius:i+self.radius+1,j-self.radius:j+self.radius+1]
		return self.computeGaussKernelOnPixelIntensities(lambdaMaxLocal)

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
#methods fot the computation of the joint bilateral filter

	# def applyJointBilateralFilterOnPixel(self,currentPixel):
	# 	"""compute the value of the joint bilateral filter on each pixel"""
	# 	gaussKernelOnLambdaIntensitiesAroundPixel=self.computeGaussKernelOnLambdaIntensitiesAroundPixel(currentPixel)

	# 	#computing the denominator den
	# 	pixelicDen=cv2.multiply(self.gaussKernelOnPixelPosition,gaussKernelOnLambdaIntensitiesAroundPixel)
	# 	den=pixelicDen.sum()
		
	# 	#computing sigmaMaxLocalAroundTheCurrentPixel
	# 	i=currentPixel[0];j=currentPixel[1]
	# 	sigmaMaxLocal=self.sigmaMax[i-self.radius:i+self.radius+1,j-self.radius:j+self.radius+1]
	# 	#computing the numerator num
	# 	pixelicNum=cv2.multiply(pixelicDen,sigmaMaxLocal)
	# 	num=pixelicNum.sum()

	# 	return num/den

	# def  applyJointBilateralFilter(self):
		
	# 	#pad all the images to ease the operations on the edge
	# 	self.lambdaMax=np.pad(self.lambdaMax,self.radius,'constant')
	# 	self.sigmaMax=np.pad(self.sigmaMax,self.radius,'constant')
	# 	self.filteredSigmaMax=np.pad(self.filteredSigmaMax,self.radius,'constant')

	# 	#compute the gaussian kernel between pixel's position
	# 	self.gaussKernelOnPixelPosition=self.computeGaussKernelOnPixelPosition()

	# 	# it = np.nditer(self.filteredSigmaMax,flags=['multi_index'],op_flags=['writeonly'])
	# 	# while not it.finished:
	# 	for i in range(self.radius,(self.filteredSigmaMax.shape[0]-1)-self.radius+1):
	# 		for j in range(self.radius,(self.filteredSigmaMax.shape[1]-1)-self.radius+1):
	# 			self.filteredSigmaMax[i,j]=self.applyJointBilateralFilterOnPixel([i,j])

	# 	#unpad all the images
	# 	self.lambdaMax=self.unpaddArray(self.lambdaMax)
	# 	self.sigmaMax=self.unpaddArray(self.sigmaMax)
	# 	self.filteredSigmaMax=self.unpaddArray(self.filteredSigmaMax)


	# def  applyJointBilateralFilterOptimized(self):
		
	# 	#pad all the images to ease the operations on the edge
	# 	self.lambdaMax=np.pad(self.lambdaMax,self.radius,'constant')
	# 	self.sigmaMax=np.pad(self.sigmaMax,self.radius,'constant')
	# 	# self.filteredSigmaMax=np.pad(self.filteredSigmaMax,self.radius,'constant')

	# 	#compute the gaussian kernel between pixel's position
	# 	self.gaussKernelOnPixelPosition=self.computeGaussKernelOnPixelPosition()

	# 	it = np.nditer(self.filteredSigmaMax,flags=['multi_index'],op_flags=['writeonly'])
	# 	while not it.finished:
	# 	# for i in range(self.radius,(self.filteredSigmaMax.shape[0]-1)-self.radius+1):
	# 		# for j in range(self.radius,(self.filteredSigmaMax.shape[1]-1)-self.radius+1):
	# 		currentPixelOnPaddedImage=[it.multi_index[0]+self.radius,it.multi_index[1]+self.radius]
	# 		it[0]=self.applyJointBilateralFilterOnPixel(currentPixelOnPaddedImage)
	# 	#unpad all the images
	# 	self.lambdaMax=self.unpaddArray(self.lambdaMax)
	# 	self.sigmaMax=self.unpaddArray(self.sigmaMax)
	# 	# self.filteredSigmaMax=self.unpaddArray(self.filteredSigmaMax)

	def iterate(self):
		# self.applyJointBilateralFilter()
		self.filteredSigmaMax=cv2.ximgproc.jointBilateralFilter(self.lambdaMax,self.sigmaMax,self.radius,self.sigmaColor,self.sigmaSpace)
		self.sigmaMax=cv2.max(self.sigmaMax,self.filteredSigmaMax)

	def iterateMultiple(self,nbIteration):
		for i in range(0,nbIteration):
			self.iterate()
			print(" iteration "+str(i)+" done")

	def iterateMultipleUntilConditionBroken(self):
		self.iterate()
		while ((self.filteredSigmaMax-self.sigmaMax).max()>0.03):
			self.iterate()
		# return filteredSigmaMax
	def unpaddArray(self,array):
		return array[self.radius:(array.shape[0]-1)-self.radius+1,self.radius:(array.shape[1]-1)-self.radius+1]

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

	def computeSpecularPart(self):
		self.specularPart=np.zeros(self.image[...,0].shape)
		image0=self.image[...,0]	
		image1=self.image[...,1]	
		image2=self.image[...,2]
		imageMax=cv2.max(cv2.max(image0,image1),image2)
		imageSum=self.computeSum()

		# epsilon=0.000000000000001
		self.specularPart=imageMax-cv2.multiply(self.sigmaMax,imageSum)
		self.specularPart=cv2.divide(self.specularPart,1-3*self.sigmaMax)
		# return specularPart
	def computeDiffusePart(self):
	# filteredSigmaMax=filteredSigmaMax.astype('float32')
		self.computeSpecularPart()
		image0=self.image[...,0]	
		image1=self.image[...,1]	
		image2=self.image[...,2]
		# imageMax=cv2.max(cv2.max(image0,image1),image2)
		# imageSum=image0+image1+image2
		# shift=imageMax-cv2.multiply(self.filteredSigmaMax,imageSum)
		# shift=cv2.divide(shift,1-3*self.filteredSigmaMax)
		self.diffuseImage[...,0]=image0-self.specularPart
		self.diffuseImage[...,1]=image1-self.specularPart
		self.diffuseImage[...,2]=image2-self.specularPart


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

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
	def printImageBeforeAndAfter(self):
		"""in a unique window,show at the left and at the right, the image before and after the filtering"""
		concatenatedImages=np.hstack((self.image,self.diffuseImage))
		cv2.namedWindow("concatenatedImages",cv2.WINDOW_NORMAL)
		cv2.imshow("concatenatedImages",concatenatedImages)
		while(True):
			k=cv2.waitKey()
			if k==27:
				cv2.destroyWindow("concatenatedImages")
				break
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


if __name__ == "__main__":
	algo=removeSpecular(sys.argv[1])
	# algo.launchAlgorithm()
	# algo.applyJointBilateralFilterRecursively()
	# algo.computeDiffusePart()
	# algo.printImageBeforeAndAfter()
	# self.filteredSigmaMax=np.zeros(self.lambdaMax.shape).astype('float32')
	# removeSpecularInstance.gaussKernelOnPixelPosition=removeSpecularInstance.computeGaussKernelOnPixelPosition()
	# removeSpecularInstance.applyJointBilateralFilter()
	# self.applyJointBilateralFilter()