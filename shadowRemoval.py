import scipy as sp
import matplotlib.colors
# def removeShadow(imageBGR):
import numpy as np
import cv2
import sys
import os
# from specularReflectionSeparation import *
def removeShadow(imageBGR):
	# imageBGR=readImage(filename)
	imageRGB=cv2.merge((imageBGR[...,2],imageBGR[...,1],imageBGR[...,0]))
	imageHSV=matplotlib.colors.rgb_to_hsv(imageRGB)
	brightness=imageHSV[...,2]
	logBritness=np.log(brightness+0.00000000001)
	fftLogBritness=np.fft.fft2(logBritness)
	H=computeButterWorthFilter(brightness.shape)
	filteredFFTLogBrightness=H*fftLogBritness
	filteredLogBrightness=np.real(np.fft.ifft2(filteredFFTLogBrightness))
	filteredBrightness=np.exp(filteredLogBrightness)

	filteredImageHSV=cv2.merge((imageHSV[...,0],imageHSV[...,1],filteredBrightness.astype('float32')))
	# filteredImageHSV=np.zeros(imageHSV.shape)
	# filteredImageHSV[...,0]=imageHSV[...,0]
	# filteredImageHSV[...,1]=imageHSV[...,1]
	# filteredImageHSV[...,2]=filteredBrightness
	filteredImageRGB=matplotlib.colors.hsv_to_rgb(filteredImageHSV)

	filteredImageBGR=cv2.merge((filteredImageRGB[...,2],filteredImageRGB[...,1],filteredImageRGB[...,0]))

	# differenceBrightness=logBritness-filteredLogBrightness
	# differenceBrightnessNormalized=(differenceBrightness-differenceBrightness.min())/(differenceBrightness.max()-differenceBrightness.min())
	# printImagesWithEscWithNormalization(logBritness,filteredLogBrightness,differenceBrightness)
	# printImagesWithEsc(logBritness,filteredLogBrightness,differenceBrightnessNormalized)
	
	return filteredImageBGR

def computeButterWorthFilter(shapeFilter):
	# shapeFilter=image.shape
	u=np.arange(-shapeFilter[0]/2,shapeFilter[0]/2).astype('float32')
	u=u.reshape(u.shape[0],1)
	u=u/(u.size)
	u=np.vstack((u[u.shape[0]/2:u.shape[0]],u[0:u.shape[0]/2]))

	v=np.arange(-shapeFilter[1]/2,shapeFilter[1]/2).astype('float32')
	v=v.reshape(v.shape[0],1)
	v=v/(v.size)
	v=np.vstack((v[v.shape[0]/2:v.shape[0]],v[0:v.shape[0]/2]))

	n=3;D0=0.05*np.pi
	H=0.53+0.85/((1+D0/cv2.sqrt(u**2+np.transpose(v**2)))**(2*n))

	##revert the form of H

	return H
def readImage(filename):
	"""read image and convert to float32"""
	imageBGR=cv2.imread(filename)
	# image=scipy.misc.imresize(image,resize)
	# image=scipy.misc.imresize(image,resize)
	imageBGR=imageBGR.astype('float32')/255.0
	return imageBGR

def writeImage(filename,filteredImageBGR):

	filenameOutput=os.path.splitext(filename)[0]+"WithoutShadow"+os.path.splitext(filename)[1]
	cv2.imwrite(filenameOutput,(filteredImageBGR*255).astype('uint8'))
# def removeShadow(imageBGR):


def printImagesWithEsc(*images):
	"""in a unique window,show at the left and at the right, the image before and after the filtering"""
		# concatenatedImages=np.hstack((self.image,self.diffuseImage))
	titleWindow="a Window"
		# for 
		# tupleImages=images
	concatenatedImages=np.hstack((images))
	cv2.namedWindow(titleWindow,cv2.WINDOW_NORMAL)
	cv2.imshow(titleWindow,concatenatedImages)
	while(True):
		k=cv2.waitKey()
		if k==27:
			cv2.destroyWindow(titleWindow)
			break

def normalizeImage(image):
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
def printImagesWithEscWithNormalization(*images):
	"""in a unique window,show at the left and at the right, the image before and after the filtering"""
	# concatenatedImages=np.hstack((self.image,self.diffuseImage))
	images=[normalizeImage(el) for el in list(images)]
	titleWindow="a Window"
	# for 
	# tupleImages=images
	concatenatedImages=np.hstack((images))
	cv2.namedWindow(titleWindow,cv2.WINDOW_NORMAL)
	cv2.imshow(titleWindow,concatenatedImages)
	k=cv2.waitKey()
	while(True):
		k=cv2.waitKey()
		if k==27:
			cv2.destroyWindow(titleWindow)
			break

if __name__ == "__main__":
	filename=sys.argv[1]
	imageBGR=readImage(filename)
	filteredImageBGR=removeShadow(imageBGR)
	# imageBGR=readImage(filename)
	# imageRGB=cv2.merge((imageBGR[...,2],imageBGR[...,1],imageBGR[...,0]))
	# imageHSV=matplotlib.colors.rgb_to_hsv(imageRGB)
	# brightness=imageHSV[...,2]
	# logBritness=np.log(brightness+0.00000000001)
	# fftLogBritness=np.fft.fft2(logBritness)
	# H=computeButterWorthFilter(brightness)
	# filteredFFTLogBrightness=H*fftLogBritness
	# filteredLogBrightness=np.real(np.fft.ifft2(filteredFFTLogBrightness))
	# filteredBrightness=np.exp(filteredLogBrightness)

	# filteredImageHSV=np.zeros(imageHSV.shape)
	# filteredImageHSV[...,0]=imageHSV[...,0]
	# filteredImageHSV[...,1]=imageHSV[...,1]
	# filteredImageHSV[...,2]=filteredBrightness
	# filteredImageRGB=matplotlib.colors.hsv_to_rgb(filteredImageHSV)

	# v=cv2.merge((filteredImageRGB[...,2],filteredImageRGB[...,1],filteredImageRGB[...,0]))

	filenameOutput=os.path.splitext(filename)[0]+"WithoutShadow"+os.path.splitext(filename)[1]
	cv2.imwrite(filenameOutput,(filteredImageBGR*255).astype('uint8'))
	print("look at your screen !")
	printImagesWithEsc(imageBGR,filteredImageBGR,imageBGR/filteredImageBGR)

