

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import math
import pdb
import subprocess


def printImages(*images):
    # """in a unique window,show at the left and at the right, the image before and after the filtering"""
    # concatenatedImages=np.hstack((self.image,self.diffuseImage))
    images=[el for el in list(images)]
    titleWindow="a Window"
    # for 
    # tupleImages=images
    concatenatedImages=np.hstack((images))
    cv2.namedWindow(titleWindow,cv2.WINDOW_NORMAL)
    cv2.imshow(titleWindow,concatenatedImages)
    k=cv2.waitKey()
    if k==27:
      cv2.destroyWindow(titleWindow)

def houghFastTransform(filenameImageInput):
  # subprocess.call("FastHough")
  filenameImageOutput="houghSpace.jpg"
  subprocess.call(['FastHoughTransform/FastHough',filenameImageInput,filenameImageOutput])
  houghSpace=cv2.imread(filenameImageOutput)
  return houghSpace[...,0]

def enhanceHoughTransform(houghTransform,radius=2):
  houghTransformPadded=np.pad(houghTransform,radius,'constant').astype("float32")/255
  houghTransformEnhanced=np.zeros(houghTransformPadded.shape)
  sumPatch=np.zeros(houghTransformPadded.shape)
  for j in range(radius,sumPatch.shape[1]-radius):
    sumPatch[:,j]=np.sum( houghTransformPadded[:,j-radius:j+radius],axis=1)
  sumPatchim1=np.vstack((sumPatch[1:sumPatch.shape[0],:],sumPatch[0,:]))
  sumPatchip1=np.vstack((sumPatch[0,:],sumPatch[0:sumPatch.shape[0]-1,:]))
  sumPatch+=sumPatchim1+ sumPatchip1
  sumPatch+=0.00000000000001
  # sumPatch[sum]
  meanPatch=sumPatch/((2*radius+1)**2)
  houghTransformEnhanced=houghTransformPadded**2/meanPatch
  return houghTransformEnhanced[radius:houghTransformEnhanced.shape[0]-radius,radius:houghTransformEnhanced.shape[1]-radius]

def getLines(enhancedHoughSpace):
  indexRow=np.arange(0,enhancedHoughSpace.shape[0])
  indexCol=np.arange(0,enhancedHoughSpace.shape[1])
  indexRow=indexRow.reshape(indexRow.shape[0],1)
  indexCol=indexCol.reshape(1,indexCol.shape[0])
  indexes=indexRow*(indexCol.max()+1)+indexCol

  indexesLines=indexes[enhancedHoughSpace>1000]
  rhoLines=indexesLines/enhancedHoughSpace.shape[1];
  thetaLines=indexesLines-rhoLines*enhancedHoughSpace.shape[1];
  rhoLines=rhoLines.reshape(rhoLines.shape[0],1)
  thetaLines=thetaLines.reshape(thetaLines.shape[0],1)

  precisionTheta=enhancedHoughSpace.shape[1]/360

  rhoThetaLines=np.hstack((rhoLines,thetaLines))

  return rhoThetaLines

if __name__ == "__main__":
  image=cv2.imread("anImage2.ppm")
  houghSpace=houghFastTransform("anImage2.ppm")
  enhancedHoughSpace=enhanceHoughTransform(houghSpace,2)
  # dmin=image.shape[0]*3/4
  # Tc=0.5*dmin
