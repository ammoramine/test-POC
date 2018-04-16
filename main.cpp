////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ximgproc/fast_hough_transform.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
int main(int argc, char *argv[])
{
  // IplImage* img = 0; 
  // int height,width,step,channels;
  // uchar *data;
  // int i,j,k;

  if(argc<3){
    printf("Usage: main <image-file-name-input> <image-file-name-output>\n");
    exit(0);
  }

  // void cv::ximgproc::FastHoughTransform   (   InputArray    src,
  //   OutputArray   dst,
  //   int   dstMatDepth,
  //   int   angleRange = ARO_315_135,
  //   int   op = FHT_ADD,
  //   int   makeSkew = HDO_DESKEW 
  // )   

  // load an image 
  // std::cout<<argv[1]<<std::endl;
  cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  // image.convertTo(image,CV_32FC1)
  // std::cout<<image<<std::endl;
  // cv::namedWindow( "image", cv::WINDOW_NORMAL );
  // cv::imshow( "image",image);
  // cv::waitKey(0);
  cv::imwrite("imageWritten.jpg",image);
  int lowThreshold;
  int const max_lowThreshold = 100;
  int ratio = 3;
  int kernel_size = 3;
  
  cv::Mat edge;
  cv::Canny( image, edge,40.0,100.0,3);// lowThreshold, lowThreshold*ratio, kernel_size );
  edge.convertTo(edge,CV_32FC1);edge=edge/255.0;
  cv::imwrite("edgeWritten.jpg",edge);
  // cv::FileStorage file1("edgeWritten", cv::FileStorage::WRITE);
  // file1 <<"edgeWritten"<< edge;
  // cv::namedWindow( "edge", cv::WINDOW_NORMAL );
  // cv::imshow( "edge",edge);
  // cv::waitKey(0);

  // std::cout<<argv[2]<<std::endl;
  cv::Mat houghTransform;
  cv::ximgproc::FastHoughTransform(edge,houghTransform,edge.depth());//,cv::ximgproc::ARO_315_135,cv::ximgproc::FHT_ADD,cv::ximgproc::HDO_DESKEW);
  // cv::imwrite("houghTransform.jpg",houghTransform);
  cv::imwrite(argv[2],houghTransform);

  //enhance the hough transform
  return 0;

}