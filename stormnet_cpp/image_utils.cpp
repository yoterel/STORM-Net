#include "image_utils.h"



using namespace cv;


float measure_blur(Mat img)
{
  Mat laplacian;
  Laplacian(img, laplacian, CV_64F);

  Mat mean, stddev;
  meanStdDev(laplacian, mean, stddev);
  return stddev.at<double>(0, 0);
}

