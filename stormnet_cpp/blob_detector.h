#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <vector>



using namespace cv;
using namespace torch::indexing;

using namespace std;



torch::Tensor get_blob_keypoints(Mat mask, int max_key_points);


