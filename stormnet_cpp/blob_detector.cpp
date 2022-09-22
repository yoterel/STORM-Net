#include "blob_detector.h"



using namespace cv;
using namespace torch::indexing;

using namespace std;



torch::Tensor get_blob_keypoints(Mat mask, int max_key_points)
{
  SimpleBlobDetector::Params params;
  params.minThreshold = 127;
  params.maxThreshold = 127;

  params.filterByArea = true;
  params.minArea = 200;

  Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
  std::vector<KeyPoint> keypoints;
  detector->detect(mask, keypoints);

  std::vector<KeyPoint> valid_keypoints;
  for(int i=0; i<keypoints.size(); ++i) {
    if(keypoints[i].pt.x > 300 && keypoints[i].pt.x < 700) {
      valid_keypoints.push_back(keypoints[i]);
    }
  }
  keypoints = move(valid_keypoints);

  std::sort(keypoints.begin(), keypoints.end(),
    [](KeyPoint& k1, KeyPoint& k2) -> bool {
      return k1.response > k2.response;
  });

  
  int num_points = min(max_key_points, (int)valid_keypoints.size());
  torch::Tensor landmarks = torch::empty({num_points, 2});
  for(int i=0; i<num_points; ++i) {
    landmarks.index({i, 0}) = keypoints[i].pt.x;
    landmarks.index({i, 1}) = keypoints[i].pt.y;
  }
  return landmarks;
}


