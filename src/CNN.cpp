#include "CNN.h"



std::string ssrlcv::segmentClouds(std::string pathToImage){
  std::string exec = "python3 src/scripts/segmentation/unet.py --predict --image=";
  exec += pathToImage;
  exec += " --weights=src/scripts/segmentation/weights/cloud_segmentation.weights";
  int exitCode = system(exec);
}
