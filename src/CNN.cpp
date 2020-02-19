#include "CNN.h"



std::string ssrlcv::segmentClouds(std::string pathToImage){
  std::string exec = "python3 src/scripts/segmentation/unet.py --predict --image=";
  exec += pathToImage;
  exec += " --weights=src/scripts/segmentation/weights/cloud_segmentation.weights";
  int exitCode = system(exec.c_str());
}



void ssrlcv::combineRGBImagesToBW(std::string r_folder, std::string g_folder, std::string b_folder, std::string bw_folder){
  //check that folders exist
  std::cout<<r_folder<<std::endl;
  std::cout<<g_folder<<std::endl;
  std::cout<<b_folder<<std::endl;
  std::cout<<bw_folder<<std::endl;
  std::vector<std::string> r_paths;
  std::vector<std::string> g_paths;
  std::vector<std::string> b_paths;
  getImagePaths(r_folder,r_paths);
  getImagePaths(g_folder,g_paths);
  getImagePaths(b_folder,b_paths);
  if(r_paths.size()!=g_paths.size() || g_paths.size() != b_paths.size()){
    std::cerr<<"FOLDERS HAVE DIFFERENT NUMBERS OF IMAGES IN THEM"<<std::endl;
    exit(0);
  }
  int numImages = r_paths.size();
  std::cout<<numImages<<" images found"<<std::endl;
  std::string bwFilePath;
  uint3 height, width, colorDepth;
  for(int i = 0; i < numImages; ++i){

    bwFilePath = bw_folder + "/bw_" + r_paths[i].substr(r_paths[i].find_last_of("/") + 5);
    std::cout<<"creating "<<bwFilePath<<std::endl;
    unsigned char* red = readImage(r_paths[i].c_str(),height.x,width.x,colorDepth.x);
    unsigned char* green = readImage(g_paths[i].c_str(),height.y,width.y,colorDepth.y);
    unsigned char* blue = readImage(b_paths[i].c_str(),height.z,width.z,colorDepth.z);
    if(height.x != height.y || height.x != height.z || 
    width.x != width.y || width.x != width.z || colorDepth.x != 1 || 
    colorDepth.y != 1 || colorDepth.z != 1){
      std::cerr<<"IMAGES CANNOT BE COMBINED"<<std::endl;
      exit(0);
    }
    unsigned char* bw = new unsigned char[width.x*height.x];
    for(int p = 0; p < height.x*width.y; ++p){
      bw[p] = rgbToBW({red[p],blue[p],green[p]});
    }
    writeImage(bwFilePath.c_str(),bw,1,width.x,height.x);
  }
}
