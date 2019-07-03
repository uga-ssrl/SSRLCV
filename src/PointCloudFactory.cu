#include "PointCloudFactory.cuh"

PointCloudFactory::PointCloudFactory() { 
}

void PointCloudFactory::generatePointCloud(PointCloud * out, Image * images, int numImages, SubPixelMatchSet * matchSet) 
{
    CameraData * cData = new CameraData();
    cData->numCameras = numImages;
    cData->cameras = new Camera[numImages]; 
    for(int i = 0; i < numImages; i++) {
      cData->cameras[i].val1 = images[i].descriptor.cam_pos.x;
      cData->cameras[i].val2 = images[i].descriptor.cam_pos.y;
      cData->cameras[i].val3 = images[i].descriptor.cam_pos.z;
      cData->cameras[i].val4 = images[i].descriptor.cam_vec.x;
      cData->cameras[i].val5 = images[i].descriptor.cam_vec.y;
      cData->cameras[i].val6 = images[i].descriptor.cam_vec.z;
    }


    FeatureMatches* reprojection_matches = new FeatureMatches();
    reprojection_matches->numMatches = matchSet->numMatches;
    reprojection_matches->matches = new float4[matchSet->numMatches];
    for(int i = 0; i < matchSet->numMatches; ++i){
      reprojection_matches->matches[i] = {matchSet->matches[i].subLocations[0].x,matchSet->matches[i].subLocations[0].y,matchSet->matches[i].subLocations[1].x,matchSet->matches[i].subLocations[1].y};
    }

    delete matchSet;

  	//execute 2 view reprojection on gpu
  	twoViewReprojection(reprojection_matches, cData, out);

    delete cData;
    delete reprojection_matches;
}