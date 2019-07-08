#include "common_includes.h"
#include "Image.cuh"
#include "FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "reprojection.cuh"
#include "surface.cuh"

//TODO across all methods in octree and surface use const __restrict__ to enable
//https://stackoverflow.com/questions/31344454/can-a-const-restrict-increase-cuda-register-usage

//TODO to have further depth make octree node keys a long

//TODO think of a better way to spread out color

//TODO convert as many LUTs to be constant as possible, use __local__, __constant__, and __shared__

//TODO add timers to copy methods?

//TODO make method for getting grid and block dimensions

//TODO use overload operators for cuda vector arithmetic in octree.cu

//TODO make octree a class not a struct with private members and functions

//TODO optimize atomics with cooperative_groups (warp aggregated)

//TODO find all temporary device result arrays and remove redundant initial cudaMemcpyHostToDevice

//TODO make normals unit vectors?

//TODO cudaFree(constant memory)????????

//TODO typedef structs

//TODO delete methods in images


int main(int argc, char *argv[]){
  try{
    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //ARG PARSING
    if(argc < 2 || argc > 4){
      std::cout<<"USAGE ./bin/dsift_parallel </path/to/image/directory/> <optional:numorientations>"<<std::endl;
      exit(-1);
    }
    std::string path = argv[1];
    std::vector<std::string> imagePaths = findFiles(path);

    int numOrientations = (argc > 2) ? std::stoi(argv[2]) : 1;
    int numImages = (int) imagePaths.size();

    /*
    DENSE SIFT
    */

    SIFT_FeatureFactory featureFactory = SIFT_FeatureFactory(numOrientations);
    Image* images = new Image[numImages];
    MemoryState pixFeatureDescriptorMemoryState[3] = {gpu,gpu,gpu};
    for(int i = 0; i < numImages; ++i){
      images[i] = Image(imagePaths[i], i, pixFeatureDescriptorMemoryState);
      images[i].convertToBW();
      printf("%s size = %dx%d\n",imagePaths[i].c_str(), images[i].descriptor.size.x, images[i].descriptor.size.y);
      featureFactory.setImage(&(images[i]));
      featureFactory.generateFeaturesDensly();
    }
    std::cout<<"image features are set"<<std::endl;

    images[0].descriptor.foc = 0.160;
    images[0].descriptor.fov = (11.4212*PI/180);
    images[0].descriptor.cam_pos = {7.81417, 0.0f, 44.3630};
    images[0].descriptor.cam_vec = {-0.173648, 0.0f, -0.984808};
    images[1].descriptor.foc = 0.160;
    images[1].descriptor.fov = (11.4212*PI/180);
    images[1].descriptor.cam_pos = {0.0f,0.0f,45.0f};
    images[1].descriptor.cam_vec = {0.0f,0.0f,-1.0f};

    //get_cam_params2view(images[0].descriptor,images[1].descriptor,"data/morpheus/params_morpheus.txt")

    MatchFactory matchFactory = MatchFactory();
    matchFactory.setCutOffRatio(0.1);
    SubPixelMatchSet* matchSet = nullptr;
    matchFactory.generateSubPixelMatchesPairwiseConstrained(&(images[0]), &(images[1]), 5.0f, matchSet, cpu);
    matchFactory.refineMatches(matchSet);

    //TODO write method to clear all image featuresand descriptors
    printf("\nParallel DSIFT took = %f seconds.\n\n",((float) clock() -  partialTimer)/CLOCKS_PER_SEC);

    /*
    REPROJECTION
    */

    std::cout<<"starting reprojection"<<std::endl;
    partialTimer = clock();

  	CameraData* cData = new CameraData();
    cData->cameras = new Camera[2];
    cData->numCameras = 2;
    cData->cameras[0].val1 = images[0].descriptor.cam_pos.x;
    cData->cameras[0].val2 = images[0].descriptor.cam_pos.y;
    cData->cameras[0].val3 = images[0].descriptor.cam_pos.z;
    cData->cameras[0].val4 = images[0].descriptor.cam_vec.x;
    cData->cameras[0].val5 = images[0].descriptor.cam_vec.y;
    cData->cameras[0].val6 = images[0].descriptor.cam_vec.z;

    cData->cameras[0].val1 = images[1].descriptor.cam_pos.x;
    cData->cameras[0].val2 = images[1].descriptor.cam_pos.x;
    cData->cameras[0].val3 = images[1].descriptor.cam_pos.x;
    cData->cameras[0].val4 = images[1].descriptor.cam_vec.x;
    cData->cameras[0].val5 = images[1].descriptor.cam_vec.x;
    cData->cameras[0].val6 = images[1].descriptor.cam_vec.x;


    FeatureMatches* reprojection_matches = new FeatureMatches();
    reprojection_matches->numMatches = matchSet->numMatches;
    reprojection_matches->matches = new float4[matchSet->numMatches];
    for(int i = 0; i < matchSet->numMatches; ++i){
      reprojection_matches->matches[i] = {matchSet->matches[i].subLocations[0].x,matchSet->matches[i].subLocations[0].y,matchSet->matches[i].subLocations[1].x,matchSet->matches[i].subLocations[1].y};
    }

    delete matchSet;

  	//execute 2 view reprojection on gpu
  	PointCloud* pCloud = nullptr;
  	twoViewReprojection(reprojection_matches, cData, pCloud);

    delete cData;
    delete reprojection_matches;

    printf("\nParallel Reprojection took %f seconds.\n\n",((float) clock() -  partialTimer)/CLOCKS_PER_SEC);

    /*
    SURFACE RECONSTRUCTION
    */
    std::cout<<"Starting surface reconstruction"<<std::endl;
    partialTimer = clock();
    int depth = 8;

    std::cout<<"depth = "<<depth<<std::endl;
    Octree octree = Octree(pCloud->numPoints, pCloud->points, depth, false);
    float3* cameraPositions = new float3[2];
    cameraPositions[0] = images[0].descriptor.cam_pos;
    cameraPositions[1] = images[1].descriptor.cam_pos;
    octree.name = "everest254";
    octree.computeNormals(3, 20, 2, cameraPositions);
    octree.writeVertexPLY();
    octree.writeEdgePLY();
    octree.writeCenterPLY();
    octree.writeNormalPLY();
    delete[] cameraPositions;
    Surface surface = Surface(&octree);
    surface.marchingCubes();

    std::cout<<"---------------------------------------------------"<<std::endl;
    printf("\nSurface Reconstruction took %f seconds.\n\n",((float) clock() -  partialTimer)/CLOCKS_PER_SEC);

    printf("\nTOTAL TIME = %f seconds.\n\n",((float) clock() - totalTimer)/CLOCKS_PER_SEC);

    return 0;
  }
  catch (const std::exception &e){
      std::cerr << "Caught exception: " << e.what() << '\n';
      std::exit(1);
  }
  catch (...){
      std::cerr << "Caught unknown exception\n";
      std::exit(1);
  }

}
