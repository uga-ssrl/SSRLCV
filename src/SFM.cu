#include "common_includes.h"
#include "Image.cuh"
#include "FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh" 
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
    camera_meta cameraMeta = readCameraMeta(path); 

    int numOrientations = (argc > 2) ? std::stoi(argv[2]) : 1;
    int numImages = (int) imagePaths.size();

    /*
    DENSE SIFT
    */

    SIFT_FeatureFactory featureFactory = SIFT_FeatureFactory(numOrientations);
    Image* images = new Image[numImages];
    MemoryState pixFeatureDescriptorMemoryState[3] = {gpu,gpu,gpu};

    for(int i = 0; i < numImages; ++i){
      std::string& path = imagePaths[i];
      image_meta imageMeta = readImageMeta(path);
      images[i] = Image(imagePaths[i], i, pixFeatureDescriptorMemoryState);
      images[i].convertToBW();

      images[i].descriptor.foc      = cameraMeta.focal;
      images[i].descriptor.fov      = cameraMeta.fov; 
      images[i].descriptor.cam_pos  = imageMeta.position;
      images[i].descriptor.cam_vec  = imageMeta.orientation;

      printf("%s size = %dx%d\n", imagePaths[i].c_str(), images[i].descriptor.size.x, images[i].descriptor.size.y);


      featureFactory.setImage(&(images[i]));
      featureFactory.generateFeaturesDensly();
    }
    std::cout<<"features are set"<<std::endl;

    MatchFactory matchFactory = MatchFactory();
    matchFactory.setCutOffRatio(0.1);
    SubPixelMatchSet* matchSet = NULL;
    matchFactory.generateSubPixelMatchesPairwiseConstrained(&(images[0]), &(images[1]), 5.0f, matchSet, cpu);
    matchFactory.refineMatches(matchSet);

    //TODO write method to clear all image featuresand descriptors
    printf("\nParallel DSIFT took = %f seconds.\n\n",((float) clock() -  partialTimer)/CLOCKS_PER_SEC);

    /*
    REPROJECTION
    */

    std::cout<<"starting reprojection"<<std::endl;
    partialTimer = clock();

    // Point Cloud Factory 
    PointCloud * pCloud = NULL; 
    PointCloudFactory pointCloudFactory = PointCloudFactory(); 
    pointCloudFactory.generatePointCloud(pCloud, images, numImages, matchSet);


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
