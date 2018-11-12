#include "common_includes.h"
#include "Image.cuh"
#include "FeatureFactory.cuh"
#include "MatchFactory.cuh"

//TODO IO
//TODO determine image support
//TODO add versatility to image_io and use that to make Image constructors versatile

//WARNING pointer_states are as follows
//0 = NULL
//1 = __host__
//2 = __device__
//3 = both

//TODO look into use __restrict__

//TODO fix problem with feature stuff and inability to use different classes from parent feature array

int main(int argc, char *argv[]){
  try{
    //ARG PARSING
    if(argc < 2 || argc > 4){
      std::cout<<"USAGE ./bin/dsift_parallel </path/to/image/directory/> <optional:numorientations>"<<std::endl;
      exit(-1);
    }
    std::string path = argv[1];
    std::vector<std::string> imagePaths = findFiles(path);

    int numOrientations = (argc > 2) ? std::stoi(argv[2]) : 1;
    int numImages = (int) imagePaths.size();

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();

    //GET PIXEL ARRAYS & CREATE SIFT_FEATURES DENSLY
    SIFT_FeatureFactory featureFactory = SIFT_FeatureFactory(numOrientations);
    Image* images = new Image[numImages];
    MemoryState pixFeatureDescriptorMemoryState[3] = {gpu,gpu,gpu};
    for(int i = 0; i < numImages; ++i){
      images[i] = Image(imagePaths[i], i, pixFeatureDescriptorMemoryState);
      images[i].convertToBW();
      images[i].descriptor.foc = 160.0f;
      images[i].descriptor.fov = 11.0f;
      printf("%s size = %dx%d\n",imagePaths[i].c_str(), images[i].descriptor.size.x, images[i].descriptor.size.y);
      featureFactory.setImage(&(images[i]));
      featureFactory.generateFeaturesDensly();
    }
    std::cout<<"image features are set"<<std::endl;

    //camera parameters for everest254
    images[0].descriptor.cam_pos = {7.81417, 0.0f, 44.3630};
    images[0].descriptor.cam_vec = {-0.173648, 0.0f, -.984808};
    images[1].descriptor.cam_pos = {0.0f, 0.0f, 45.0f};
    images[1].descriptor.cam_vec = {0.0f, 0.0f, -1.0f};

    MatchFactory matchFactory = MatchFactory();
    SubPixelMatchSet* matchSet = NULL;
    matchFactory.generateSubPixelMatchesPairwiseConstrained(&(images[0]), &(images[1]), 10.0f, matchSet, cpu);
    delete matchSet;

    printf("\nParallel DSIFT took = %f seconds.\n\n",((float) clock() -  totalTimer)/CLOCKS_PER_SEC);

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
