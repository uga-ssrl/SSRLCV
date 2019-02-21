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

// TODO put this somewhere else
// probably shouldn't make another file and class structure around this in the future
__global__ void disparity(float2 *matches0, float2 *matches1, float3 *points, int n){
  // yeet on that id
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if (id >= n) return; // out of bounds

  points[id].x = matches0[id].x;
  points[id].y = matches0[id].y;
  // disparity here
  
  points[id].z = 0.0f;
} // disparity

// main method
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
      printf("%s size = %dx%d\n",imagePaths[i].c_str(), images[i].descriptor.size.x, images[i].descriptor.size.y);
      featureFactory.setImage(&(images[i]));
      featureFactory.generateFeaturesDensly();
    }
    std::cout<<"image features are set"<<std::endl;

    //camera parameters for everest254
    images[0].descriptor.foc = 0.160;
    images[0].descriptor.fov = (11.4212*PI/180);
    images[0].descriptor.cam_pos = {7.81417, 0.0f, 44.3630};
    images[0].descriptor.cam_vec = {-0.173648, 0.0f, -0.984808};
    images[1].descriptor.foc = 0.160;
    images[1].descriptor.fov = (11.4212*PI/180);
    images[1].descriptor.cam_pos = {0.0f,0.0f,45.0f};
    images[1].descriptor.cam_vec = {0.0f,0.0f,-1.0f};

    MatchFactory matchFactory = MatchFactory();
    SubPixelMatchSet* matchSet = NULL;
    matchFactory.generateSubPixelMatchesPairwiseConstrained(&(images[0]), &(images[1]), 5.0f, matchSet, cpu);
    printf("\nParallel DSIFT took = %f seconds.\n\n",((float) clock() -  totalTimer)/CLOCKS_PER_SEC);

    // outputs raw matches
    // just for testing, not really needed
    std::string newFile = "./data/img/everest254/everest254_matches.txt";
    std::ofstream matchstream(newFile);
    if(matchstream.is_open()){
      std::string line;
      for(int i = 0; i < matchSet->numMatches; ++i){
        line = std::to_string(matchSet->matches[i].subLocations[0].x) + ",";
        line += std::to_string(matchSet->matches[i].subLocations[0].y) + ",";
        line += std::to_string(matchSet->matches[i].subLocations[1].x) + ",";
        line += std::to_string(matchSet->matches[i].subLocations[1].y) + "\n";
        matchstream << line;
      }
    }
    else{
      std::cout<<"ERROR cannot write match file"<<std::endl;
    }

    // begin histogram filtering ...
    std::cout << "Histogram filtering matches..." << std::endl;
    int b_size = 100; // size of each bin
    int h_size = (128*256)/b_size; // size of the max error / bin size
    int histogram[h_size] = {0};
    int max_bin = 0;
    // grow hist
    for (int i = 0; i < matchSet->numMatches; ++i){
      int bin = ((int) matchSet->matches[i].distance[1]) % b_size; //bins of size 10
      histogram[bin]++;
      // find the peak of the histogram as we fill it
      if (histogram[bin] > histogram[max_bin]){
          max_bin = bin;
      }
    }
    std::cout << "Histogram peak at: " << max_bin << std::endl;
    std::cout << "Calculating sigma..." << std::endl;
    float numerator = 0.0f;
    int sigma;
    for (int i = max_bin; i > 0; i--){
      numerator += (float) histogram[i];
      // The goal is to calculate sigma for
      // https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
      if ((numerator / (float) matchSet->numMatches) >= 0.341 ){
        sigma = i;
        break;
      }
    }
    float max_error = sigma * b_size;
    std::cout << "1 sigma is: " << sigma << ", matches with error over " << max_error << " are rejected" << std::endl;


    // actually filter the matches now
    
    std::vector<float2> match0_v;
    std::vector<float2> match1_v;

    for (int i = 0; i < matchSet->numMatches; i++){
      if (matchSet->matches[i].distance[1] <= max_error) {
	float2 temp0;
	float2 temp1;
	temp0.x = matchSet->matches[i].subLocations[0].x;
	temp0.y = matchSet->matches[i].subLocations[0].y;
	temp1.x = matchSet->matches[i].subLocations[1].x;
	temp1.y = matchSet->matches[i].subLocations[1].y;
	match0_v.push_back(temp0);
	match1_v.push_back(temp1);
      }
    } // filling our boi

    int n = matchSet->numMatches - match0_v.size(); 
    int removed = matchSet->numMatches - n;
    std::cout << "removed " << removed << " matches due to error" << std::endl;

    // prep for disparity

    // matches
    float2 *h_matches0 = match0_v.data();
    float2 *h_matches1 = match1_v.data();
    // depth point cloud
    float3 *h_points;

    // matches
    float2 *d_matches0;
    float2 *d_matches1;
    // depth point cloud
    float3 *d_points;

    // the data sizes
    size_t match_size = n*sizeof(float2);
    size_t point_size = n*sizeof(float3);

    // copy mem
    cudaMemcpy( d_matches0, h_matches0, match_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_matches1, h_matches1, match_size, cudaMemcpyHostToDevice);

    // Number of thread blocks in grid
    int blockSize = 1024;
    int gridSize = (int)ceil((float)n/blockSize);

    disparity<<<gridSize, blockSize>>>(d_matches0, d_matches1, d_points, n);

    // get data back
    cudaMemcpy( h_points, d_points, point_size, cudaMemcpyDeviceToHost );

    // TODO write data to PLY

    
    // cleanup and return
    cudaFree(d_matches0);
    cudaFree(d_matches1);
    cudaFree(d_points);

    free(h_matches0);
    free(h_matches1);
    free(h_points);
    
    delete matchSet;
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


























// yee
