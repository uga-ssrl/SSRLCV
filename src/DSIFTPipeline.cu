#include "common_includes.h"
#include "cuda_util.cuh"
#include "Image.cuh"
#include "FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include <new>
#include <sstream>

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
  if (id < n){
    points[id].x = matches0[id].x;
    points[id].y = matches0[id].y;
    // TODO make the scale factor - in this case its 64.0f - based on the optical system
    points[id].z = sqrtf(64.0f * ((matches0[id].x - matches1[id].x)*(matches0[id].x - matches1[id].x) + (matches0[id].y - matches1[id].y)*(matches0[id].y - matches1[id].y)));
  }
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
    // NOTE: this is chosen by the user.
    // TODO: Abstract this away
    int segment_size = 500; // this is only the approximate goal size
    bool segmenting = true;
    // only true if we are trying to histogram filter
    // bool filtering = false;

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();

    //GET PIXEL ARRAYS & CREATE SIFT_FEATURES DENSLY
    SIFT_FeatureFactory featureFactory = SIFT_FeatureFactory(numOrientations);
    Image* images = new Image[numImages];
    MemoryState pixFeatureDescriptorMemoryState[3] = {gpu,gpu,gpu};



    // segment images before descriptors are made
    if (segmenting) {
      // load the images
      for(int i = 0; i < numImages; ++i){
        images[i] = Image(imagePaths[i], i, pixFeatureDescriptorMemoryState);
        images[i].convertToBW();
        // images[i].segment(x_segments,y_segments,x_segment_size,y_segment_size);
        printf("%s size = %dx%d\n",imagePaths[i].c_str(), images[i].descriptor.size.x, images[i].descriptor.size.y);
      }
      // segment the images here into blocks
      // find a close divisor:
      int seg_size = images[0].descriptor.size.x;
      while (seg_size > segment_size) { seg_size /= 2; }
      int x_segment_size = seg_size;
      seg_size = images[0].descriptor.size.y;
      while (seg_size > segment_size) { seg_size /= 2; }
      int y_segment_size = seg_size;
      // segment images for faster match generation
      // default segment size goal is 500 or less
      int x_segments = images[0].descriptor.size.x / x_segment_size;
      int y_segments = images[0].descriptor.size.y / y_segment_size;

      for(int i = 0; i < x_segments*y_segments; ++i){
        images[i].descriptor.foc = 0.160;
        images[i].descriptor.fov = (11.4212*PI/180);
        images[i].descriptor.cam_pos = {7.81417, 0.0f, 44.3630};
        images[i].descriptor.cam_vec = {-0.173648, 0.0f, -0.984808};
        images[i].descriptor.segment_size.x = x_segment_size;
        images[i].descriptor.segment_size.y = y_segment_size;
        images[i].descriptor.segment_num.x = x_segments;
        images[i].descriptor.segment_num.y = y_segments;
        images[i].segment_helper.is_segment = false;
        std::cout << "pre-segment" << std::endl;
        images[i].segment(x_segments,y_segments,x_segment_size,y_segment_size);
        std::cout << "segmented image " << i << " ... " << std::endl;
      }

      std::cout << "Segmented with: " << (x_segments*y_segments) << " total segments of size: " << segment_size << std::endl;


    } else {
        //GET PIXEL ARRAYS & CREATE SIFT_FEATURES DENSLY
        // SIFT_FeatureFactory featureFactory = SIFT_FeatureFactory(numOrientations);
        // images = new Image[numImages];
        // MemoryState pixFeatureDescriptorMemoryState[3] = {gpu,gpu,gpu};
        for(int i = 0; i < numImages; ++i){
          images[i] = Image(imagePaths[i], i, pixFeatureDescriptorMemoryState);
          images[i].convertToBW();

          printf("%s size = %dx%d\n",imagePaths[i].c_str(), images[i].descriptor.size.x, images[i].descriptor.size.y);

          //camera parameters for everest254
          images[0].descriptor.foc = 0.160;
          images[0].descriptor.fov = (11.4212*PI/180);
          images[0].descriptor.cam_pos = {7.81417, 0.0f, 44.3630};
          images[0].descriptor.cam_vec = {-0.173648, 0.0f, -0.984808};
          images[0].descriptor.segment_size.x = -1;
          images[0].descriptor.segment_size.y = -1;
          images[0].descriptor.segment_num.x = -1;
          images[0].descriptor.segment_num.y = -1;
          images[0].segment_helper.is_segment = false;
          images[1].descriptor.foc = 0.160;
          images[1].descriptor.fov = (11.4212*PI/180);
          images[1].descriptor.cam_pos = {0.0f,0.0f,45.0f};
          images[1].descriptor.cam_vec = {0.0f,0.0f,-1.0f};
          images[1].descriptor.segment_size.x = -1;
          images[1].descriptor.segment_size.y = -1;
          images[1].descriptor.segment_num.x = -1;
          images[1].descriptor.segment_num.y = -1;
          images[1].segment_helper.is_segment = false;

          featureFactory.setImage(&(images[i]));
          featureFactory.generateFeaturesDensly();
        }
    }

    std::cout<<"image features are set"<<std::endl<<std::endl;

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
    // NOTE / TODO simple percentage filter cutoff may also work
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
    // 3 sigma means we're basically keeping everything
    float mul = 1.0f;
    float max_error = (max_bin * b_size) + (mul * sigma * b_size);
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

    int removed = matchSet->numMatches - match0_v.size();
    int n = match0_v.size();
    //int n = 100;
    std::cout << "removed " << removed << " matches due to error | ";
    std::cout << "keeping " << n << " matches" << std::endl;

    // matches
    float2 *h_matches0;
    float2 *h_matches1;
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

    h_points = (float3*) malloc(point_size);

    cudaMalloc((void**) &d_matches0, match_size);
    cudaMalloc((void**) &d_matches1, match_size);
    cudaMalloc((void**) &d_points, point_size);

    // set the pointer to the vectors
    h_matches0 = &match0_v[0];
    h_matches1 = &match1_v[0];

    // copy mem
    cudaMemcpy( d_matches0, h_matches0, match_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_matches1, h_matches1, match_size, cudaMemcpyHostToDevice);

    // Number of thread blocks in grid
    int blockSize = 1024;
    int gridSize = (int)ceil((float)n/blockSize);

    disparity<<<gridSize, blockSize>>>(d_matches0, d_matches1, d_points, n);

    // get data back
    cudaMemcpy(h_points, d_points, point_size, cudaMemcpyDeviceToHost );

    // TODO write data to PLY

    //savePly(h_points, n);

    std::ofstream outputFile1("test.ply");
    outputFile1 << "ply\nformat ascii 1.0\nelement vertex ";
    outputFile1 << n << "\n";
    outputFile1 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
    outputFile1 << "end_header\n";

    for(int i = 0; i < n; i++){
            outputFile1 << h_points[i].x << " " << h_points[i].y << " " << h_points[i].z << " " << 0 << " " << 254 << " " << 0 << "\n";
    }
    std::cout<<"test.ply has been written to repo root"<<std::endl;

    // cleanup and return
    cudaFree(d_matches0);
    cudaFree(d_matches1);
    cudaFree(d_points);

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
