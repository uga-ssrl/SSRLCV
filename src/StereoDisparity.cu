#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

int main(int argc, char *argv[]){
  try{
    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //ARG PARSING
    if(argc < 2 || argc > 4){
      std::cout<<"USAGE ./bin/SFM </path/to/image/directory/>"<<std::endl;
      exit(-1);
    }
    std::string path = argv[1];
    std::vector<std::string> imagePaths = ssrlcv::findFiles(path);

    int numImages = (int) imagePaths.size();

    /*
    DENSE SIFT
    */

    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory();
    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    unsigned int convertColorDepthTo = 1;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],convertColorDepthTo,i);
      //sift border is 24 due to 1xbin would normally be 12
      image->quadtree->setNodeFlags({12.0f+image->quadtree->border.x,12.0f+image->quadtree->border.y},true);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeaturesDensly(image);
      allFeatures.push_back(features);
      images.push_back(image);
    }

    ssrlcv::MatchFactory matchFactory = ssrlcv::MatchFactory();
    std::cout << "Starting matching, this will take a while ..." << std::endl;
    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.generateMatchesBruteForce(images[0],allFeatures[0],images[1],allFeatures[1]);

    matches->transferMemoryTo(ssrlcv::cpu);

    int n = matches->numElements;
    // refile to float2
    // TODO maybe add this to the match factory?
    // TODO or add to point cloud factory
    std::cout << "Copying matches" << std::endl;
    float2* matches0;
    float2* matches1;
    size_t match_size = n*sizeof(float2);
    matches0 = (float2*) malloc(match_size);
    matches1 = (float2*) malloc(match_size);
    for (int i = 0; i < n; i++){
      matches0[i] = matches->host[i].features[0].loc;
      matches1[i] = matches->host[i].features[1].loc;
      //if (!(i%100)){
      //   std::cout << matches0[i].x << " | " << matches1[i].x << std::endl;
      //   std::cout << matches0[i].y << " | " << matches1[i].y << std::endl;
      // //}
    }

    std::cout << "starting disparity with " << n << " matches ..." << std::endl;
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();

    float3* points;
    size_t points_size = n*sizeof(float3);
    points = (float3*) malloc(points_size);
    points = demPoints.stereo_disparity(matches0,matches1,points,n,1.0);

    free(matches0);
    free(matches1);

    // TODO use something else here for saving the PLY
    std::ofstream outputFile1("test.ply");
    outputFile1 << "ply\nformat ascii 1.0\nelement vertex ";
    outputFile1 << n << "\n";
    outputFile1 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
    outputFile1 << "end_header\n";

    for(int i = 0; i < n; i++){
            outputFile1 << points[i].x << " " << points[i].y << " " << points[i].z << " " << 0 << " " << 254 << " " << 0 << "\n";
    }
    std::cout<<"test.ply has been written to repo root"<<std::endl;

    free(points);

    return 0;
  }
  catch (const std::exception &e){
      std::cerr << "Caught exception: " << e.what() << '\n';
      std::exit(1);
  }
  catch (const ssrlcv::UnityException &e){
      std::cerr << "Caught exception: " << e.what() << '\n';
      std::exit(1);
  }
  catch (...){
      std::cerr << "Caught unknown exception\n";
      std::exit(1);
  }

}
