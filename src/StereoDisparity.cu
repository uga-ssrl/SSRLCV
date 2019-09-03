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

    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(true,1);
    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    featureFactory.setDescriptorContribWidth(6.0f);
    featureFactory.setOrientationContribWidth(1.5f);
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image);
      allFeatures.push_back(features);
      images.push_back(image);
    }
    allFeatures[0]->transferMemoryTo(ssrlcv::cpu);
    // for(int i = 0; i < allFeatures[0]->numElements; ++i){
    //   printf("%d-%f,%f\n",i,allFeatures[0]->host[i].loc.x,allFeatures[0]->host[i].loc.y);
    //   std::cout<<std::endl;
    // }
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>();
    std::cout << "Starting matching, this will take a while ..." << std::endl;
    ssrlcv::Unity<ssrlcv::Match<ssrlcv::SIFT_Descriptor>>* matches = matchFactory.generateMatchesBruteForce(images[0],allFeatures[0],images[1],allFeatures[1]);

    //matchFactory.refineMatches(matches, 0.0001);

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
    std::ofstream outputFileMatch("./data/img/everest254/everest254_matches.txt");
    for (int i = 0; i < n; i++){
      outputFileMatch << matches->host[i].features[0].loc.x<<",";
      outputFileMatch << matches->host[i].features[0].loc.y<<",";
      outputFileMatch << matches->host[i].features[1].loc.x<<",";
      outputFileMatch << matches->host[i].features[1].loc.y<<"\n";

      matches0[i] = matches->host[i].features[0].loc;
      matches1[i] = matches->host[i].features[1].loc;
    }
    std::cout << "starting disparity with " << n << " matches ..." << std::endl;
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();

    float3* points;
    size_t points_size = n*sizeof(float3);
    points = (float3*) malloc(points_size);
    points = demPoints.stereo_disparity(matches0,matches1,points,n,64.0f);

    free(matches0);
    free(matches1);

    // TODO use something else here for saving the PLY
    std::ofstream outputFile1("./out/test.ply");
    outputFile1 << "ply\nformat ascii 1.0\nelement vertex ";
    outputFile1 << n << "\n";
    outputFile1 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
    outputFile1 << "end_header\n";

    for(int i = 0; i < n; i++){
            outputFile1 << points[i].x << " " << points[i].y << " " << points[i].z << " " << 0 << " " << 254 << " " << 0 << "\n";
    }
    std::cout<<"test.ply has been written to ./out/"<<std::endl;

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
