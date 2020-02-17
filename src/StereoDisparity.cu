#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
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
      std::cout<<"USAGE ./bin/StereoDisparity </path/to/image/directory/> </path/to/optional/seedimage.png>"<<std::endl;
      exit(-1);
    }
    std::string path = argv[1];
    std::vector<std::string> imagePaths = ssrlcv::findFiles(path);

    int numImages = (int) imagePaths.size();
    if(numImages != 2){
      std::cerr<<"ERROR: this executable only accepts 2 images other than seed"<<std::endl;
      exit(0);
    }

    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,250.0f*250.0f);

    /*
    FEATURE EXTRACTION
    */
    //seed features extraction

    ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* seedFeatures = nullptr;
    if(argc == 3){
      std::string seedPath = argv[2];
      ssrlcv::Image* seed = new ssrlcv::Image(seedPath,-1);
      seedFeatures = featureFactory.generateFeatures(seed,false,2,0.8);
      matchFactory.setSeedFeatures(seedFeatures);
      delete seed;
    }

    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,2,0.8);
      features->transferMemoryTo(ssrlcv::cpu);
      images.push_back(image);
      allFeatures.push_back(features);
    }

    // TODO this needs tp be handled by setting stuff with the binary camera
    // param reader and done at about the same time as image loading
    images[0]->id = 0;
    images[0]->camera.size = {1024,1024};
    images[0]->camera.cam_pos = {781.417,0.0,4436.30};//{7.81417,0.0,44.3630};
    images[0]->camera.cam_rot = {-0.173648,0.0,-0.984808};
    images[0]->camera.fov = {(11.4212 * (M_PI/180.0)),(11.4212 * (M_PI/180.0))}; // 11.4212 degrees
    images[0]->camera.foc = 0.16; // 160mm, 0.16m
    // this can also be a passthru (and really should be)
    images[0]->camera.dpix.x = (images[0]->camera.foc * tanf(images[0]->camera.fov.x / 2.0f)) / (images[0]->camera.size.x / 2.0f );
    images[0]->camera.dpix.y = images[0]->camera.dpix.y;
    std::cout << "Estimated pixel size: " << images[0]->camera.dpix.y << std::endl;

    images[1]->id = 1;
    images[1]->camera.size = {1024,1024};
    images[1]->camera.cam_pos = {0.0,0.0,4500.0};//{0.0,0.0,45.0};
    images[1]->camera.cam_rot = {0.0, 0.0,-1.0};
    images[1]->camera.fov = {(11.4212 * (M_PI/180.0)),(11.4212 * (M_PI/180.0))};
    images[1]->camera.foc = 0.16; //160mm, 1.6cm, 0.16m
    // this can also be a passthru (and really should be)
    images[1]->camera.dpix.x = (images[1]->camera.foc * tanf(images[1]->camera.fov.x / 2.0f)) / (images[1]->camera.size.x / 2.0f );
    images[1]->camera.dpix.y = images[1]->camera.dpix.y;

    // make the camera array
    size_t cam_bytes = images.size()*sizeof(ssrlcv::Image::Camera);
    ssrlcv::Image::Camera* cameras;
    cameras = (ssrlcv::Image::Camera*) malloc(cam_bytes);
    cameras[0] = images[0]->camera;
    cameras[1] = images[1]->camera;

    /*
    MATCHING
    */
    //seeding with false photo

    std::cout << "Starting matching..." << std::endl;
    ssrlcv::Unity<float>* seedDistances = (argc == 3) ? matchFactory.getSeedDistances(allFeatures[0]) : nullptr;
    ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1],seedDistances);
    if(seedDistances != nullptr) delete seedDistances;

    distanceMatches->transferMemoryTo(ssrlcv::cpu);
    float maxDist = 0.0f;
    for(int i = 0; i < distanceMatches->size(); ++i){
      if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
    }
    printf("%f\n",maxDist);
    if(distanceMatches->getMemoryState() != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);
    delete distanceMatches;
    std::string delimiter = "/";
    std::string matchFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter)) + "/matches.txt";
    ssrlcv::writeMatchFile(matches, matchFile);

    /*
    STEREODISPARITY
    */
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,cameras);

    delete matches;
    ssrlcv::writePLY("out/test.ply",points);
    delete points;

    for(int i = 0; i < imagePaths.size(); ++i){
      delete images[i];
      delete allFeatures[i];
    }

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
