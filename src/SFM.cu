#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

//TODO fix gaussian operators - currently creating very low values


int main(int argc, char *argv[]){
  try{

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //ARG PARSING
    
    std::map<std::string,ssrlcv::arg*> args = ssrlcv::parseArgs(argc,argv);
    if(args.find("dir") == args.end()){
      std::cerr<<"ERROR: SFM executable requires a directory of images"<<std::endl;
      exit(-1);
    }
    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,250.0f*250.0f);
    bool seedProvided = false;
    ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* seedFeatures = nullptr;
    if(args.find("seed") != args.end()){
      seedProvided = true;
      std::string seedPath = ((ssrlcv::img_arg*)args["seed"])->path;
      ssrlcv::Image* seed = new ssrlcv::Image(seedPath,-1);
      seedFeatures = featureFactory.generateFeatures(seed,false,2,0.8);
      matchFactory.setSeedFeatures(seedFeatures);
      delete seed;
    }
    std::vector<std::string> imagePaths = ((ssrlcv::img_dir_arg*)args["dir"])->paths;
    int numImages = (int) imagePaths.size();
    std::cout<<"found "<<numImages<<" in directory given"<<std::endl;

    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,2,0.8);
      features->transferMemoryTo(ssrlcv::cpu);
      images.push_back(image);
      allFeatures.push_back(features);
    }

    // NEEDS TO BE REPLACED TO AUTO READ IN CAMERA PARAMS
    //=======================================================================================================================================
    // TODO this needs tp be handled by setting stuff with the binary camera
    // param reader and done at about the same time as image loading
    images[0]->id = 0;
    images[0]->camera.size = {1024,1024};
    images[0]->camera.cam_pos = {781.417,0.0,4436.30};//{7.81417,0.0,44.3630};
    // images[0]->camera.cam_vec = {-0.173648,0.0,-0.984808}; WAS {-0.173648,0.0,-0.984808}
    images[0]->camera.cam_vec = {0.0,10.0 * (M_PI/180.0),0.0}; // rotation in the y direction
    images[0]->camera.axangle = 0.0f;
    images[0]->camera.fov = (11.4212 * (M_PI/180.0)); // 11.4212 degrees
    images[0]->camera.foc = 0.16; // 160mm, 0.16m
    // this can also be a passthru (and really should be)
    images[0]->camera.dpix.x = (images[0]->camera.foc * tanf(images[0]->camera.fov / 2.0f)) / (images[0]->camera.size.x / 2.0f );
    images[0]->camera.dpix.y = images[0]->camera.dpix.y;
    std::cout << "Estimated pixel size: " << images[0]->camera.dpix.y << std::endl;
    // images[0]->camera.dpix.x = 0.014; // 14nm
    // images[0]->camera.dpix.y = 0.014; // 14nm

    images[1]->id = 1;
    images[1]->camera.size = {1024,1024};
    images[1]->camera.cam_pos = {0.0,0.0,4500.0};//{0.0,0.0,45.0};
    // images[1]->camera.cam_vec = {0.0, 0.0,-1.0}; // WAS {0.0,0.0,-1.0}
    images[1]->camera.cam_vec = {0.0,0.0,0.0}; // There was no rotation!
    images[1]->camera.axangle = 0.0f;
    images[1]->camera.fov = (11.4212 * (M_PI/180.0));// 11.4212 degress
    images[1]->camera.foc = 0.16; //160mm, 1.6cm, 0.16m
    // this can also be a passthru (and really should be)
    images[1]->camera.dpix.x = (images[1]->camera.foc * tanf(images[1]->camera.fov / 2.0f)) / (images[1]->camera.size.x / 2.0f );
    images[1]->camera.dpix.y = images[1]->camera.dpix.y;
    // images[1]->camera.dpix.x = 0.000014; // 14um
    // images[1]->camera.dpix.y = 0.000014; // 14um
    // NEEDS TO BE REPLACED TO AUTO READ IN CAMERA PARAMS
    //=======================================================================================================================================

    /*
    MATCHING
    */
    //seeding with false photo

    std::cout << "Starting matching..." << std::endl;
    
    ssrlcv::Unity<float>* seedDistances = (seedProvided) ? matchFactory.getSeedDistances(allFeatures[0]) : nullptr;
    ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1],seedDistances);
    if(seedDistances != nullptr) delete seedDistances;

    distanceMatches->transferMemoryTo(ssrlcv::cpu);
    float maxDist = 0.0f;
    for(int i = 0; i < distanceMatches->numElements; ++i){
      if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
    }
    printf("max euclidean distance between features = %f\n",maxDist);
    if(distanceMatches->state != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);
    delete distanceMatches;
    std::string delimiter = "/";
    std::string matchFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter)) + "/matches.txt";
    ssrlcv::writeMatchFile(matches, matchFile);

    // HARD CODED FOR 2 VIEW
    // Need to fill into to MatchSet boi
    std::cout << "Generating MatchSet ..." << std::endl;
    ssrlcv::MatchSet matchSet;
    matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matches->numElements*2,ssrlcv::cpu);
    matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matches->numElements,ssrlcv::cpu);
    matches->setMemoryState(ssrlcv::cpu);
    for(int i = 0; i < matchSet.matches->numElements; i++){
      matchSet.keyPoints->host[i*2] = matches->host[i].keyPoints[0];
      matchSet.keyPoints->host[i*2 + 1] = matches->host[i].keyPoints[1];
      matchSet.matches->host[i] = {2,i*2};
    }
    std::cout << "Generated MatchSet ..." << std::endl << "Total Matches: " << matches->numElements << std::endl << std::endl;


    /*
    2 View Reprojection
    */
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();

    // bunlde adjustment loop would be here. images_vec woudl be modified to minimize the boi
    unsigned long long int* linearError = (unsigned long long int*)malloc(sizeof(unsigned long long int));
    ssrlcv::BundleSet bundleSet = demPoints.generateBundles(&matchSet,images);

    // the version that will be used normally
    ssrlcv::Unity<float3>* points = demPoints.twoViewTriangulate(bundleSet, linearError);



    std::cout << "Total Linear Error: " << *linearError << std::endl;


    // optional stereo disparity here
    // /*
    // STEREODISPARITY
    // */
    // ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    // ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,8.0);
    //

    delete matches;
    ssrlcv::writePLY("out/test.ply",points);
    delete points;

    // clean up the images
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

























































// yeet
