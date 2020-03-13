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
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,200.0f*200.0f);
    bool seedProvided = false;
    ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* seedFeatures = nullptr;
    if(args.find("seed") != args.end()){
      seedProvided = true;
      std::string seedPath = ((ssrlcv::img_arg*)args["seed"])->path;
      ssrlcv::Image* seed = new ssrlcv::Image(seedPath,-1);
      seedFeatures = featureFactory.generateFeatures(seed,false,3,0.8);
      matchFactory.setSeedFeatures(seedFeatures);
      delete seed;
    }
    std::vector<std::string> imagePaths = ((ssrlcv::img_dir_arg*)args["dir"])->paths;
    int numImages = (int) imagePaths.size();
    std::cout<<"found "<<numImages<<" in directory given"<<std::endl;
    if(numImages != 2){
      std::cerr<<"ERROR this executable only accepts 2 images other than seed currently"<<std::endl;
      exit(0);
    }

    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,3,0.8);
      features->transferMemoryTo(ssrlcv::cpu);
      images.push_back(image);
      allFeatures.push_back(features);
    }

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
    for(int i = 0; i < distanceMatches->size(); ++i){
      if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
    }
    printf("max euclidean distance between features = %f\n",maxDist);
    if(distanceMatches->getMemoryState() != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);
    delete distanceMatches;

    std::string delimiter = "/";
    std::string matchFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter)) + "/matches.txt";
    ssrlcv::writeMatchFile(matches, matchFile);

    // HARD CODED FOR 2 VIEW
    // Need to fill into to MatchSet boi
    std::cout << "Generating MatchSet ..." << std::endl;
    ssrlcv::MatchSet matchSet;
    matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matches->size()*2,ssrlcv::cpu);
    matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matches->size(),ssrlcv::cpu);
    matches->setMemoryState(ssrlcv::cpu);
    for(int i = 0; i < matchSet.matches->size(); i++){
      matchSet.keyPoints->host[i*2] = matches->host[i].keyPoints[0];
      matchSet.keyPoints->host[i*2 + 1] = matches->host[i].keyPoints[1];
      matchSet.matches->host[i] = {2,i*2};
    }
    std::cout << "Generated MatchSet ..." << std::endl << "Total Matches: " << matches->size() << std::endl << std::endl;



    /*
    2 View Reprojection
    */
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();

    // bunlde adjustment loop would be here. images_vec woudl be modified to minimize the boi
    float* linearError = (float*)malloc(sizeof(float));
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


    ssrlcv::Octree* octree = new ssrlcv::Octree(points,10,true);
    octree->name = "everetst_test";
    octree->writeEdgePLY();
    //ssrlcv::MeshFactory mesher = ssrlcv::MeshFactory(octree);//would call poisson 

    delete octree;

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
