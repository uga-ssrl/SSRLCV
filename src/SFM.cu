
// _______________________________________________________________________________________________________________
//  _____/\\\\\\\\\\\_______/\\\\\\\\\\\______/\\\\\\\\\______/\\\____________________/\\\\\\\\\__/\\\________/\\\_
//   ___/\\\/////////\\\___/\\\/////////\\\__/\\\///////\\\___\/\\\_________________/\\\////////__\/\\\_______\/\\\_
//    __\//\\\______\///___\//\\\______\///__\/\\\_____\/\\\___\/\\\_______________/\\\/___________\//\\\______/\\\__
//     ___\////\\\___________\////\\\_________\/\\\\\\\\\\\/____\/\\\______________/\\\______________\//\\\____/\\\___
//      ______\////\\\___________\////\\\______\/\\\//////\\\____\/\\\_____________\/\\\_______________\//\\\__/\\\____
//       _________\////\\\___________\////\\\___\/\\\____\//\\\___\/\\\_____________\//\\\_______________\//\\\/\\\_____
//        __/\\\______\//\\\___/\\\______\//\\\__\/\\\_____\//\\\__\/\\\______________\///\\\______________\//\\\\\______
//         _\///\\\\\\\\\\\/___\///\\\\\\\\\\\/___\/\\\______\//\\\_\/\\\\\\\\\\\\\\\____\////\\\\\\\\\______\//\\\_______
//          ___\///////////_______\///////////_____\///________\///__\///////////////________\/////////________\///________
//           _______________________________________________________________________________________________________________

#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

/**
 * the safe shutdown methods is initiated when a SIGINT is captured, but can be extended
 * to many other types of exeption handleing. Here we should makes sure that
 * memory is safely shutting down, CPU threads are killed, and whatever else is desired.
 */
void safeShutdown(int sig){
  std::cout << "Safely Ending SSRLCV ..." << std::endl;
  logger.logState("safeShutdown");
  logger.stopBackgroundLogging();
  exit(sig); // exit with the same signal
}

int main(int argc, char *argv[]){
  try{

    // register the SIGINT safe shutdown
    std::signal(SIGINT, safeShutdown);

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    // initialize the logger, this should ONLY HAPPEN ONCE
    // the logger requires that a "safes shutdown" signal handler is created
    // so that the logger.shutdown() method can be called.
    logger.logState("start"); // these can be used to time parts of the pipeline afterwards and correlate it with ofther stuff
    logger.startBackgoundLogging(1); // write a voltage, current, power log every 5 seconds

    //ARG PARSING

    logger.logState("SEED");
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
      seedFeatures = featureFactory.generateFeatures(seed,false,2,0.8);
      matchFactory.setSeedFeatures(seedFeatures);
      delete seed;
    }
    std::vector<std::string> imagePaths = ((ssrlcv::img_dir_arg*)args["dir"])->paths;
    int numImages = (int) imagePaths.size();
    std::cout<<"found "<<numImages<<" in directory given"<<std::endl;
    logger.logState("SEED");

    logger.logState("FEATURES");
    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      // logger.logState("generating features");
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,2,0.8);
      features->transferMemoryTo(ssrlcv::cpu);
      images.push_back(image);
      allFeatures.push_back(features);
      // logger.logState("done generating features");
    }
    logger.logState("FEATURES");

    //
    // MATCHING
    //

    std::cout << "Starting matching..." << std::endl;

    logger.logState("MATCHING");
    // logger.logState("generating seed matches");
    ssrlcv::Unity<float>* seedDistances = (seedProvided) ? matchFactory.getSeedDistances(allFeatures[0]) : nullptr;
    ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1],seedDistances);
    if(seedDistances != nullptr) delete seedDistances;
    // logger.logState("done generating seed matches");

    distanceMatches->transferMemoryTo(ssrlcv::cpu);
    float maxDist = 0.0f;
    for(int i = 0; i < distanceMatches->size(); ++i){
      if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
    }
    printf("max euclidean distance between features = %f\n",maxDist);
    if(distanceMatches->getMemoryState() != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);
    delete distanceMatches;

    // Need to fill into to MatchSet boi
    std::cout << "Generating MatchSet ..." << std::endl;
    ssrlcv::MatchSet matchSet;

    if (images.size() == 2){
      //
      // 2 View Case
      //
      logger.logState("matching images");
      matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matches->size()*2,ssrlcv::cpu);
      matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matches->size(),ssrlcv::cpu);
      matches->setMemoryState(ssrlcv::cpu);
      matchSet.matches->setMemoryState(ssrlcv::cpu);
      matchSet.keyPoints->setMemoryState(ssrlcv::cpu);
      logger.logState("done matching images");
      for(int i = 0; i < matchSet.matches->size(); i++){
        matchSet.keyPoints->host[i*2] = matches->host[i].keyPoints[0];
        matchSet.keyPoints->host[i*2 + 1] = matches->host[i].keyPoints[1];
        matchSet.matches->host[i] = {2,i*2};
      }
      std::cout << "Generated MatchSet ..." << std::endl << "Total Matches: " << matches->size() << std::endl << std::endl;
    } else {
      //
      // N View Case
      //
      logger.logState("matching images");
      matchSet = matchFactory.generateMatchesExaustive(images,allFeatures);
      matches->setMemoryState(ssrlcv::cpu);
      matchSet.matches->setMemoryState(ssrlcv::cpu);
      matchSet.keyPoints->setMemoryState(ssrlcv::cpu);
      logger.logState("done matching images");

      // optional to save output
      // matchSet.keyPoints->checkpoint(0,"out/kp");
      // matchSet.matches->checkpoint(0,"out/m");
    }
    logger.logState("MATCHING");

    // the bois
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::MeshFactory meshBoi = ssrlcv::MeshFactory();
    ssrlcv::MeshFactory finalMesh = ssrlcv::MeshFactory();
    ssrlcv::Unity<float3>* points;
    ssrlcv::Unity<float>* errors;
    ssrlcv::BundleSet bundleSet;

    if (images.size() == 2){
      //
      // 2 View Case
      //

      // ============= Initial Triangulation

      std::cout << "Attempting 2-view Triangulation" << std::endl;

      logger.logState("TRIANGULATE");

      float* linearError = (float*)malloc(sizeof(float));
      bundleSet = demPoints.generateBundles(&matchSet,images);

      points = demPoints.twoViewTriangulate(bundleSet, linearError);
      std::cout << "\tUnfiltered Linear Error: " << std::fixed << std::setprecision(12) << *linearError << std::endl;
      logger.logState("TRIANGULATE");

      // ============= Filtering

      logger.logState("FILTER");

      demPoints.linearCutoffFilter(&matchSet,images, 100.0); // <--- removes linear errors over 100 km

      // first time
      float sigma_filter = 1.0;
      demPoints.deterministicStatisticalFilter(&matchSet,images, sigma_filter, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.twoViewTriangulate(bundleSet, linearError);
      std::cout << "Filted " << sigma_filter  << " Linear Error: " << std::fixed << std::setprecision(12) << *linearError << std::endl;

      // second time
      sigma_filter = 3.0;
      demPoints.deterministicStatisticalFilter(&matchSet,images, sigma_filter, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.twoViewTriangulate(bundleSet, linearError);
      std::cout << "Filted " << sigma_filter  << " Linear Error: " << std::fixed << std::setprecision(12) << *linearError << std::endl;

      // neighbor filter
      demPoints.scalePointCloud(1000.0,points); // scales from km into meters
      float3 rotation = {0.0f, PI, 0.0f};
      demPoints.rotatePointCloud(rotation, points);
      // set the mesh points
      finalMesh.setPoints(points);
      //finalMesh.filterByNeighborDistance(3.0); // <--- filter bois past 3.0 sigma (about 99.5% of points) if 2 view is good then this is usually good
      finalMesh.savePoints("ssrlcv-filtered");

      logger.logState("FILTER");

      // ============= Bundle Adjustment

      logger.logState("BA");
      points = demPoints.BundleAdjustTwoView(&matchSet,images, 10, "");
      finalMesh.setPoints(points);
      //finalMesh.filterByNeighborDistance(3.0); // <--- filter bois past 3.0 sigma (about 99.5% of points) if 2 view is good then this is usually good
      finalMesh.savePoints("ssrlcv-BA-final");
      logger.logState("BA");

    } else {
      //
      // N View Case
      //

      // ============= Initial Triangulation

      std::cout << "Attempting N-view Triangulation" << std::endl;

      logger.logState("TRIANGULATE");

      float* angularError = (float*)malloc(sizeof(float));
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.nViewTriangulate(bundleSet, angularError);

      logger.logState("TRIANGULATE");

      // ============= Filtering

      logger.logState("FILTER");

      for (int i = 0; i < 10; i++) {
        demPoints.deterministicStatisticalFilter(&matchSet,images, 3.0, 0.1); // <---- samples 10% of points and removes anything past 1.0 sigma
        bundleSet = demPoints.generateBundles(&matchSet,images);
        points = demPoints.nViewTriangulate(bundleSet, angularError);
        std::cout << "Filted " << 0.1  << " Linear Error: " << std::fixed << std::setprecision(12) << *angularError << std::endl;
      }
      finalMesh.setPoints(points);
      //finalMesh.filterByNeighborDistance(3.0); // <--- filter bois past 3.0 sigma (about 99.5% of points) if 2 view is good then this is usually good
      finalMesh.savePoints("ssrlcv-filtered");

      logger.logState("FILTER");

      logger.logState("BA");
      logger.logState("BA");

    }

    // cleanup
    delete points;
    delete matches;
    delete matchSet.matches;
    delete matchSet.keyPoints;
    delete bundleSet.bundles;
    delete bundleSet.lines;
    for(int i = 0; i < imagePaths.size(); ++i){
      delete images[i];
      delete allFeatures[i];
    }

    logger.logState("end");
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
