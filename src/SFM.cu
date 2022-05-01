
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

#include "common_includes.hpp"
#include "SFM.cuh"
#include "Image.cuh"
#include "io_util.hpp"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

/**
 * \brief Example of safe shutdown method caused by a signal.
 * \details the safe shutdown methods is initiated when a SIGINT is captured, but can be extended
 * to many other types of exeption handleing. Here we should makes sure that
 * memory is safely shutting down, CPU threads are killed, and whatever else is desired.
 * \note ssrlcv::Unity<T>::checkpoint() is a great way to keep progress, but the Unity must be 
 * global to call this in any signal capturing method
 */
void safeShutdown(int sig){
  std::cout << "Safely Ending SSRLCV ..." << std::endl;
  logger.logState("safeShutdown");
  logger.stopBackgroundLogging();
  exit(sig); // exit with the same signal
}

void ssrlcv::doFeatureGeneration(ssrlcv::FeatureGenerationInput *in, ssrlcv::FeatureGenerationOutput *out) {
  ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);

  logger.logState("SEED");
  if (in->seedPath.size() > 0) {
    // new image with path and ID
    std::shared_ptr<ssrlcv::Image> seed = std::make_shared<ssrlcv::Image>(in->seedPath,-1);
    // array of features containing sift descriptors at every point

    out->seedFeatures = featureFactory.generateFeatures(seed,false,2,0.8);
  }
  logger.logState("SEED");

  logger.logState("FEATURES");
  for (int i = 0; i < in->numImages; i ++) {
    // new image with path and ID
    std::shared_ptr<ssrlcv::Image> image = std::make_shared<ssrlcv::Image>(in->imagePaths[i], i);
    // array of features containing sift descriptors at every point
    std::shared_ptr<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> features =
              featureFactory.generateFeatures(image,false,2,0.8);
    features->transferMemoryTo(ssrlcv::cpu);
    out->images.push_back(image);
    out->allFeatures.push_back(features);
  }
  logger.logState("FEATURES");

}

void ssrlcv::doFeatureMatching(ssrlcv::FeatureMatchingInput *in, ssrlcv::FeatureMatchingOutput *out) {
  std::cout << "Starting matching..." << std::endl;
  ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,200.0f*200.0f);
  logger.logState("MATCHING");
  // logger.logState("generating seed matches");
  if (in->seedFeatures != nullptr)
    matchFactory.setSeedFeatures(in->seedFeatures);
  std::shared_ptr<ssrlcv::Unity<float>> seedDistances = (in->seedFeatures != nullptr) ? matchFactory.getSeedDistances(in->allFeatures[0]) : nullptr;
  std::shared_ptr<ssrlcv::Unity<ssrlcv::DMatch>> distanceMatches = matchFactory.generateDistanceMatches(in->images[0], in->allFeatures[0], in->images[1], in->allFeatures[1], seedDistances);
  // logger.logState("done generating seed matches");

  distanceMatches->transferMemoryTo(ssrlcv::cpu);
  float maxDist = 0.0f;
  for(int i = 0; i < distanceMatches->size(); ++i){
    if(maxDist < distanceMatches->host.get()[i].distance) maxDist = distanceMatches->host.get()[i].distance;
  }
  printf("max euclidean distance between features = %f\n",maxDist);
  if(distanceMatches->getMemoryState() != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
  std::shared_ptr<ssrlcv::Unity<ssrlcv::Match>> matches = matchFactory.getRawMatches(distanceMatches);

  // Need to fill into to MatchSet boi
  std::cout << "Generating MatchSet ..." << std::endl;

  if (in->images.size() == 2){
    //
    // 2 View Case
    //
    logger.logState("matching images");
    out->matchSet.keyPoints = std::make_shared<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,matches->size()*2,ssrlcv::cpu);
    out->matchSet.matches = std::make_shared<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,matches->size(),ssrlcv::cpu);
    matches->setMemoryState(ssrlcv::cpu);
    out->matchSet.matches->setMemoryState(ssrlcv::cpu);
    out->matchSet.keyPoints->setMemoryState(ssrlcv::cpu);
    logger.logState("done matching images");
    for(int i = 0; i < out->matchSet.matches->size(); i++){
      out->matchSet.keyPoints->host.get()[i*2] = matches->host.get()[i].keyPoints[0];
      out->matchSet.keyPoints->host.get()[i*2 + 1] = matches->host.get()[i].keyPoints[1];
      out->matchSet.matches->host.get()[i] = {2,i*2};
    }
    std::cout << "Generated MatchSet ..." << std::endl << "Total Matches: " << matches->size() << std::endl << std::endl;
  } else {
    //
    // N View Case
    //
    logger.logState("matching images");
    out->matchSet = matchFactory.generateMatchesExaustive(in->images, in->allFeatures);
    matches->setMemoryState(ssrlcv::cpu);
    out->matchSet.matches->setMemoryState(ssrlcv::cpu);
    out->matchSet.keyPoints->setMemoryState(ssrlcv::cpu);
    logger.logState("done matching images");

    // optional to save output
    // matchSet.keyPoints->checkpoint(0,"out/kp");
    // matchSet.matches->checkpoint(0,"out/m");
  }
  logger.logState("MATCHING");

}

int main(int argc, char *argv[]){
  try{

    // register the SIGINT safe shutdown
    std::signal(SIGINT, safeShutdown);

    // CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    // initialize the logger, this should ONLY HAPPEN ONCE
    // the logger requires that a "safes shutdown" signal handler is created
    // so that the logger.shutdown() method can be called.
    logger.logState("start"); // these can be used to time parts of the pipeline afterwards and correlate it with ofther stuff
    logger.startBackgoundLogging(1); // write a voltage, current, power log every 5 seconds

    // ARG PARSING

    std::map<std::string, ssrlcv::arg*> args = ssrlcv::parseArgs(argc, argv);

    if(args.find("dir") == args.end()){
      std::cerr << "ERROR: SFM executable requires a directory of images" << std::endl;
      exit(-1);
    }

    std::string seedPath;
    if(args.find("seed") != args.end()){
      seedPath = ((ssrlcv::img_arg *)args["seed"])->path;
    }
    std::vector<std::string> imagePaths = ((ssrlcv::img_dir_arg *)args["dir"])->paths;
    int numImages = (int) imagePaths.size();
    std::cout<<"Found " << numImages << " images in directory given" << std::endl;
    logger.logState("SEED");

    //
    //
    //

    ssrlcv::FeatureGenerationInput featureGenInput(seedPath, imagePaths, numImages);
    ssrlcv::FeatureGenerationOutput featureGenOutput;
    ssrlcv::doFeatureGeneration(&featureGenInput, &featureGenOutput);
    
    //
    // FEATURE MATCHING
    //

    ssrlcv::FeatureMatchingInput featureMatchInput(featureGenOutput.seedFeatures, featureGenOutput.allFeatures, featureGenOutput.images);
    ssrlcv::FeatureMatchingOutput featureMatchOutput;
    ssrlcv::doFeatureMatching(&featureMatchInput, &featureMatchOutput);

    std::vector<std::shared_ptr<ssrlcv::Image>> images = featureGenOutput.images;
    ssrlcv::MatchSet matchSet = featureMatchOutput.matchSet;
    std::vector<std::shared_ptr<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures = featureGenOutput.allFeatures;


    // the bois
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::MeshFactory meshBoi = ssrlcv::MeshFactory();
    ssrlcv::MeshFactory finalMesh = ssrlcv::MeshFactory();
    std::shared_ptr<ssrlcv::Unity<float3>> points;
    std::shared_ptr<ssrlcv::Unity<float>> errors;
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
      std::cout << "Filtered " << sigma_filter  << " Linear Error: " << std::fixed << std::setprecision(12) << *linearError << std::endl;

      // second time
      sigma_filter = 3.0;
      demPoints.deterministicStatisticalFilter(&matchSet,images, sigma_filter, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.twoViewTriangulate(bundleSet, linearError);
      std::cout << "Filtered " << sigma_filter  << " Linear Error: " << std::fixed << std::setprecision(12) << *linearError << std::endl;
      free(linearError);

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
        std::cout << "Filtered " << 0.1  << " Linear Error: " << std::fixed << std::setprecision(12) << *angularError << std::endl;
      }
      finalMesh.setPoints(points);
      //finalMesh.filterByNeighborDistance(3.0); // <--- filter bois past 3.0 sigma (about 99.5% of points) if 2 view is good then this is usually good
      finalMesh.savePoints("ssrlcv-filtered");

      logger.logState("FILTER");

      logger.logState("BA");
      logger.logState("BA");

    }

    // cleanup
    for (ssrlcv::arg_pair p : args) {
      delete p.second; 
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
