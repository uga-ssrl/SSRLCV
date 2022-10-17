
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
#include "Pipeline.cuh"
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
  logger.info << "Safely Ending SSRLCV ...";
  logger.logState("safeShutdown");
  logger.stopBackgroundLogging();
  exit(sig); // exit with the same signal
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
    logger.info.printf("Found %d images in directory given", numImages);
    logger.logState("SEED");

    // off-precision distance for epipolar matching (in pixels)
    float epsilon = 5.0;
    if (args.find("epsilon") != args.end()) {
      epsilon = ((ssrlcv::flt_arg *)args["epsilon"])->val;

      logger.info.printf("Setting delta (for epipolar geometry) to %f kilometers.", epsilon);
    }

    // off-precision distance for orbital SfM (in kilometers)
    float delta = 0.0;
    if (args.find("delta") != args.end()) {
      delta = ((ssrlcv::flt_arg *)args["delta"])->val;

      logger.info.printf("Setting delta (for earth-centered epipolar geometry) to %f kilometers.", delta);
    }

    //
    // FEATURE GENERATION
    //

    ssrlcv::FeatureGenerationInput featureGenInput = {seedPath, imagePaths, numImages};
    ssrlcv::FeatureGenerationOutput featureGenOutput;
    ssrlcv::doFeatureGeneration(&featureGenInput, &featureGenOutput);
    
    //
    // FEATURE MATCHING
    //

    ssrlcv::FeatureMatchingInput featureMatchInput = {featureGenOutput.seedFeatures, featureGenOutput.allFeatures, featureGenOutput.images, epsilon, delta};
    ssrlcv::FeatureMatchingOutput featureMatchOutput;
    ssrlcv::doFeatureMatching(&featureMatchInput, &featureMatchOutput);

    //
    // TRIANGULATION
    //

    ssrlcv::TriangulationInput triangulationInput = {featureMatchOutput.matchSet, featureGenOutput.images};
    ssrlcv::TriangulationOutput triangulationOutput;
    ssrlcv::doTriangulation(&triangulationInput, &triangulationOutput);

    //
    // FILTERING
    //

    ssrlcv::FilteringInput filteringInput = {triangulationInput.matchSet, featureGenOutput.images};
    ssrlcv::FilteringOutput filteringOutput;
    ssrlcv::doFiltering(&filteringInput, &filteringOutput);

    //
    // BUNDLE ADJUSTMENT
    //

    ssrlcv::BundleAdjustInput bundleAdjustInput = {filteringInput.matchSet, featureGenOutput.images};
    ssrlcv::BundleAdjustOutput bundleAdjustOutput;
    ssrlcv::doBundleAdjust(&bundleAdjustInput, &bundleAdjustOutput);


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
