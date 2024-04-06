
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
#include <sys/stat.h>
#include <stdio.h>

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

enum cp_stage {
  fg = 1, // feature generation
  fm = 2, // feature matching
  tri = 3, // triangulation
  filt = 4, // filtering
  ba = 5 // bundle adjustment
};

std::string getCheckpointDirForStage(std::string rootDir, int stage) {
  return rootDir + "/outputs/sfm-stage" + std::to_string(stage) + "/";
}

void mkdirIfAbsent(std::string dir) {
  struct stat st = {0};

  if (stat(dir.c_str(), &st) == -1) {
      mkdir(dir.c_str(), 0777);
  }
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

    if(args.find("dir") == args.end() && args.find("cpdir") == args.end()){
      std::cerr << "ERROR: SFM executable requires a directory of images" << std::endl;
      exit(-1);
    }

    bool checkpoint;
    ssrlcv::img_dir_arg *imageDir;
    std::string checkpointRootDir;
    if(args.find("cpdir") == args.end()) {
      logger.info.printf("No checkpointing");
      checkpoint = false;
      imageDir = (ssrlcv::img_dir_arg *)args["dir"];
    } else {
      logger.info.printf("Checkpointing after every stage");
      checkpoint = true;
      imageDir = (ssrlcv::img_dir_arg *)args["cpdir"];
      checkpointRootDir = imageDir->rootPath;
      logger = ssrlcv::Logger((checkpointRootDir).c_str());
      mkdirIfAbsent(checkpointRootDir + "/outputs");
    }

    std::string seedPath;
    if(args.find("seed") != args.end()){
      seedPath = ((ssrlcv::img_arg *)args["seed"])->path;
    }
    std::vector<std::string> imagePaths = imageDir->paths;
    int numImages = (int) imagePaths.size();
    logger.info.printf("Found %d images in directory given", numImages);
    logger.logState("SEED");

    // off-precision distance for epipolar matching (in pixels)
    float epsilon = 5.0;
    if (args.find("epsilon") != args.end()) {
      epsilon = ((ssrlcv::flt_arg *)args["epsilon"])->val;

      logger.info.printf("Setting epsilon (for epipolar geometry) to %f pixels.", epsilon);
    }

    // off-precision distance for orbital SfM (in kilometers)
    float delta = 0.0;
    if (args.find("delta") != args.end()) {
      delta = ((ssrlcv::flt_arg *)args["delta"])->val;

      logger.info.printf("Setting delta (for earth-centered epipolar geometry) to %f kilometers.", delta);
    }

    ssrlcv::FeatureGenerationInput featureGenInput = {seedPath, imagePaths, numImages};
    ssrlcv::FeatureGenerationOutput featureGenOutput;
    ssrlcv::FeatureMatchingInput featureMatchInput;
    ssrlcv::FeatureMatchingOutput featureMatchOutput;
    ssrlcv::TriangulationInput triangulationInput;
    ssrlcv::TriangulationOutput triangulationOutput;
    ssrlcv::FilteringInput filteringInput;
    ssrlcv::FilteringOutput filteringOutput;
    ssrlcv::BundleAdjustInput bundleAdjustInput;
    ssrlcv::BundleAdjustOutput bundleAdjustOutput;

    if (checkpoint) {
      struct stat buf;

      int cur_stage;
      for (cur_stage = 1; cur_stage < 5; cur_stage ++) {
        if (stat((checkpointRootDir + "/outputs/sfm-stage" + std::to_string(cur_stage) + "/done").c_str(), &buf) != 0) break;
      }
      logger.info.printf("Already checkpointed %d stage(s).", cur_stage - 1);

      switch (cur_stage) {
        case cp_stage::fg: // Feature generation
          // nothing checkpointed yet, continue as normal (start with feature generation)
          break;
        case cp_stage::fm: // Feature matching
          featureMatchInput.fromCheckpoint(getCheckpointDirForStage(checkpointRootDir, cp_stage::fg), numImages, epsilon, delta);
          goto FEATURE_MATCHING;
        case cp_stage::tri:
          triangulationInput.fromCheckpoint(
            getCheckpointDirForStage(checkpointRootDir, cp_stage::fg),
            getCheckpointDirForStage(checkpointRootDir, cp_stage::fm),
            numImages
          );
          goto TRIANGULATION;
        case cp_stage::filt:
          filteringInput.fromCheckpoint(
            getCheckpointDirForStage(checkpointRootDir, cp_stage::fg),
            getCheckpointDirForStage(checkpointRootDir, cp_stage::fm),
            numImages
          );
          goto FILTERING;
          break;
        case cp_stage::ba:
          bundleAdjustInput.fromCheckpoint(
            getCheckpointDirForStage(checkpointRootDir, cp_stage::fg),
            getCheckpointDirForStage(checkpointRootDir, cp_stage::filt),
            numImages
          );
          goto BUNDLE_ADJUSTMENT;
          break;
        default:
          break;
      }
    }

    //
    // FEATURE GENERATION
    //

    ssrlcv::doFeatureGeneration(&featureGenInput, &featureGenOutput);
    if (checkpoint) {
      std::string cpdir = getCheckpointDirForStage(checkpointRootDir, cp_stage::fg);
      mkdirIfAbsent(cpdir);
      if (featureGenOutput.seedFeatures != nullptr) {
        featureGenOutput.seedFeatures->checkpoint(-1, cpdir);
      }
      for (int i = 0; i < numImages; i ++) {
        featureGenOutput.images.at(i)->checkpoint(cpdir);
        featureGenOutput.allFeatures.at(i)->checkpoint(i, cpdir);
      }
      fclose(fopen((cpdir + "/done").c_str(), "w"));
    }
    
    //
    // FEATURE MATCHING
    //

    featureMatchInput.fromFeatureGeneration(&featureGenOutput, epsilon, delta);
    FEATURE_MATCHING:
    ssrlcv::doFeatureMatching(&featureMatchInput, &featureMatchOutput);
    if (checkpoint) {
      std::string cpdir = getCheckpointDirForStage(checkpointRootDir, cp_stage::fm);
      mkdirIfAbsent(cpdir);
      featureMatchOutput.matchSet.keyPoints->checkpoint(0, cpdir);
      featureMatchOutput.matchSet.matches->checkpoint(0, cpdir);
      fclose(fopen((cpdir + "/done").c_str(), "w"));
    }

    //
    // TRIANGULATION
    //

    triangulationInput.fromPreviousStage(&featureMatchInput, &featureMatchOutput);
    TRIANGULATION:
    ssrlcv::doTriangulation(&triangulationInput, &triangulationOutput);
    if (checkpoint) {
      std::string cpdir = getCheckpointDirForStage(checkpointRootDir, cp_stage::tri);
      mkdirIfAbsent(cpdir);
      // Copy PLY file
      std::ifstream in("out/ssrlcv-initial.ply", std::ios::in | std::ios::binary);
      std::ofstream out((cpdir + "/ssrlcv-initial.ply").c_str(), std::ios::out | std::ios::binary);
      out << in.rdbuf();
      fclose(fopen((cpdir + "/done").c_str(), "w"));
    }

    //
    // FILTERING
    //

    filteringInput.fromPreviousStage(&triangulationInput);
    FILTERING:
    ssrlcv::doFiltering(&filteringInput, &filteringOutput);
    if (checkpoint) {
      std::string cpdir = getCheckpointDirForStage(checkpointRootDir, cp_stage::filt);
      mkdirIfAbsent(cpdir);
      // Copy PLY file
      std::ifstream in("out/ssrlcv-filtered.ply", std::ios::in | std::ios::binary);
      std::ofstream out((cpdir + "/ssrlcv-filtered.ply").c_str(), std::ios::out | std::ios::binary);
      out << in.rdbuf();

      // Write new matches
      filteringInput.matchSet.keyPoints->checkpoint(0, cpdir);
      filteringInput.matchSet.matches->checkpoint(0, cpdir);

      fclose(fopen((cpdir + "/done").c_str(), "w"));
    }

    //
    // BUNDLE ADJUSTMENT
    //

    bundleAdjustInput.fromPreviousStage(&filteringInput);
    BUNDLE_ADJUSTMENT:
    ssrlcv::doBundleAdjust(&bundleAdjustInput, &bundleAdjustOutput);

    //
    // FINAL OUTPUT (no need to checkpoint Bundle Adjustment raw structures as it's the last stage)
    //
    if (checkpoint) {
      std::string cpdir(checkpointRootDir + "/outputs/final");
      mkdirIfAbsent(cpdir);

      // Copy PLY file from final BA stage
      std::ifstream in("out/ssrlcv-BA-final.ply", std::ios::in | std::ios::binary);
      std::ofstream out((cpdir + "/ssrlcv-BA-final.ply").c_str(), std::ios::out | std::ios::binary);
      out << in.rdbuf();

      // Copy PLY file from filtering checkpoint
      std::string filtPlyPath = getCheckpointDirForStage(checkpointRootDir, cp_stage::filt) + "/ssrlcv-filtered.ply";
      std::ifstream inFilt(filtPlyPath.c_str(), std::ios::in | std::ios::binary);
      std::ofstream outFilt((cpdir + "/ssrlcv-filtered.ply").c_str(), std::ios::out | std::ios::binary);
      outFilt << inFilt.rdbuf();

      // Copy PLY file from triangulation checkpoint
      std::string triPlyPath = getCheckpointDirForStage(checkpointRootDir, cp_stage::tri) + "/ssrlcv-initial.ply";
      std::ifstream inTri(triPlyPath.c_str(), std::ios::in | std::ios::binary);
      std::ofstream outTri((cpdir + "/ssrlcv-initial.ply").c_str(), std::ios::out | std::ios::binary);
      outTri << inTri.rdbuf();

      // Copy Log file
      std::ifstream inLog((checkpointRootDir + "/ssrlcv.log").c_str(), std::ios::in | std::ios::binary);
      std::ofstream outLog((cpdir + "/ssrlcv.log").c_str(), std::ios::out | std::ios::binary);
      outLog << inLog.rdbuf();
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
