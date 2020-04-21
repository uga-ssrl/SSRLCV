
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
#include "Logger.hpp"

/**
 * The global logger
 */
ssrlcv::Logger logger;

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
    logger = ssrlcv::Logger("out"); // log in the out directory
    logger.logState("start"); // these can be used to time parts of the pipeline afterwards and correlate it with ofther stuff
    logger.startBackgoundLogging(1); // write a voltage, current, power log every 5 seconds

    //ARG PARSING

    logger.logState("reading images");
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
    logger.logState("done reading images");

    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      logger.logState("generating features");
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,2,0.8);
      features->transferMemoryTo(ssrlcv::cpu);
      images.push_back(image);
      allFeatures.push_back(features);
      logger.logState("done generating features");
    }

    //
    // MATCHING
    //

    std::cout << "Starting matching..." << std::endl;

    logger.logState("generating seed matches");
    ssrlcv::Unity<float>* seedDistances = (seedProvided) ? matchFactory.getSeedDistances(allFeatures[0]) : nullptr;
    ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1],seedDistances);
    if(seedDistances != nullptr) delete seedDistances;
    logger.logState("done generating seed matches");

    distanceMatches->transferMemoryTo(ssrlcv::cpu);
    float maxDist = 0.0f;
    for(int i = 0; i < distanceMatches->size(); ++i){
      if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
    }
    printf("max euclidean distance between features = %f\n",maxDist);
    if(distanceMatches->getMemoryState() != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);
    delete distanceMatches;

    /*
    std::string delimiter = "/";
    std::string matchFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter)) + "/matches.txt";
    ssrlcv::writeMatchFile(matches, matchFile);
    */

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
      std::cout << "Attempting 2-view Triangulation" << std::endl;

      // if we are checkout errors
      errors = new ssrlcv::Unity<float>(nullptr,matchSet.matches->size(),ssrlcv::cpu);

      logger.logState("triangulation");
      float* linearError = (float*)malloc(sizeof(float));
      bundleSet = demPoints.generateBundles(&matchSet,images);

      // the bundles can also be printed
      Unity<float3>* testpositions = new Unity<float3>(nullptr,bundleSet.lines->size(),gpu);
      Unity<float3>* testnormals   = new Unity<float3>(nullptr,bundleSet.lines->size(),gpu);
      for (int b = 0; b < bundleSet.lines->size(); b++) {
        testpositions->host[b] = bundleSet.lines->host[b].pnt;
        testnormals->host[b]   = bundleSet.lines->host[b].vec;
      }
      ssrlcv::writePLY("pushbroomBundles",testpositions,testnormals);

      points = demPoints.twoViewTriangulate(bundleSet, errors, linearError);
      logger.logState("done triangulation");
      ssrlcv::writePLY("unfiltered",points);
      std::cout << "\tUnfiltered Linear Error: " << std::fixed << std::setprecision(12) << *linearError << std::endl;
      ssrlcv::writeCSV(errors, "initial2ViewErrors");
      finalMesh.setPoints(points);
      // ssrlcv::Unity<float>* neighborDists = finalMesh.calculateAverageDistancesToNeighbors(6);
      float avgDist = finalMesh.calculateAverageDistanceToNeighbors(6);
      std::cout << "\tAverage Distance to 6 neighbors: " << avgDist << std::endl;
      demPoints.saveDebugLinearErrorCloud(&matchSet,images, "linearErrorsColored");
      // it's good to do a cutoff filter first how this is chosen is mostly based on ur gut
      // if a poor estimate is chosen then you will have to statistical filter multiple times
      // option 1: pick a fixed value
        // unless scaled, the point cloud is in km. This is the maximum "missmatch" distance between lines to allow in km
        demPoints.linearCutoffFilter(&matchSet,images, 100.0); // <--- removes linear errors over 100 km
      // option 2: tie the initial cutoff to some fraction of the initial linear error
        // demPoints.linearCutoffFilter(&matchSet,images,*linearError / (bundleSet.bundles->size() * 3));
      // option 3: don't use the linear cutoff at all and just use multiple statistical filters (it is safer)
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.twoViewTriangulate(bundleSet, errors, linearError);
      ssrlcv::writePLY("linearCutoff",points);
      logger.logState("start filter");
      // here you can filter points in a number of ways before bundle adjustment or triangulation
      float sigma_filter = 1.0;
      demPoints.deterministicStatisticalFilter(&matchSet,images, sigma_filter, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.twoViewTriangulate(bundleSet, errors, linearError);
      std::cout << "Filted " << sigma_filter  << " Linear Error: " << std::fixed << std::setprecision(12) << *linearError << std::endl;
      finalMesh.setPoints(points);
      ssrlcv::writeCSV(errors, "initial2ViewErrors");
      // ssrlcv::Unity<float>* neighborDists = finalMesh.calculateAverageDistancesToNeighbors(6);
      avgDist = finalMesh.calculateAverageDistanceToNeighbors(6);
      std::cout << "\tAverage Distance to 6 neighbors: " << avgDist << std::endl;
      bundleSet = demPoints.generateBundles(&matchSet,images);
      logger.logState("end filter");

      /*
      // OPTIONAL
      // a second filter can re-filter the new error histogram
      // this is usually a good idea, as there will be new relative extrema to remove
      // doing this too many times will simply over filter the point cloud
      demPoints.deterministicStatisticalFilter(&matchSet,images, 2.0, 0.1); // <---- samples 10% of points and removes anything past 2.0 sigma
      bundleSet = demPoints.generateBundles(&matchSet,images);
      */

      /*
      // Planar filtering is very good at removing noise that is not close to the estimated model.
      demPoints.planarCutoffFilter(&matchSet, images, 10.0f); // <---- this will remove any points more than +/- 10 km from the  estimated plane
      bundleSet = demPoints.generateBundles(&matchSet,images);
      */

      /*
      // OPTIONAL
      // a sensitivity analysis allows one to view the functions and camera parameter derivates pre bundles adjustment
      // this should not be used in produciton and is really only useful for debugging optimizers used in bundle adjustment
      std::string temp_filename = "sensitivity";
      demPoints.generateSensitivityFunctions(&matchSet,images,temp_filename);
      */

      /*
      // could output pre mesh related stuff:
      ssrlcv::writePLY("pointcloud01", points);
      ssrlcv::writeCSV(points, "pointcloud01");
      */

      /*
      // OPTIONAL
      // to compare a points cloud with a ground truth model the first need to be scaled
      // the distance values here are in km but most truth models are in meters
      demPoints.scalePointCloud(1000.0,points); // scales from km into meters
      // rotate pi around the y axis
      float3 rotation = {0.0f, PI, 0.0f};
      demPoints.rotatePointCloud(rotation, points);
      // OPTIONAL
      // to visualize the estimated plane which the structure lies within you can use
      // the demPoints.visualizePlaneEstimation() method like so:
      demPoints.visualizePlaneEstimation(points, images, "planeEstimation", 10000); // usually in km, this is now only 10 km bc of scaling
      // load the example mesh to do the comparison, here I assume we are using the everst PLY
      meshBoi.loadMesh("data/truth/Everest_ground_truth.ply");
        // to save a mesh as a PLY simply:
        // meshBoi.saveMesh("testMesh");
      // to calculate the "missmatch" between the point cloud and the ground truth you can use this method:
      float error = meshBoi.calculateAverageDifference(points, {0.0f , 0.0f, 10.0f}); // (0,0,1) is the Normal to the X-Y plane, which the point cloud and mesh are on
      std::cout << "Average error to ground truth is: " << error << " km, " << (error * 1000) << " meters" << std::endl;
      // this methods saves the error on each point
      ssrlcv::Unity<float>* truthErrors = meshBoi.calculatePerPointDifference(points, {0.0f , 0.0f, 1.0f});
      // then you can save these errors in a CSV
      ssrlcv::writeCSV(truthErrors, "resolutionErrors");
      // you can also save them as color coded
      ssrlcv::writePLY("resolutionErrors",points, truthErrors, 300); // NOTE it has already been scaled to meters, set error the cutoff to 300 meters
      */

      /*
      // OPTIONAL
      // Tests can be done with bundle adjustment to check bounds on how
      // well it performs
      ssrlcv::Unity<float>* noise = new ssrlcv::Unity<float>(nullptr,6,ssrlcv::cpu);
      noise->host[0] = 0.0; // X
      noise->host[1] = 0.2; // Y
      noise->host[2] = 0.0; // Z
      noise->host[3] = 0.0; // X^
      noise->host[4] = 0.0; // Y^
      noise->host[5] = 0.0; // Z^
      demPoints.testBundleAdjustmentTwoView(&matchSet,images, 10, noise);
      */

      // starting bundle adjustment here
      // std::cout << "Starting Bundle Adjustment Loop ..." << std::endl;
      // points = demPoints.BundleAdjustTwoView(&matchSet,images, 10);


      // begin mesh-level tasks, there are no more cameras or matches after this stage

      /*
      // set the mesh points
      finalMesh.setPoints(points);


      // you can filter these points and view their distributions in multiple ways
      ssrlcv::Unity<float>* neighborDists = finalMesh.calculateAverageDistancesToNeighbors(6); // calculate average distance to 6 neighbors
      ssrlcv::writeCSV(neighborDists, "neighborDistances");
      float avgDist = finalMesh.calculateAverageDistanceToNeighbors(6); // the average distance from any even node to another
      std::cout << "Average Distance to 6 neighbors is: " << avgDist << std::endl;
      ssrlcv::writePLY("neighborDistancesColored",points,neighborDists,(2.0f * avgDist)); // a point cloud with colored neighbor dists


      // to only keep points within a certain sigma of neighbor distance use the following filter
      finalMesh.filterByNeighborDistance(3.0); // <--- filter bois past 3.0 sigma (about 99.5% of points) if 2 view is good then this is usually good
      finalMesh.savePoints("densityFiltered");


      //  try a VSFM compare
      ssrlcv::MeshFactory vsfm = ssrlcv::MeshFactory();
      vsfm.loadPoints("../vsfm-test.ply");
      demPoints.scalePointCloud(1000.0,vsfm.points); //
      float3 rotation2 = {0.0f, PI, 0.0f};
      demPoints.rotatePointCloud(rotation2,vsfm.points);
      // now try to translate it back where it should go
      ssrlcv::Unity<float3>* point1 = demPoints.getAveragePoint(meshBoi.points);
      ssrlcv::Unity<float3>* point2 = demPoints.getAveragePoint(vsfm.points);
      point1->host[0] -= point2->host[0];
      demPoints.translatePointCloud(point2->host[0], vsfm.points);
      // save the new VSFM scaled points
      ssrlcv::writePLY("vsfm", vsfm.points);
      // compare to ground truth
      float error2 = meshBoi.calculateAverageDifference(vsfm.points, {0.0f , 0.0f, 1.0f}); // (0,0,1) is the Normal to the X-Y plane, which the point cloud and mesh are on
      std::cout << "VSFM average error to ground truth is: " << error2 << " meters" << std::endl;
      */



    } else {
      //
      // N View Case
      //
      std::cout << "Attempting N-view Triangulation" << std::endl;

      // if we are checkout errors
      errors = new ssrlcv::Unity<float>(nullptr,matchSet.matches->size(),ssrlcv::cpu);

      logger.logState("triangulation");
      float* angularError = (float*)malloc(sizeof(float));
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.nViewTriangulate(bundleSet, errors, angularError);
      std::cout << "\t >>>>>>>> TOTOAL POINTS: " << points->size() << std::endl;
      ssrlcv::writePLY("unfiltered",points);
      logger.logState("done triangulation");

      demPoints.saveDebugLinearErrorCloud(&matchSet,images, "linearErrorsColored");
      demPoints.saveViewNumberCloud(&matchSet,images, "ViewNumbers");
      ssrlcv::writeCSV(errors, "nViewInitialErrors");

      std::cout << "\tUnfiltered Linear Error: " << std::fixed << std::setprecision(12) << *angularError << std::endl;
      //ssrlcv::writeCSV(errors->host, (int) errors->size(), "individualAngularErrors1");

      finalMesh.setPoints(points);
      // ssrlcv::Unity<float>* neighborDists = finalMesh.calculateAverageDistancesToNeighbors(6);
      float avgDist = finalMesh.calculateAverageDistanceToNeighbors(6);
      std::cout << "\tAverage Distance to 6 neighbors: " << avgDist << std::endl;

      ssrlcv::writePLY("attemptAtSomeStuff",points, errors, 300);


      /*
      logger.logState("start filter");
      demPoints.linearCutoffFilter(&matchSet, images, 100.0); // unless scaled, the point cloud is in km. This is the maximum "missmatch" distance between lines to allow in km
      bundleSet = demPoints.generateBundles(&matchSet,images);
      */

      // Planar filtering is very good at removing noise that is not close to the estimated model.
      //demPoints.planarCutoffFilter(&matchSet, images, 10.0f); // <---- this will remove any points more than +/- 10 km from the  estimated plane
      //bundleSet = demPoints.generateBundles(&matchSet,images);

      // multiple filters are needed, because outlier points are discovered in stages
      // decreasing sigma over time is best because the real "mean" error becomes more
      // accurate as truely noisey points are removed
      float sigma_filter = 3.0;
      demPoints.deterministicStatisticalFilter(&matchSet,images, sigma_filter, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      bundleSet = demPoints.generateBundles(&matchSet,images);
      errors = new ssrlcv::Unity<float>(nullptr,matchSet.matches->size(),ssrlcv::cpu);
      points = demPoints.nViewTriangulate(bundleSet, angularError);
      std::cout << "\t >>>>>>>> TOTOAL POINTS: " << points->size() << std::endl;
      std::cout << "Filted " << sigma_filter  << " Linear Error: " << std::fixed << std::setprecision(12) << *angularError << std::endl;
      // demPoints.saveDebugLinearErrorCloud(&matchSet,images, "linearErrorsColored2");
      finalMesh.setPoints(points);
      // ssrlcv::Unity<float>* neighborDists = finalMesh.calculateAverageDistancesToNeighbors(6);
      avgDist = finalMesh.calculateAverageDistanceToNeighbors(6);
      std::cout << "\tAverage Distance to 6 neighbors: " << avgDist << std::endl;
      // ssrlcv::writeCSV(errors, "nViewFilteredErrors");
      ssrlcv::writePLY("filtered",points);


      // float sigma_filter = 1.0;
      // demPoints.deterministicStatisticalFilter(&matchSet,images, sigma_filter, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      // bundleSet = demPoints.generateBundles(&matchSet,images);
      // points = demPoints.twoViewTriangulate(bundleSet, errors, linearError);
      // std::cout << "Filted " << sigma_filter  << " Linear Error: " << std::fixed << std::setprecision(12) << *linearError << std::endl;
      // finalMesh.setPoints(points);
      // ssrlcv::writeCSV(errors, "initial2ViewErrors");
      // // ssrlcv::Unity<float>* neighborDists = finalMesh.calculateAverageDistancesToNeighbors(6);
      // avgDist = finalMesh.calculateAverageDistanceToNeighbors(6);
      // std::cout << "\tAverage Distance to 6 neighbors: " << avgDist << std::endl;
      // bundleSet = demPoints.generateBundles(&matchSet,images);


      // for (int i = 0; i < 3; i++){
      //   demPoints.deterministicStatisticalFilter(&matchSet,images, 3.0, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      //   bundleSet = demPoints.generateBundles(&matchSet,images);
      // }
      // for (int i = 0; i < 6; i++){
      //   demPoints.deterministicStatisticalFilter(&matchSet,images, 1.0, 0.1); // <---- samples 10% of points and removes anything past 1.0 sigma
      //   bundleSet = demPoints.generateBundles(&matchSet,images);
      // }
      // // then, if the cloud is large enough still, one last filter
      // demPoints.deterministicStatisticalFilter(&matchSet,images, 0.2, 0.1); // <---- samples 10% of points and removes anything past 0.2 sigma
      // bundleSet = demPoints.generateBundles(&matchSet,images);
      logger.logState("end filter");

      /*
      // now redo triangulation with the newlyfiltered boi
      points = demPoints.nViewTriangulate(bundleSet, errors, angularError);

      // OPTIONAL
      // to compare a points cloud with a ground truth model the first need to be scaled
      // the distance values here are in km but most truth models are in meters
      demPoints.scalePointCloud(1000.0,points); // scales from km into meters
      // rotate pi around the y axis
      float3 rotation = {0.0f, PI, 0.0f};
      demPoints.rotatePointCloud(rotation, points);
      // OPTIONAL
      // to visualize the estimated plane which the structure lies within you can use
      // the demPoints.visualizePlaneEstimation() method like so:
      demPoints.visualizePlaneEstimation(points, images, "planeEstimation", 10000); // usually in km, this is now only 10 km bc of scaling
      // you can compare to a "ground truth" mesh
      // load the example mesh to do the comparison, here I assume we are using the everst PLY
      meshBoi.loadMesh("data/truth/Everest_ground_truth.ply");
      float error = meshBoi.calculateAverageDifference(points, {0.0f , 0.0f, 1.0f}); // (0,0,1) is the Normal to the X-Y plane, which the point cloud and mesh are on
      std::cout << "Average error to ground truth is: " << error << " km, " << (error * 1000) << " meters" << std::endl;
      ssrlcv::Unity<float>* truthErrors = meshBoi.calculatePerPointDifference(points, {0.0f , 0.0f, 1.0f});
      // then you can save these errors in a CSV
      ssrlcv::writeCSV(truthErrors, "resolutionErrors");
      // you can also save them as color coded
      ssrlcv::writePLY("resolutionErrors",points, truthErrors, 300); // NOTE it has already been scaled to meters, set error the cutoff to 300 meters


      //ssrlcv::writeCSV(errors->host, (int) errors->size(), "individualAngularErrors2");
      demPoints.saveDebugCloud(points, bundleSet, images);

      // begin mesh-level tasks, there are no more cameras or matches after this stage

      // set the mesh points
      finalMesh.setPoints(points);


      // you can filter these points and view their distributions in multiple ways
      ssrlcv::Unity<float>* neighborDists = finalMesh.calculateAverageDistancesToNeighbors(6); // calculate average distance to 6 neighbors
      ssrlcv::writeCSV(neighborDists, "neighborDistances");
      float avgDist = finalMesh.calculateAverageDistanceToNeighbors(6); // the average distance from any even node to another
      std::cout << "Average Distance to 6 neighbors is: " << avgDist << std::endl;
      ssrlcv::writePLY("neighborDistancesColored",points,neighborDists,(2.0f * avgDist)); // a point cloud with colored neighbor dists


      // to only keep points within a certain sigma of neighbor distance use the following filter
      // with the nview case the noise goes up the number of views you add, unless you
      finalMesh.filterByNeighborDistance(1.0); // <--- filter bois past 1.5 sigma (about 95% of points)
      finalMesh.savePoints("densityFiltered"); //
      */

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
