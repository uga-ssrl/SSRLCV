
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

    //
    // MATCHING
    //

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
    // ssrlcv::writeMatchFile(matches, matchFile);

    // Need to fill into to MatchSet boi
    std::cout << "Generating MatchSet ..." << std::endl;
    ssrlcv::MatchSet matchSet;

    if (images.size() == 2){
      //
      // 2 View Case
      //
      matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matches->size()*2,ssrlcv::cpu);
      matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matches->size(),ssrlcv::cpu);
      matches->setMemoryState(ssrlcv::cpu);
      matchSet.matches->setMemoryState(ssrlcv::cpu);
      matchSet.keyPoints->setMemoryState(ssrlcv::cpu);
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
      matchSet = matchFactory.generateMatchesExaustive(images,allFeatures);
      matches->setMemoryState(ssrlcv::cpu);
      matchSet.matches->setMemoryState(ssrlcv::cpu);
      matchSet.keyPoints->setMemoryState(ssrlcv::cpu);

      // optional to save output
      // matchSet.keyPoints->checkpoint(0,"out/kp");
      // matchSet.matches->checkpoint(0,"out/m");
    }

    // the bois
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::MeshFactory meshBoi = ssrlcv::MeshFactory();
    ssrlcv::Unity<float3>* points;
    ssrlcv::Unity<float>* errors;
    ssrlcv::BundleSet bundleSet;

    if (images.size() == 2){
      //
      // 2 View Case
      //
      std::cout << "Attempting 2-view Triangulation" << std::endl;

      float* linearError = (float*)malloc(sizeof(float));
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.twoViewTriangulate(bundleSet, linearError);
      ssrlcv::writePLY("out/unfiltered.ply",points);
      demPoints.saveDebugLinearErrorCloud(&matchSet,images, "linearErrorsColored");
      // it's good to do a cutoff filter first how this is chosen is mostly based on ur gut
      // if a poor estimate is chosen then you will have to statistical filter multiple times
      // option 1: pick a fixed value
        demPoints.linearCutoffFilter(&matchSet,images,1000); // <--- removes linear errors over 1000
      // option 2: tie the initial cutoff to some fraction of the initial linear error
        // demPoints.linearCutoffFilter(&matchSet,images,*linearError / (bundleSet.bundles->size() * 3));
      // option 3: don't use the linear cutoff at all and just use multiple statistical filters (it is safer)
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.twoViewTriangulate(bundleSet, linearError);
      ssrlcv::writePLY("out/linearCutoff.ply",points);
      // here you can filter points in a number of ways before bundle adjustment or triangulation
      demPoints.deterministicStatisticalFilter(&matchSet,images, 1.0, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      bundleSet = demPoints.generateBundles(&matchSet,images);

      /*
      // OPTIONAL
      // a second filter can re-filter the new error histogram
      // this is usually a good idea, as there will be new relative extrema to remove
      // doing this too many times will simply over filter the point cloud
      demPoints.deterministicStatisticalFilter(&matchSet,images, 2.0, 0.1); // <---- samples 10% of points and removes anything past 2.0 sigma
      bundleSet = demPoints.generateBundles(&matchSet,images);
      */

      // Planar filtering is very good at removing noise that is not close to the estimated model.
      demPoints.planarCutoffFilter(&matchSet, images, 10.0f); // <---- this will remove any points more than +/- 10 km from the  estimated plane

      // OPTIONAL
      // to visualize the estimated plane which the structure lies within you can use
      // the demPoints.visualizePlaneEstimation() method like so:
      demPoints.visualizePlaneEstimation(points, images, "planeEstimation");

      // the version that will be used normally
      points = demPoints.twoViewTriangulate(bundleSet, linearError);
      std::cout << "Total Linear Error: " << std::fixed << std::setprecision(12) << *linearError << std::endl;

      /*
      // OPTIONAL
      // a sensitivity analysis allows one to view the functions and camera parameter derivates pre bundles adjustment
      // this should not be used in produciton and is really only useful for debugging optimizers used in bundle adjustment
      std::string temp_filename = "sensitivity";
      demPoints.generateSensitivityFunctions(&matchSet,images,temp_filename);
      */

      /*
      // OPTIONAL
      // Varoius scaling examples:
        demPoints.scalePointCloud(10.0,points);
        ssrlcv::writePLY("out/scaledx10.ply",points);
        demPoints.scalePointCloud(10.0,points);
        ssrlcv::writePLY("out/scaledx100.ply",points);
        demPoints.scalePointCloud(10.0,points);
        ssrlcv::writePLY("out/scaledx1000.ply",points);
      */

      // OPTIONAL
      // to compare a points cloud with a ground truth model the first need to be scaled
      // the distance values here are in km but most truth models are in meters
      demPoints.scalePointCloud(1000.0,points); // scales from km into meters
      // rotate pi around the y axis
      float3 rotation = {0.0f, PI, 0.0f};
      demPoints.rotatePointCloud(rotation, points);
      // load the example mesh to do the comparison, here I assume we are using the everst PLY
      meshBoi.loadMesh("data/truth/Everest_ground_truth.ply");
        // to save a mesh as a PLY simply:
        // meshBoi.saveMesh("testMesh");
      // to calculate the "missmatch" between the point cloud and the ground truth you can use this method:
      float error = meshBoi.calculateAverageDifference(points, {0.0f , 0.0f, 1.0f}); // (0,0,1) is the Normal to the X-Y plane, which the point cloud and mesh are on
      std::cout << "Average error to ground truth is: " << error << " km, " << (error * 1000) << " meters" << std::endl;
      // this methods saves the error on each point
      ssrlcv::Unity<float>* truthErrors = meshBoi.calculatePerPointDifference(points, {0.0f , 0.0f, 1.0f});
      // then you can save these errors in a CSV
      ssrlcv::writeCSV(truthErrors, "resolutionErrors");
      // you can also save them as color coded
      ssrlcv::writePLY("resolutionErrors",points, truthErrors, 300); // NOTE it has already been scaled to meters, set error the cutoff to 300 meters

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


    } else {
      //
      // N View Case
      //
      std::cout << "Attempting N-view Triangulation" << std::endl;

      // if we are checkout errors
      errors = new ssrlcv::Unity<float>(nullptr,matchSet.matches->size(),ssrlcv::cpu);

      float* angularError = (float*)malloc(sizeof(float));
      bundleSet = demPoints.generateBundles(&matchSet,images);
      points = demPoints.nViewTriangulate(bundleSet, errors, angularError);
      ssrlcv::writePLY("out/unfiltered.ply",points);

      demPoints.saveDebugLinearErrorCloud(&matchSet,images, "linearErrorsColored");
      demPoints.saveViewNumberCloud(&matchSet,images, "ViewNumbers");

      std::cout << "Initial Angular Error: " << *angularError << std::endl;
      //ssrlcv::writeCSV(errors->host, (int) errors->size(), "individualAngularErrors1");

      demPoints.linearCutoffFilter(&matchSet,images,300);
      bundleSet = demPoints.generateBundles(&matchSet,images);

      // Planar filtering is very good at removing noise that is not close to the estimated model.
      demPoints.planarCutoffFilter(&matchSet, images, 10.0f); // <---- this will remove any points more than +/- 10 km from the  estimated plane

      // OPTIONAL
      // to visualize the estimated plane which the structure lies within you can use
      // the demPoints.visualizePlaneEstimation() method like so:
      demPoints.visualizePlaneEstimation(points, images, "planeEstimation");

      // multiple filters are needed, because outlier points are discovered in stages
      // decreasing sigma over time is best because the real "mean" error becomes more
      // accurate as truely noisey points are removed
      for (int i = 0; i < 3; i++){
        demPoints.deterministicStatisticalFilter(&matchSet,images, 3.0, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
        bundleSet = demPoints.generateBundles(&matchSet,images);
      }
      for (int i = 0; i < 6; i++){
        demPoints.deterministicStatisticalFilter(&matchSet,images, 1.0, 0.1); // <---- samples 10% of points and removes anything past 1.0 sigma
        bundleSet = demPoints.generateBundles(&matchSet,images);
      }
      // then, if the cloud is large enough still, one last filter
      demPoints.deterministicStatisticalFilter(&matchSet,images, 0.2, 0.1); // <---- samples 10% of points and removes anything past 0.2 sigma
      bundleSet = demPoints.generateBundles(&matchSet,images);

      // now redo triangulation with the newlyfiltered boi
      points = demPoints.nViewTriangulate(bundleSet, errors, angularError);

      // OPTIONAL
      // to compare a points cloud with a ground truth model the first need to be scaled
      // the distance values here are in km but most truth models are in meters
      demPoints.scalePointCloud(1000.0,points); // scales from km into meters
      // rotate pi around the y axis
      float3 rotation = {0.0f, PI, 0.0f};
      demPoints.rotatePointCloud(rotation, points);
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

    }

    std::cout << "writing final PLY ..." << std::endl;
    ssrlcv::writePLY("final",points);

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
