#include "Pipeline.cuh"

  template <typename T>
  std::string getCheckpoint(std::string directory, int id) {
    return directory + "/" + std::to_string(id) + "_" + typeid(T).name() + ".uty";
  }

  std::string getImgCheckpoint(std::string directory, int id) {
    return directory + "/" + std::to_string(id) + "_" + typeid(ssrlcv::Image).name() + ".cpimg";
  }

  //////////////////////////////
  // FEATURE GENERATION
  //////////////////////////////

  void ssrlcv::doFeatureGeneration(ssrlcv::FeatureGenerationInput *in, ssrlcv::FeatureGenerationOutput *out) {
    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);
  
    logger.logState("SEED");
    if (in->seedPath.size() > 0) {
      // new image with path and ID
      ssrlcv::ptr::value<ssrlcv::Image> seed = ssrlcv::ptr::value<ssrlcv::Image>(in->seedPath,-1);
      // array of features containing sift descriptors at every point
  
      out->seedFeatures = featureFactory.generateFeatures(seed,false,2,0.8);
    }
    logger.logState("SEED");
  
    logger.logState("FEATURES");
    
    float3 offset;

    for (int i = 0; i < in->numImages; i ++) {
      // new image with path and ID
      ssrlcv::ptr::value<ssrlcv::Image> image = ssrlcv::ptr::value<ssrlcv::Image>(in->imagePaths[i], i);
      
      if (i == 0)
        offset = image->camera.cam_pos;
      image->camera.ecef_offset = offset;
      image->camera.cam_pos -= offset;

      // array of features containing sift descriptors at every point
      ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> features =
                featureFactory.generateFeatures(image,false,2,0.8);
      features->transferMemoryTo(ssrlcv::cpu);
      out->images.push_back(image);
      out->allFeatures.push_back(features);
    }
    logger.logState("FEATURES");
  
  }

  //////////////////////////////
  // POSE ESTIMATION
  //////////////////////////////

  void ssrlcv::PoseEstimationInput::fromCheckpoint(std::string featureGenDirectory, int numImages) {
    struct stat buf;
    std::string seedCpPath = getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(featureGenDirectory, -1);
    if (stat(seedCpPath.c_str(), &buf) == 0) {
      seedFeatures = seedCpPath;
    }
    for (int i = 0; i < numImages; i ++) {
      images.push_back(ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(featureGenDirectory, i), i, true));
      allFeatures.push_back(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(featureGenDirectory, i));
    }
  }
  void ssrlcv::PoseEstimationInput::fromPreviousStage(FeatureGenerationOutput *featureGenOutput) {
    this->seedFeatures = featureGenOutput->seedFeatures;
    this->allFeatures = featureGenOutput->allFeatures;
    this->images = featureGenOutput->images;
  }

  void ssrlcv::doPoseEstimation(ssrlcv::PoseEstimationInput *in, ssrlcv::PoseEstimationOutput *out) {
    logger.info << "Starting pose estimation...";
    logger.logState("POSE");

    // TODO: Put in loop for N-View (only works for 2-view right now)

    //ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> matches(std::string("tmp/0_N6ssrlcv5MatchE.uty"));

    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,10.0f*10.0f);

    if (in->seedFeatures != nullptr)
      matchFactory.setSeedFeatures(in->seedFeatures);

    out->seedDistances = (in->seedFeatures != nullptr) ? matchFactory.getSeedDistances(in->allFeatures[0]) : nullptr;

    logger.logState("done generating seed matches");

    logger.logState("matching images");
    //ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> matches = matchFactory.generateMatches(in->images[0], in->allFeatures[0], in->images[1], in->allFeatures[1], out->seedDistances);
    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> matches = matchFactory.generateMatchesDoubleConstrained(in->images[0], in->allFeatures[0], in->images[1], in->allFeatures[1], 100, 3, out->seedDistances);
    logger.logState("done matching images");

    //matches->checkpoint(0, "tmp/");


    matches->transferMemoryTo(ssrlcv::cpu);
    ssrlcv::PoseEstimator estim(in->images.at(0), in->images.at(1), matches);
    //ssrlcv::Pose pose = estim.estimatePoseRANSAC();

    ssrlcv::Pose pose;

    float3 pos = in->images.at(0)->camera.cam_pos - in->images.at(1)->camera.cam_pos;
    pos = rotatePointArbitrary(pos, {0, 0, 1}, - in->images.at(0)->camera.cam_rot.z);
    pos = rotatePointArbitrary(pos, {0, 1, 0}, - in->images.at(0)->camera.cam_rot.y);
    pos = rotatePointArbitrary(pos, {1, 0, 0}, - in->images.at(0)->camera.cam_rot.x);
    pose.x = pos.x;
    pose.y = pos.y;
    pose.z = pos.z;

    float C0[3][3], C0t[3][3], C1[3][3], relative[3][3];
    ssrlcv::getRotationMatrix(in->images.at(0)->camera.cam_rot, C0);
    ssrlcv::transpose(C0, C0t);
    ssrlcv::getRotationMatrix(in->images.at(1)->camera.cam_rot, C1);
    ssrlcv::multiply(C0t, C1, relative);
    float3 rot = ssrlcv::getAxisRotations(relative);
    pose.roll = rot.x;
    pose.pitch = rot.y;
    pose.yaw = rot.z;
    printf("Original pose: %f %f %f\n", pose.roll, pose.pitch, pose.yaw);

    estim.LM_optimize(&pose);


    float R1[3][3], R2[3][3], R[3][3];
    logger.info.printf("Original Position: %f %f %f", in->images.at(1)->camera.cam_pos.x, in->images.at(1)->camera.cam_pos.y, in->images.at(1)->camera.cam_pos.z);
    in->images.at(1)->camera.cam_pos = in->images.at(0)->camera.cam_pos + ssrlcv::rotatePoint({1000 * pose.x, 1000 * pose.y, 1000 * pose.z}, in->images.at(0)->camera.cam_rot);
    ssrlcv::getRotationMatrix({pose.roll, pose.pitch, pose.yaw}, R1);
    ssrlcv::getRotationMatrix(in->images.at(0)->camera.cam_rot, R2);
    ssrlcv::multiply(R2, R1, R);
    in->images.at(1)->camera.cam_rot = ssrlcv::getAxisRotations(R);
    logger.info.printf("Rotation: %f %f %f", in->images.at(1)->camera.cam_rot.x, in->images.at(1)->camera.cam_rot.y, in->images.at(1)->camera.cam_rot.z);
    logger.info.printf("Position: %f %f %f", in->images.at(1)->camera.cam_pos.x, in->images.at(1)->camera.cam_pos.y, in->images.at(1)->camera.cam_pos.z);

    logger.logState("POSE");
  }

  //////////////////////////////
  // FEATURE MATCHING
  //////////////////////////////

  void ssrlcv::FeatureMatchingInput::fromCheckpoint(std::string featureGenDirectory, std::string poseDirectory, int numImages, float epsilon, float delta) {
    struct stat buf;
    std::string seedCpPath = getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(featureGenDirectory, -1);
    if (stat(seedCpPath.c_str(), &buf) == 0) {
      seedFeatures = seedCpPath;
    }
    for (int i = 0; i < numImages; i ++) {
      images.push_back(ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(featureGenDirectory, i), i, true));
      allFeatures.push_back(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(featureGenDirectory, i));
    }
    this->epsilon = epsilon;
    this->delta = delta;
    // Pose estimation only supported in 2 view
    if (numImages == 2) {
      std::string seedDistPath = getCheckpoint<float>(poseDirectory, 0);
      if (stat(seedDistPath.c_str(), &buf) == 0) {
        seedDistances = seedDistPath;
      }
    }
  }
  void ssrlcv::FeatureMatchingInput::fromPreviousStage(PoseEstimationInput *poseInput, PoseEstimationOutput *poseOutput, float epsilon, float delta) {
    this->seedFeatures = poseInput->seedFeatures;
    this->allFeatures = poseInput->allFeatures;
    this->images = poseInput->images;
    this->seedDistances = poseOutput->seedDistances;
    this->epsilon = epsilon;
    this->delta = delta;
  }
  
  void ssrlcv::doFeatureMatching(ssrlcv::FeatureMatchingInput *in, ssrlcv::FeatureMatchingOutput *out) {
    logger.info << "Starting matching...";
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,200.0f*200.0f);
    logger.logState("MATCHING");
    // logger.logState("generating seed matches");
    if (in->seedFeatures != nullptr)
      matchFactory.setSeedFeatures(in->seedFeatures);

    if (in->seedDistances == nullptr)
      in->seedDistances = (in->seedFeatures != nullptr) ? matchFactory.getSeedDistances(in->allFeatures[0]) : nullptr;
  
    if (in->images.size() == 2){
      //
      // 2 View Case
      //
      logger.logState("done generating seed matches");

      logger.logState("matching images");
      #if GEO_ORBIT == 1
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> distanceMatches = matchFactory.generateDistanceMatchesDoubleConstrained(in->images[0], in->allFeatures[0], in->images[1], in->allFeatures[1], in->epsilon, in->delta, in->seedDistances);
      #else
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> distanceMatches = matchFactory.generateDistanceMatches(in->images[0], in->allFeatures[0], in->images[1], in->allFeatures[1], in->seedDistances);
      #endif
      logger.logState("done matching images");
    
      distanceMatches->transferMemoryTo(ssrlcv::cpu);
      float maxDist = 0.0f;
      ssrlcv::DMatch *dhost = distanceMatches->host.get();
      for(int i = 0; i < distanceMatches->size(); ++i){
        if(maxDist < dhost[i].distance) maxDist = dhost[i].distance;
      }
      logger.info.printf("max euclidean distance between features = %f",maxDist);
      if(distanceMatches->getMemoryState() != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
      ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> matches = matchFactory.getRawMatches(distanceMatches);
      
      // Need to fill into to MatchSet boi
      logger.info << "Generating MatchSet ...";
      out->matchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,matches->size()*2,ssrlcv::cpu);
      out->matchSet.matches = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,matches->size(),ssrlcv::cpu);
      matches->setMemoryState(ssrlcv::cpu);
      out->matchSet.matches->setMemoryState(ssrlcv::cpu);
      out->matchSet.keyPoints->setMemoryState(ssrlcv::cpu);
      ssrlcv::KeyPoint *okhost = out->matchSet.keyPoints->host.get();
      ssrlcv::Match *mhost = matches->host.get();
      ssrlcv::MultiMatch *omhost = out->matchSet.matches->host.get();
      for(int i = 0; i < out->matchSet.matches->size(); i++){
        okhost[i*2] = mhost[i].keyPoints[0];
        okhost[i*2 + 1] = mhost[i].keyPoints[1];
        omhost[i] = {2,i*2};
      }
      logger.info << "Generated MatchSet ...";
      logger.info << "Total Matches: " + std::to_string(matches->size());
    } else {
      //
      // N View Case
      //
      logger.logState("matching images");
      out->matchSet = matchFactory.generateMatchesExhaustive(in->images, in->allFeatures, in->epsilon, in->delta);
      out->matchSet.matches->setMemoryState(ssrlcv::cpu);
      out->matchSet.keyPoints->setMemoryState(ssrlcv::cpu);
      logger.logState("done matching images");
  
      // optional to save output
      // matchSet.keyPoints->checkpoint(0,"out/kp");
      // matchSet.matches->checkpoint(0,"out/m");
    }
    logger.logState("MATCHING");
  
  }

  //////////////////////////////
  // TRIANGULATION
  //////////////////////////////

  void ssrlcv::TriangulationInput::fromCheckpoint(std::string featureGenDir, std::string featureMatchDir, int numImages) {
    for (int i = 0; i < numImages; i ++) {
      images.push_back(ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(featureGenDir, i), i, true));
    }
    matchSet.keyPoints = getCheckpoint<ssrlcv::KeyPoint>(featureMatchDir, 0);
    matchSet.matches = getCheckpoint<ssrlcv::MultiMatch>(featureMatchDir, 0);
  }

  void ssrlcv::TriangulationInput::fromPreviousStage(FeatureMatchingInput *featureMatchingInput, FeatureMatchingOutput *featureMatchingOutput) {
    this->images = featureMatchingInput->images;
    this->matchSet = featureMatchingOutput->matchSet;
  }

  void ssrlcv::doTriangulation(ssrlcv::TriangulationInput *in, ssrlcv::TriangulationOutput *out) {
    ssrlcv::PointCloudFactory pointCloudFactory = ssrlcv::PointCloudFactory();
    typedef ssrlcv::ptr::value<ssrlcv::Unity<float3>> (ssrlcv::PointCloudFactory::*TriFunc)(ssrlcv::BundleSet, float*);
    TriFunc triangulate = (in->images.size() == 2) ? TriFunc(&ssrlcv::PointCloudFactory::twoViewTriangulate) : TriFunc(&ssrlcv::PointCloudFactory::nViewTriangulate);
  
    logger.info << "Attempting Triangulation";
  
    logger.logState("TRIANGULATE");
  
    float error; // linear for 2-view, angular for N-view
    ssrlcv::BundleSet bundleSet = pointCloudFactory.generateBundles(&in->matchSet,in->images);
    out->points = (pointCloudFactory.*triangulate)(bundleSet, &error);
    std::stringstream ss;
    ss << "\tUnfiltered Error: " << std::fixed << std::setprecision(12) << error;
    logger.info << ss.str();

    ssrlcv::writePLY("ssrlcv-initial", out->points);
    logger.logState("TRIANGULATE");
  }

  //////////////////////////////
  // FILTERING
  //////////////////////////////

  void ssrlcv::FilteringInput::fromCheckpoint(std::string featureGenDir, std::string featureMatchDir, int numImages) {
    for (int i = 0; i < numImages; i ++) {
      images.push_back(ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(featureGenDir, i), i, true));
    }
    matchSet.keyPoints = getCheckpoint<ssrlcv::KeyPoint>(featureMatchDir, 0);
    matchSet.matches = getCheckpoint<ssrlcv::MultiMatch>(featureMatchDir, 0);
  }

  void ssrlcv::FilteringInput::fromPreviousStage(TriangulationInput *triangulationInput) {
    this->images = triangulationInput->images;
    this->matchSet = triangulationInput->matchSet;
  }
  
  void ssrlcv::doFiltering(ssrlcv::FilteringInput *in, ssrlcv::FilteringOutput *out) {
    ssrlcv::PointCloudFactory pointCloudFactory;
    ssrlcv::MeshFactory meshFactory;
  
    logger.logState("FILTER");
  
    if (in->images.size() == 2) {
      float linearError;
  
      pointCloudFactory.linearCutoffFilter(&in->matchSet,in->images, 100.0); // <--- removes linear errors over 100 km
  
      // first time
      float sigma_filter = 3.0;
      pointCloudFactory.deterministicStatisticalFilter(&in->matchSet,in->images, sigma_filter, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      ssrlcv::BundleSet bundleSet = pointCloudFactory.generateBundles(&in->matchSet,in->images);
      out->points = pointCloudFactory.twoViewTriangulate(bundleSet, &linearError);
      std::stringstream ss;
      ss << "Filtered " << sigma_filter  << " Linear Error: " << std::fixed << std::setprecision(12) << linearError;
      logger.info << ss.str();
  
      // second time
      /*
      sigma_filter = 3.0;
      pointCloudFactory.deterministicStatisticalFilter(&in->matchSet,in->images, sigma_filter, 0.1); // <---- samples 10% of points and removes anything past 3.0 sigma
      bundleSet = pointCloudFactory.generateBundles(&in->matchSet,in->images);
      out->points = pointCloudFactory.twoViewTriangulate(bundleSet, &linearError);
      ss.str("");
      ss << "Filtered " << sigma_filter  << " Linear Error: " << std::fixed << std::setprecision(12) << linearError;
      logger.info << ss.str();
      */
  
      // neighbor filter
      //pointCloudFactory.scalePointCloud(1000.0,out->points); // scales from km into meters
      //float3 rotation = {0.0f, PI, 0.0f};
      //pointCloudFactory.rotatePointCloud(rotation, out->points);
    } else {
      float angularError;
  
      for (int i = 0; i < 1; i++) { // could increase for more aggressive filtering
        pointCloudFactory.deterministicStatisticalFilter(&in->matchSet,in->images, 3.0, 0.1); // <---- samples 10% of points and removes anything past 5.0 sigma
        ssrlcv::BundleSet bundleSet = pointCloudFactory.generateBundles(&in->matchSet,in->images);
        out->points = pointCloudFactory.nViewTriangulate(bundleSet, &angularError);
        std::stringstream ss;
        ss << "Filtered " << 0.1  << " Linear Error: " << std::fixed << std::setprecision(12) << angularError;
        logger.info << ss.str();
      }
    }
  
    // set the mesh points
    meshFactory.setPoints(out->points);
    //finalMesh.filterByNeighborDistance(3.0); // <--- filter bois past 3.0 sigma (about 99.5% of points) if 2 view is good then this is usually good
    meshFactory.savePoints("ssrlcv-filtered");
  
    logger.logState("FILTER");
  
  }

  //////////////////////////////
  // BUNDLE ADJUSTMENT
  //////////////////////////////

  void ssrlcv::BundleAdjustInput::fromCheckpoint(std::string featureGenDir, std::string filteringDir, int numImages) {
    for (int i = 0; i < numImages; i ++) {
      images.push_back(ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(featureGenDir, i), i, true));
    }
    matchSet.keyPoints = getCheckpoint<ssrlcv::KeyPoint>(filteringDir, 0);
    matchSet.matches = getCheckpoint<ssrlcv::MultiMatch>(filteringDir, 0);
  }

  void ssrlcv::BundleAdjustInput::fromPreviousStage(FilteringInput *filteringInput) {
    this->images = filteringInput->images;
    this->matchSet = filteringInput->matchSet;
  }
  
  void ssrlcv::doBundleAdjust(ssrlcv::BundleAdjustInput *in, ssrlcv::BundleAdjustOutput *out) {
    if (in->images.size() != 2)
      return; // not yet implemented for N-View
  
    ssrlcv::PointCloudFactory pointCloudFactory;
    ssrlcv::MeshFactory meshFactory;
  
    logger.logState("BA");
    out->points = pointCloudFactory.BundleAdjustTwoView(&in->matchSet,in->images, 10, "");
    meshFactory.setPoints(out->points);
    //finalMesh.filterByNeighborDistance(3.0); // <--- filter bois past 3.0 sigma (about 99.5% of points) if 2 view is good then this is usually good
    meshFactory.savePoints("ssrlcv-BA-final");
    logger.logState("BA");
  }