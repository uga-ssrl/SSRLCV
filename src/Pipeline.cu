#include "Pipeline.cuh"

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
  
  void ssrlcv::doFeatureMatching(ssrlcv::FeatureMatchingInput *in, ssrlcv::FeatureMatchingOutput *out) {
    logger.info << "Starting matching...";
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,200.0f*200.0f);
    logger.logState("MATCHING");
    // logger.logState("generating seed matches");
    if (in->seedFeatures != nullptr)
      matchFactory.setSeedFeatures(in->seedFeatures);
  
    if (in->images.size() == 2){
      //
      // 2 View Case
      //
      ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances = (in->seedFeatures != nullptr) ? matchFactory.getSeedDistances(in->allFeatures[0]) : nullptr;
      logger.logState("done generating seed matches");

      logger.logState("matching images");
      #if GEO_ORBIT == 1
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> distanceMatches = matchFactory.generateDistanceMatchesDoubleConstrained(in->images[0], in->allFeatures[0], in->images[1], in->allFeatures[1], in->epsilon, in->delta, seedDistances);
      #else
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> distanceMatches = matchFactory.generateDistanceMatches(in->images[0], in->allFeatures[0], in->images[1], in->allFeatures[1], seedDistances);
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

    ssrlcv::MeshFactory meshFactory;
    meshFactory.setPoints(out->points);
    meshFactory.savePoints("ssrlcv-initial");
  
    logger.logState("TRIANGULATE");
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