#include <gtest/gtest.h>
#include "Pipeline.cuh"

std::string getPath(bool threeView = false) {
  if (threeView)
    return "test/checkpoints/Pipeline3View/";
  return "test/checkpoints/Pipeline2View/";
}

template <typename T>
std::string getCheckpoint(int id, bool threeView = false) {
  return getPath(threeView) + std::to_string(id) + "_" + typeid(T).name() + ".uty";
}

std::string getImgCheckpoint(int id, bool threeView = false) {
  return getPath(threeView) + std::to_string(id) + "_" + typeid(ssrlcv::Image).name() + ".cpimg";
}

template <typename T>
bool sameFeatures(ssrlcv::Unity<ssrlcv::Feature<T>> x, ssrlcv::Unity<ssrlcv::Feature<T>> y) {

  ssrlcv::MemoryState origin = x.getMemoryState();
  if (origin != ssrlcv::cpu) {
    x.transferMemoryTo(ssrlcv::cpu);
    y.transferMemoryTo(ssrlcv::cpu);
  }

  for (int i = 0; i < x.size(); i ++) {
    if (x.host.get()[i].loc != y.host.get()[i].loc) {
      std::cout << "Locations are not equal" << std::endl;
      return false;
    }
    if (x.host.get()[i].descriptor.distProtocol(y.host.get()[i].descriptor) > 20) {
      std::cout << "Descriptors are too far apart" << std::endl;
      return false;
    }
  }

  return true;
}

bool sameKeypoints(ssrlcv::Unity<ssrlcv::KeyPoint> x, ssrlcv::Unity<ssrlcv::KeyPoint> y) {

  ssrlcv::MemoryState origin = x.getMemoryState();
  if (origin != ssrlcv::cpu) {
    x.transferMemoryTo(ssrlcv::cpu);
    y.transferMemoryTo(ssrlcv::cpu);
  }

  for (int i = 0; i < x.size(); i ++) {
    if (x.host.get()[i].loc != y.host.get()[i].loc) {
      std::cout << "Locations are not equal" << std::endl;
      return false;
    }
    if (x.host.get()[i].parentId != y.host.get()[i].parentId) {
      std::cout << "ParentIds are not equal" << std::endl;
      return false;
    }
  }

  return true;
}

bool sameMatches(ssrlcv::Unity<ssrlcv::MultiMatch> x, ssrlcv::Unity<ssrlcv::MultiMatch> y) {

  ssrlcv::MemoryState origin = x.getMemoryState();
  if (origin != ssrlcv::cpu) {
    x.transferMemoryTo(ssrlcv::cpu);
    y.transferMemoryTo(ssrlcv::cpu);
  }

  for (int i = 0; i < x.size(); i ++) {
    if (x.host.get()[i].numKeyPoints != y.host.get()[i].numKeyPoints) {
      std::cout << "Numbers of KeyPoints are not equal" << std::endl;
      return false;
    }
    if (x.host.get()[i].index != y.host.get()[i].index) {
      std::cout << "Indices are not equal" << std::endl;
      return false;
    }
  }

  return true;
}

bool samePoints(ssrlcv::Unity<float3> x, ssrlcv::Unity<float3> y) {

  ssrlcv::MemoryState origin = x.getMemoryState();
  if (origin != ssrlcv::cpu) {
    x.transferMemoryTo(ssrlcv::cpu);
    y.transferMemoryTo(ssrlcv::cpu);
  }

  for (int i = 0; i < x.size(); i ++) {
    if (x.host.get()[i] != y.host.get()[i]) {
      std::cout << "Points are not equal" << std::endl;
      return false;
    }
  }

  return true;
}

TEST(PipelineTest, FeatureGeneration2View) {
  std::string seedPath("/work/demlab/sfm/SSRLCV-Sample-Data/seeds/seed_spongebob.png");
  std::vector<std::string> imagePaths;
  imagePaths.push_back("/work/demlab/sfm/SSRLCV-Sample-Data/everest1024/2view/1.png");
  imagePaths.push_back("/work/demlab/sfm/SSRLCV-Sample-Data/everest1024/2view/2.png");
  int numImages = (int) imagePaths.size();
  ssrlcv::FeatureGenerationInput featureGenInput = {seedPath, imagePaths, numImages};
  ssrlcv::FeatureGenerationOutput featureGenOutput;
  ssrlcv::doFeatureGeneration(&featureGenInput, &featureGenOutput);
  // featureGenOutput.seedFeatures->checkpoint(-1, getPath());
  // featureGenOutput.allFeatures.at(0)->checkpoint(0, getPath());
  // featureGenOutput.allFeatures.at(1)->checkpoint(1, getPath());
  featureGenOutput.images.at(0)->checkpoint(getPath());
  featureGenOutput.images.at(1)->checkpoint(getPath());

  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> realSeedFeatures(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(-1));
  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> realFeatures0(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(0));
  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> realFeatures1(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(1));

  // Make sure two images/features
  EXPECT_EQ(featureGenOutput.allFeatures.size(), 2);
  EXPECT_EQ(featureGenOutput.images.size(), 2);
  // Make sure number of features is the same
  EXPECT_EQ(featureGenOutput.seedFeatures->size(), realSeedFeatures.size());
  EXPECT_EQ(featureGenOutput.allFeatures.at(0)->size(), realFeatures0.size());
  EXPECT_EQ(featureGenOutput.allFeatures.at(1)->size(), realFeatures1.size());
  // Check that memory states are equal
  EXPECT_EQ(featureGenOutput.seedFeatures->getMemoryState(), realSeedFeatures.getMemoryState());
  EXPECT_EQ(featureGenOutput.allFeatures.at(0)->getMemoryState(), realFeatures0.getMemoryState());
  EXPECT_EQ(featureGenOutput.allFeatures.at(1)->getMemoryState(), realFeatures1.getMemoryState());
  // Check that data is equal
  EXPECT_EQ(sameFeatures(*featureGenOutput.seedFeatures, realSeedFeatures), true);
  EXPECT_EQ(sameFeatures(*featureGenOutput.allFeatures.at(0), realFeatures0), true);
  EXPECT_EQ(sameFeatures(*featureGenOutput.allFeatures.at(1), realFeatures1), true);
}

TEST(PipelineTest, FeatureGeneration3View) {
  std::string seedPath("/work/demlab/sfm/SSRLCV-Sample-Data/seeds/seed_spongebob.png");
  std::vector<std::string> imagePaths;
  imagePaths.push_back("/work/demlab/sfm/SSRLCV-Sample-Data/everest1024/3view/1.png");
  imagePaths.push_back("/work/demlab/sfm/SSRLCV-Sample-Data/everest1024/3view/2.png");
  imagePaths.push_back("/work/demlab/sfm/SSRLCV-Sample-Data/everest1024/3view/3.png");
  int numImages = (int) imagePaths.size();
  ssrlcv::FeatureGenerationInput featureGenInput = {seedPath, imagePaths, numImages};
  ssrlcv::FeatureGenerationOutput featureGenOutput;
  ssrlcv::doFeatureGeneration(&featureGenInput, &featureGenOutput);
  // featureGenOutput.seedFeatures->checkpoint(-1, getPath(true));
  // featureGenOutput.allFeatures.at(0)->checkpoint(0, getPath(true));
  // featureGenOutput.allFeatures.at(1)->checkpoint(1, getPath(true));
  // featureGenOutput.allFeatures.at(2)->checkpoint(2, getPath(true));
  featureGenOutput.images.at(0)->checkpoint(getPath(true));
  featureGenOutput.images.at(1)->checkpoint(getPath(true));
  featureGenOutput.images.at(2)->checkpoint(getPath(true));

  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> realSeedFeatures(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(-1, true));
  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> realFeatures0(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(0, true));
  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> realFeatures1(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(1, true));
  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> realFeatures2(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(2, true));

  // Make sure two images/features
  EXPECT_EQ(featureGenOutput.allFeatures.size(), 3);
  EXPECT_EQ(featureGenOutput.images.size(), 3);
  // Make sure number of features is the same
  EXPECT_EQ(featureGenOutput.seedFeatures->size(), realSeedFeatures.size());
  EXPECT_EQ(featureGenOutput.allFeatures.at(0)->size(), realFeatures0.size());
  EXPECT_EQ(featureGenOutput.allFeatures.at(1)->size(), realFeatures1.size());
  EXPECT_EQ(featureGenOutput.allFeatures.at(2)->size(), realFeatures2.size());
  // Check that memory states are equal
  EXPECT_EQ(featureGenOutput.seedFeatures->getMemoryState(), realSeedFeatures.getMemoryState());
  EXPECT_EQ(featureGenOutput.allFeatures.at(0)->getMemoryState(), realFeatures0.getMemoryState());
  EXPECT_EQ(featureGenOutput.allFeatures.at(1)->getMemoryState(), realFeatures1.getMemoryState());
  EXPECT_EQ(featureGenOutput.allFeatures.at(2)->getMemoryState(), realFeatures2.getMemoryState());
  // Check that data is equal
  EXPECT_EQ(sameFeatures(*featureGenOutput.seedFeatures, realSeedFeatures), true);
  EXPECT_EQ(sameFeatures(*featureGenOutput.allFeatures.at(0), realFeatures0), true);
  EXPECT_EQ(sameFeatures(*featureGenOutput.allFeatures.at(1), realFeatures1), true);
  EXPECT_EQ(sameFeatures(*featureGenOutput.allFeatures.at(2), realFeatures2), true);
}

TEST(PipelineTest, FeatureMatching2View) {
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> seedFeatures(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(-1));

  std::vector<ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures;
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> features0 = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(0));
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> features1 = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(1));
  allFeatures.push_back(features0);
  allFeatures.push_back(features1);

  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
  ssrlcv::ptr::value<ssrlcv::Image> image0 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(0), 0);
  ssrlcv::ptr::value<ssrlcv::Image> image1 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(1), 1);
  images.push_back(image0);
  images.push_back(image1);

  ssrlcv::FeatureMatchingInput featureMatchInput = {seedFeatures, allFeatures, images, 25, 5};
  ssrlcv::FeatureMatchingOutput featureMatchOutput;
  ssrlcv::doFeatureMatching(&featureMatchInput, &featureMatchOutput);
  // featureMatchOutput.matchSet.keyPoints->checkpoint(0, getPath());
  // featureMatchOutput.matchSet.matches->checkpoint(0, getPath());

  ssrlcv::Unity<ssrlcv::KeyPoint> realKeyPoints(getCheckpoint<ssrlcv::KeyPoint>(0));
  ssrlcv::Unity<ssrlcv::MultiMatch> realMatches(getCheckpoint<ssrlcv::MultiMatch>(0));

  // Make sure sizes are the same
  EXPECT_EQ(featureMatchOutput.matchSet.keyPoints->size(), realKeyPoints.size());
  EXPECT_EQ(featureMatchOutput.matchSet.matches->size(), realMatches.size());
  // Check that memory states are equal
  EXPECT_EQ(featureMatchOutput.matchSet.keyPoints->getMemoryState(), realKeyPoints.getMemoryState());
  EXPECT_EQ(featureMatchOutput.matchSet.matches->getMemoryState(), realMatches.getMemoryState());
  // Check that data is equal
  EXPECT_EQ(sameKeypoints(*featureMatchOutput.matchSet.keyPoints, realKeyPoints), true);
  EXPECT_EQ(sameMatches(*featureMatchOutput.matchSet.matches, realMatches), true);
}

TEST(PipelineTest, FeatureMatching3View) {
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> seedFeatures(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(-1, true));

  std::vector<ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures;
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> features0 = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(0, true));
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> features1 = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(1, true));
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> features2 = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>(getCheckpoint<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>(2, true));
  allFeatures.push_back(features0);
  allFeatures.push_back(features1);
  allFeatures.push_back(features2);

  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
  ssrlcv::ptr::value<ssrlcv::Image> image0 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(0, true));
  ssrlcv::ptr::value<ssrlcv::Image> image1 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(1, true));
  ssrlcv::ptr::value<ssrlcv::Image> image2 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(2, true));
  images.push_back(image0);
  images.push_back(image1);
  images.push_back(image2);

  ssrlcv::FeatureMatchingInput featureMatchInput = {seedFeatures, allFeatures, images, 0.0};
  ssrlcv::FeatureMatchingOutput featureMatchOutput;
  ssrlcv::doFeatureMatching(&featureMatchInput, &featureMatchOutput);
  // featureMatchOutput.matchSet.keyPoints->checkpoint(0, getPath(true));
  // featureMatchOutput.matchSet.matches->checkpoint(0, getPath(true));

  ssrlcv::Unity<ssrlcv::KeyPoint> realKeyPoints(getCheckpoint<ssrlcv::KeyPoint>(0, true));
  ssrlcv::Unity<ssrlcv::MultiMatch> realMatches(getCheckpoint<ssrlcv::MultiMatch>(0, true));

  // Make sure sizes are the same
  EXPECT_EQ(featureMatchOutput.matchSet.keyPoints->size(), realKeyPoints.size());
  EXPECT_EQ(featureMatchOutput.matchSet.matches->size(), realMatches.size());
  // Check that memory states are equal
  EXPECT_EQ(featureMatchOutput.matchSet.keyPoints->getMemoryState(), realKeyPoints.getMemoryState());
  EXPECT_EQ(featureMatchOutput.matchSet.matches->getMemoryState(), realMatches.getMemoryState());
  // Check that data is equal
  EXPECT_EQ(sameKeypoints(*featureMatchOutput.matchSet.keyPoints, realKeyPoints), true);
  EXPECT_EQ(sameMatches(*featureMatchOutput.matchSet.matches, realMatches), true);
}

TEST(PipelineTest, Triangulation2View) {
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
  ssrlcv::ptr::value<ssrlcv::Image> image0 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(0));
  ssrlcv::ptr::value<ssrlcv::Image> image1 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(1));
  images.push_back(image0);
  images.push_back(image1);

  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>> keyPoints(getCheckpoint<ssrlcv::KeyPoint>(0));
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>> matches(getCheckpoint<ssrlcv::MultiMatch>(0));
  ssrlcv::MatchSet matchSet = {keyPoints, matches};

  ssrlcv::TriangulationInput triangulationInput = {matchSet, images};
  ssrlcv::TriangulationOutput triangulationOutput;
  ssrlcv::doTriangulation(&triangulationInput, &triangulationOutput);
  // triangulationOutput.points->checkpoint(0, getPath());

  // Match Set Shouldn't Change
  ssrlcv::Unity<ssrlcv::KeyPoint> realKeyPoints(getCheckpoint<ssrlcv::KeyPoint>(0));
  ssrlcv::Unity<ssrlcv::MultiMatch> realMatches(getCheckpoint<ssrlcv::MultiMatch>(0));
  ssrlcv::Unity<float3> realPoints(getCheckpoint<float3>(0));

  // Make sure sizes are the same
  EXPECT_EQ(triangulationInput.matchSet.keyPoints->size(), realKeyPoints.size());
  EXPECT_EQ(triangulationInput.matchSet.matches->size(), realMatches.size());
  EXPECT_EQ(triangulationOutput.points->size(), realPoints.size());
  // Check that memory states are equal
  EXPECT_EQ(triangulationInput.matchSet.keyPoints->getMemoryState(), realKeyPoints.getMemoryState());
  EXPECT_EQ(triangulationInput.matchSet.matches->getMemoryState(), realMatches.getMemoryState());
  EXPECT_EQ(triangulationOutput.points->getMemoryState(), realPoints.getMemoryState());
  // Check that data is equal
  EXPECT_EQ(sameKeypoints(*triangulationInput.matchSet.keyPoints, realKeyPoints), true);
  EXPECT_EQ(sameMatches(*triangulationInput.matchSet.matches, realMatches), true);
  EXPECT_EQ(samePoints(*triangulationOutput.points, realPoints), true);
}

TEST(PipelineTest, Triangulation3View) {
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
  ssrlcv::ptr::value<ssrlcv::Image> image0 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(0, true));
  ssrlcv::ptr::value<ssrlcv::Image> image1 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(1, true));
  ssrlcv::ptr::value<ssrlcv::Image> image2 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(2, true));
  images.push_back(image0);
  images.push_back(image1);
  images.push_back(image2);

  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>> keyPoints(getCheckpoint<ssrlcv::KeyPoint>(0, true));
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>> matches(getCheckpoint<ssrlcv::MultiMatch>(0, true));
  ssrlcv::MatchSet matchSet = {keyPoints, matches};

  ssrlcv::TriangulationInput triangulationInput = {matchSet, images};
  ssrlcv::TriangulationOutput triangulationOutput;
  ssrlcv::doTriangulation(&triangulationInput, &triangulationOutput);
  //triangulationOutput.points->checkpoint(0, getPath(true));

  // Match Set Shouldn't Change
  ssrlcv::Unity<ssrlcv::KeyPoint> realKeyPoints(getCheckpoint<ssrlcv::KeyPoint>(0, true));
  ssrlcv::Unity<ssrlcv::MultiMatch> realMatches(getCheckpoint<ssrlcv::MultiMatch>(0, true));
  ssrlcv::Unity<float3> realPoints(getCheckpoint<float3>(0, true));

  // Make sure sizes are the same
  EXPECT_EQ(triangulationInput.matchSet.keyPoints->size(), realKeyPoints.size());
  EXPECT_EQ(triangulationInput.matchSet.matches->size(), realMatches.size());
  EXPECT_EQ(triangulationOutput.points->size(), realPoints.size());
  // Check that memory states are equal
  EXPECT_EQ(triangulationInput.matchSet.keyPoints->getMemoryState(), realKeyPoints.getMemoryState());
  EXPECT_EQ(triangulationInput.matchSet.matches->getMemoryState(), realMatches.getMemoryState());
  EXPECT_EQ(triangulationOutput.points->getMemoryState(), realPoints.getMemoryState());
  // Check that data is equal
  EXPECT_EQ(sameKeypoints(*triangulationInput.matchSet.keyPoints, realKeyPoints), true);
  EXPECT_EQ(sameMatches(*triangulationInput.matchSet.matches, realMatches), true);
  EXPECT_EQ(samePoints(*triangulationOutput.points, realPoints), true);
}

TEST(PipelineTest, Filtering2View) {
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
  ssrlcv::ptr::value<ssrlcv::Image> image0 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(0));
  ssrlcv::ptr::value<ssrlcv::Image> image1 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(1));
  images.push_back(image0);
  images.push_back(image1);

  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>> keyPoints(getCheckpoint<ssrlcv::KeyPoint>(0));
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>> matches(getCheckpoint<ssrlcv::MultiMatch>(0));
  ssrlcv::MatchSet matchSet = {keyPoints, matches};

  ssrlcv::FilteringInput filteringInput = {matchSet, images};
  ssrlcv::FilteringOutput filteringOutput;
  ssrlcv::doFiltering(&filteringInput, &filteringOutput);
  // filteringOutput.points->checkpoint(1, getPath());
  // filteringInput.matchSet.keyPoints->checkpoint(1, getPath());
  // filteringInput.matchSet.matches->checkpoint(1, getPath());

  ssrlcv::Unity<ssrlcv::KeyPoint> realKeyPoints(getCheckpoint<ssrlcv::KeyPoint>(1));
  ssrlcv::Unity<ssrlcv::MultiMatch> realMatches(getCheckpoint<ssrlcv::MultiMatch>(1));
  ssrlcv::Unity<float3> realPoints(getCheckpoint<float3>(1));

  // Make sure sizes are the same
  EXPECT_EQ(filteringInput.matchSet.keyPoints->size(), realKeyPoints.size());
  EXPECT_EQ(filteringInput.matchSet.matches->size(), realMatches.size());
  EXPECT_EQ(filteringOutput.points->size(), realPoints.size());
  // Check that memory states are equal
  EXPECT_EQ(filteringInput.matchSet.keyPoints->getMemoryState(), realKeyPoints.getMemoryState());
  EXPECT_EQ(filteringInput.matchSet.matches->getMemoryState(), realMatches.getMemoryState());
  EXPECT_EQ(filteringOutput.points->getMemoryState(), realPoints.getMemoryState());
  // Check that data is equal
  EXPECT_EQ(sameKeypoints(*filteringInput.matchSet.keyPoints, realKeyPoints), true);
  EXPECT_EQ(sameMatches(*filteringInput.matchSet.matches, realMatches), true);
  EXPECT_EQ(samePoints(*filteringOutput.points, realPoints), true);
}

TEST(PipelineTest, Filtering3View) {
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
  ssrlcv::ptr::value<ssrlcv::Image> image0 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(0, true));
  ssrlcv::ptr::value<ssrlcv::Image> image1 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(1, true));
  ssrlcv::ptr::value<ssrlcv::Image> image2 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(2, true));
  images.push_back(image0);
  images.push_back(image1);
  images.push_back(image2);

  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>> keyPoints(getCheckpoint<ssrlcv::KeyPoint>(0, true));
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>> matches(getCheckpoint<ssrlcv::MultiMatch>(0, true));
  ssrlcv::MatchSet matchSet = {keyPoints, matches};

  ssrlcv::FilteringInput filteringInput = {matchSet, images};
  ssrlcv::FilteringOutput filteringOutput;
  ssrlcv::doFiltering(&filteringInput, &filteringOutput);
  // filteringOutput.points->checkpoint(1, getPath(true));
  // filteringInput.matchSet.keyPoints->checkpoint(1, getPath(true));
  // filteringInput.matchSet.matches->checkpoint(1, getPath(true));

  ssrlcv::Unity<ssrlcv::KeyPoint> realKeyPoints(getCheckpoint<ssrlcv::KeyPoint>(1, true));
  ssrlcv::Unity<ssrlcv::MultiMatch> realMatches(getCheckpoint<ssrlcv::MultiMatch>(1, true));
  ssrlcv::Unity<float3> realPoints(getCheckpoint<float3>(1, true));

  // Make sure sizes are the same
  EXPECT_EQ(filteringInput.matchSet.keyPoints->size(), realKeyPoints.size());
  EXPECT_EQ(filteringInput.matchSet.matches->size(), realMatches.size());
  EXPECT_EQ(filteringOutput.points->size(), realPoints.size());
  // Check that memory states are equal
  EXPECT_EQ(filteringInput.matchSet.keyPoints->getMemoryState(), realKeyPoints.getMemoryState());
  EXPECT_EQ(filteringInput.matchSet.matches->getMemoryState(), realMatches.getMemoryState());
  EXPECT_EQ(filteringOutput.points->getMemoryState(), realPoints.getMemoryState());
  // Check that data is equal
  EXPECT_EQ(sameKeypoints(*filteringInput.matchSet.keyPoints, realKeyPoints), true);
  EXPECT_EQ(sameMatches(*filteringInput.matchSet.matches, realMatches), true);
  EXPECT_EQ(samePoints(*filteringOutput.points, realPoints), true);
}

TEST(PipelineTest, BundleAdjust2View) {
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
  ssrlcv::ptr::value<ssrlcv::Image> image0 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(0));
  ssrlcv::ptr::value<ssrlcv::Image> image1 = ssrlcv::ptr::value<ssrlcv::Image>(getImgCheckpoint(1));
  images.push_back(image0);
  images.push_back(image1);

  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>> keyPoints(getCheckpoint<ssrlcv::KeyPoint>(1));
  ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>> matches(getCheckpoint<ssrlcv::MultiMatch>(1));
  ssrlcv::MatchSet matchSet = {keyPoints, matches};

  ssrlcv::BundleAdjustInput bundleAdjustInput = {matchSet, images};
  ssrlcv::BundleAdjustOutput bundleAdjustOutput;
  ssrlcv::doBundleAdjust(&bundleAdjustInput, &bundleAdjustOutput);
  // bundleAdjustOutput.points->checkpoint(2, getPath());

  // Match Set shouldn't change
  ssrlcv::Unity<ssrlcv::KeyPoint> realKeyPoints(getCheckpoint<ssrlcv::KeyPoint>(1));
  ssrlcv::Unity<ssrlcv::MultiMatch> realMatches(getCheckpoint<ssrlcv::MultiMatch>(1));
  ssrlcv::Unity<float3> realPoints(getCheckpoint<float3>(2));

  // Make sure sizes are the same
  EXPECT_EQ(bundleAdjustInput.matchSet.keyPoints->size(), realKeyPoints.size());
  EXPECT_EQ(bundleAdjustInput.matchSet.matches->size(), realMatches.size());
  EXPECT_EQ(bundleAdjustOutput.points->size(), realPoints.size());
  // Check that memory states are equal
  EXPECT_EQ(bundleAdjustInput.matchSet.keyPoints->getMemoryState(), realKeyPoints.getMemoryState());
  EXPECT_EQ(bundleAdjustInput.matchSet.matches->getMemoryState(), realMatches.getMemoryState());
  EXPECT_EQ(bundleAdjustOutput.points->getMemoryState(), realPoints.getMemoryState());
  // Check that data is equal
  EXPECT_EQ(sameKeypoints(*bundleAdjustInput.matchSet.keyPoints, realKeyPoints), true);
  EXPECT_EQ(sameMatches(*bundleAdjustInput.matchSet.matches, realMatches), true);
  EXPECT_EQ(samePoints(*bundleAdjustOutput.points, realPoints), true);
}