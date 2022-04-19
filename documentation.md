# SFM

## Description

The Structure from Motion process starts with feature extraction. A Feature Factory object is created and is then used to generate the features needed for the feature matching process. The generateFeatures function (in SIFT_FeatureFactory.cu) converts the inputted image into black and white. Important features/extrema in the image are detected by checking the RGB values.      

The next part of the process is feature matching. The features generated during feature extraction are used. The distances between features (from different images) are generated. These distances (seedDistances), as well as queryFeatures and targetFeaures, are then used to find matching pixels. There are two cases of how matching is done, a 2-view case or an n-view case.   

Initial triangulation begins after feature matching. First the bundles are generated. An object bundleSet is used to store lines and sets of lines as bundles. There are two cases for bundle generation, standard projection or use of a pushbroom camera. The bundle set is used to set up a point cloud and then triangulate using skew lines to find their closest interception.  

## Functions

#### Feature Extraction

**SIFT_FeatureFactory(orientationContriWidth, descriptorContriWidth)**   
creates FeatureFactory object.  
FeatureFactory object is used to generate/create features from the images inputted as well as check key points.   

**MatchFactory<SIFT_Descriptor>(relativeThreshold, absoluteThreshold)**   
creates MatchFactory object.    
MatchFactory object is used to set the seed features as well as validate, refine, and sort matches.   


## Data Types


## Dependencies

  * libpng-dev
  * libtiff-dev
  * g++
  * gcc
  * nvcc
  * CUDA 10.0