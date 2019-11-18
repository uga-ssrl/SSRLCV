struct Spline{
      float coeff[6][6][4][4];
    };
    typedef struct Spline Spline;

    struct SubpixelM7x7{
      float M1[9][9];
      float M2[9][9];
    };
    typedef struct SubpixelM7x7 SubpixelM7x7;

/*
    METHODS IN MATCHFACTORY BELOW THIS ONLY WORK FOR DENSE FEATURES THAT HAVE NOT BEEN FILTERED
    */
    /**
    * \brief Generates subpixel matches between sift features
    * \warning This only works for dense features
    */
    Unity<FeatureMatch<T>>* generateSubPixelMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures);
    /**
    * \brief Generates subpixel matches between sift features constrained by the epipolar line
    * \warning This only works for dense features
    * \warning This method requires Images to have filled out Camera variable
    */
    Unity<FeatureMatch<T>>* generateSubPixelMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon);

 //subpixel kernels
  template<typename T>
  __global__ void initializeSubPixels(unsigned long numMatches, FeatureMatch<T>* matches, SubpixelM7x7* subPixelDescriptors,
    uint2 querySize, unsigned long numFeaturesQuery, Feature<T>* featuresQuery,
    uint2 targetSize, unsigned long numFeaturesTarget, Feature<T>* featuresTarget);

  __global__ void fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, Spline* splines);
  template<typename T>
  __global__ void determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, FeatureMatch<T>* matches, Spline* splines);
