#include "FeatureFactory.cuh"

/*
HOST METHODS
*/
//Base feature factory


ssrlcv::FeatureFactory::FeatureFactory(){

}
ssrlcv::FeatureFactory::~FeatureFactory(){

}
ssrlcv::FeatureFactory::ScaleSpace::ScaleSpace(){

}
ssrlcv::FeatureFactory::ScaleSpace::ScaleSpace(unsigned int numOctaves, int startingOctave, unsigned int numBlurs, Image* image){

}
ssrlcv::FeatureFactory::ScaleSpace::~ScaleSpace(){

}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(){
  
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(unsigned int numBlurs, float* sigmas){

}
ssrlcv::FeatureFactory::ScaleSpace::Octave::~Octave(){

}
