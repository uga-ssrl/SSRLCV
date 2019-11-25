#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "FeatureFactory.cuh"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "matrix_util.cuh"


ssrlcv::PointCloudFactory localizer = ssrlcv::PointCloudFactory();  

//fundamental for parallel images
float fundamental[3][3] = {
  {0.0f,0.0f,0.0f},
  {0.0f,0.0f,-1.0f},
  {0.0f,1.0f,0.0f}
};

void writeNanoDiagnosticFile(std::string folder,unsigned long time,float powerDraw,float temp){

}

ssrlcv::Image readPGM(std::string pathToFile){
  std::cout<<"reading pgm file for disparities"<<std::endl;
  int numrows = 0, numcols = 0;
  std::ifstream infile(pathToFile);
  std::stringstream ss;
  std::string inputLine = "";

  // First line : version
  getline(infile,inputLine);
  std::cout << "Version : " << inputLine << std::endl;

  // Continue with a stringstream
  ss << infile.rdbuf();
  // Third line : size
  ss >> numcols >> numrows;
  std::cout << numcols << " columns and " << numrows << " rows" << std::endl;

  int max_val;
  ss >> max_val;
  std::cout<<"max_val = "<<max_val<<std::endl;

  ssrlcv::Unity<unsigned char>* pixels = new ssrlcv::Unity<unsigned char>(nullptr, numcols*numrows,ssrlcv::cpu);

  // Following lines : data
  for(int i = 0; i < numrows*numcols; ++i){ 
    ss.read((char*)&pixels->host[i],1);
  }
  
  infile.close();
  return ssrlcv::Image({numcols,numrows},1,pixels);
}
ssrlcv::Unity<ssrlcv::Match>* readPFM(std::string pathToFile, uint2 &size){
  std::cout<<"reading pfm file for disparities"<<std::endl;
  std::ifstream file(pathToFile.c_str(),std::ios::in|std::ios::binary);
  if(file){
    char type[2];
    file.read(type,2*sizeof(char));
    char eol;
    file.read(&eol,sizeof(char));
    std::cout<<type<<eol;
    file >> size.x >> size.y;
    file.read(&eol,sizeof(char));
    std::cout<<size.x<<","<<size.y<<eol;
    int colorDepth = 1;
    if(type[1] == 'F'){
      colorDepth = 3;
    }
    eol = 0;
    float scale = 0.0f;
    file >> scale;
    while(eol != 0x0a){
      file.read(&eol,sizeof(char));
    }
    std::cout<<scale<<eol;
    bool littleEndian = scale < 0.0f;
    if(littleEndian) scale *= -1;
    float pixel;
    float** pixels = new float*[size.y];
    float min = FLT_MAX;
    float max = -FLT_MAX;
    std::vector<int2> infPixels;
    char* floatToConvert;
    char* returnFloat;
    float raw = 0;
    for(int row = 0; row < size.y; ++row){
      if(littleEndian) pixels[size.y-row-1] = new float[size.x*colorDepth];
      else pixels[row] = new float[size.x*colorDepth];
      for(int col = 0; col < size.x; ++col){
        for(int color = 0; color < colorDepth; ++color){
          file.read((char*)&pixel, sizeof(float));
          if(isinf(pixel)){
            if(littleEndian) infPixels.push_back({col,size.y - row - 1});
            else infPixels.push_back({col,row});
          } 
          else{
            if(pixel < min) min = pixel;
            if(pixel > max) max = pixel;
          }
          if(littleEndian) pixels[size.y-row-1][col*colorDepth + color] = pixel;
          else pixels[row][col*colorDepth + color] = pixel;
          
        }
      }
    }
    //if inf that likely means edge
    bool foundContrib = false;
    for(auto coord = infPixels.begin(); coord != infPixels.end(); ++coord){
      foundContrib = false;
      for(int i = 1; i <= 5 && !foundContrib; ++i){
        for(int y = -i; y <= i && !foundContrib; ++y){
          for(int x = -i; x <= i && !foundContrib; ++x){
            if(y != 0 && x != 0 && 
              coord->y + y > 0 && coord->y + y < size.y && 
              coord->x + x > 0 && coord->x + x < size.x && 
              isfinite(pixels[coord->y + y][coord->x + x])){
              pixels[coord->y][coord->x] = pixels[coord->y + y][coord->x + x];
              foundContrib = true;
            }
          }
        }     
      }
    }
    printf("min = %f, max = %f\n",min,max);
    ssrlcv::Unity<ssrlcv::Match>* matches = new ssrlcv::Unity<ssrlcv::Match>(nullptr,size.x*size.y,ssrlcv::cpu);
    float disp = 0.0f;
    for(int row = 0; row < size.y; ++row){
      for(int col = 0; col < size.x; ++col){
        if(colorDepth == 3){
          disp = pixels[row][col*3]*0.25f;
          disp += pixels[row][col*3 + 1]*0.5f;
          disp += pixels[row][col*3 + 2]*0.25f;
        }
        else{
          disp = pixels[row][col];
        }
        ssrlcv::Match match;
        match.invalid = isinf(disp);
        match.keyPoints[0].parentId = 0;
        match.keyPoints[1].parentId = 1;
        match.keyPoints[0].loc = {col + disp,row};
        match.keyPoints[1].loc = {col,row};
        matches->host[row*size.x + col] = match;
      }
    }
    return matches;
  }
  else{
    std::cerr<<"ERROR: cannot open "<<pathToFile<<std::endl;
    exit(-1);
  }
}
float getMatchDisparity(const ssrlcv::Match &match){
  return match.keyPoints[0].loc.x - match.keyPoints[1].loc.x;
}
void binMatches(ssrlcv::Unity<ssrlcv::Match>* matches, uint2 imageSize){
  ssrlcv::Match* binnedMatches = new ssrlcv::Match[matches->numElements/4];
  if(matches->state != ssrlcv::cpu) matches->setMemoryState(ssrlcv::cpu);
  float disp = 0.0f;
  for(int row = 0; row < imageSize.y/2; ++row){
    for(int col = 0; col < imageSize.x/2; ++col){
      disp = 0.0f;
      disp += getMatchDisparity(matches->host[row*2*imageSize.x + (col*2)]);
      disp += getMatchDisparity(matches->host[row*2*imageSize.x + (col*2+1)]);
      disp += getMatchDisparity(matches->host[(row*2+1)*imageSize.x + (col*2)]);
      disp += getMatchDisparity(matches->host[(row*2+1)*imageSize.x + (col*2+1)]);
      disp /= 8.0f;
      
      //8 because /= 4 for bin algo and /=2 for image size getting smaller
      ssrlcv::Match match;
      match.invalid = false;
      match.keyPoints[0].parentId = 0;
      match.keyPoints[1].parentId = 1;
      match.keyPoints[0].loc = {col + disp,row};
      match.keyPoints[1].loc = {col,row};
      binnedMatches[row*(imageSize.x/2) + col] = match;
    }
  }
  matches->setData(binnedMatches,matches->numElements/4,ssrlcv::cpu);
}
//note the folder with the pfmFile in it has to have /full, /half and /quarter subdirectories
void createMatchTestSet(std::string pfmFile){
  uint2 size;
  ssrlcv::Unity<ssrlcv::Match>* matches = readPFM(pfmFile,size);
  std::string folder = pfmFile.substr(0,pfmFile.find_last_of("/")+1);
  ssrlcv::writeMatchFile(matches,folder+"full/matches.txt");
  binMatches(matches,size);
  ssrlcv::writeMatchFile(matches,folder+"half/matches.txt");
  binMatches(matches,size/2);
  ssrlcv::writeMatchFile(matches,folder+"quarter/matches.txt");
  delete matches;
}
void runPipelineOn(std::string matchFile, float3 calib){
  std::string resultFolder = matchFile.substr(0,matchFile.find_last_of("/")+1);
  ssrlcv::Unity<ssrlcv::Match>* matches = ssrlcv::readMatchFile(matchFile);
  ssrlcv::writeMatchFile(matches,resultFolder + "matches.bin",calib,true);
  matches->setMemoryState(ssrlcv::gpu);
  ssrlcv::Unity<float3>* points = localizer.stereo_disparity(matches,calib.x,calib.y,calib.z);
  delete matches;
  ssrlcv::writePLY((resultFolder + "disparity.ply").c_str(),points);
  ssrlcv::writeDisparityImage(points,0,resultFolder + "disparity.png");
  delete points;
}
//hard coded locations
void blockMatchingTest(){
  float3 australiaCalib = {1133.249,285.3,121.745};//australia map 717x492
  unsigned int maxDisparities = 73;
  unsigned int windowSize = 15;
  ssrlcv::Direction direction = ssrlcv::left;
  ssrlcv::Image image[2] = {
    ssrlcv::Image("data/disparityTest/Australia/im0.png",1,0),
    ssrlcv::Image("data/disparityTest/Australia/im1.png",1,1)
  };
  ssrlcv::Unity<ssrlcv::Match>* australiaMatches = ssrlcv::generateDiparityMatches(
    image[0].size,image[0].pixels,image[1].size,image[1].pixels,
    fundamental,maxDisparities,windowSize,direction
  );
  ssrlcv::Unity<float3>* australiaPoints = localizer.stereo_disparity(australiaMatches,australiaCalib.x,australiaCalib.y,australiaCalib.z);
  ssrlcv::writeMatchFile(australiaMatches,"data/disparityTest/Australia/matches.txt");
  ssrlcv::writePLY("data/disparityTest/Australia/Australia.ply",australiaPoints);
  ssrlcv::writeDisparityImage(australiaPoints,100,"data/disparityTest/Australia/disparity.png");
  delete australiaMatches;
  delete australiaPoints;
}
bool fileExists(const char* fileName){
  std::ifstream test(fileName); 
  return (test) ? true : false;
}
void runfullSIFTPipeline(std::string folder, std::string seedImage){
  std::vector<std::string> imagePaths = ssrlcv::findFiles(folder);

  int numImages = (int) imagePaths.size();

  ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);
  ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,250.0f*250.0f);

  /*
  FEATURE EXTRACTION
  */
  //seed features extraction

  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* seedFeatures = nullptr;
  bool seedExists = fileExists(seedImage.c_str());
  if(seedExists){
    ssrlcv::Image* seed = new ssrlcv::Image(seedImage,-1);
    seedFeatures = featureFactory.generateFeatures(seed,false,2,0.8); 
    matchFactory.setSeedFeatures(seedFeatures);
    delete seed;
  } 

  std::vector<ssrlcv::Image*> images;
  std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
  for(int i = 0; i < numImages; ++i){
    ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
    ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,2,0.8);
    images.push_back(image);
    allFeatures.push_back(features);
  }
    
  /*
  MATCHING
  */
  //seeding with false photo

  std::cout << "Starting matching..." << std::endl;
  ssrlcv::Unity<float>* seedDistances = (seedExists) ? matchFactory.getSeedDistances(allFeatures[0]) : nullptr;    
  ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1],seedDistances);
  if(seedDistances != nullptr) delete seedDistances;

  distanceMatches->transferMemoryTo(ssrlcv::cpu);
  float maxDist = 0.0f;
  for(int i = 0; i < distanceMatches->numElements; ++i){
    if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
  }
  printf("%f\n",maxDist);
  if(distanceMatches->state != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
  ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);
  delete distanceMatches;
  std::string delimiter = "/";
  std::string matchFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter)) + "/matches.txt";
  ssrlcv::writeMatchFile(matches, matchFile);
    
  /*
  STEREODISPARITY
  */
  ssrlcv::Unity<float3>* points = localizer.stereo_disparity(matches,64.0f);
  std::string disparityFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter));
  disparityFile = disparityFile.substr(0,disparityFile.rfind(delimiter))  + "/disparity.png";
  ssrlcv::writeDisparityImage(points,0,disparityFile);

  delete matches;
  ssrlcv::writePLY("out/test.ply",points);
  delete points;

  for(int i = 0; i < imagePaths.size(); ++i){
    delete images[i];
    delete allFeatures[i];
  }


}

int main(int argc, char *argv[]){
  try{
    //datasets coming from http://vision.middlebury.edu/stereo/data/scenes2014/
    //Make sure that matches.txt exists for all tests
    // createMatchTestSet("data/disparityTest/Adirondack/disp0.pfm");
    // createMatchTestSet("data/disparityTest/Mask/disp0.pfm");
    // createMatchTestSet("data/disparityTest/Playroom/disp0.pfm");
    // createMatchTestSet("data/disparityTest/Jadeplant/disp0.pfm");
    // createMatchTestSet("data/disparityTest/Pipes/disp0.pfm");

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();
  
    /*
    Few examples are from reading ground truth files to calculate z in mm
    NOTE float3 calibs have = {foc,baseline,doffset} with foc and doffset in pixels and baseline in mm
    */
    float3 adirondackCalib = {4161.221,176.252,209.059};//adirondack 2880x1988 - full size
    runPipelineOn("data/disparityTest/Adirondack/full/matches.txt",adirondackCalib);
    adirondackCalib = adirondackCalib/2.0f;
    runPipelineOn("data/disparityTest/Adirondack/half/matches.txt",adirondackCalib);
    adirondackCalib = adirondackCalib/2.0f;
    runPipelineOn("data/disparityTest/Adirondack/quarter/matches.txt",adirondackCalib);

    float3 maskCalib = {4844.97,170.458,162.296};//mask 2792x2008 - full size
    runPipelineOn("data/disparityTest/Mask/full/matches.txt",maskCalib);
    maskCalib = maskCalib/2.0f;
    runPipelineOn("data/disparityTest/Mask/half/matches.txt",maskCalib);
    maskCalib = maskCalib/2.0f;
    runPipelineOn("data/disparityTest/Mask/quarter/matches.txt",maskCalib);

    float3 playroomCalib = {4029.299,342.789,270.821};//playroom 2800x1908 - full size
    runPipelineOn("data/disparityTest/Playroom/full/matches.txt",playroomCalib);
    playroomCalib = playroomCalib/2.0f;
    runPipelineOn("data/disparityTest/Playroom/half/matches.txt",playroomCalib);
    playroomCalib = playroomCalib/2.0f;
    runPipelineOn("data/disparityTest/Playroom/quarter/matches.txt",playroomCalib);

    float3 jadeplantCalib = {7315.238,380.135,809.195};//jadeplant 2800x1908
    runPipelineOn("data/disparityTest/Jadeplant/full/matches.txt",jadeplantCalib);
    jadeplantCalib = jadeplantCalib/2.0f;
    runPipelineOn("data/disparityTest/Jadeplant/half/matches.txt",jadeplantCalib);
    jadeplantCalib = jadeplantCalib/2.0f;
    runPipelineOn("data/disparityTest/Jadeplant/quarter/matches.txt",jadeplantCalib);

    float3 pipesCalib = {3968.297,236.922,77.215};//pipes 2960x1924
    runPipelineOn("data/disparityTest/Pipes/full/matches.txt",pipesCalib);
    pipesCalib = pipesCalib/2.0f;
    runPipelineOn("data/disparityTest/Pipes/half/matches.txt",pipesCalib);
    pipesCalib = pipesCalib/2.0f;
    runPipelineOn("data/disparityTest/Pipes/quarter/matches.txt",pipesCalib);

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
