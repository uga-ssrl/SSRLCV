#include "Image.cuh"

// =============================================================================================================
//
// Constructors and Destructors
//
// =============================================================================================================

__device__ __host__ ssrlcv::Image::Camera::Camera(){
  this->cam_rot = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->fov = {0.0f,0.0f};
  this->foc = 0;
  this->dpix = {0.0f,0.0f};
  this->size = {0,0};
}
__device__ __host__ ssrlcv::Image::Camera::Camera(uint2 size){
  this->cam_rot = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->fov = {0.0f,0.0f};
  this->foc = 0;
  this->dpix = {0.0f,0.0f};
  this->size = {0,0};
}
__device__ __host__ ssrlcv::Image::Camera::Camera(uint2 size, float3 cam_pos, float3 cam_rot){
  this->cam_pos = cam_pos;
  this->cam_rot = cam_rot;
  this->fov = {0.0f,0.0f};
  this->foc = 0;
  this->dpix = {0.0f,0.0f};
  this->size = size;
}

ssrlcv::Image::Image(){
  this->id = -1;
  this->filePath = "n/a";
}

ssrlcv::Image::Image(uint2 size, unsigned int colorDepth, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels){
  this->filePath = "n/a";
  this->id = -1;
  this->colorDepth = colorDepth;
  this->pixels = pixels;
  this->camera.size = size;
  this->size = size;
}

ssrlcv::Image::Image(std::string filePath, int id) {
  if (getFileExtension(filePath) == "cpimg") {
    std::ifstream cp(filePath.c_str(), std::ifstream::binary);
    char buf[sizeof(ssrlcv::Image)];
    cp.read(buf, sizeof(ssrlcv::Image));
    this->camera = ((ssrlcv::Image *)buf)->camera;
    this->id = ((ssrlcv::Image *)buf)->id;
    this->size = ((ssrlcv::Image *)buf)->size;
    this->colorDepth = ((ssrlcv::Image *)buf)->colorDepth;
    this->pushbroom = ((ssrlcv::Image *)buf)->pushbroom;
    ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> thing(getFolderFromFilePath(filePath) + "/pixels/" + std::to_string(this->id) + "_" + typeid(unsigned char).name() + ".uty");
    this->pixels = thing;
    cp.close();
    return;
  }

  std::string filename = getFileFromFilePath(filePath);
  this->filePath = filePath;
  this->id = id;
  this->colorDepth = 1;
  // find the image extension
  ssrlcv::ptr::host<unsigned char> pixels_host = readImage(filePath.c_str(), this->size.y, this->size.x, this->colorDepth);

  // set some initial params
  this->camera.size = this->size;
  this->size = size;

  this->pixels = ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>>(pixels_host,this->size.y*this->size.x*this->colorDepth,cpu);

  // this check is just in case Middleburry data is input
  std::string file_no_digits = filename;
  int s = 0;
  while (s < file_no_digits.length()){
    if (isdigit(file_no_digits[s])) {
      file_no_digits.erase(s);
    } else {
      s++;
    }
  }

  // read additional params, and if the param requirement is removed then don't do any of this
  // checks that the image is not a seed image. extra params are not needed for seed images
  if (id != -1){
    std::string params_path = getFolderFromFilePath(filePath);
    // defaults to reading ascii params if both exist
    if (fileExists(params_path + "/params.csv")){// read in the file as an ASCII enoding
      logger.info << "Reading ASCII encoded camera parameters ...";
      logger.info << "Looking for matching file " + filename;
      // you know, this could be cleaner and generalized but idk if we wil ever want a csv reader other than here
      std::ifstream file(params_path + "/params.csv"); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
      std::string value;
      while (file.good()){
          // wait until we find the filename, or maybe we don't and it was empty. in that case nothing happens
          getline(file,value,','); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
          // sanitize the input
          value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());
          if (filename == value){ // if we have a match, read in the parameters one by one
            getline(file,value,',');
            if ("pushbroom" == value) {
              //
              // The pushbroom camera params case
              //
              this->isPushbroom = true;
              // see https://hirise-pds.lpl.arizona.edu/PDS/DOCUMENT/HIRISE_RDR_SIS.PDF
              // example file: https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/ORB_063400_063499/ESP_063462_1985/ESP_063462_1985_RED.LBL
              logger.info << "Detected a pushbroom camera system";
              this->pushbroom.size = this->camera.size;
              getline(file,value,',');
              this->pushbroom.projection_center.x = std::atof(value.c_str()); // latitude
              getline(file,value,',');
              this->pushbroom.projection_center.y = std::atof(value.c_str()); // longitude
              getline(file,value,',');
              this->pushbroom.axis_radius = std::atof(value.c_str()); // radius of the body at the projection center
              getline(file,value,',');
              this->pushbroom.roll = std::atof(value.c_str()); // roll of the pushbroom
              getline(file,value,',');
              this->pushbroom.altitude = std::atof(value.c_str()); // the altitude of the pushbroom
              getline(file,value,',');
              this->pushbroom.foc = std::atof(value.c_str()); // the foc of the sensor
              // calcualte the fov
              getline(file,value,',');
              this->pushbroom.gsd = std::atof(value.c_str()) / 1000.0f; // in meter, convert to km   // convert below to meters
              // this->pushbroom.fov = 2.0f * tanf(this->pushbroom.gsd / (this->pushbroom.altitude));
              // this->pushbroom.fov = 2.0f * atanf( (this->pushbroom.gsd * (this->camera.size.x / 2.0f)) / this->pushbroom.altitude );
              getline(file,value,',');
              this->pushbroom.fov = std::atof(value.c_str()) * (PI / 180.0f); // the swath field of view in degrees
              // calculate dpix
              this->pushbroom.dpix.x = (this->pushbroom.foc * tanf(this->pushbroom.fov / 2.0f)) / (this->pushbroom.size.x / 2.0f );
              this->pushbroom.dpix.y = this->pushbroom.dpix.y;
              // done!
              file.close();
              break;
            } else {
              //
              // The projective camera params case
              //
              logger.info << "Detected a standard projective camera system";
              this->isPushbroom = false;
              this->camera.cam_pos.x = std::atof(value.c_str());
              getline(file,value,',');
              this->camera.cam_pos.y = std::atof(value.c_str());
              getline(file,value,',');
              this->camera.cam_pos.z = std::atof(value.c_str());
              getline(file,value,',');
              this->camera.cam_rot.x = std::atof(value.c_str());
              getline(file,value,',');
              this->camera.cam_rot.y = std::atof(value.c_str());
              getline(file,value,',');
              this->camera.cam_rot.z = std::atof(value.c_str());
              getline(file,value,',');
              this->camera.fov.x     = std::atof(value.c_str());
              getline(file,value,',');
              this->camera.fov.y     = std::atof(value.c_str());
              getline(file,value,',');
              this->camera.foc       = std::atof(value.c_str());
              getline(file,value,',');
              // this->camera.dpix.x    = std::atof(value.c_str());
              // uses pinhole camera assumption
              this->camera.dpix.x = (this->camera.foc * tanf(this->camera.fov.x / 2.0f)) / (this->camera.size.x / 2.0f );
              getline(file,value,',');
              // this->camera.dpix.y    = std::atof(value.c_str());
              // uses pinhole camera assumption
              this->camera.dpix.y = this->camera.dpix.x;
              getline(file,value,',');
              this->camera.timeStamp = std::strtoll(value.c_str(), NULL, 0);
              getline(file,value,',');
              // camera.size.x was already set
              getline(file,value,',');
              // camera.side.y was already set
              file.close();
              break;
            } // end parsing pinhole and pushbroom
          } // end reading file
      } // end while
      file.close();
    } else if (fileExists(params_path + "/params.bcp")) {
      logger.info << "Reading BCP encoded camera parameters ...";
      logger.warn << "WARN: BCP camera reading is not yet supported";
      // TODO read in binary incoded guys here
    } else if (fileExists(params_path + "/" + file_no_digits + "_par.txt")) {
      logger.info << "Middleburry ACSII camera params found for " + filename;
      this->isPushbroom = false;
      std::ifstream file(params_path + "/" + file_no_digits + "_par.txt"); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
      std::string value;

      while (file.good()){
        // check if we have found the files
        getline(file,value,' ');
        if (filename == value){
          // now parse the Middleburry file, see http://vision.middlebury.edu/mview/data/
          getline(file,value,' ');
          this->camera.foc = std::atof(value.c_str()) / 1000.0f; // k11
          this->camera.fov.x = 2.0 * atan( 1.0 / this->camera.foc) * (180.0/PI); // pls
          this->camera.dpix.x = (this->camera.foc * tanf(this->camera.fov.x / 2.0f)) / (this->camera.size.x / 2.0f );
          this->camera.dpix.y = this->camera.dpix.x;
          this->camera.fov.y = tan( (this->camera.dpix.x * this->camera.size.y * 0.5f) / this->camera.foc);

          getline(file,value,' '); //k12
          getline(file,value,' '); //k13
          getline(file,value,' '); //k21
          getline(file,value,' '); //k22
          getline(file,value,' '); //k23
          getline(file,value,' '); //k31
          getline(file,value,' '); //k32
          getline(file,value,' '); //k33

          float R[3][3];
          getline(file,value,' '); // R11
          R[0][0] = std::atof(value.c_str());
          getline(file,value,' '); // R12
          R[0][1] = std::atof(value.c_str());
          getline(file,value,' '); // R13
          R[0][2] = std::atof(value.c_str());
          getline(file,value,' '); // R21
          R[1][0] = std::atof(value.c_str());
          getline(file,value,' '); // R22
          R[1][1] = std::atof(value.c_str());
          getline(file,value,' '); // R23
          R[1][2] = std::atof(value.c_str());
          getline(file,value,' '); // R31
          R[2][0] = std::atof(value.c_str());
          getline(file,value,' '); // R32
          R[2][1] = std::atof(value.c_str());
          getline(file,value,' '); // R33
          R[2][2] = std::atof(value.c_str());

          // fill the rotations
          // helpful link 1: https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
          // helpful link 2: https://gamedev.stackexchange.com/questions/50963/how-to-extract-euler-angles-from-transformation-matrix
          this->camera.cam_rot.x = atan2f(R[2][1], R[2][2]);
          this->camera.cam_rot.y = atan2f(-1.0f * R[2][0], sqrtf((R[2][1] * R[2][1]) + (R[2][2] * R[2][2])));
          this->camera.cam_rot.z = atan2f(R[1][0], R[0][0]);

          getline(file,value,' '); // t1
          this->camera.cam_pos.x = std::atof(value.c_str());
          getline(file,value,' '); // t2
          this->camera.cam_pos.y = std::atof(value.c_str());
          getline(file,value,' '); // t3
          this->camera.cam_pos.z = std::atof(value.c_str());

          file.close();
          break;
        } // end file check
      } // end while

      file.close();
    } else { // if no config file was found!
      logger.warn << "WARN: NO CAMERA PARAM FILE FOUND, at least an empty params.csv or params.bcp is required. To disable this requirement use the flag -np or -noparams";
      // std::throw -1; // TODO make this throw an exception
      logger.err << "\t given   filePath: " << filePath;
      logger.err << "\t given   filename: " << filename;
      logger.err << "\t given param_path: " << params_path;
      logger.err << "\t cleaned filename: " << file_no_digits;
    }
  }
  logger.info << "filePath: " + filePath;
}

void ssrlcv::Image::checkpoint(std::string dirPath) {
  std::string path = dirPath + std::to_string(this->id) + "_" + typeid(ssrlcv::Image).name() + ".cpimg";
  std::ofstream cp(path.c_str(), std::ofstream::binary);
  size_t hash_code = typeid(ssrlcv::Image).hash_code();
  if(cp.is_open()){
    ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> temp = this->pixels;
    std::string temp_str = this->filePath;
    this->filePath.clear();
    this->pixels = nullptr;
    cp.write((char *)this,sizeof(ssrlcv::Image));
    this->filePath = temp_str;
    this->pixels = temp;
    cp.close();
    if(cp.good()){
      logger.info<<"checkpoint " + path + " successfully written";
    }
    else{
      path = "could not write Image checkpoint: " + path;
      throw ssrlcv::CheckpointException(path);
    }
  }
  else{
    path = "could not open for writing: " + path;
    throw ssrlcv::CheckpointException(path);
  }
  mkdir((dirPath + "/pixels").c_str(), 0755);
  this->pixels->checkpoint(this->id, dirPath + "/pixels/");
}

ssrlcv::Image::Image(std::string filePath, unsigned int convertColorDepthTo, int id){
  this->filePath = filePath;
  this->id = id;
  this->colorDepth = 1;
  ssrlcv::ptr::host<unsigned char> pixels_host = readImage(filePath.c_str(), this->size.y, this->size.x, this->colorDepth);

  this->camera.size = this->size;
  this->size = size;
  this->pixels = ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>>(pixels_host,this->size.y*this->size.x*this->colorDepth,cpu);
  //for(int i = 0; i < this->pixels->size(); ++i){
    //std::cout<<this->pixels->host.get()[i]<<"\n";
  //}
  if(convertColorDepthTo == 1){
    convertToBW(this->pixels, this->colorDepth);
    this->colorDepth = 1;
  }
  else if(convertColorDepthTo != 0){
    logger.err<<"ERROR: Image() does not currently support conversion to anything but BW";
    exit(-1);
  }
}

// =============================================================================================================
//
// Image Host Methods
//
// =============================================================================================================

void ssrlcv::Image::convertColorDepthTo(unsigned int colorDepth){
  logger.info.printf("Converting pixel depth to %d from %d", colorDepth, this->colorDepth);
  if(colorDepth == 1){
    convertToBW(this->pixels,this->colorDepth);
    this->colorDepth = 1;
  }
  else if (colorDepth == 3){
    convertToRGB(this->pixels,this->colorDepth);
    this->colorDepth = 3;
  }
  else{
    logger.err<<std::to_string(colorDepth)<<" is currently not supported in convertColorDepthTo";
    exit(-1);
  }
}
ssrlcv::ptr::value<ssrlcv::Unity<int2>> ssrlcv::Image::getPixelGradients(){
  return generatePixelGradients(this->size,this->pixels);
}
void ssrlcv::Image::alterSize(int scalingFactor){
  if(scalingFactor == 0){
    logger.err<<"using a binDepth of 0 results in no binning or upsampling";
    logger.err<<"use binDepth>0 for binning and binDepth<0 for upsampling";
    return;
  }
  else if((float)this->size.x/powf(2.0f,scalingFactor) < 1.0f ||(float)this->size.y/powf(2.0f,scalingFactor) < 1.0f){
    logger.err<<"ERROR binning "<<std::to_string(scalingFactor)<<" many times cannot be done on and image of size "<<std::to_string(this->size.x)<<"x"<<std::to_string(this->size.y);
    exit(-1);
  }
  MemoryState origin = this->pixels->getMemoryState();
  if(origin != gpu) this->pixels->setMemoryState(gpu);

  uint2 scaler = {2,2};
  if(scalingFactor < 0){//upsampling
    for(int i = 0; i < abs(scalingFactor); ++i){
      this->pixels->setData(upsample(this->size,this->pixels)->device,this->size.x*this->size.y*this->colorDepth*4,gpu);
      this->size = this->size*scaler;
    }
  }
  else{//downsampling
    for(int i = 0; i < scalingFactor; ++i){
      this->pixels->setData(bin(this->size,this->pixels)->device,(this->size.x*this->size.y*this->colorDepth)/4,gpu);
      this->size = this->size/scaler;
    }
  }
  if(origin != gpu) this->pixels->setMemoryState(origin);
}

/**
* retuns the camera paramters as a float vector where all data types are cast to floats
* removes the unix time stamp
* @returns array of floats representing the camera parameters in the order X pos, Y pos, Z pos, X rot, Y rot, Z rot, fov X, fov Y, foc, dpix x, dpix y
*/
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::Image::getFloatVector(){
  ssrlcv::ptr::value<ssrlcv::Unity<float>> params = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,11,ssrlcv::cpu);
  params->host.get()[0 ] = this->camera.cam_pos.x;
  params->host.get()[1 ] = this->camera.cam_pos.y;
  params->host.get()[2 ] = this->camera.cam_pos.z;
  params->host.get()[3 ] = this->camera.cam_rot.x;
  params->host.get()[4 ] = this->camera.cam_rot.y;
  params->host.get()[5 ] = this->camera.cam_rot.z;
  params->host.get()[6 ] = this->camera.fov.x    ;
  params->host.get()[7 ] = this->camera.fov.y    ;
  params->host.get()[8 ] = this->camera.foc      ;
  params->host.get()[9 ] = this->camera.dpix.x   ;
  params->host.get()[10] = this->camera.dpix.y   ;
  return params;
}

/**
* retuns the camera paramters as a float vector of length len, where all data types are cast to floats
* removes the unix time stamp
* @param len the desired lentgth of the vector, has the effect of extracting less parameters
* @returns array of floats representing the camera parameters in the order X pos, Y pos, Z pos, X rot, Y rot, Z rot, fov X, fov Y, foc, dpix x, dpix y
*/
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::Image::getFloatVector(int len){
  ssrlcv::ptr::value<ssrlcv::Unity<float>> params = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,len,ssrlcv::cpu);
  switch(len){
    case 11:
      params->host.get()[10] = this->camera.dpix.y    ;
    case 10:
      params->host.get()[9 ] = this->camera.dpix.x    ;
    case 9:
      params->host.get()[8 ] = this->camera.foc       ;
    case 8:
      params->host.get()[7 ] = this->camera.fov.y     ;
    case 7:
      params->host.get()[6 ] = this->camera.fov.x     ;
    case 6:
      params->host.get()[5 ] = this->camera.cam_rot.z ;
    case 5:
      params->host.get()[4 ] = this->camera.cam_rot.y ;
    case 4:
      params->host.get()[3 ] = this->camera.cam_rot.x ;
    case 3:
      params->host.get()[2 ] = this->camera.cam_pos.z ;
    case 2:
      params->host.get()[1 ] = this->camera.cam_pos.y ;
    case 1:
      params->host.get()[0 ] = this->camera.cam_pos.x ;
      break;
    default:
      logger.err << "ERROR: the requested camera float vector is out of bounds or non-standard!";
      break;
  }
  return params;
}

/**
* updates the camera parameters from a float vector representing camera parameters
* if there are less than 11 params the camera will still be updated, retaining values for params not included
* @param array of floats which should update the current parameters in the order X pos, Y pos, Z pos, X rot, Y rot, Z rot, fov X, fov Y, foc, dpix x, dpix y
*/
void ssrlcv::Image::setFloatVector(ssrlcv::ptr::value<ssrlcv::Unity<float>> params){
  switch(params->size()){
    case 11:
      this->camera.dpix.y    = params->host.get()[10];
    case 10:
      this->camera.dpix.x    = params->host.get()[9];
    case 9:
      this->camera.foc       = params->host.get()[8];
    case 8:
      this->camera.fov.y     = params->host.get()[7];
    case 7:
      this->camera.fov.x     = params->host.get()[6];
    case 6:
      this->camera.cam_rot.z = params->host.get()[5];
    case 5:
      this->camera.cam_rot.y = params->host.get()[4];
    case 4:
      this->camera.cam_rot.x = params->host.get()[3];
    case 3:
      this->camera.cam_pos.z = params->host.get()[2];
    case 2:
      this->camera.cam_pos.y = params->host.get()[1];
    case 1:
      this->camera.cam_pos.x = params->host.get()[0];
    default:
      break;
  }
}

/**
* calculates a vector represnting the difference between the input extinsic camera parameters and the local extinsic camera paramters
* caclulated for the cam_pos and cam_rot vectors as local camera - input camera
* @param other the other camera's extrinsic parameters
* @returns difference the difference between the extinsic camera params
*/
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::Image::getExtrinsicDifference(Camera other){
  ssrlcv::ptr::value<ssrlcv::Unity<float>> diff = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,6,ssrlcv::cpu);
  diff->host.get()[0] = camera.cam_pos.x - other.cam_pos.x;
  diff->host.get()[1] = camera.cam_pos.y - other.cam_pos.y;
  diff->host.get()[2] = camera.cam_pos.z - other.cam_pos.z;
  diff->host.get()[3] = camera.cam_rot.x - other.cam_rot.x;
  diff->host.get()[4] = camera.cam_rot.y - other.cam_rot.y;
  diff->host.get()[5] = camera.cam_rot.z - other.cam_rot.z;
  return diff;
}

/**
 * Sets P to a projection matrix that maps a point in homogenous R3 (ECEF) to its projection
 * onto this image's camera (as a point in homogenous R2)
 * 
 * @param P pass-by-value matrix that is set to the projection matrix
 */
void ssrlcv::Image::getProjectionMatrix(float4 (&P)[3]) {
  float3 K[3];
  float4 R[3];

  K[0] = {this->camera.foc / this->camera.dpix.x, 0, this->size.x / 2.0f};
  K[1] = {0, this->camera.foc / this->camera.dpix.y, this->size.y / 2.0f};
  K[2] = {0, 0, 1};

  R[0] = {
    cosf(this->camera.cam_rot.z) * cosf(this->camera.cam_rot.y),
    sinf(this->camera.cam_rot.z) * cosf(this->camera.cam_rot.y),
    -1 * sinf(this->camera.cam_rot.y),
    0
  };

  R[1] = {
    cosf(this->camera.cam_rot.z) * sinf(this->camera.cam_rot.y) * sinf(this->camera.cam_rot.x) - sinf(this->camera.cam_rot.z) * cosf(this->camera.cam_rot.x),
    sinf(this->camera.cam_rot.z) * sinf(this->camera.cam_rot.y) * sinf(this->camera.cam_rot.x) + cosf(this->camera.cam_rot.z) * cosf(this->camera.cam_rot.x),
    cosf(this->camera.cam_rot.y) * sinf(this->camera.cam_rot.x),
    0
  };

  R[2] = {
    cosf(this->camera.cam_rot.z) * sinf(this->camera.cam_rot.y) * cosf(this->camera.cam_rot.x) + sinf(this->camera.cam_rot.z) * sinf(this->camera.cam_rot.x),
    sinf(this->camera.cam_rot.z) * sinf(this->camera.cam_rot.y) * cosf(this->camera.cam_rot.x) - cosf(this->camera.cam_rot.z) * sinf(this->camera.cam_rot.x),
    cosf(this->camera.cam_rot.y) * cosf(this->camera.cam_rot.x),
    0
  };

  float3 ecef_cent = {
    this->camera.cam_pos.x + this->camera.ecef_offset.x,
    this->camera.cam_pos.y + this->camera.ecef_offset.y,
    this->camera.cam_pos.z + this->camera.ecef_offset.z
  };

  for (int i = 0; i < 3; i ++) {
    R[i].w -= R[i].x * ecef_cent.x + R[i].y * ecef_cent.y + R[i].z * ecef_cent.z;
  }

  multiply(K, R, P);

}
// =============================================================================================================
//
// Other Host Methods
//
// =============================================================================================================

ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> ssrlcv::addBufferBorder(uint2 size, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, int2 border){
  if(border.x == 0 && border.y == 0){
    logger.err<<"ERROR border cannot be 0";
    exit(-1);
  }
  if(border.x*2 + (int) size.x < 0 || border.y*2 + (int)size.y < 0){
    logger.err<<"ERROR border causes negative dimensions";
    exit(-1);
  }
  if(pixels->size()%((int)size.x*size.y) != 0){
    logger.err<<"ERROR color depth cannot be determined due to pixels->size()%(size.x*size.y) != 0";
  }
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);
  uint2 newSize = {size.x + (border.x*2),size.y + (border.y*2)};
  int colorDepth = pixels->size()/((int)size.x*size.y);
  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> bufferedPixels = ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>>(nullptr,newSize.x*newSize.y*colorDepth,gpu);
  for(int y = border.y; y < (int)size.y + border.y; ++y){
    CudaSafeCall(cudaMemcpy(bufferedPixels->device.get() + (y*newSize.x) + border.x,pixels->device.get() + (y*size.x),size.x*sizeof(unsigned char),cudaMemcpyDeviceToDevice));
  }
  if(origin != gpu){
    bufferedPixels->setMemoryState(origin);
    pixels->setMemoryState(origin);
  }
  return bufferedPixels;
}
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::addBufferBorder(uint2 size, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, int2 border){
  if(border.x == 0 && border.y == 0){
    logger.err<<"ERROR border cannot be 0";
    exit(-1);
  }
  if(border.x*2 + (int) size.x < 0 || border.y*2 + (int)size.y < 0){
    logger.err<<"ERROR border causes negative dimensions";
    exit(-1);
  }
  if(pixels->size()%((int)size.x*size.y) != 0){
    logger.err<<"ERROR color depth cannot be determined due to pixels->size()%(size.x*size.y) != 0";
  }
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);

  uint2 newSize = {size.x + (border.x*2),size.y + (border.y*2)};
  int colorDepth = pixels->size()/((int)size.x*size.y);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> bufferedPixels = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,newSize.x*newSize.y*colorDepth,gpu);
  for(int y = 0; y < (int)size.y; ++y){
    CudaSafeCall(cudaMemcpy(bufferedPixels->device.get() + ((y+border.y)*newSize.x) + border.x,pixels->device.get() + (y*size.x),size.x*sizeof(float),cudaMemcpyDeviceToDevice));
  }
  if(origin != gpu){
    bufferedPixels->setMemoryState(origin);
    pixels->setMemoryState(origin);
  }
  return bufferedPixels;
}

ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> ssrlcv::convertImageToChar(ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels){
  ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels_cpy = ssrlcv::ptr::value<ssrlcv::Unity<float>>(pixels);
  normalizeImage(pixels_cpy);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixels_cpy->size(),grid,block,convertToCharImage);
  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> castPixels = ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>>(nullptr,pixels_cpy->size(),gpu);
  convertToCharImage<<<grid,block>>>(pixels_cpy->size(),castPixels->device.get(),pixels_cpy->device.get());
  cudaDeviceSynchronize();
  CudaCheckError();
  return castPixels;
}
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::convertImageToFlt(ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels){
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixels->size(),grid,block,convertToFltImage);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> castPixels = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,pixels->size(),gpu);
  convertToFltImage<<<grid,block>>>(pixels->size(),pixels->device.get(),castPixels->device.get());
  cudaDeviceSynchronize();
  CudaCheckError();
  if(origin != gpu){
    pixels->setMemoryState(origin);
    castPixels->setMemoryState(origin);
  }
  return castPixels;
}

//todo use cuda reduction instead of cpu loop for min max finding
//todo add support for color depth
void ssrlcv::normalizeImage(ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels){
  MemoryState origin = pixels->getMemoryState();
  float2 minMax = {FLT_MAX,-FLT_MAX};
  if(pixels->getFore() != both) pixels->setMemoryState(both);
  for(int i = 0; i < pixels->size(); ++i){
      if(minMax.x > pixels->host.get()[i]) minMax.x = pixels->host.get()[i];
      if(minMax.y < pixels->host.get()[i]) minMax.y = pixels->host.get()[i];
  }
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned long, float*, float2) = &normalize;
  getFlatGridBlock(pixels->size(),grid,block,fp);
  normalize<<<grid,block>>>(pixels->size(),pixels->device.get(),minMax);
  cudaDeviceSynchronize();
  CudaCheckError();
  pixels->setFore(gpu);
  if(origin != pixels->getMemoryState()) pixels->setMemoryState(origin);
}
void ssrlcv::normalizeImage(ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, float2 minMax){
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned long, float*, float2) = &normalize;
  getFlatGridBlock(pixels->size(),grid,block,fp);
  normalize<<<grid,block>>>(pixels->size(),pixels->device.get(),minMax);
  pixels->setFore(gpu);
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin != gpu) pixels->setMemoryState(origin);
}

void ssrlcv::convertToBW(ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, unsigned int colorDepth){
  if(colorDepth == 1){
    logger.warn<<"Pixels are already bw";
    return;
  }

  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);

  unsigned int numPixels = (pixels->size()/colorDepth);

  ssrlcv::ptr::device<unsigned char> bwPixels_device( numPixels);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numPixels, grid, block,generateBW);

  generateBW<<<grid,block>>>(numPixels, colorDepth, pixels->device.get(), bwPixels_device.get());
  cudaDeviceSynchronize();
  CudaCheckError();

  pixels->setData(bwPixels_device, numPixels, gpu);
  if(origin != gpu) pixels->setMemoryState(origin);
}
void ssrlcv::convertToRGB(ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, unsigned int colorDepth){
  if(colorDepth == 3){
    logger.warn<<"Pixels are already rgb";
    return;
  }

  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);

  unsigned int numPixels = (pixels->size()/colorDepth);

  ssrlcv::ptr::device<unsigned char> rgbPixels_device( numPixels*3);

  dim3 grid;
  dim3 block;
  getFlatGridBlock(numPixels, grid, block,generateRGB);
  generateRGB<<<grid,block>>>(numPixels, colorDepth, pixels->device.get(), rgbPixels_device.get());
  cudaDeviceSynchronize();
  CudaCheckError();

  pixels->setData(rgbPixels_device, 3*numPixels, gpu);
  if(origin != gpu) pixels->setMemoryState(origin);
}
//TODO implement
void calcFundamentalMatrix_2View(float cam0[3][3], float cam1[3][3], float (&F)[3][3]){

}
void ssrlcv::calcFundamentalMatrix_2View(Image* query, Image* target, float3 (&F)[3]){
  if(query->camera.foc != target->camera.foc){
    logger.err<<"ERROR calculating fundamental matrix for 2view needs to bet taken with same camera (foc&fov are same)";
    exit(-1);
  }
  float angle1;
  if(abs(query->camera.cam_rot.z) < .00001) {
    if(query->camera.cam_rot.y > 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle1 = atan(query->camera.cam_rot.y / query->camera.cam_rot.z);
    if(query->camera.cam_rot.z<0 && query->camera.cam_rot.y>=0) {
      angle1 += PI;
    }
    if(query->camera.cam_rot.z<0 && query->camera.cam_rot.y<0) {
      angle1 -= PI;
    }
  }
  float3 A1[3] = {
    {1, 0, 0},
    {0, cos(angle1), -sin(angle1)},
    {0, sin(angle1), cos(angle1)}
  };

  float3 temp = {0.0f,0.0f,0.0f};
  multiply(A1, query->camera.cam_rot,temp);

  float angle2 = 0.0f;
  if(abs(temp.z) < .00001) {
    if(temp.x <= 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle2 = atan(-1*temp.x / temp.z);
    if(temp.z<0 && temp.x<0) {
      angle1 += PI;
    }
    if(temp.z<0 && temp.x>0) {
      angle2 -= PI;
    }
  }
  float3 B1[3] = {
    {cos(angle2), 0, sin(angle2)},
    {0, 1, 0},
    {-sin(angle2), 0, cos(angle2)}
  };

  float3 temp2 = {0.0f,0.0f,0.0f};
  multiply(B1, temp, temp2);
  float3 rot1[3];
  multiply(B1, A1, rot1);
  float3 rot1Transpose[3];
  transpose(rot1,rot1Transpose);
  multiply(rot1Transpose, temp2, temp);

  angle1 = 0.0f;
  if(abs(target->camera.cam_rot.z) < .00001) {
    if(target->camera.cam_rot.y > 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle1 = atan(target->camera.cam_rot.y / target->camera.cam_rot.z);
    if(target->camera.cam_rot.z<0 && target->camera.cam_rot.y>=0) {
      angle1 += PI;
    }
    if(target->camera.cam_rot.z<0 && target->camera.cam_rot.y<0) {
      angle1 -= PI;
    }
  }
  float3 A2[3] = {
    {1, 0, 0},
    {0, cos(angle1), -sin(angle1)},
    {0, sin(angle1), cos(angle1)}
  };
  multiply(A2, target->camera.cam_rot,temp2);

  angle2 = 0.0f;
  if(abs(temp2.z) < .00001) {
    if(temp2.x <= 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle2 = atan(-1*temp2.x / temp2.z);
    if(temp2.z<0 && temp2.x<0) {
      angle1 += PI;
    }
    if(temp2.z<0 && temp2.x>0) {
      angle2 -= PI;
    }
  }
  float3 B2[3] = {
    {cos(angle2), 0, sin(angle2)},
    {0, 1, 0},
    {-sin(angle2), 0, cos(angle2)}
  };

  multiply(B2, temp2, temp);

  float3 rot2[3];
  multiply(B2, A2, rot2);
  float3 rot2Transpose[3];
  transpose(rot2, rot2Transpose);

  multiply(rot2Transpose, temp, temp2);

  float3 K[3] = {
    {query->camera.foc, 0, ((float)query->size.x)/2.0f},//NOTE the foc was divided by dpix.x and dpix.y but currently using foc in pixels
    {0, query->camera.foc, ((float)query->size.y)/2.0f},//NOTE the foc was divided by dpix.x and dpix.y but currently using foc in pixels
    {0, 0, 1}
  };
  float3 K_inv[3];
  inverse(K,K_inv);
  float3 K_invTranspose[3];
  transpose(K_inv,K_invTranspose);

  float3 R[3];
  multiply(rot2Transpose, rot1, R);
  float3 S[3] = {
    {0, query->camera.cam_pos.z - target->camera.cam_pos.z, target->camera.cam_pos.y - query->camera.cam_pos.y},
    {query->camera.cam_pos.z - target->camera.cam_pos.z,0, query->camera.cam_pos.x - target->camera.cam_pos.x},
    {query->camera.cam_pos.y - target->camera.cam_pos.y, target->camera.cam_pos.x - query->camera.cam_pos.x, 0}
  };
  float3 E[3];
  multiply(R,S,E);
  float3 tempF[3];
  multiply(K_invTranspose, E,tempF);
  multiply(tempF, K_inv, F);
  logger.info.printf("between image %d and %d the final fundamental matrix result is", query->id, target->id);
  for(int r = 0; r < 3; ++r) {
    std::stringstream ss;
    ss << F[r].x << "  " << F[r].y << " "<<  F[r].z;
    logger.info << ss.str();
  }
}
void ssrlcv::get_cam_params2view(Image* cam1, Image* cam2, std::string infile){
  std::ifstream input(infile);
  std::string line;
  float res = 0.0f;
  while(std::getline(input, line)) {
    std::istringstream iss(line);
    std::string param;
    float arg1;
    float arg2;
    float arg3;
    iss >> param >> arg1;
    if(param.compare("foc") == 0) {
      cam1->camera.foc = arg1;
      cam2->camera.foc = arg1;
    }
    else if(param.compare("fov") == 0) {
      //cam1->camera.fov = arg1;
      //cam2->camera.fov = arg1;
    }
    else if(param.compare("res") == 0) {
      res = arg1;
    }
    else if(param.compare("cam1C") == 0) {
      iss >> arg2 >> arg3;
      cam1->camera.cam_pos.x = arg1;
      cam1->camera.cam_pos.y = arg2;
      cam1->camera.cam_pos.z = arg3;
    }
    else if(param.compare("cam1V") == 0) {
      iss >> arg2 >> arg3;
      cam1->camera.cam_rot.x = arg1;
      cam1->camera.cam_rot.y = arg2;
      cam1->camera.cam_rot.z = arg3;
    }
    else if(param.compare("cam2C") == 0) {
      iss >> arg2 >> arg3;
      cam2->camera.cam_pos.x = arg1;
      cam2->camera.cam_pos.y = arg2;
      cam2->camera.cam_pos.z = arg3;
    }
    else if(param.compare("cam2V") == 0) {
      iss >> arg2 >> arg3;
      cam2->camera.cam_rot.x = arg1;
      cam2->camera.cam_rot.y = arg2;
      cam2->camera.cam_rot.z = arg3;
    }
  }
  res = res;//remove compile time warning while this is not used lol
  cam1->camera.dpix = {cam1->camera.foc*tan(cam1->camera.fov.x/2)/(cam1->size.x/2),
    cam1->camera.foc*tan(cam1->camera.fov.y/2)/(cam1->size.y/2)};
  cam2->camera.dpix = {cam2->camera.foc*tan(cam2->camera.fov.x/2)/(cam2->size.x/2),
    cam2->camera.foc*tan(cam2->camera.fov.y/2)/(cam2->size.y/2)};
}

ssrlcv::ptr::value<ssrlcv::Unity<int2>> ssrlcv::generatePixelGradients(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels){
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);

  ssrlcv::ptr::device<int2> gradients_device(pixels->size());
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(uint2,unsigned char*,int2*) = &calculatePixelGradients;
  getFlatGridBlock(pixels->size(),grid,block,fp);
  calculatePixelGradients<<<grid,block>>>(imageSize,pixels->device.get(),gradients_device.get());
  CudaCheckError();

  if(origin != gpu) pixels->setMemoryState(origin);

  return ssrlcv::ptr::value<Unity<int2>>(gradients_device,pixels->size(),gpu);
}
ssrlcv::ptr::value<ssrlcv::Unity<float2>> ssrlcv::generatePixelGradients(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels){
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);

  ssrlcv::ptr::device<float2> gradients_device(pixels->size());
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(uint2,float*,float2*) = &calculatePixelGradients;
  getFlatGridBlock(pixels->size(),grid,block,fp);
  calculatePixelGradients<<<grid,block>>>(imageSize,pixels->device.get(),gradients_device.get());
  CudaCheckError();

  if(origin == cpu) pixels->setMemoryState(cpu);

  return ssrlcv::ptr::value<Unity<float2>>(gradients_device,pixels->size(),gpu);
}

void ssrlcv::makeBinnable(uint2 &size, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, int plannedDepth){
  MemoryState origin = pixels->getMemoryState();
  int numResize = (int)pow(2, plannedDepth);
  int dimOffset[2] = {(int)size.x%numResize,(int)size.y%numResize};
  if(dimOffset[0] || dimOffset[1]){
    if(origin != gpu) pixels->setMemoryState(gpu);
    bool mustSizeUp = size.x%2 || size.y%2;
    if(mustSizeUp){
      pixels->setData(upsample(size,pixels)->device,pixels->size()*4,gpu);
      size = size*2;numResize *= 2;
      dimOffset[0] = size.x%numResize;
      dimOffset[1] = size.y%numResize;
    }
    int2 border = {
        dimOffset[0] ? (numResize-((int)size.x%numResize))/2 : 0,
        dimOffset[1] ? (numResize-((int)size.y%numResize))/2 : 0
    };
    uint2 newSize = {border.x*2 + size.x, border.y*2 + size.y};
    pixels->setData(addBufferBorder(size,pixels,border)->device,newSize.x*newSize.y,gpu);
    size = newSize;
    pixels->setData(bin(size,pixels)->device,pixels->size()/4,gpu);
    size = size/2;
    if(origin != gpu) pixels->setMemoryState(origin);
  }
  else{
    logger.info.printf("no resize necessary for binning to depth %d", plannedDepth);
  }
}
void ssrlcv::makeBinnable(uint2 &size, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, int plannedDepth){
  MemoryState origin = pixels->getMemoryState();
  int numResize = (int)pow(2, plannedDepth);
  int dimOffset[2] = {(int)size.x%numResize,(int)size.y%numResize};
  if(dimOffset[0] || dimOffset[1]){
    if(origin != gpu) pixels->setMemoryState(gpu);
    bool mustSizeUp = size.x%2 || size.y%2;
    if(mustSizeUp){
      pixels->setData(upsample(size,pixels)->device,pixels->size()*4,gpu);
      size = size*2;numResize *= 2;
      dimOffset[0] = size.x%numResize;
      dimOffset[1] = size.y%numResize;
    }
    int2 border = {
        dimOffset[0] ? (numResize-((int)size.x%numResize))/2 : 0,
        dimOffset[1] ? (numResize-((int)size.y%numResize))/2 : 0
    };
    uint2 newSize = {border.x*2 + size.x, border.y*2 + size.y};
    pixels->setData(addBufferBorder(size,pixels,border)->device,newSize.x*newSize.y,gpu);
    size = newSize;
    if(mustSizeUp){
        pixels->setData(bin(size,pixels)->device,pixels->size()/4,gpu);
        size = size/2;
    }
    if(origin != gpu) pixels->setMemoryState(origin);
  }
  else{
    logger.info.printf("no resize necessary for binning to depth %d", plannedDepth);
  }
}

ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> ssrlcv::bin(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels){
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);
  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> binnedImage = ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>>(nullptr, pixels->size()/4, gpu);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(uint2, unsigned int, unsigned char*, unsigned char*) = &binImage;
  get2DGridBlock(imageSize/2,grid,block,fp);
  int colorDepth = pixels->size()/((int)imageSize.x*imageSize.y);
  binImage<<<grid,block>>>(imageSize,colorDepth,pixels->device.get(),binnedImage->device.get());
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin != gpu){
    pixels->setMemoryState(origin);
    binnedImage->setMemoryState(origin);
  }
  return binnedImage;
}
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::bin(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels){
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);

  ssrlcv::ptr::value<ssrlcv::Unity<float>> binnedImage = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr, pixels->size()/4, gpu);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(uint2, unsigned int, float*, float*) = &binImage;
  get2DGridBlock(imageSize/2,grid,block,fp);
  int colorDepth = pixels->size()/((int)imageSize.x*imageSize.y);
  binImage<<<grid,block>>>(imageSize,colorDepth,pixels->device.get(),binnedImage->device.get());
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin != gpu){
    pixels->setMemoryState(origin);
    binnedImage->setMemoryState(origin);
  }
  return binnedImage;
}
ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> ssrlcv::upsample(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels){
  MemoryState origin = pixels->getMemoryState();

  if(origin != gpu) pixels->setMemoryState(gpu);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(uint2, unsigned int, unsigned char*, unsigned char*) = &upsampleImage;
  get2DGridBlock(imageSize*2,grid,block,fp);
  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> upsampledImage = ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>>(nullptr, pixels->size()*4, gpu);
  int colorDepth = pixels->size()/((int)imageSize.x*imageSize.y);
  upsampleImage<<<grid,block>>>(imageSize,colorDepth,pixels->device.get(),upsampledImage->device.get());
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin != gpu){
    pixels->setMemoryState(origin);
    upsampledImage->setMemoryState(origin);
  }
  return upsampledImage;

}
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::upsample(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels){
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(uint2, unsigned int, float*, float*) = &upsampleImage;
  get2DGridBlock(imageSize*2,grid,block,fp);
  int colorDepth = pixels->size()/((int)imageSize.x*imageSize.y);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> upsampledImage = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr, pixels->size()*4, gpu);
  upsampleImage<<<grid,block>>>(imageSize,colorDepth,pixels->device.get(),upsampledImage->device.get());
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin != gpu){
    pixels->setMemoryState(origin);
    upsampledImage->setMemoryState(origin);
  }
  return upsampledImage;
}
ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> ssrlcv::scaleImage(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, float outputPixelWidth){
  MemoryState origin = pixels->getMemoryState();

  if(origin != gpu) pixels->setMemoryState(gpu);


  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(uint2, unsigned int, unsigned char*, unsigned char*, float) = &bilinearInterpolation;
  get2DGridBlock((imageSize/outputPixelWidth) + 1, grid,block,fp);
  int colorDepth = pixels->size()/((int)imageSize.x*imageSize.y);
  ssrlcv::ptr::device<unsigned char> sampledImage_device(pixels->size()*4);
  bilinearInterpolation<<<grid,block>>>(imageSize,colorDepth,pixels->device.get(),sampledImage_device.get(),outputPixelWidth);
  cudaDeviceSynchronize();
  CudaCheckError();

  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> sampledImage = ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>>(sampledImage_device, (unsigned long) (pixels->size()/(outputPixelWidth*outputPixelWidth)), gpu);

  if(origin != gpu){
    pixels->setMemoryState(origin);
    sampledImage->setMemoryState(origin);
  }

  return sampledImage;
}
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::scaleImage(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, float outputPixelWidth){
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);


  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(uint2, unsigned int, float*, float*, float) = &bilinearInterpolation;
  get2DGridBlock((imageSize/outputPixelWidth) + 1, grid,block,fp);
  int colorDepth = pixels->size()/((int)imageSize.x*imageSize.y);
  ssrlcv::ptr::device<float> sampledImage_device(pixels->size()*4);
  bilinearInterpolation<<<grid,block>>>(imageSize,colorDepth,pixels->device.get(),sampledImage_device.get(),outputPixelWidth);
  cudaDeviceSynchronize();
  CudaCheckError();

  ssrlcv::ptr::value<ssrlcv::Unity<float>> sampledImage = ssrlcv::ptr::value<ssrlcv::Unity<float>>(sampledImage_device, (unsigned long) (pixels->size()/(outputPixelWidth*outputPixelWidth)), gpu);

  if(origin != gpu){
    pixels->setMemoryState(origin);
    sampledImage->setMemoryState(origin);
  }

  return sampledImage;
}


ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::convolve(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, int2 kernelSize, float* kernel, bool symmetric){
  if(kernelSize.x%2 == 0 || kernelSize.y%2 == 0){
    logger.err<<"ERROR kernel for image convolution must have an odd dimension";
    exit(-1);
  }
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);
  int colorDepth = pixels->size()/((int)imageSize.x*imageSize.y);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> convolvedImage = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,pixels->size(),gpu);
  ssrlcv::ptr::device<float> kernel_device(kernelSize.x*kernelSize.y);
  CudaSafeCall(cudaMemcpy(kernel_device.get(),kernel,kernelSize.x*kernelSize.y*sizeof(float),cudaMemcpyHostToDevice));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};

  if(symmetric){
    void (*fp)(uint2, unsigned char*, unsigned int, int2, float*, float*) = &convolveImage_symmetric;
    get2DGridBlock(imageSize,grid,block,fp);
    convolveImage_symmetric<<<grid,block>>>(imageSize, pixels->device.get(), colorDepth, kernelSize, kernel_device.get(), convolvedImage->device.get());
  }
  else{
    void (*fp)(uint2, unsigned char*, unsigned int, int2, float*, float*) = &convolveImage;
    get2DGridBlock(imageSize,grid,block,fp);
    convolveImage<<<grid,block>>>(imageSize, pixels->device.get(), colorDepth, kernelSize, kernel_device.get(), convolvedImage->device.get());
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin != gpu){
    convolvedImage->setMemoryState(origin);
    pixels->setMemoryState(origin);
  }
  return convolvedImage;
}
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::convolve(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, int2 kernelSize, float* kernel, bool symmetric){
  if(kernelSize.x%2 == 0 || kernelSize.y%2 == 0){
    logger.err<<"ERROR kernel for image convolution must have an odd dimension";
    exit(-1);
  }
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);
  int colorDepth = pixels->size()/((int)imageSize.x*imageSize.y);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> convolvedImage = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,pixels->size(),gpu);
  ssrlcv::ptr::device<float> kernel_device(kernelSize.x*kernelSize.y);
  CudaSafeCall(cudaMemcpy(kernel_device.get(),kernel,kernelSize.x*kernelSize.y*sizeof(float),cudaMemcpyHostToDevice));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};

  if(symmetric){
    void (*fp)(uint2, float*, unsigned int, int2, float*, float*) = &convolveImage_symmetric;
    get2DGridBlock(imageSize,grid,block,fp);
    convolveImage_symmetric<<<grid,block>>>(imageSize, pixels->device.get(), colorDepth, kernelSize, kernel_device.get(), convolvedImage->device.get());
  }
  else{
    void (*fp)(uint2, float*, unsigned int, int2, float*, float*) = &convolveImage;
    get2DGridBlock(imageSize,grid,block,fp);
    convolveImage<<<grid,block>>>(imageSize, pixels->device.get(), colorDepth, kernelSize, kernel_device.get(), convolvedImage->device.get());
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin != gpu){
    convolvedImage->setMemoryState(origin);
    pixels->setMemoryState(origin);
  }
  return convolvedImage;
}

ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::convolveSeparable(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, int kernelSize, float* kernel, bool symmetric){
  if(kernelSize%2 == 0){
    logger.err<<"ERROR kernel for image convolution must have an odd dimension";
    exit(-1);
  }
  MemoryState origin = pixels->getMemoryState();
  if(origin != gpu) pixels->setMemoryState(gpu);
  int colorDepth = pixels->size()/((int)imageSize.x*imageSize.y);
  ssrlcv::ptr::device<float> kernel_device(kernelSize);
  CudaSafeCall(cudaMemcpy(kernel_device.get(),kernel,kernelSize*sizeof(float),cudaMemcpyHostToDevice));
  ssrlcv::ptr::value<ssrlcv::Unity<float>> convolvedImage = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,pixels->size(),gpu);
  ssrlcv::ptr::device<float> halfConvolvedImage = ssrlcv::ptr::device<float>(pixels->size());

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};

  if(symmetric){
    void (*fp)(uint2, float*, unsigned int, int, float*, float*, bool) = &convolveImage1D_symmetric;
    get2DGridBlock(imageSize,grid,block,fp);
    convolveImage1D_symmetric<<<grid,block>>>(imageSize, pixels->device.get(), colorDepth, kernelSize, kernel_device.get(), halfConvolvedImage.get(), false);
    cudaDeviceSynchronize();
    CudaCheckError();
    convolveImage1D_symmetric<<<grid,block>>>(imageSize, halfConvolvedImage.get(), colorDepth, kernelSize, kernel_device.get(), convolvedImage->device.get(), true);
    cudaDeviceSynchronize();
    CudaCheckError();
  }
  else{
    void (*fp)(uint2, float*, unsigned int, int, float*, float*, bool) = &convolveImage1D;
    get2DGridBlock(imageSize,grid,block,fp);
    convolveImage1D<<<grid,block>>>(imageSize, pixels->device.get(), colorDepth, kernelSize, kernel_device.get(), halfConvolvedImage.get(), false);
    cudaDeviceSynchronize();
    CudaCheckError();
    convolveImage1D<<<grid,block>>>(imageSize, halfConvolvedImage.get(), colorDepth, kernelSize, kernel_device.get(), convolvedImage->device.get(), true);
    cudaDeviceSynchronize();
    CudaCheckError();
  }

  if(origin != gpu){
    pixels->setMemoryState(origin);
    convolvedImage->setMemoryState(origin);
  }
  return convolvedImage;
}


// =============================================================================================================
//
// Device Kernels
//
// =============================================================================================================

__device__ __host__ __forceinline__ int ssrlcv::getSymmetrizedCoord(int i, unsigned int l){
  int ll = 2*l;
  i = (i+ll)%ll;
  return (i>l-1) ? i = ll - 1 - i : i;
}
__device__ __host__ unsigned char ssrlcv::bwaToBW(const uchar2 &color){
  return (1-color.y)*color.x + color.y*color.x;
}
__device__ __host__ unsigned char ssrlcv::rgbToBW(const uchar3 &color){
  return (color.x/4) + (color.y/2) + (color.z/4);
}
__device__ __host__ unsigned char ssrlcv::rgbaToBW(const uchar4 &color){
  return rgbToBW(rgbaToRGB(color));
}
__device__ __host__ uchar3 ssrlcv::bwToRGB(const unsigned char &color){
  int colorTemp = (int) color*10;
  return {(unsigned char)(colorTemp/4),(unsigned char)(colorTemp/2),(unsigned char)(colorTemp/4)};
}
__device__ __host__ uchar3 ssrlcv::bwaToRGB(const uchar2 &color){
  return {color.x,color.y,(unsigned char)((color.x/3)*2 + (color.y/3))};
}
__device__ __host__ uchar3 ssrlcv::rgbaToRGB(const uchar4 &color){
  return {
    (unsigned char)((1-color.w)*color.x + color.w*color.x),
    (unsigned char)((1-color.w)*color.y + color.w*color.y),
    (unsigned char)((1-color.w)*color.z + color.w*color.z),
  };
}

__global__ void ssrlcv::generateBW(int numPixels, unsigned int colorDepth, unsigned char* colorPixels, unsigned char* pixels){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    int numValues = (int) colorDepth;
    switch(numValues){
      case 2:
        pixels[globalID] = bwaToBW({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1]});
        break;
      case 3:
        pixels[globalID] = rgbToBW({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1], colorPixels[globalID*numValues + 2]});
        break;
      case 4:
        pixels[globalID] = rgbaToBW({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1], colorPixels[globalID*numValues + 2], colorPixels[globalID*numValues + 3]});
        break;
      default:
        printf("ERROR colorDepth of %u is not supported",numValues);
        asm("trap;");
    }
  }
}
__global__ void ssrlcv::generateRGB(int numPixels, unsigned int colorDepth, unsigned char* colorPixels, unsigned char* pixels){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    int numValues = colorDepth;
    uchar3 value;
    switch(numValues){
      case 1:
        value = bwToRGB(colorPixels[globalID]);
        break;
      case 2:
        value = bwaToRGB({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1]});
        break;
      case 4:
        value = rgbaToRGB({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1], colorPixels[globalID*numValues + 2], colorPixels[globalID*numValues + 3]});
        break;
      default:
        printf("ERROR colorDepth of %u is not supported",numValues);
        asm("trap;");
    }
    pixels[globalID*3] = value.x;
    pixels[globalID*3 + 1] = value.y;
    pixels[globalID*3 + 2] = value.z;
  }
}

__global__ void ssrlcv::binImage(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* binnedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x < imageSize.x/2 && y < imageSize.y/2){
    for(int d = 0; d < colorDepth; ++d){
      float sumPix = pixels[y*colorDepth*2*imageSize.x + (x*2*colorDepth) + d] +
      pixels[(y*2+1)*colorDepth*imageSize.x + (x*2*colorDepth) + d] +
      pixels[y*2*colorDepth*imageSize.x + ((x*2+1)*colorDepth) + d] +
      pixels[(y*2+1)*colorDepth*imageSize.x + ((x*2+1)*colorDepth) + d];
      binnedImage[y*colorDepth*(imageSize.x/2) + (x*colorDepth) + d] = (unsigned char) roundf(sumPix/4.0f);
    }
  }
}
__global__ void ssrlcv::upsampleImage(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* upsampledImage){
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i < imageSize.x*2 && j < imageSize.y*2){
    float x = i*0.5f;
    float y = j*0.5f;
    int xm = getSymmetrizedCoord((int)x,imageSize.x);
    int xp = getSymmetrizedCoord((int)x + 1,imageSize.x);
    int ym = getSymmetrizedCoord((int)y,imageSize.y);
    int yp = getSymmetrizedCoord((int)y + 1,imageSize.y);

    float2 interLocDiff = {x-floor(x),y-floor(y)};

    for(int d = 0; d < colorDepth; ++d){
      float sumPix = interLocDiff.x*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1.0f-interLocDiff.x)*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xm*colorDepth + d]);
      sumPix += interLocDiff.x*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1-interLocDiff.x)*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xm*colorDepth + d]);
      upsampledImage[j*colorDepth*(imageSize.x*2) + i*colorDepth + d] = (unsigned char) sumPix;
    }
  }
}
__global__ void ssrlcv::bilinearInterpolation(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* outputPixels, float outputPixelWidth){
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i < imageSize.x/outputPixelWidth && j < imageSize.y/outputPixelWidth){
    float x = i*outputPixelWidth;
    float y = j*outputPixelWidth;
    int xm = getSymmetrizedCoord((int)x,imageSize.x);
    int xp = getSymmetrizedCoord((int)x + 1,imageSize.x);
    int ym = getSymmetrizedCoord((int)y,imageSize.y);
    int yp = getSymmetrizedCoord((int)y + 1,imageSize.y);

    float2 interLocDiff = {x-floor(x),y-floor(y)};

    for(int d = 0; d < colorDepth; ++d){
      float sumPix = interLocDiff.x*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1.0f-interLocDiff.x)*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xm*colorDepth + d]);
      sumPix += interLocDiff.x*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1-interLocDiff.x)*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xm*colorDepth + d]);
      outputPixels[j*colorDepth*llroundf(imageSize.x/outputPixelWidth) + i*colorDepth + d] = (unsigned char) sumPix;
    }
  }
}

__global__ void ssrlcv::binImage(uint2 imageSize, unsigned int colorDepth, float* pixels, float* binnedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x < imageSize.x/2 && y < imageSize.y/2){
    for(int d = 0; d < colorDepth; ++d){
      float sumPix = pixels[y*colorDepth*2*imageSize.x + x*2*colorDepth + d] +
      pixels[(y*2+1)*colorDepth*imageSize.x + x*2*colorDepth + d] +
      pixels[y*2*colorDepth*imageSize.x + (x*2+1)*colorDepth + d] +
      pixels[(y*2+1)*colorDepth*imageSize.x + (x*2+1)*colorDepth + d];
      binnedImage[y*colorDepth*(imageSize.x/2) + x*colorDepth + d] = sumPix/4.0f;
    }
  }
}
__global__ void ssrlcv::upsampleImage(uint2 imageSize, unsigned int colorDepth, float* pixels, float* upsampledImage){
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i < imageSize.x*2 && j < imageSize.y*2){
    float x = i*0.5f;
    float y = j*0.5f;
    int xm = getSymmetrizedCoord((int)x,imageSize.x);
    int xp = getSymmetrizedCoord((int)x + 1,imageSize.x);
    int ym = getSymmetrizedCoord((int)y,imageSize.y);
    int yp = getSymmetrizedCoord((int)y + 1,imageSize.y);

    float2 interLocDiff = {x-floor(x),y-floor(y)};

    for(int d = 0; d < colorDepth; ++d){
      float sumPix = interLocDiff.x*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1.0f-interLocDiff.x)*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xm*colorDepth + d]);
      sumPix += interLocDiff.x*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1-interLocDiff.x)*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xm*colorDepth + d]);
      upsampledImage[j*colorDepth*(imageSize.x*2) + i*colorDepth + d] = sumPix;
    }
  }
}
__global__ void ssrlcv::bilinearInterpolation(uint2 imageSize, unsigned int colorDepth, float* pixels, float* outputPixels, float outputPixelWidth){
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i < imageSize.x/outputPixelWidth && j < imageSize.y/outputPixelWidth){
    float x = i*outputPixelWidth;
    float y = j*outputPixelWidth;
    int xm = getSymmetrizedCoord((int)x,imageSize.x);
    int xp = getSymmetrizedCoord((int)x + 1,imageSize.x);
    int ym = getSymmetrizedCoord((int)y,imageSize.y);
    int yp = getSymmetrizedCoord((int)y + 1,imageSize.y);

    float2 interLocDiff = {x-floor(x),y-floor(y)};

    for(int d = 0; d < colorDepth; ++d){
      float sumPix = interLocDiff.x*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1.0f-interLocDiff.x)*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xm*colorDepth + d]);
      sumPix += interLocDiff.x*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1-interLocDiff.x)*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xm*colorDepth + d]);
      outputPixels[j*colorDepth*llroundf(imageSize.x/outputPixelWidth) + i*colorDepth + d] = sumPix;
    }
  }
}

__global__ void ssrlcv::convolveImage(uint2 imageSize, unsigned char* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    if(x + (kernelSize.x/2) >= imageSize.x || x < kernelSize.x/2 || y + (kernelSize.y/2) >= imageSize.y || y < kernelSize.y/2){
      convolvedImage[(y*imageSize.x + x)*colorDepth + color] = 0;
    }
    else{
      float sum = 0.0f;
      for(int ky = -kernelSize.y/2; ky <= kernelSize.y/2; ++ky){
        for(int kx = -kernelSize.x/2; kx <= kernelSize.x/2; ++kx){
          sum += ((float)pixels[((y+ky)*imageSize.x + (x+kx))*colorDepth + color])*kernel[(ky+(kernelSize.y/2))*kernelSize.x + (kx+(kernelSize.x/2))];
        }
      }
      convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
    }
  }
}
__global__ void ssrlcv::convolveImage(uint2 imageSize, float* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    if(x + (kernelSize.x/2) >= imageSize.x || x < kernelSize.x/2 || y + (kernelSize.y/2) >= imageSize.y || y < kernelSize.y/2){
      convolvedImage[(y*imageSize.x + x)*colorDepth + color] = 0;
    }
    else{
      float sum = 0.0f;
      for(int ky = -kernelSize.y/2; ky <= kernelSize.y/2; ++ky){
        for(int kx = -kernelSize.x/2; kx <= kernelSize.x/2; ++kx){
          sum += pixels[((y+ky)*imageSize.x + (x+kx))*colorDepth + color]*kernel[(ky+(kernelSize.y/2))*kernelSize.x + (kx+(kernelSize.x/2))];
        }
      }
      convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
    }
  }
}
__global__ void ssrlcv::convolveImage1D(uint2 imageSize, float* pixels, unsigned int colorDepth, int kernelSize, float* kernel, float* convolvedImage, bool vertical){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    float sum = 0.0f;
    if (!vertical) {
      for(int kx = -kernelSize/2; kx <= kernelSize/2; ++kx){
        sum += pixels[((y)*imageSize.x + (x+kx))*colorDepth + color]*kernel[kx+(kernelSize/2)];
      }
    } else {
      for(int ky = -kernelSize/2; ky <= kernelSize/2; ++ky){
        sum += pixels[((y+ky)*imageSize.x + (x))*colorDepth + color]*kernel[ky+(kernelSize/2)];
      }
    }
    convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
  }
}
__global__ void ssrlcv::convolveImage_symmetric(uint2 imageSize, unsigned char* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    int2 symmetricCoord = {0,0};
    float sum = 0.0f;
    for(int ky = -kernelSize.y/2; ky <= kernelSize.y/2; ++ky){
      for(int kx = -kernelSize.x/2; kx <= kernelSize.x/2; ++kx){
        symmetricCoord = {getSymmetrizedCoord(x+kx,(int)imageSize.x),getSymmetrizedCoord(y+ky,(int)imageSize.y)};
        sum += ((float)pixels[((symmetricCoord.y)*imageSize.x + (symmetricCoord.x))*colorDepth + color])*kernel[(ky+(kernelSize.y/2))*kernelSize.x + (kx+(kernelSize.x/2))];
      }
    }
    convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
  }
}
__global__ void ssrlcv::convolveImage_symmetric(uint2 imageSize, float* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    int2 symmetricCoord = {0,0};
    float sum = 0.0f;
    for(int ky = -kernelSize.y/2; ky <= kernelSize.y/2; ++ky){
      for(int kx = -kernelSize.x/2; kx <= kernelSize.x/2; ++kx){
        symmetricCoord = {getSymmetrizedCoord(x+kx,(int)imageSize.x),getSymmetrizedCoord(y+ky,(int)imageSize.y)};
        sum += pixels[((symmetricCoord.y)*imageSize.x + (symmetricCoord.x))*colorDepth + color]*kernel[(ky+(kernelSize.y/2))*kernelSize.x + (kx+(kernelSize.x/2))];
      }
    }
    convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
  }
}
__global__ void ssrlcv::convolveImage1D_symmetric(uint2 imageSize, float* pixels, unsigned int colorDepth, int kernelSize, float* kernel, float* convolvedImage, bool vertical){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    int2 symmetricCoord = {0,0};
    float sum = 0.0f;
    if (!vertical) {
      for(int kx = -kernelSize/2; kx <= kernelSize/2; ++kx){
        symmetricCoord = {getSymmetrizedCoord(x+kx,(int)imageSize.x),(int)y};
        sum += pixels[((symmetricCoord.y)*imageSize.x + (symmetricCoord.x))*colorDepth + color]*kernel[kx+(kernelSize/2)];
      }
    } else {
      for(int ky = -kernelSize/2; ky <= kernelSize/2; ++ky){
        symmetricCoord = {(int)x,getSymmetrizedCoord(y+ky,(int)imageSize.y)};
        sum += pixels[((symmetricCoord.y)*imageSize.x + (symmetricCoord.x))*colorDepth + color]*kernel[ky+(kernelSize/2)];
      }
    }
    convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
  }
}

__global__ void ssrlcv::convertToCharImage(unsigned int numPixels, unsigned char* pixels, float* fltPixels){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    pixels[globalID] = (unsigned char) 255.0f*fltPixels[globalID];
  }
}
__global__ void ssrlcv::convertToFltImage(unsigned int numPixels, unsigned char* pixels, float* fltPixels){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    fltPixels[globalID] = (float) pixels[globalID];
  }
}
__global__ void ssrlcv::normalize(unsigned long numPixels, float* pixels, float2 minMax){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    pixels[globalID] = (pixels[globalID] - minMax.x)/(minMax.y - minMax.x);
  }
}

__global__ void ssrlcv::calculatePixelGradients(uint2 imageSize, unsigned char* pixels, int2* gradients){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < imageSize.x*imageSize.y){
    int2 loc = {(int)(globalID%imageSize.x),(int)(globalID/imageSize.x)};
    int2 xContrib = {loc.x + 1,loc.x - 1};
    int2 yContrib = {loc.y + 1,loc.y - 1};
    if(xContrib.y == -1) xContrib = xContrib + 1;
    else if(xContrib.x == imageSize.x) xContrib = xContrib - 1;
    if(yContrib.y == -1) yContrib = yContrib + 1;
    else if(yContrib.x == imageSize.y) yContrib = yContrib - 1;
    gradients[globalID] = {
      (int)pixels[loc.y*imageSize.x + xContrib.x] - (int)pixels[loc.y*imageSize.x + xContrib.y],
      (int)pixels[yContrib.x*imageSize.x + loc.x] - (int)pixels[yContrib.y*imageSize.x + loc.x]
    };
  }
}
__global__ void ssrlcv::calculatePixelGradients(uint2 imageSize, float* pixels, float2* gradients){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < imageSize.x*imageSize.y){
    int2 loc = {(int)(globalID%imageSize.x),(int)(globalID/imageSize.x)};
    int2 xContrib = {loc.x + 1,loc.x - 1};
    int2 yContrib = {loc.y + 1,loc.y - 1};
    if(xContrib.y == -1) xContrib = xContrib + 1;
    else if(xContrib.x == imageSize.x) xContrib = xContrib - 1;
    if(yContrib.y == -1) yContrib = yContrib + 1;
    else if(yContrib.x == imageSize.y) yContrib = yContrib - 1;
    gradients[globalID] = {
      pixels[loc.y*imageSize.x + xContrib.x] - pixels[loc.y*imageSize.x + xContrib.y],
      pixels[yContrib.x*imageSize.x + loc.x] - pixels[yContrib.y*imageSize.x + loc.x]
    };
  }
}
