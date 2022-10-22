#include "io_util.hpp"

// \brief these are the args for the main program
// "flag" "identifier"
std::map<std::string, std::string> ssrlcv::cl_args = {
  {"-i","img"}, // for single images
  {"--image","img"}, // for single images
  {"-d","dir"}, // for directories
  {"--directory","dir"}, // for directories
  {"-s","seed"}, // for seed images
  {"--seed","seed"}, // for seed images
  {"-np","noparams"}, // to disable the requirement of a params.csv or params.bcp file
  {"--noparams","noparams"},  // to disable the requirement of a params.csv or params.bcp file
  {"--epsilon","epsilon"}, // epipolar geometry
  {"--delta","delta"} // epipolar geometry
};

void ssrlcv::toLower(std::string &str){
  std::locale loc;
  for (std::string::size_type i = 0; i < str.length(); ++i){
    str[i] = std::tolower(str[i], loc);
  }
}

bool ssrlcv::fileExists(std::string fileName){
    std::ifstream infile(fileName);
    return infile.good();
}
bool ssrlcv::directoryExists(std::string dirPath){
    if(dirPath.c_str() == NULL) return false;
    DIR *dir;
    bool bExists = false;
    dir = opendir(dirPath.c_str());
    if(dir != NULL){
      bExists = true;
      (void) closedir(dir);
    }
    return bExists;
}

/*
 * retunrs the file extension of a give file
 * @param path a string of a filepath
 * @return string a string of the end of the file
 */
std::string ssrlcv::getFileExtension(std::string path){
  std::string type = path.substr(path.find_last_of(".") + 1);
  toLower(type);
  return type;
}

/*
 * Returns the folder of a file give a fully qualified filepath
 * @param path a string representing a fully qualified filepath
 * @return string
 */
std::string ssrlcv::getFolderFromFilePath(std::string path){
  return path.substr(0, path.find_last_of("\\/"));
}

/*
 * Returns the filename from a fully qualified filepath
 * @param path a string representing a fully qualified filepath
 * @return string which is the filename only
 */
std::string ssrlcv::getFileFromFilePath(std::string path){
  return path.substr(path.find_last_of("\\/") + 1);
}

void ssrlcv::getImagePaths(std::string dirPath, std::vector<std::string> &imagePaths){
  DIR* dir;
  if(dirPath.back() != '/') dirPath += "/";
  if (nullptr == (dir = opendir(dirPath.c_str()))){
    logger.err.printf("Error : Failed to open input directory %s",dirPath.c_str());
    exit(-1);
  }
  struct dirent* in_file;
  std::string extension;
  while((in_file = readdir(dir)) != nullptr){
    std::string currentFileName = in_file->d_name;
    extension = getFileExtension(currentFileName);
    if(extension == "png" ||
    extension == "jpg" || extension == "jpeg" ||
    extension == "tif" || extension == "tiff"){
      currentFileName = dirPath + currentFileName;
      imagePaths.push_back(currentFileName);
    }
  }
  std::sort(imagePaths.begin(),imagePaths.end());
  closedir(dir);
}
void ssrlcv::getFilePaths(std::string dirPath, std::vector<std::string> &paths, std::string extension)
{
  DIR *dir;
  if (dirPath.back() != '/')
    dirPath += "/";
  if (nullptr == (dir = opendir(dirPath.c_str())))
  {
    logger.err.printf("Error : Failed to open input directory %s", dirPath.c_str());
    exit(-1);
  }
  struct dirent *in_file;
  while ((in_file = readdir(dir)) != nullptr)
  {
    std::string currentFileName = in_file->d_name;
    if (extension != "all" && extension != getFileExtension(currentFileName))
    {
      continue;
    }
    currentFileName = dirPath + currentFileName;
    paths.push_back(currentFileName);
  }
  std::sort(paths.begin(), paths.end());
  closedir(dir);
}
//will be removed soon
std::vector<std::string> ssrlcv::findFiles(std::string path){
  std::vector<std::string> imagePaths;
  if(path.find(".png") != std::string::npos){
    imagePaths.push_back(path);
  }
  else{
    if(path.substr(path.length() - 1, 1) != "/") path += "/";
    getImagePaths(path, imagePaths);
  }
  return imagePaths;
}

ssrlcv::img_arg::img_arg(char* path){
  this->path = path;
  if(!fileExists(this->path)){
    logger.err<<"ERROR: "<<this->path<<" does not exist";
    exit(-1);
  }
}
ssrlcv::img_dir_arg::img_dir_arg(char* path){
  if(directoryExists(path)){
    getImagePaths(path,this->paths);
    if(this->paths.size() == 0){
      logger.err<<"ERROR: no images found in "<<path;
      exit(-1);
    }
  }
  else{
    logger.err<<"ERROR: "<<path<<" does not exist";
    exit(-1);
  }
}

ssrlcv::flt_arg::flt_arg(char* val){
  this->val = std::stof(val);
}

ssrlcv::int_arg::int_arg(char* val){
  this->val = std::stoi(val);
}

/*
 * Arguments from the main executable are parsed here. These are set above in the cl_args map
 * @param
 */
std::map<std::string, ssrlcv::arg*> ssrlcv::parseArgs(int numArgs, char* args[]){
  if(numArgs < 3){
    logger.err<<"USAGE ./bin/<executable> -d </path/to/image/directory/> -i </path/to/image> -s </path/to/seed/image>";
    exit(0);
  }
  std::map<std::string, arg*> arg_map;
  for(int a = 1; a < numArgs - 1; ++a){
    std::string temp = cl_args[args[a]];
    logger.info<<"found "+temp+" in arguments";
    if(temp == "image" || temp == "seed"){
      arg_map.insert(arg_pair(temp,new img_arg(args[++a])));
    }
    else if(temp == "dir"){
      if(arg_map.find("dir") != arg_map.end()){
        getImagePaths(args[++a],((img_dir_arg*)arg_map["dir"])->paths);
      }
      else{
        arg_map.insert(arg_pair(temp, new img_dir_arg(args[++a])));
      }
    }
    else if (temp == "epsilon" || temp == "delta"){
      arg_map.insert(arg_pair(temp, new flt_arg(args[++a])));
    }
  }
  if(arg_map.find("img") == arg_map.end() && arg_map.find("dir") == arg_map.end()){
    logger.err<<"ERROR must include atleast one image other than seed for processing";
    logger.err<<"USAGE ./bin/<executable> -d </path/to/image/directory/> -i </path/to/image> -s </path/to/seed/image>";
    exit(0);
  }
  return arg_map;
}



ssrlcv::ptr::host<unsigned char> ssrlcv::getPixelArray(unsigned char** &row_pointers, const unsigned int &width, const unsigned int &height, const int numValues){
  if(numValues == 0){
    logger.err<<"ERROR: png color type not supported in parallel DSIFT";
    exit(-1);
  }
  ssrlcv::ptr::host<unsigned char> imageMatrix(height*width*numValues);
  for (unsigned int r=0; r < height; ++r){
    for(unsigned int c=0; c < width; ++c){
      for(int p=0; p < numValues; ++p){
        imageMatrix.get()[(r*width + c)*numValues + p] = row_pointers[r][c*numValues + p];
      }
    }
    delete[] row_pointers[r];
  }
  delete[] row_pointers;
  logger.info<<"Pixel data aquired";
  return imageMatrix;
}

ssrlcv::ptr::host<unsigned char> ssrlcv::readPNG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int& colorDepth){
  /* open file and test for it being a png */
  FILE* fp = fopen(filePath, "rb");
  logger.info<<"READING "+std::string(filePath);

  unsigned char header[8];

  if (!fp){
    logger.err<<"[read_png_file] File could not be opened for reading: " + std::string(filePath);
    exit(-1);
  }

  fread(header, 1, 8, fp);

  if (png_sig_cmp(header, 0, 8)){
    logger.err << "[read_png_file] File is not recognized as a PNG file: " + std::string(filePath);
    exit(-1);
  }

  /* initialize stuff */
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr){
    logger.err<<"[read_png_file] png_create_read_struct failed";
    exit(-1);
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);

  if (!info_ptr){
    logger.err<<"[read_png_file] png_create_info_struct failed";
    exit(-1);
  }

  if (setjmp(png_jmpbuf(png_ptr))){
    logger.err<<"[read_png_file] Error during init_io";
    exit(-1);
  }

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  //png_byte color_type = png_get_color_type(png_ptr, info_ptr);//unused
  //png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);//unused
  int numChannels = png_get_channels(png_ptr, info_ptr);
  png_read_update_info(png_ptr, info_ptr);
  unsigned char** row_pointers = new unsigned char*[height];
  for (unsigned int r=0; r < height; ++r){
    row_pointers[r] = new unsigned char[width*numChannels]();
  }
  png_read_image(png_ptr, row_pointers);
  fclose(fp);
  colorDepth = numChannels;
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  return getPixelArray(row_pointers, width, height, numChannels);
}

void ssrlcv::writePNG(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height){
  /* create file */
  FILE *fp = fopen(filePath, "wb");
  if(!fp){
    logger.err<<"[write_png_file] File %s could not be opened for writing "+std::string(filePath);
  }

  /* initialize stuff */
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if(!png_ptr){
    logger.err<<"[write_png_file] png_create_write_struct failed ";
  }


  png_infop info_ptr = png_create_info_struct(png_ptr);

  if (!info_ptr){
    logger.err<<"[write_png_file] png_create_info_struct failed ";
  }

  if (setjmp(png_jmpbuf(png_ptr))){
    logger.err<<"[write_png_file] Error during init_io ";
  }

  png_init_io(png_ptr, fp);

  /* write header */
  if (setjmp(png_jmpbuf(png_ptr))){
    logger.err<<"[write_png_file] Error during writing header ";
  }

  int colorType = PNG_COLOR_TYPE_GRAY;
  if(colorDepth == 2){
    colorType = PNG_COLOR_TYPE_GRAY_ALPHA;
  }
  else if(colorDepth == 3){
    colorType = PNG_COLOR_TYPE_RGB;
  }
  else if(colorDepth == 4){
    colorType = PNG_COLOR_TYPE_RGB_ALPHA;
  }

  png_set_IHDR(png_ptr, info_ptr, width, height,
               8, colorType, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  /* write bytes */
  if (setjmp(png_jmpbuf(png_ptr))){
    logger.err<<"[write_png_file] Error during writing bytes ";
  }

  unsigned char** row_pointers = new unsigned char*[height];
  for(unsigned int i = 0; i < height; ++i){
    row_pointers[i] = new unsigned char[width*colorDepth];
    std::memcpy(row_pointers[i], image + i*width*colorDepth, width*colorDepth*sizeof(unsigned char));
  }

  png_write_image(png_ptr, row_pointers);

  /* end write */
  if (setjmp(png_jmpbuf(png_ptr))){
    logger.err<<"[write_png_file] Error during end of write ";
  }

  png_write_end(png_ptr, nullptr);
  fclose(fp);
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  logger.info<<std::string(filePath)+" has been written";
}

ssrlcv::ptr::host<unsigned char> ssrlcv::readJPEG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth){
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  logger.info<<"attempting to read "+std::string(filePath);
  FILE *infile = fopen(filePath, "rb");
  if (!infile){
    logger.err.printf("can't open %s", filePath);
    exit(-1);
  }
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  (void) jpeg_read_header(&cinfo, true);
  (void) jpeg_start_decompress(&cinfo);

  width = cinfo.output_width;
  height = cinfo.output_height;
  colorDepth = cinfo.output_components;

  int scanlineSize = width * colorDepth;
  ssrlcv::ptr::host<unsigned char> pixels(height * scanlineSize);
  unsigned char **buffer = new unsigned char*[1];
  buffer[0] = new unsigned char[scanlineSize];

  for(unsigned int row = 0; row < height; ++row){
    (void)jpeg_read_scanlines(&cinfo, buffer, 1);
    std::memcpy(pixels.get() + (row * scanlineSize), buffer[0], scanlineSize * sizeof(unsigned char));
  }

  (void)jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  fclose(infile);

  return pixels;
}
void ssrlcv::writeJPEG(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height){
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE *outfile = fopen(filePath, "wb");
  if (!outfile)
  {
    logger.err.printf("can't open %s", filePath);
    exit(-1);
  }
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = colorDepth;
  if(colorDepth == 1){
    cinfo.in_color_space = JCS_GRAYSCALE;
  }
  else{
    cinfo.in_color_space = JCS_RGB;
  }
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo,75,true);
  (void) jpeg_start_compress(&cinfo,true);

  int scanlineSize = width * colorDepth;
  unsigned char** buffer = new unsigned char*[1];
  buffer[0] = new unsigned char[scanlineSize];

  for (unsigned int row = 0; row < height; ++row){
    buffer[0] = &image[row*width*colorDepth];
    (void)jpeg_write_scanlines(&cinfo, buffer, 1);
  }

  (void)jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  fclose(outfile);
  logger.info<<std::string(filePath)+" has been written";
}

ssrlcv::ptr::host<unsigned char> ssrlcv::readImage(const char *filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth){
  std::string str_filePath = filePath;
  std::string fileType = getFileExtension(str_filePath);
  ssrlcv::ptr::host<unsigned char> pixels(nullptr);

  if (fileType == "png"){
    pixels = readPNG(filePath,height,width,colorDepth);
  }
  else if (fileType == "jpg" || fileType == "jpeg"){
    pixels = readJPEG(filePath,height,width,colorDepth);
  }
  else{
    throw UnsupportedImageException(str_filePath);
  }
  return pixels;
}

void ssrlcv::writeImage(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height){
  std::string str_filePath = filePath;
  std::string fileType = getFileExtension(str_filePath);

  if (fileType == "png"){
    writePNG(filePath,image,colorDepth,width,height);
  }
  else if (fileType == "jpg" || fileType == "jpeg"){
    writeJPEG(filePath,image,colorDepth,width,height);
  }
  else{
    throw UnsupportedImageException(str_filePath);
  }
}

  /**
 * Replaces the file's extension with .bcp (binary camera parameters)
 * @return NEW std::string with the resultant file path
 */
std::string getImageMetaName(std::string imgpath)
{
  for(std::string::iterator it = imgpath.end(); it != imgpath.begin(); it--) {
    if(*it == '.') return std::string(imgpath.replace(it, imgpath.end(), ".bcp"));
  }
  return std::string();
}


/**
 * Reads the binary
 *
 */
bool ssrlcv::readImageMeta(std::string imgpath, bcpFormat & out)
{

  std::string path = getImageMetaName(imgpath);
  if(path.empty()) return false;

  FILE * file = fopen(path.c_str(), "r");
  if(file == nullptr) {
    logger.err << "Couldn't open " << path;
    return false;
  }




  // Reading the file - for now, the official spec of this format is the layout of the struct bcpFormat,
  //  detailed in io_util.h
  //
  // Try to keep this code as smart and modular as possible to make it easy to tinker with later on

  //
  // I have the below macros to help this.
  // They should be able to be moved around, inserted and deleted as needed to match the spec in io_util.h
  // They will clean up and return false on error

  /*
   * BIN_VAL(name)          - Reads the next value into the struct field named `name`
   * EOF_CHECK              - Checks for end-of-file - Call this after every BIN_* read except the last one
   * EOF_EXPECT             - Expects end-of-file.  Call this after the last read
   */

// -- Macros -- //
#define BIN_VAL(name) \
  fread((void *) &(out.name), sizeof(out.name), 1, file); \
  if(ferror(file)) { logger.err << "Error reading " << path; fclose(file); return false; }

#define EOF_CHECK if(feof(file)) { logger.err << "Error in " << path << ": Unexpected EOF"; fclose(file); return false; }
#define EOF_EXPECT \
  bool __fread; \
  fread((void *) &__fread, sizeof(__fread), 1, file); \
  if(! feof(file)) { logger.err << "Error in " << path << ": Expected EOF.  Format invalid, ignoring this file"; fclose(file); return false; }
// -- -- -- //


  BIN_VAL(pos)
  EOF_CHECK

  BIN_VAL(vec)
  EOF_CHECK

  BIN_VAL(fov)
  EOF_CHECK

  BIN_VAL(foc)
  EOF_CHECK

  BIN_VAL(dpix)
  EOF_EXPECT


  fclose(file);
  return true;

#undef BIN_VAL
#undef EOF_CHECK
#undef EOF_EXPECT


  // I guess we could try some sort of binary reading/writing the entire struct,
  //  but I think it's best to shy away from that as we don't know how the compiler will handle struct padding
}


// =============================================================================================================
//
// PLY IO
//
// =============================================================================================================

/**
 * Reads an input ASCII encoded PLY and
 * @param filePath the relative path to the input file
 * @return points the points of the point cloud in a float3 unity
 */
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::readPLY(const char* filePath){
  // disable both of these to remove print statements
  bool local_debug   = false;
  bool local_verbose = true;

  if (local_verbose || local_debug) logger.info << "Reading Mesh ... ";

  // temp storage
  std::vector<float3> tempPoints;
  // std::vector<int> tempFaces;
  std::ifstream input(filePath);
  unsigned int numPoints = 0;
  bool inData   = false;

  // assuming ASCII encoding
  std::string line;
  while (std::getline(input, line)){
    std::istringstream iss(line);

    if (!inData){ // parse the header

      std::string tag;
      iss >> tag;

      //
      // Handle elements here
      //
      if (!tag.compare("element")){
        if(local_debug) logger.info << "element found";
        // temp vars for strings
        std::string elem;
        std::string type;
        int num;

        iss >> type;
        iss >> num;

        // set the correct value
        if (!type.compare("vertex")){
          numPoints = num;
          if(local_debug) logger.info << "detected " + std::to_string(num) + " Points";
        } else if (!type.compare("face")) {
          if(local_debug) logger.info << "detected " + std::to_string(num) + " Faces";
        } else if (!type.compare("edge")) {
          // TODO read in edges if desired
          logger.warn << "\tWARNING: edge reading is not currently supported in MeshFactory";
          if(local_debug) logger.info << "detected " + std::to_string(num) + " Edges";
        }

      }

      // header is ending
      if (!tag.compare("end_header")){
        inData = true;
      }
    } else { // parse the data

      //
      // Handle the Data reading here
      //

      if (tempPoints.size() < numPoints && numPoints) {
        //
        // add the point
        //

        float3 point;
        iss >> point.x;
        iss >> point.y;
        iss >> point.z;
        tempPoints.push_back(point);
        if (local_debug) logger.info << "\t" + std::to_string(point.x) + ", " + std::to_string(point.y) + ", " + std::to_string(point.z);
      }

    } // end data reading

  } // end while

  input.close(); // close the stream

  // save the values to the mesh
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> points = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr,tempPoints.size(),ssrlcv::cpu);
  for (unsigned int i = 0; i < points->size(); i++) {
    points->host.get()[i] = tempPoints[i];
  }

  if (local_verbose || local_debug) {
    logger.info << "Done reading PLY!";
    logger.info << "\t Total Points Loaded:  " + std::to_string(points->size());
  }

  return points;
}

void ssrlcv::writePLY(const char* filePath, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, bool binary){
  logger.info << "saving " << std::to_string(points->size()) << " points ...";
  MemoryState origin = points->getMemoryState();
  if(origin == gpu) points->transferMemoryTo(cpu);
  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, points->size(), reinterpret_cast<uint8_t*>(points->host.get()), tinyply::Type::INVALID, 0);

  std::filebuf fb_binary;
  if(binary){
    fb_binary.open(filePath, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to write ply");
    ply.write(outstream_binary, true);
  }
  else{
    std::filebuf fb_ascii;
  	fb_ascii.open(filePath, std::ios::out);
  	std::ostream outstream_ascii(&fb_ascii);
  	if (outstream_ascii.fail()) throw std::runtime_error("failed to write ply");
    ply.write(outstream_ascii, false);
  }
  logger.info<<std::string(filePath)+" has successfully been written";

  if(origin == gpu) points->setMemoryState(gpu);
}

void ssrlcv::writePLY(std::string filename, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, bool binary){
  logger.info << "saving " + std::to_string(points->size()) + " points ...";
  MemoryState origin = points->getMemoryState();
  if(origin == gpu) points->transferMemoryTo(cpu);
  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, points->size(), reinterpret_cast<uint8_t*>(points->host.get()), tinyply::Type::INVALID, 0);

  std::filebuf fb_binary;
  if(binary){
    fb_binary.open(filename.c_str(), std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to write ply");
    ply.write(outstream_binary, true);
  }
  else{
    std::filebuf fb_ascii;
  	fb_ascii.open(filename.c_str(), std::ios::out);
  	std::ostream outstream_ascii(&fb_ascii);
  	if (outstream_ascii.fail()) throw std::runtime_error("failed to write ply");
    ply.write(outstream_ascii, false);
  }
  logger.info << filename + " has successfully been written";

  if(origin == gpu) points->setMemoryState(gpu);
}

/**
 * @brief a simple ASCII PLY writing method that does not require the tinyPLY external lib
 * @param filename the name of the file to be saved in the /out directory
 * @param points the points to save as a PLY
 */
void ssrlcv::writePLY(const char* filename, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points){
  std::ofstream of;
  std::string fname = filename;
  of.open ("out/" + fname + ".ply");
  of << "ply\nformat ascii 1.0\n";
  of << "comment author: Caleb Adams & Jackson Parker\n";
  of << "comment SSRL CV simple PLY writer\n";
  of << "element vertex " << points->size() << "\n";
  of << "property float x\nproperty float y\nproperty float z\n"; // the elements in the guy
  of << "end_header\n";
  // start writing the values
  for (unsigned int i = 0; i < points->size(); i++){
    of << points->host.get()[i].x << " " << points->host.get()[i].y << " " << points->host.get()[i].z << "\n";
  }
  of.close(); // done with the file building
}

/**
 * @brief a simple ASCII PLY writing method that does not require the tinyPLY external lib
 * @param filename the name of the file to be saved in the /out directory
 * @param points the points to save as a PLY
 */
void ssrlcv::writePLY(std::string filename, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points){
  std::ofstream of;
  of.open ("out/" + filename + ".ply");
  of << "ply\nformat ascii 1.0\n";
  of << "comment author: Caleb Adams & Jackson Parker\n";
  of << "comment SSRL CV simple PLY writer\n";
  of << "element vertex " << points->size() << "\n";
  of << "property float x\nproperty float y\nproperty float z\n"; // the elements in the guy
  of << "end_header\n";
  // start writing the values
  for (unsigned int i = 0; i < points->size(); i++){
    of << points->host.get()[i].x << " " << points->host.get()[i].y << " " << points->host.get()[i].z << "\n";
  }
  of.close(); // done with the file building
}

// colored PLY writing
void ssrlcv::writePLY(std::string filename, colorPoint* cpoint, int size){
  std::ofstream of;
  of.open ("out/" + filename + ".ply");
  of << "ply\nformat ascii 1.0\n";
  of << "comment author: Caleb Adams & Jackson Parker\n";
  of << "comment SSRL CV color PLY writer\n";
  of << "element vertex " << size << "\n";
  of << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"; // the elements in the guy
  of << "end_header\n";
  // start writing the values
  for (int i = 0; i < size; i++){
    of << cpoint[i].x << " " << cpoint[i].y << " " << cpoint[i].z << " " << cpoint[i].r << " " << cpoint[i].g << " " << cpoint[i].b << "\n";
  }
  of.close(); // done with the file building
}

// colored PLY writing
void ssrlcv::writePLY(const char* filePath, colorPoint* cpoint, int size){
  std::string filename = filePath;
  std::ofstream of;
  of.open ("out/" + filename + ".ply");
  of << "ply\nformat ascii 1.0\n";
  of << "comment author: Caleb Adams & Jackson Parker\n";
  of << "comment SSRL CV color PLY writer\n";
  of << "element vertex " << size << "\n";
  of << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"; // the elements in the guy
  of << "end_header\n";
  // start writing the values
  for (int i = 0; i < size; i++){
    of << std::fixed << std::setprecision(32) << cpoint[i].x << " " << cpoint[i].y << " " << cpoint[i].z << " " << (unsigned int) cpoint[i].r << " " << (unsigned int) cpoint[i].g << " " << (unsigned int) cpoint[i].b << "\n";
  }
  of.close(); // done with the file building
}

void ssrlcv::writePLY(const char* filePath, ssrlcv::ptr::value<ssrlcv::Unity<colorPoint>> cpoint){
  std::string filename = filePath;
  std::ofstream of;
  of.open ("out/" + filename + ".ply");
  of << "ply\nformat ascii 1.0\n";
  of << "comment author: Caleb Adams & Jackson Parker\n";
  of << "comment SSRL CV color PLY writer\n";
  of << "element vertex " << cpoint->size() << "\n";
  of << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"; // the elements in the guy
  of << "end_header\n";
  // start writing the values
  for (unsigned int i = 0; i < cpoint->size(); i++){
    of << std::fixed << std::setprecision(32) << cpoint->host.get()[i].x << " " << cpoint->host.get()[i].y << " " << cpoint->host.get()[i].z << " " << (unsigned int) cpoint->host.get()[i].r << " " << (unsigned int) cpoint->host.get()[i].g << " " << (unsigned int) cpoint->host.get()[i].b << "\n";
  }
  of.close(); // done with the file building
}

/**
* writes a mesh with colors
* @param filename the filename
* @param points the points
* @param faceList the faces
* @param faceEncoding the face encoding
* @param colors the colors of the points
*/
void ssrlcv::writePLY(const char* filename, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, ssrlcv::ptr::value<ssrlcv::Unity<int>> faceList, int faceEncoding, ssrlcv::ptr::value<ssrlcv::Unity<uchar3>> colors){
  // TODO

}

/**
 * @brief writes a Mesh PLY file that also contains a surface
 * writes a PLY file that includes a surface along with the points
 * @param filename the desired name of the output file
 * @param points a set of points in the mesh
 * @param faceList a list of "faces" which are indices for point location
 * @param faceEncoding an int where 3 means trianglar and 4 mean quadrilateral
 */
void ssrlcv::writePLY(const char* filename, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, ssrlcv::ptr::value<ssrlcv::Unity<int>> faceList, int faceEncoding){
  std::string fname = filename;
  // we need triangles or quadrilaterals
  if (!(faceEncoding == 3 || faceEncoding == 4)){
    logger.err << "ERROR: error writing mesh based PLY, unsupported face encoding of " << std::to_string(faceEncoding) ;
    return;
  }
  std::ofstream of;
  // build header
  of.open ("out/" + fname + ".ply");
  of << "ply\nformat ascii 1.0\n";
  of << "comment author: Caleb Adams & Jackson Parker\n";
  of << "comment SSRL CV PLY writer\n";
  of << "element vertex " << points->size() << "\n";
  of << "property float x\nproperty float y\nproperty float z\n"; // the elements in the guy
  of << "element face " << (faceList->size() / faceEncoding) << "\n"; // the numer of faces
  of << "property list uchar uint vertex_indices\n";
  of << "end_header\n";
  // loop thru the points
  for (unsigned int i = 0; i < points->size(); i++){
    of << std::fixed << std::setprecision(32) << points->host.get()[i].x << " " << points->host.get()[i].y << " " << points->host.get()[i].z << "\n";
  }
  // loop thru the faces
  for (unsigned int i = 0; i < faceList->size(); i += faceEncoding){
    of << faceEncoding << " ";
    for (int j = 0; j < faceEncoding; j++){
      of << faceList->host.get()[i + j] << " ";
    }
    of << "\n";
  }
  of.close();
}

/**
 * @brief write a PLY that is color coded along the associated gradient points passed in
 * @param filename is the desired name of the output PLY file
 * @param points is the collection of points to color with the gradient
 * @param gradient the values that represent the "variance" of values to be colored with a gradient
 */
void ssrlcv::writePLY(const char* filename, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, ssrlcv::ptr::value<ssrlcv::Unity<float>> gradient){

   // build the helpers to make the colors
  uchar3 colors[2000];
  float3 good = {108,255,221};
  float3 meh  = {251,215,134};
  float3 bad  = {247,121,125};
  float3 gr1  = (meh - good)/1000;
  float3 gr2  = (bad - meh )/1000;
  // initialize the gradient "mapping"
  float3 temp;
  // std::cout << "building gradient" << "\n";
  for (int i = 0; i < 2000; i++){
    if (i < 1000){
      temp = good + gr1*i;
      colors[i].x = (unsigned char) floor(temp.x);
      colors[i].y = (unsigned char) floor(temp.y);
      colors[i].z = (unsigned char) floor(temp.z);
    } else {
      temp = meh  + gr2*i;
      colors[i].x = (unsigned char) floor(temp.x);
      colors[i].y = (unsigned char) floor(temp.y);
      colors[i].z = (unsigned char) floor(temp.z);
    }
  }

  struct colorPoint* cpoints = (colorPoint*)  malloc(points->size() * sizeof(struct colorPoint));

  float max = 0.0; // it would be nice to have a better way to get the max, but because this is only for debug idc
  for (unsigned int i = 0; i < gradient->size(); i++){
    if (gradient->host.get()[i] > max){
      max = gradient->host.get()[i];
    }
  }
  // now fill in the color point locations
  for (unsigned int i = 0; i < points->size(); i++){
    // i assume that the errors and the points will have the same indices
    cpoints[i].x = points->host.get()[i].x; //
    cpoints[i].y = points->host.get()[i].y;
    cpoints[i].z = points->host.get()[i].z;
    int j = floor(gradient->host.get()[i] * (2000 / max));
    cpoints[i].r = colors[j].x;
    cpoints[i].g = colors[j].y;
    cpoints[i].b = colors[j].z;
  }

  // save the file
  writePLY(filename, cpoints, points->size());
}

/**
 * @brief write a PLY that is color coded along the associated gradient points passed in
 * @param filename is the desired name of the output PLY file
 * @param points is the collection of points to color with the gradient
 * @param gradient the values that represent the "variance" of values to be colored with a gradient
 * @param cutoff the max gradient value, where the gradient should end. all points after this will be the same color
 */
void ssrlcv::writePLY(const char* filename, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, ssrlcv::ptr::value<ssrlcv::Unity<float>> gradient, float cutoff){

  // build the helpers to make the colors
 uchar3 colors[2000];
 float3 good = {108,255,221};
 float3 meh  = {251,215,134};
 float3 bad  = {247,121,125};
 float3 gr1  = (meh - good)/1000;
 float3 gr2  = (bad - meh )/1000;
 // initialize the gradient "mapping"
 float3 temp;
 // std::cout << "building gradient" << "\n";
 for (int i = 0; i < 2000; i++){
   if (i < 1000){
     temp = good + gr1*i;
     colors[i].x = (unsigned char) floor(temp.x);
     colors[i].y = (unsigned char) floor(temp.y);
     colors[i].z = (unsigned char) floor(temp.z);
   } else {
     temp = meh  + gr2*i;
     colors[i].x = (unsigned char) floor(temp.x);
     colors[i].y = (unsigned char) floor(temp.y);
     colors[i].z = (unsigned char) floor(temp.z);
   }
 }

 struct colorPoint* cpoints = (colorPoint*)  malloc(points->size() * sizeof(struct colorPoint));

 // now fill in the color point locations
 for (unsigned int i = 0; i < points->size(); i++){
   // i assume that the errors and the points will have the same indices
   cpoints[i].x = points->host.get()[i].x; //
   cpoints[i].y = points->host.get()[i].y;
   cpoints[i].z = points->host.get()[i].z;
   int j = floor(gradient->host.get()[i] * (2000.0f / cutoff));
   if (j > 1999) j = 1999; // sets to max cutoff no matter what
   cpoints[i].r = colors[j].x;
   cpoints[i].g = colors[j].y;
   cpoints[i].b = colors[j].z;
 }

 // save the file
 writePLY(filename, cpoints, points->size());

}

/**
 * @brief write a PLY that is a point cloud including normals
 * @param filename is the desired name of the output PLY file
 * @param points is the collection of points
 * @param normals are the normal vectors (assumed to have been normalized) for each of the point cloud's points
 */
void ssrlcv::writePLY(const char* filename, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, ssrlcv::ptr::value<ssrlcv::Unity<float3>> normals){
  std::ofstream of;
  std::string fname = filename;
  of.open ("out/" + fname + ".ply");
  of << "ply\nformat ascii 1.0\n";
  of << "comment author: Caleb Adams & Jackson Parker\n";
  of << "comment SSRL CV points with normals PLY writer\n";
  of << "element vertex " << points->size() << "\n";
  of << "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n"; // the elements in the guy
  of << "end_header\n";
  // start writing the values
  for (unsigned int i = 0; i < points->size(); i++){
    of << std::fixed << std::setprecision(32) << points->host.get()[i].x << " " << points->host.get()[i].y << " " << points->host.get()[i].z << " " << normals->host.get()[i].x << " " << normals->host.get()[i].y << " " << normals->host.get()[i].z << "\n";
  }
  of.close(); // done with the file building
}

// =============================================================================================================
//
// CSV and Misc IO
//
// =============================================================================================================

/*
 * Takes in an array of floats and writes them to a CSV
 * @param values a set of float elements as a float array that are written in csv format on one line
 * @param num the number of elements in the float array
 * @param filename a string representing the desired filename of the csv output
 */
void ssrlcv::writeCSV(float* values, int num, std::string filename){
  std::ofstream outfile;
  outfile.open("out/" + filename + ".csv");
  // the stupid method of doing this would be to just write it all on the same line ... that's what I'm going to do!
  // other overloaded versions of this method will handle more robust types of inputs and saving and so on.
  for(int i = 0; i < num; i++) outfile << std::fixed << std::setprecision(32) << std::to_string(values[i]) << ",";
  outfile.close();
}

/*
 * Takes in a c++ vector and prints it all on one line of a csv
 * @param v a vector of float guys
 * @param filename a string representing the desired filename of the csv output
 */
void ssrlcv::writeCSV(std::vector<float> v, std::string filename){
  std::ofstream outfile;
  outfile.open("out/" + filename + ".csv");
  for (unsigned int i = 0; i < v.size(); i++) outfile << std::fixed << std::setprecision(32) << v[i] << ",";
  outfile.close();
}

/*
 * Takes in two c++ vectors and writes their values as:
 * `x,y` on a single line for all values in a CSV encoeded format
 * all pairs are on a new line. Assumes the vectors are the same size
 * @param x a vector of x float values
 * @param y a vector of y float values
 * @param filename a string representing the desired filename of the csv output
 */
void ssrlcv::writeCSV(std::vector<float> x, std::vector<float> y, std::string filename){
  if (x.size() != y.size()){
    logger.err << "CSV ERROR: Vectors are not the same size!" ;
    return;
  }
  std::ofstream outfile;
  outfile.open("out/" + filename + ".csv");
  for (unsigned int i = 0; i < x.size(); i++) {
      outfile << std::fixed << std::setprecision(32) << x[i] << "," << y[i] << "\n";
  }
  outfile.close();
}

/*
 * Takes in two c++ vectors and writes their values as:
 * `x,y,z` on a single line for all values in a CSV encoeded format
 * all pairs are on a new line. Assumes the vectors are the same size
 * @param v a vector of float3 that is used to save `x,y,z`
 */
void ssrlcv::writeCSV(std::vector<float3> v, const char* filename){
  std::ofstream outfile;
  std::string fname = filename;
  outfile.open("out/" + fname + ".csv");
  for (unsigned int i = 0; i < v.size(); i++) {
      outfile << std::fixed << std::setprecision(32) << v[i].x << "," << v[i].y << "," << v[i].z << "\n";
  }
  outfile.close();
}

/*
 * saves a CSV file with a unity input
 * @param values a unity float input
 * @param filename the desired filename
 */
void ssrlcv::writeCSV(ssrlcv::ptr::value<ssrlcv::Unity<float>> values, const char* filename){
  std::ofstream outfile;
  std::string fname = filename;
  outfile.open("out/" + fname + ".csv");
  for (unsigned int i = 0; i < values->size(); i++) outfile << std::fixed << std::setprecision(32) << values->host.get()[i] << ",";
  outfile.close();
}

/*
 * Takes in two c++ vectors and writes their values as:
 * `x,y,z` on a single line for all values in a CSV encoeded format
 * all pairs are on a new line. Assumes the vectors are the same size
 * @param v a unity float3 that is used to save `x,y,z`
 */
void ssrlcv::writeCSV(ssrlcv::ptr::value<ssrlcv::Unity<float3>> v, const char* filename){
  std::ofstream outfile;
  std::string fname = filename;
  outfile.open("out/" + fname + ".csv");
  for (unsigned int i = 0; i < v->size(); i++) {
      outfile << std::fixed << std::setprecision(32) << v->host.get()[i].x << "," << v->host.get()[i].y << "," << v->host.get()[i].z << "\n";
  }
  outfile.close();
}


















// yee
