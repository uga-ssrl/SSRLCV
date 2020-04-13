#include "io_util.h"

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
  {"--noparams","noparams"}  // to disable the requirement of a params.csv or params.bcp file
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
    printf("Error : Failed to open input directory %s\n",dirPath.c_str());
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
    printf("Error : Failed to open input directory %s\n", dirPath.c_str());
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
    std::cerr<<"ERROR: "<<this->path<<" does not exist"<<std::endl;
    exit(-1);
  }
}
ssrlcv::img_dir_arg::img_dir_arg(char* path){
  if(directoryExists(path)){
    getImagePaths(path,this->paths);
    if(this->paths.size() == 0){
      std::cerr<<"ERROR: no images found in "<<path<<std::endl;
      exit(-1);
    }
  }
  else{
    std::cerr<<"ERROR: "<<path<<" does not exist"<<std::endl;
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
    std::cout<<"USAGE ./bin/<executable> -d </path/to/image/directory/> -i </path/to/image> -s </path/to/seed/image>"<<std::endl;
    exit(0);
  }
  std::map<std::string, arg*> arg_map;
  for(int a = 1; a < numArgs - 1; ++a){
    std::string temp = cl_args[args[a]];
    std::cout<<"found "<<temp<<" in arguments"<<std::endl;
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
  }
  if(arg_map.find("img") == arg_map.end() && arg_map.find("dir") == arg_map.end()){
    std::cerr<<"ERROR must include atleast one image other than seed for processing"<<std::endl;
    std::cout<<"USAGE ./bin/<executable> -d </path/to/image/directory/> -i </path/to/image> -s </path/to/seed/image>"<<std::endl;
    exit(0);
  }
  return arg_map;
}



unsigned char* ssrlcv::getPixelArray(unsigned char** &row_pointers, const unsigned int &width, const unsigned int &height, const int numValues){
  if(numValues == 0){
    std::cout<<"ERROR: png color type not supported in parallel DSIFT"<<std::endl;
    exit(-1);
  }
  unsigned char* imageMatrix = new unsigned char[height*width*numValues];
  for (unsigned int r=0; r < height; ++r){
    for(unsigned int c=0; c < width; ++c){
      for(int p=0; p < numValues; ++p){
        imageMatrix[(r*width + c)*numValues + p] = row_pointers[r][c*numValues + p];
      }
    }
    delete[] row_pointers[r];
  }
  delete[] row_pointers;
  std::cout<<"Pixel data aquired"<<std::endl;
  return imageMatrix;
}

unsigned char* ssrlcv::readPNG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int& colorDepth){
  /* open file and test for it being a png */
  FILE* fp = fopen(filePath, "rb");
  std::cout<<"READING "<<filePath<<std::endl;

  unsigned char header[8];

  if (!fp){
    std::cout<<"[read_png_file] File could not be opened for reading: "<< filePath<<std::endl;
    exit(-1);
  }

  fread(header, 1, 8, fp);

  if (png_sig_cmp(header, 0, 8)){
    std::cout << "[read_png_file] File is not recognized as a PNG file: " << filePath << std::endl;
    exit(-1);
  }

  /* initialize stuff */
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr){
    std::cout<<"[read_png_file] png_create_read_struct failed"<<std::endl;
    exit(-1);
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);

  if (!info_ptr){
    std::cout<<"[read_png_file] png_create_info_struct failed"<<std::endl;
    exit(-1);
  }

  if (setjmp(png_jmpbuf(png_ptr))){
    std::cout<<"[read_png_file] Error during init_io"<<std::endl;
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
  return getPixelArray(row_pointers, width, height, numChannels);
}

void ssrlcv::writePNG(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height){
  /* create file */
  FILE *fp = fopen(filePath, "wb");
  if(!fp){
    std::cout<<"[write_png_file] File %s could not be opened for writing "<<filePath<<std::endl;
  }

  /* initialize stuff */
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if(!png_ptr){
    std::cout<<"[write_png_file] png_create_write_struct failed "<<std::endl;
  }


  png_infop info_ptr = png_create_info_struct(png_ptr);

  if (!info_ptr){
    std::cout<<"[write_png_file] png_create_info_struct failed "<<std::endl;
  }

  if (setjmp(png_jmpbuf(png_ptr))){
    std::cout<<"[write_png_file] Error during init_io "<<std::endl;
  }

  png_init_io(png_ptr, fp);

  /* write header */
  if (setjmp(png_jmpbuf(png_ptr))){
    std::cout<<"[write_png_file] Error during writing header "<<std::endl;
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
    std::cout<<"[write_png_file] Error during writing bytes "<<std::endl;
  }

  unsigned char** row_pointers = new unsigned char*[height];
  for(unsigned int i = 0; i < height; ++i){
    row_pointers[i] = new unsigned char[width*colorDepth];
    std::memcpy(row_pointers[i], image + i*width*colorDepth, width*colorDepth*sizeof(unsigned char));
  }

  png_write_image(png_ptr, row_pointers);

  /* end write */
  if (setjmp(png_jmpbuf(png_ptr))){
    std::cout<<"[write_png_file] Error during end of write "<<std::endl;
  }

  png_write_end(png_ptr, nullptr);
  fclose(fp);
  std::cout<<filePath<<" has been written"<<std::endl;
}

unsigned char* ssrlcv::readJPEG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth){
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  std::cout<<"attempting to read "<<filePath<<std::endl;
  FILE *infile = fopen(filePath, "rb");
  if (!infile){
    fprintf(stderr, "can't open %s\n", filePath);
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
  unsigned char *pixels = new unsigned char[height * scanlineSize];
  unsigned char **buffer = new unsigned char*[1];
  buffer[0] = new unsigned char[scanlineSize];

  for(unsigned int row = 0; row < height; ++row){
    (void)jpeg_read_scanlines(&cinfo, buffer, 1);
    std::memcpy(pixels + (row * scanlineSize), buffer[0], scanlineSize * sizeof(unsigned char));
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
    fprintf(stderr, "can't open %s\n", filePath);
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
  std::cout<<filePath<<" has been written"<<std::endl;
}

unsigned char* ssrlcv::readImage(const char *filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth){
  std::string str_filePath = filePath;
  std::string fileType = getFileExtension(str_filePath);
  unsigned char* pixels = nullptr;

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
    std::cerr << "Couldn't open " << path << std::endl;
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
  if(ferror(file)) { std::cerr << "Error reading " << path << std::endl; fclose(file); return false; }

#define EOF_CHECK if(feof(file)) { std::cerr << "Error in " << path << ": Unexpected EOF" << std::endl; fclose(file); return false; }
#define EOF_EXPECT \
  bool __fread; \
  fread((void *) &__fread, sizeof(__fread), 1, file); \
  if(! feof(file)) { std::cerr << "Error in " << path << ": Expected EOF.  Format invalid, ignoring this file" << std::endl; fclose(file); return false; }
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

void ssrlcv::writePLY(const char* filePath, Unity<float3>* points, bool binary){
  std::cout << "saving " << points->size() << " points ..." << std::endl;
  MemoryState origin = points->getMemoryState();
  if(origin == gpu) points->transferMemoryTo(cpu);
  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, points->size(), reinterpret_cast<uint8_t*>(points->host), tinyply::Type::INVALID, 0);

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
  std::cout<<filePath<<" has successfully been written"<<std::endl;

  if(origin == gpu) points->setMemoryState(gpu);
}

void ssrlcv::writePLY(std::string filename, Unity<float3>* points, bool binary){
  std::cout << "saving " << points->size() << " points ..." << std::endl;
  MemoryState origin = points->getMemoryState();
  if(origin == gpu) points->transferMemoryTo(cpu);
  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, points->size(), reinterpret_cast<uint8_t*>(points->host), tinyply::Type::INVALID, 0);

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
  std::cout << filename.c_str() << " has successfully been written" << std::endl;

  if(origin == gpu) points->setMemoryState(gpu);
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

void ssrlcv::writePLY(const char* filePath, Unity<colorPoint>* cpoint){
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
    of << std::fixed << std::setprecision(32) << cpoint->host[i].x << " " << cpoint->host[i].y << " " << cpoint->host[i].z << " " << (unsigned int) cpoint->host[i].r << " " << (unsigned int) cpoint->host[i].g << " " << (unsigned int) cpoint->host[i].b << "\n";
  }
  of.close(); // done with the file building
}

// mesh
/**
 * @brief writes a Mesh PLY file that also contains a surface
 * writes a PLY file that includes a surface along with the points
 * @param filename the desired name of the output file
 * @param points a set of points in the mesh
 * @param faceList a list of "faces" which are just encoded triangles
 */
void ssrlcv::writePLY(const char* filename, Unity<float3>* points, Unity<int3>* faceList){

  std::cerr << "PLY writing with triangular faces not yet supported" << std::endl;

}

/**
 * @brief writes a Mesh PLY file that also contains a surface
 * writes a PLY file that includes a surface along with the points
 * @param filename the desired name of the output file
 * @param points a set of points in the mesh
 * @param faceList a list of "faces" which are just encoded quadrilaterals
 */
void ssrlcv::writePLY(const char* filename, Unity<float3>* points, Unity<int4>* faceList){

  std::cerr << "PLY writing with quadrilateral faces not yet supported" << std::endl;

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
  for (int i = 0; i < v.size(); i++) outfile << std::fixed << std::setprecision(32) << v[i] << ",";
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
    std::cerr << "CSV ERROR: Vectors are not the same size!" << std::endl;
    return;
  }
  std::ofstream outfile;
  outfile.open("out/" + filename + ".csv");
  for (int i = 0; i < x.size(); i++) {
      outfile << std::fixed << std::setprecision(32) << x[i] << "," << y[i] << std::endl;
  }
  outfile.close();
}























// yee
