#include "io_util.h"

#include <stdio.h>


void ssrlcv::getImagePaths(std::string dirPath, std::vector<std::string> &imagePaths){
  DIR* dir;
  if (nullptr == (dir = opendir(dirPath.c_str()))){
    printf("Error : Failed to open input directory %s\n",dirPath.c_str());
    exit(-1);
  }
  struct dirent* in_file;
  while((in_file = readdir(dir)) != nullptr){
    std::string currentFileName = in_file->d_name;

    if (currentFileName == "." || currentFileName == ".." ||
      currentFileName.substr(currentFileName.length() - 3) != "png") continue;

    currentFileName = dirPath + currentFileName;
    imagePaths.push_back(currentFileName);
  }
  closedir(dir);
  std::cout<<"found "<<imagePaths.size()<<std::endl;
}
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

unsigned char* ssrlcv::getPixelArray(unsigned char** &row_pointers, const unsigned int &width, const unsigned int &height, const int numValues){
  if(numValues == 0){
    std::cout<<"ERROR: png color type not supported in parallel DSIFT"<<std::endl;
    exit(-1);
  }
  unsigned char* imageMatrix = new unsigned char[height*width*numValues];
  for (int r=0; r < height; ++r){
    for(int c=0; c < width; ++c){
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
    std::cout<<"[read_png_file] File %s could not be opened for reading "<< filePath<<std::endl;
    exit(-1);
  }

  fread(header, 1, 8, fp);

  if (png_sig_cmp(header, 0, 8)){
    std::cout<<"[read_png_file] File %s is not recognized as a PNG file "<< filePath<<std::endl;
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
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);
  //png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);//unused
  int numChannels = png_get_channels(png_ptr, info_ptr);
  png_read_update_info(png_ptr, info_ptr);
  unsigned char** row_pointers = new unsigned char*[height];
  for (int r=0; r < height; ++r){
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
  for(int i = 0; i < height; ++i){
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



//
// Binary files - Gitlab #58
//

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


//
// Old PLY code
//

void ssrlcv::writePLY(const char* filePath, Unity<float3>* points, bool binary){
  MemoryState origin = points->state;
  if(origin == gpu) points->transferMemoryTo(cpu);
  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, points->numElements, reinterpret_cast<uint8_t*>(points->host), tinyply::Type::INVALID, 0);

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
