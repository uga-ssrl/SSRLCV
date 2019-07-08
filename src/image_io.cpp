#include "image_io.h"


void getImagePaths(std::string dirPath, std::vector<std::string> &imagePaths){
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
std::vector<std::string> findFiles(std::string path){
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

unsigned char* getPixelArray(unsigned char** &row_pointers, const int &width, const int &height, const int numValues){
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

unsigned char* readPNG(const char* filePath, int &height, int &width, unsigned int& colorDepth){
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

void writePNG(const char* filePath, const unsigned char* &image, const int &width, const int &height){

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

  png_set_IHDR(png_ptr, info_ptr, width, height,
               8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  /* write bytes */
  if (setjmp(png_jmpbuf(png_ptr))){
    std::cout<<"[write_png_file] Error during writing bytes "<<std::endl;
  }

  unsigned char** row_pointers = new unsigned char*[height];
  for(int i = 0; i < height; ++i){
    row_pointers[i] = new unsigned char[width];
    std::memcpy(row_pointers[i], image + i*width, width*sizeof(unsigned char));
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
