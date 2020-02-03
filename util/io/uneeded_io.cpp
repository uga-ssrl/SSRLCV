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