#include "Feature.cuh"

/*
STRUCTURE METHODS
*/

__device__ __host__ ssrlcv::SIFT_Descriptor::SIFT_Descriptor(){
  this->theta = 0.0f;
  this->sigma = 0.0f;
}
__device__ __host__ ssrlcv::SIFT_Descriptor::SIFT_Descriptor(float theta){
  this->theta = theta;
  this->sigma = 0.0f;
}
__device__ __host__ ssrlcv::SIFT_Descriptor::SIFT_Descriptor(float theta, unsigned char values[128]){
  this->theta = theta;
  this->sigma = 0.0f;
  for(int i = 0; i < 128; ++i){
    this->values[i] = values[i];
  }
}
__device__ __host__ void ssrlcv::SIFT_Descriptor::print(){
  printf("%f,%f\n",this->sigma,this->theta);
  for(int x = 0,d = 0; x < 4; ++x){
    printf("\n");
    for(int y = 0; y < 4; ++y){
      printf("  ");
      for(int a = 0; a < 8; ++a){
          printf("%d",(int) this->values[d++]);
          if(a < 8) printf(",");
      }
    }
  }
  printf("\n\n");
}
__device__ float ssrlcv::SIFT_Descriptor::distProtocol(const SIFT_Descriptor& b, const float &bestMatch){
  float dist = 0.0f;
  for(int i = 0; i < 128 && dist < bestMatch; ++i){
    dist += ((float)this->values[i]-b.values[i])*((float)this->values[i]-b.values[i]);
  }
  return dist;
} 
__device__ __host__ ssrlcv::Window_3x3::Window_3x3(){
  for(int x = 0; x < 3; ++x){
    for(int y = 0; y < 3; ++y){
      this->values[x][y] = 0;
    }
  }
}
__device__ __host__ ssrlcv::Window_3x3::Window_3x3(unsigned char values[3][3]){
 for(int x = 0; x < 3; ++x){
    for(int y = 0; y < 3; ++y){
      this->values[x][y] = values[x][y];
    }
  }
}
__device__ float ssrlcv::Window_3x3::distProtocol(const Window_3x3& b, const float &bestMatch){
  float absDiff = 0;
  for(int x = 0; x < 3 && absDiff < bestMatch; ++x){
    for(int y = 0; y < 3 && absDiff < bestMatch; ++y){
      absDiff += abs((float)this->values[x][y]-(float)b.values[x][y]);
    }
  }
  return absDiff;
}
__device__ __host__ ssrlcv::Window_9x9::Window_9x9(){
  for(int x = 0; x < 9; ++x){
    for(int y = 0; y < 9; ++y){
      this->values[x][y] = 0;
    }
  }
}
__device__ __host__ ssrlcv::Window_9x9::Window_9x9(unsigned char values[9][9]){
 for(int x = 0; x < 9; ++x){
    for(int y = 0; y < 9; ++y){
      this->values[x][y] = values[x][y];
    }
  }
}
__device__ float ssrlcv::Window_9x9::distProtocol(const Window_9x9& b, const float &bestMatch){
  float absDiff = 0;
  for(int x = 0; x < 9 && absDiff < bestMatch; ++x){
    for(int y = 0; y < 9 && absDiff < bestMatch; ++y){
      absDiff += abs((float)this->values[x][y]-(float)b.values[x][y]);
    }
  }
  return absDiff;
}
__device__ __host__ ssrlcv::Window_15x15::Window_15x15(){
  for(int x = 0; x < 15; ++x){
    for(int y = 0; y < 15; ++y){
      this->values[x][y] = 0;
    }
  }
}
__device__ __host__ ssrlcv::Window_15x15::Window_15x15(unsigned char values[15][15]){
 for(int x = 0; x < 15; ++x){
    for(int y = 0; y < 15; ++y){
      this->values[x][y] = values[x][y];
    }
  }
}
__device__ float ssrlcv::Window_15x15::distProtocol(const Window_15x15& b, const float &bestMatch){
  float absDiff = 0;
  for(int x = 0; x < 15 && absDiff < bestMatch; ++x){
    for(int y = 0; y < 15 && absDiff < bestMatch; ++y){
      absDiff += abs((float)this->values[x][y]-(float)b.values[x][y]);
    }
  }
  return absDiff;
}
__device__ __host__ ssrlcv::Window_25x25::Window_25x25(){
  for(int x = 0; x < 25; ++x){
    for(int y = 0; y < 25; ++y){
      this->values[x][y] = 0;
    }
  }
}
__device__ __host__ ssrlcv::Window_25x25::Window_25x25(unsigned char values[25][25]){
 for(int x = 0; x < 3; ++x){
    for(int y = 0; y < 3; ++y){
      this->values[x][y] = values[x][y];
    }
  }
}
__device__ float ssrlcv::Window_25x25::distProtocol(const Window_25x25& b, const float &bestMatch){
  float absDiff = 0;
  for(int x = 0; x < 25 && absDiff < bestMatch; ++x){
    for(int y = 0; y < 25 && absDiff < bestMatch; ++y){
      absDiff += abs((float)this->values[x][y]-(float)b.values[x][y]);
    }
  }
  return absDiff;
}
__device__ __host__ ssrlcv::Window_31x31::Window_31x31(){
  for(int x = 0; x < 31; ++x){
    for(int y = 0; y < 31; ++y){
      this->values[x][y] = 0;
    }
  }
}
__device__ __host__ ssrlcv::Window_31x31::Window_31x31(unsigned char values[31][31]){
 for(int x = 0; x < 31; ++x){
    for(int y = 0; y < 31; ++y){
      this->values[x][y] = values[x][y];
    }
  }
}
__device__ float ssrlcv::Window_31x31::distProtocol(const Window_31x31& b, const float &bestMatch){
  float absDiff = 0;
  for(int x = 0; x < 31 && absDiff < bestMatch; ++x){
    for(int y = 0; y < 31 && absDiff < bestMatch; ++y){
      absDiff += abs((float)this->values[x][y]-(float)b.values[x][y]);
    }
  }
  return absDiff;
}
