/**
* \file example_unity.cu
* \author  Jackson Parker
* \date    Feb 8 2020
* \brief Example on the usage of the Unity structure.
* \example 
* \see Unity
*/
#include "../../include/common_includes.h"
/*
NOTE: This example kernel is VERY simple compared to most and never has numElements hit grid.x limits
It is meant as a Unity example
*/
__global__ void add_100(int numElements, int* data){
  if(blockIdx.x < numElements){
    data[blockIdx.x] += 100;
  }
}

void add_100(ssrlcv::Unity<int>* i_nums){
  //check where the data is 
  ssrlcv::MemoryState origin = i_nums->getMemoryState();
  //make sure i_nums->device has the most up to date memory
  if(origin == ssrlcv::cpu || i_nums->getFore() == ssrlcv::cpu){
    i_nums->transferMemoryTo(ssrlcv::gpu);//this is making i_nums->state = i_nums->fore = ssrlcv::both
  }
  add_100<<<i_nums->size(),1>>>(i_nums->size(),i_nums->device);
  cudaDeviceSynchronize();//global threadfence
  CudaCheckError();//cuda error checker
  i_nums->setFore(ssrlcv::gpu);//tell Unity where the updated memory is
  if(origin == ssrlcv::cpu){
    i_nums->setMemoryState(ssrlcv::cpu);//returning i_nums with state = cpu
  }
  else if(origin == ssrlcv::both){
    i_nums->transferMemoryTo(ssrlcv::cpu);//just make sure i_nums->fore = both
  }
  //else origin was on the gpu so no need to do anything
}

int main(int argc, char *argv[]){
  try{
    //Instantiate a Unity of a certain length from a nullptr.
    ssrlcv::Unity<int>* i_nums = new ssrlcv::Unity<int>(nullptr,100,ssrlcv::cpu);
    
    //Fill it with information.
    for(int i = 0; i < 100; ++i){
      i_nums->host[i] = i-100;
    } 
    
    //Transfer to gpu.
    i_nums->transferMemoryTo(ssrlcv::gpu);
    
    //Now i_nums information is on gpu and cpu.
    add_100<<<i_nums->size(),1>>>(i_nums->size(),i_nums->device);
    
    //Now make sure Unity knows that I have changed values on the device because memory is also on the CPU.
    i_nums->setFore(ssrlcv::gpu);
    
    //now I want the memory on the cpu but not on the gpu anymore
    //NOTE: due to not calling cudaDeviceSynchronize() this is the threadfence as it uses cudaMemcpy
    i_nums->setMemoryState(ssrlcv::cpu);
    CudaCheckError();//cuda error checker

    /*
    Because we setFore Unity knew that I recently updated 
    i_nums->device so it transfered that update to i_nums->host 
    before deleting i_nums->device.
    */
    for(int i = 0; i < 100; ++i){
      std::cout<<i_nums->host[i]<<std::endl;
    }
    //Now lets delete and replace i_nums data.
    int* replacement = new int[1000]();
    for(int i = 0; i < 1000; ++i){
      replacement[i] = -i-100;
    }
    i_nums->setData(replacement,1000,ssrlcv::cpu);

    /*
    Now lets use a user function that assumes i_nums should return in 
    the same state it was in when passed to it. 
    */
    add_100(i_nums);
    for(int i = 0; i < 1000; ++i){
      std::cout<<i_nums->host[i]<<std::endl;
    }

    /*
    Thats the basic concept of Unity! 
    And it can hold ANY type, just 
    replace <int> with <T> T=type of your data.
    */

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


