/**
* \file example_unity.cu
* \author  Jackson Parker
* \date    Feb 8 2020
* \brief Example on the usage of the Unity structure.
* \details This can also act as an executable for unit testing.
* \example 
* \see Unity
*/
#include "../../common_includes.h"

/*
NOTE: This example kernel is VERY simple compared to most 
and never has numElements hit grid.x limits
It is meant as a Unity example
*/
__global__ void add_100(int numElements, int* data){
  if(blockIdx.x < numElements){
    data[blockIdx.x] += 100;
  }
}

void add_100(jax::Unity<int>* i_nums){
  //check where the data is 
  jax::MemoryState origin = i_nums->getMemoryState();
  //make sure i_nums.device has the most up to date memory
  if(origin == jax::cpu || i_nums->getFore() == jax::cpu){
    i_nums->transferMemoryTo(jax::gpu);//this is making i_nums.state = i_nums.fore = jax::both
  }
  add_100<<<i_nums->size(),1>>>(i_nums->size(),i_nums->device);
  cudaDeviceSynchronize();//global threadfence
  CudaCheckError();//cuda error checker
  i_nums->setFore(jax::gpu);//tell Unity where the updated memory is
  if(origin == jax::cpu){
    i_nums->setMemoryState(jax::cpu);//returning i_nums with state = cpu
  }
  else if(origin == jax::both){
    i_nums->transferMemoryTo(jax::cpu);//just make sure i_nums.fore = both
  }
  //else origin was on the gpu so no need to do anything
}
template<typename T>
bool printTest(unsigned long numElements, T* data, T* truth){
  for(int i = 0; i < numElements; ++i){
    if(data[i] != truth[i]){
      std::cout<<"test failed"<<std::endl;
      return false;
    }
  }
  std::cout<<"test passed"<<std::endl;
  return true;
}

int main(int argc, char *argv[]){
  try{
    int fullpass = 0;
    /*
    Instantiate a Unity of a certain length from a nullptr.
    */
    std::cout<<"test proper fore usage, Unity<T>::transferMemoryTo and Unity<T>::setMemoryState\n";
    jax::Unity<int> i_nums = jax::Unity<int>(nullptr,100,jax::cpu);
    std::vector<int> truth;
    /*
    Fill host with information.
    */
    for(int i = 0; i < 100; ++i){
      i_nums.host[i] = i-100;
      truth.push_back((i-100)+100);
    } 
    
    /*
    Transfer to gpu.
    */      
    i_nums.transferMemoryTo(jax::gpu);
    std::cout<<"\tafter i_nums.transferMemoryTo(jax::gpu) ";
    i_nums.printInfo();
    add_100<<<i_nums.size(),1>>>(i_nums.size(),i_nums.device);
    
    /*
    Now make sure Unity knows that I have changed values on 
    the device because memory is also on the CPU.
    */
    i_nums.setFore(jax::gpu);
    std::cout<<"\tafter i_nums.setFore(jax::gpu) ";
    i_nums.printInfo();
    
    /*
    Now I want the memory on the cpu but not on the gpu anymore.
    NOTE: due to not calling cudaDeviceSynchronize() 
    this is the threadfence as it uses cudaMemcpy.
    */
    i_nums.setMemoryState(jax::cpu);
    CudaCheckError();//cuda error checker
    std::cout<<"\tafter i_nums.setMemoryState(jax::cpu) ";
    i_nums.printInfo();
    std::cout<<"\t";
    /*
    Because we setFore Unity knew that I recently updated 
    i_nums.device so it transfered that update to i_nums.host 
    before deleting i_nums.device.
    */
    fullpass += printTest<int>(i_nums.size(),i_nums.host,&truth[0]);

    /*
    Now lets delete and replace i_nums data. Unity<T>::setData
    does not copy the passed data it actually uses it and sets 
    device or host to it.
    */
    std::cout<<"testing Unity<T>::setData"<<std::endl;
    int* replacement = new int[1000]();
    for(int i = 0; i < 1000; ++i){
      replacement[i] = -i-100;
      if(i < 100) truth[i] = -i;
      else truth.push_back(-i);
    }
    i_nums.setData(replacement,1000,jax::cpu);
    std::cout<<"\tafter i_nums.setData(replacement,1000,jax::cpu) ";
    i_nums.printInfo();
    std::cout<<"\t";
    fullpass += printTest<int>(i_nums.size(),i_nums.host,replacement);

    /*
    Now lets use a user function that assumes i_nums should return in 
    the same state it was in when passed to it. 
    */
    std::cout<<"testing user function handling state properly\n\t";
    add_100(&i_nums);
    fullpass += printTest<int>(i_nums.size(),i_nums.host,&truth[0]);

    
    /*
    Now lets resize the Unity so that only the first 
    10 elements are kept. 
    */
    std::cout<<"testing Unity<T>::resize\n\t";
    i_nums.transferMemoryTo(jax::gpu);//Transfer to gpu, setting i_nums.state & fore to both
    std::cout<<"after i_nums.transferMemoryTo(jax::gpu) ";
    i_nums.printInfo();
    std::cout<<"\t";
    i_nums.resize(10);
    std::cout<<"after i_nums.resize(10) ";
    i_nums.printInfo();
    std::cout<<"\ttest "<<((i_nums.size() == 10) ? "passed" : "failed")<<std::endl;
    

    /*
    Now lets test the zeroOut feature, which is essentially clear() 
    without deleting device or host. 
    */
    std::cout<<"testing Unity<T>::zeroOut\n\t";
    i_nums.zeroOut(jax::cpu);
    std::cout<<"after i_nums.zeroOut(jax::cpu) ";
    i_nums.printInfo();
    truth.clear();
    for(int i = 0; i < 10; ++i){
      truth.push_back(0);
    }
    std::cout<<"\t";
    fullpass += printTest<int>(i_nums.size(),i_nums.host,&truth[0]);
    
    /*
    As the zeroOut function is setting i_nums.fore = cpu, we 
    can setFore(gpu) and any transfer other than clear will 
    give host back the original data. This also shows 
    how improperly tracking and setting fore can lead to 
    changes being overwritten. Unity<T>::fore is used to inform 
    Unity about changes to a particular memory address and is 
    vital in utilizing Unity. 
    NOTE: Unity will also not allow you to set 
    fore to both manually, this is done within unity by using transferMemoryState() 
    or setMemoryStateTo(both).
    */
    std::cout<<"testing result of improper fore tracking with Unity<T>::transferMemoryTo\n\t";
    i_nums.setFore(jax::gpu);
    std::cout<<"after i_nums.setFore(jax::gpu) ";
    i_nums.printInfo();
    i_nums.transferMemoryTo(jax::cpu);
    std::cout<<"\tafter i_nums.transferMemoryTo(jax::cpu) ";
    i_nums.printInfo();
    std::cout<<"\t";
    for(int i = 0; i < 10; ++i){
      truth[i] = -i;
    }
    fullpass += printTest<int>(i_nums.size(),i_nums.host,&truth[0]);

    /*
    Another example where fore messes things up
    */
    std::cout<<"testing result of improper fore tracking with Unity<T>::setMemoryState\n\t";

    i_nums.zeroOut(jax::gpu);//now gpu = fore as Unity sets it when zeroOut is called
    std::cout<<"after i_nums.zeroOut(jax::gpu) ";
    i_nums.printInfo();
    for(int i = 0; i < i_nums.size(); ++i){
      i_nums.host[i] = i;
      truth[i] = 0;
    }

    /*
    So now i_nums.host = {0,1,2,3,4,5,6,7,8,9}, butttttt
    NOTE: if you try and transfer memory to the same state as fore 
    nothing will happen and a warning will be logged.
    */
    i_nums.transferMemoryTo(jax::gpu);//example of transfer doing nothing due to fore being set to gpu
    std::cout<<"\tafter i_nums.transferMemoryTo(jax::gpu) ";
    i_nums.printInfo();
    i_nums.setMemoryState(jax::cpu);//as gpu is fore this will transfer gpu to cpu before deleting i_nums.device
    std::cout<<"\tafter i_nums.setMemoryState(jax::cpu) ";
    i_nums.printInfo();
    std::cout<<"\t";
    fullpass += printTest<int>(i_nums.size(),i_nums.host,&truth[0]);


    /*
    If i_nums was a pointer allocated with new, you would have to delete i_nums.
    Here destructor takes care of it.  
    */

    if(fullpass == 6){
      std::cout<<"ALL TESTS PASSED"<<std::endl;
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


