/**
* \file example_checkpoint_signals.cu
* \author  Jackson Parker
* \date    March 4 2020
* \brief Example on the usage of the Unity structure's checkpointing and ways to use c signals
* \details This can also act as an executable for unit testing.
* \example 
* \see example_unity.cu
* \see Unity
* \todo add signal examples for SIGABRT, SIGFPE, SIGILL, SIGSEGV, SIGTERM
*/
#include "../../include/common_includes.h"
#include "../../include/io_util.h"
#include <csignal>
#include <typeinfo>
#include <unistd.h>

/*
UTILITY
*/

template<typename T>
bool printTest(ssrlcv::Unity<T>* original, ssrlcv::Unity<T>* checkpoint){
    bool passed = true;
    ssrlcv::MemoryState origin[2] = {original->getMemoryState(), checkpoint->getMemoryState()};
    if(origin[0] != origin[1]){
        std::cout<<"state is different"<<std::endl;
        passed = false;
    }
    unsigned long size[2] = {original->size(), checkpoint->size()};
    if(size[0] != size[1]){
        std::cout<<"size is different"<<std::endl;
        return false;//should not check data then or risk segfault
    }
    if(origin[0] == ssrlcv::gpu){
        original->transferMemoryTo(ssrlcv::cpu);
    }
    if(origin[1] == ssrlcv::gpu){
        checkpoint->transferMemoryTo(ssrlcv::cpu);
    }
    int numdiff = 0;
    for(int i = 0; i < size[0]; ++i){
        if(original->host[i] != checkpoint->host[i]) numdiff++;
    }
    if(!numdiff){
        std::cout<<"data is equivalent"<<std::endl;
    }
    else{
        std::cout<<numdiff<<" elements differ"<<std::endl;
        passed = false;
    }
    return passed;
}

/*
SIGNAL HANDLING 
*/

/*
due to the way that signals work in C++ global variables are required for 
signal handlers to access variables
*/



ssrlcv::Unity<int> i_nums;

void sigintHandler(int signal){
    std::cout<<"Interrupted by signal ("<<signal<<")"<<std::endl;
    std::cout<<"Writing checkpoint for Unity<int>"<<std::endl;
    i_nums.checkpoint(0,"./");//will write i_nums data and state information
    //exit(signal);//usually you would want to exit after an interrupt
    //if you do not want to exit do this
    //reset handler 
    //signal(SIGINT, sigintHandler);//will handle signals 
    //fflush(stdout);
}


int main(int argc, char *argv[]){
  try{
    signal(SIGINT, sigintHandler);//will handle signals 

    i_nums = ssrlcv::Unity<int>(nullptr,100,ssrlcv::cpu);

    for(int i = 0; i < 100; ++i){
      i_nums.host[i] = i;
    } 
    //no need to setFore as state == cpu so fore == cpu
    i_nums.transferMemoryTo(ssrlcv::gpu);

    /*
    now when interrupted i_nums should be checkpointed with 
    data and state information so that Unity<T>(std::string path)
    can reinstantiate from the checkpoint
    */
    while(1){
        std::cout<<"Waiting for interrupt..."<<std::endl;
        sleep(1);
    }

    ssrlcv::Unity<int> i_nums_cpt;

    bool successfull = false;
    std::vector<std::string> checkpointList;
    ssrlcv::getFilePaths("./",checkpointList,".uty");

    for(auto file = checkpointList.begin(); file != checkpointList.end(); ++file){
        try{
            ssrlcv::Unity<int> tmp = ssrlcv::Unity<int>(*file);
            i_nums_cpt = tmp;
            successfull = true;
        }
        catch (const ssrlcv::CheckpointException &e){} 
    }
    if(!successfull){
        std::cout<<"could not find or read a checkpoint"<<std::endl;
        std::cout<<"checkpointing unity test failed"<<std::endl;
        exit(-1);
    }

    if(printTest(&i_nums,&i_nums_cpt)){
        std::cout<<"TEST PASSED"<<std::endl;
    }
    else{
        std::cout<<"TEST FAILED"<<std::endl;
    }

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


