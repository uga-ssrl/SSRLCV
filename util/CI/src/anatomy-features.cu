#include "io_fmt_anatomy.cuh"
#include "unit-testing.h" 

#include <fstream>

using namespace ssrlcv; 

int main(int argc, char ** argv) { 
    const char * testfile = "data/morpheus1_sift.txt";
    std::ifstream file; 

    file.open(testfile, std::ifstream::in); 
    if(file.fail()) throw errno; 
    Unity<Feature<SIFT_Descriptor>> * features = ssrlcv::io::anatomy::readFeatures(file);
    TEST(features->state == cpu, "Features expected to be on CPU"); 

    // Only testing the first line.  This is pretty weak tbh but confirms the concept
    const int SIZE = 128; 

    unsigned char * testcase = features->host[0].descriptor.values; 
    short expected[SIZE] = { 
        121, 10, 0, 0, 0, 0, 0, 3, 136, 4, 0, 0, 0, 0, 0, 14, 106, 5, 0, 0, 0, 0, 0, 19, 66, 13, 0, 0, 0, 0, 0, 3, 152, 6, 0, 0, 0, 0, 0, 22, 148, 0, 0, 0, 0, 0, 0, 59, 152, 7, 0, 0, 0, 0, 0, 28, 135, 4, 0, 0, 0, 0, 0, 25, 152, 0, 0, 0, 0, 0, 0, 15, 152, 0, 0, 0, 0, 0, 0, 51, 152, 7, 0, 0, 0, 0, 0, 18, 152, 0, 0, 0, 0, 0, 0, 28, 54, 1, 0, 0, 0, 0, 0, 3, 72, 0, 0, 0, 0, 0, 0, 11, 88, 0, 0, 0, 0, 0, 0, 7, 77, 2, 1, 0, 0, 0, 0, 5
    };

    for(int i = 0; i < SIZE; i++) { 
        TEST((unsigned char) expected[i] == testcase[i], "Mismatch at index %d", i);
    }

    delete features; 
    return 0; 
}