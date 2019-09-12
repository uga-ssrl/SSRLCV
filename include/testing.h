#ifndef __TESTING_H__
#define __TESTING_H__

#include <iostream>
using namespace std;


void TEST(bool test, const char * message) {
  if(! test) {
    cerr << message << endl;
    exit(1);
  }
}

#endif
