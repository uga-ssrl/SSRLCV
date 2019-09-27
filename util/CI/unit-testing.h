#ifndef __TESTING_H__
#define __TESTING_H__

#include <iostream>


void TEST(bool test, const char * format, ...) {
  va_list args; 
  if(! test) {
    printf(format, args); 
    exit(1);
  }
}

#define TESTING_CATCH \
catch(const int x) { \
  std::cerr << "OS error: " << strerror(x) << std::endl; \
  throw; \
} 

#endif
