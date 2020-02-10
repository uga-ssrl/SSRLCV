/** 
 * \file CVExceptions.hpp
 * \brief File for custom exceptions within the library
 * \details This file includes all exceptions that could be thrown
 * in the ssrlcv library that do not involve CUDA or the Unity class.
*/
#ifndef CVEXCEPTIONS_HPP
#define CVEXCEPTIONS_HPP

#include <stdio.h>
#include <string>
#include <cstring>
#include <iostream> 

namespace ssrlcv{
  /**
   * \addtogroup error_util
   * \{
   */
  /**
   * \brief Base SSRLCV exception.
   * \details All other SSRLCV exceptions should be 
   * children of this struct.
  */
  struct CVException : std::exception{
    std::string msg;
    CVException(){
      msg = "Unknown Exception";
    }
    CVException(std::string msg) : msg("SSRLCV Exception: " + msg){}
    virtual const char* what() const throw(){
      return msg.c_str();
    }
  };




  /**
   * \}
   */
}






#endif /* CVEXCEPTIONS_HPP */
