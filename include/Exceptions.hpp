/** 
 * \file CVExceptions.hpp
 * \brief File for custom exceptions within the library
 * \details This file includes all exceptions that could be thrown
 * in the ssrlcv library that do not involve CUDA or the Unity class.
*/
#pragma once
#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

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
  struct SSRLCVException : std::exception{
    std::string msg;
    SSRLCVException()
    {
      msg = "SSRLCV Exception";
    }
    SSRLCVException(std::string msg) : msg("SSRLCV Exception: " + msg) {}
    virtual const char* what() const throw(){
      return msg.c_str();
    }
  };
  /**
   * \brief Base SSRLCV File IO Exception
   * \details All exceptions related to file IO should be 
   * derived from this struct.  
   */
  struct FileIOExpection : SSRLCVException{
    std::string msg;
    FileIOExpection(){
      msg = "File IO Exception";
    }
    FileIOExpection(std::string msg) : msg("File IO Exception: " + msg) {}
    virtual const char *what() const throw(){
      return msg.c_str();
    }
  };
  /**
   * \brief Exception thrown when there is an attempt to use 
   * an unsupported image type. (tiff is one of these right now)
   */
  struct UnsupportedImageException : FileIOExpection{
    std::string msg;
    UnsupportedImageException(){
      msg = "Unsupported Image Type Exception (supported = png,jpg,tiff)";
    }
    UnsupportedImageException(std::string msg) : msg("Unsupported Image Type Exception (supported = png,jpg): " + msg) {}
    virtual const char *what() const throw(){
      return msg.c_str();
    }
  };

  /**
   * \}
   */
}






#endif /* EXCEPTIONS_HPP */
