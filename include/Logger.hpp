/** \file logger.hpp
* \brief this contains helpful ways to log data
*/
#ifndef LOGGER_HPP
#define LOGGER_HPP

#include "common_includes.h"
#include "Image.cuh"
#include "MatchFactory.cuh"
#include "Unity.cuh"
#include "Octree.cuh"
#include "io_util.h"


namespace ssrlcv{


  class Logger{

  public:

    // public variables here

    // =============================================================================================================
    //
    // Constructors and Destructors
    //
    // =============================================================================================================

    /**
     * Default constructor
     */
    Logger();

    /**
     * Constructor with path to desired log directory
     */
    Logger(const char* logPath);

    /**
     * Default destructor
     */
    ~Logger();

    // =============================================================================================================
    //
    // Logging Methods
    //
    // =============================================================================================================

    /**
     * write the input string to the log
     * @param input a string to write to the log
     */
    void log(const char* input);

    /**
     * write the input string to the log
     * @param input a string to write to the log
     */
    void log(std::string input);

  private:

    // TODO make this a global path
    // The path to the log directory
    std::string logPath;

    // the default log filename
    const char* logName = "ssrlcv.log";

    // the path and filename
    std::string logFileLocation;

    // the out stream
    std::fstream* stream;

  }; // end Logger class
}


#endif /* LOGGER_HPP */
