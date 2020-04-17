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

    // public variables

    // the mutex lock for safe logging, publicly availible
    mutable std::mutex mtx;

    // =============================================================================================================
    //
    // Constructors, Destructors, and Operators
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
     * Overloaded Copy Constructor, needed to keep unified mutex locking
     */
    Logger(Logger const &loggerCopy);

    /**
     * Overloaded Assignment Operator, needed to keep unitied mutex locking
     */
    Logger &operator=(Logger const &loggerCopy);

    /**
     * Default destructor
     */
    ~Logger();

    // =============================================================================================================
    //
    // Direct Logging Methods
    //
    // =============================================================================================================

    /**
     * write the input string to the log, tags it as a comment
     * @param input a string to write to the log
     */
    void log(const char* input);

    /**
     * write the input string to the log, tags it as a comment
     * @param input a string to write to the log
     */
    void log(std::string input);

    /**
     * logs a state with a state tag, it is expected that the programmer has a set of pre-defined states
     * @param state a string to be tagged as a state
     */
    void logState(const char* state);

    /**
     * logs a state with a state tag, it is expected that the programmer has a set of pre-defined states
     * @param state a string to be tagged as a state
     */
    void logState(std::string state);

    /**
     * logs the CPU names
     * for details see https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_tx2_32.html%23wwpID0E0ZS0HA
     */
    void logCPUnames();

    /**
     * logs the system voltage
     * for details see https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_tx2_32.html%23wwpID0E05H0HA
     */
    void logVoltage();

    /**
     * logs the system current
     * for details see https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_tx2_32.html%23wwpID0E05H0HA
     */
    void logCurrent();

    /**
     * logs the system power
     * for details see https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_tx2_32.html%23wwpID0E05H0HA
     */
    void logPower();

  private:

    // private variables

    // TODO make this a global path
    // The path to the log directory
    std::string logPath;

    // the default log filename
    const char* logName = "ssrlcv.log";

    // the path and filename
    std::string logFileLocation;

    // this is enabled when a new logging thread starts
    bool isLogging;

    // this is enabled when the user wants to kill a background log
    bool killLogging;

    // logging thread
    std::thread background_thread;

  }; // end Logger class
}


#endif /* LOGGER_HPP */
