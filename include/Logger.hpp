/** \file logger.hpp
* \brief this contains helpful ways to log data
* \note The global logger is declared externally at the bottom  of this file
*/
#pragma once
#ifndef LOGGER_HPP
#define LOGGER_HPP

#if LOG_LEVEL >= 4
#define LOG_MEM
#endif


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
// #include <cstring>
// #include <ctype.h>
#include <stdio.h>
#include <unistd.h>
#include <iomanip>
#include <thread>
#include <mutex>


namespace ssrlcv{


  class Logger{
  

  public:

    /**
     * Structure to handle error messages
     */
    struct Error{
      Logger* logger;
      Error();
      Error(Logger* logger);
      /**
       * Overloaded Bit Shift Left Operator, needed to keep united mutex locking
       */
      Error &operator<<(const char* input);
      Error &operator<<(std::string input);

      template <typename... Args>
      void printf(const char *format, Args&&... args){
      #if LOG_LEVEL >= 1
        char buffer[200];
        int num = snprintf(buffer, 200, format, std::forward<Args>(args)...);
        if (num >= 0 && num < 200) {
          this->logger->logError(buffer);
        } else {
          this->logger->logError("Printf statement too long");
        }
      #endif
      }
    };
    /**
     * Structure to handle warning messages
     */
    struct Warning{
      Logger* logger;
      Warning();
      Warning(Logger* logger);
      /**
       * Overloaded Bit Shift Left Operator, needed to keep united mutex locking
       */
      Warning &operator<<(const char *input);
      Warning &operator<<(std::string input);

      template <typename... Args>
      void printf(const char *format, Args&&... args){
      #if LOG_LEVEL >= 2
        char buffer[200];
        int num = snprintf(buffer, 200, format, std::forward<Args>(args)...);
        if (num >= 0 && num < 200) {
          this->logger->logWarning(buffer);
        } else {
          this->logger->logError("Printf statement too long");
        }
      #endif
      }
    };
    /**
     * Structure to handle info messages
     */
    struct Info{
      Logger* logger;
      Info();
      Info(Logger* logger);
      /**
       * Overloaded Bit Shift Left Operator, needed to keep united mutex locking
       */
      Info &operator<<(const char *input);
      Info &operator<<(std::string input);

      template <typename... Args>
      void printf(const char *format, Args&&... args){
      #if LOG_LEVEL >= 3
        char buffer[200];
        int num = snprintf(buffer, 200, format, std::forward<Args>(args)...);
        if (num >= 0 && num < 200) {
          this->logger->logInfo(buffer);
        } else {
          this->logger->logError("Printf statement too long");
        }
      #endif
      }
    };

    struct Memory {
      Logger* logger;
      Memory();
      Memory(Logger* logger);

    #ifdef LOG_MEM
      std::string memoryLogLocation;

      long device_mem = 0;
      long host_pinned_mem = 0;
      long host_unpinned_mem = 0;
    #endif

      void logHostPinned(long change);
      void logHostUnpinned(long change);
      void logDevice(long change);
    };

    enum Verbosity{
      nothing = 0,
      errors = 1,
      warnings = 2,
      comments = 3
    };

    Error err;
    Warning warn;
    Info info;
  #ifdef LOG_MEM
    Memory mem;
  #endif

    int verbosity;//

    // public variables

    // the mutex lock for safe logging, publicly availible
    mutable std::mutex mtx;

    // TODO make this a global path
    // The path to the log directory
    std::string logPath;

    // the path and filename
    std::string logFileLocation;

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
     * Overloaded Bit Shift Left Operator, needed to keep united mutex locking
     */
    Logger &operator<<(const char* input);

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
     * logs a message with a warning tag
     * @param input a string to write to the log
     */
    void logWarning(const char *input);

    /**
     * logs a message with a warning tag
     * @param input a string to write to the log
     */
    void logWarning(std::string input);

    /**
     * logs a message with an info tag
     * @param input a string to write to the log
     */
    void logInfo(const char *input);

    /**
     * logs a message with an info tag
     * @param input a string to write to the log
     */
    void logInfo(std::string input);

    /**
     * logs a message with an error tag
     * @param input a string to write to the log
     */
    void logError(const char* input);

    /**
     * logs a message with an error tag
     * @param input a string to write to the log
     */
    void logError(std::string input);

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

    /**
     * starts logging in the backgound
     * for now this only logs Voltage, Power, and Current
     * @param rate is the number of seconds in between logging
     */
    void startBackgoundLogging(int rate);

    /**
     * stops the backgound logging if it is running
     */
    void stopBackgroundLogging();

    std::string format(const char* format, ...);

  private:

    // private variables

    // the default log filename
    const char* logName = "ssrlcv.log";

    // this is enabled when a new logging thread starts
    bool isLogging;

    // this is enabled when the user wants to kill a background log
    bool killLogging;

    // the seconds in between logging
    int logDelay;

    // logging thread
    std::thread background_thread;

    // =============================================================================================================
    //
    // Private Logging Methods
    //
    // =============================================================================================================

    /**
     * Used my startBackgoundLogging to keep logging in the backgound
     * @param @param delay the time in seconds to wait in between logging
     */
    void looper(int delay);

    /**
     * Used to return a high resolution timestamp for dense resolution timing of algorithms
     * this is not nessesary needed for a final logger and coult be substituted for a less
     * precise time measurement
     * @return timestamp the number of milliseconds sense January 1st 1970
     */
    unsigned long getTime();

  }; // end Logger class

}

/**
 * The global logger
 */
extern ssrlcv::Logger logger;

#endif /* LOGGER_HPP */
