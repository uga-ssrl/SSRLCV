#include "Logger.hpp"

// =============================================================================================================
//
// Constructors and Destructors
//
// =============================================================================================================

/**
 * Default constructor
 */
ssrlcv::Logger::Logger(){
  this->logPath = ".";
  // check if the log file exists
  this->logFileLocation = this->logPath + "/" + this->logName;
  std::ifstream exist(this->logFileLocation.c_str());
  if (!exist.good()){
    // make the file
    std::ofstream temp;
    temp.open(this->logFileLocation);
    temp << "Log File Created at: " << this->logFileLocation << std::endl;
    temp.close();
  }
  this->stream = new fstream();
}

/**
 * Constructor with path to desired log location
 */
ssrlcv::Logger::Logger(const char* logPath){
  this->logPath = logPath;
  // check if the log file exists
  if (this->logPath.back == '/'){
    this->logFileLocation = this->logPath + this->logName;
  } else {
    this->logFileLocation = this->logPath + "/" + this->logName;
  }
  std::ifstream exist(this->logFileLocation.c_str());
  if (!exist.good()){
    // make the file
    std::ofstream temp;
    temp.open(this->logFileLocation);
    temp << "Log File Created at: " << this->logFileLocation << std::endl;
    temp.close();
  }
  this->stream = new fstream();
}

/**
 * Default destructor
 */
ssrlcv::Logger::~Logger(){

}

// =============================================================================================================
//
// Logging Methods
//
// =============================================================================================================

/**
 * write the input string to the log
 * @param input a string to write to the log
 */
void ssrlcv::Logger::log(const char* input){
  // TODO add time stamp
  this->stream->open(this->logFileLocation, std::ofstream::app);
  &this->stream << input << std::endl;
  this->stream->close();
}

/**
 * write the input string to the log
 * @param input a string to write to the log
 */
void ssrlcv::Logger::log(std::string input){
  // TODO add time stamp
  this->stream->open(this->logFileLocation, std::ofstream::app);
  &this->stream << input << std::endl;
  this->stream->close();
}
