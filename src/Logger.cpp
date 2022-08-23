#include "Logger.hpp"

// log in the out directory;
ssrlcv::Logger logger = ssrlcv::Logger("out");

// =============================================================================================================
//
// Constructors and Destructors
//
// =============================================================================================================


/**
 * Default constructor
 */
ssrlcv::Logger::Logger(){
  this->logPath = "./out";
  // check if the log file exists
  this->logFileLocation = this->logPath + "/" + this->logName;
  std::ifstream exist(this->logFileLocation.c_str());
  if (!exist.good()){
    // make the file
    std::ofstream temp;
    temp.open(this->logFileLocation);
    // ALWAYS LOG THE TIME!
    temp << std::fixed << std::setprecision(32) << getTime() << ",";
    // now print the real junk
    temp << "Log Generated," << this->logFileLocation << std::endl;
    temp.close();
  }
  // default backgound states
  this->isLogging = false;
  this->killLogging = false;
  // setting up specific "stream" handlers
  this->err = Error(this);
  this->warn = Warning(this);
  this->info = Info(this);
#ifdef LOG_MEM
  this->mem = Memory(this);
#endif
  // this->verbosity = 0;
  std::cout << "Logging to " + this->logFileLocation << std::endl;
}

/**
 * Constructor with path to desired log location
 */
ssrlcv::Logger::Logger(const char* logPath){
  this->logPath = logPath;
  // check if the log file exists
  if (this->logPath.back() == '/'){
    this->logFileLocation = this->logPath + this->logName;
  } else {
    this->logFileLocation = this->logPath + "/" + this->logName;
  }
  std::ifstream exist(this->logFileLocation.c_str());
  if (!exist.good()){
    // make the file
    std::ofstream temp;
    temp.open(this->logFileLocation);
    // ALWAYS LOG THE TIME!
    temp << std::fixed << std::setprecision(32) << getTime() << ",";
    // now print the real junk
    temp << "Log Generated," << this->logFileLocation << std::endl;
    temp.close();
  }
  // default backgound states
  this->isLogging = false;
  this->killLogging = false;
  // setting up specific "stream" handlers
  this->err = Error(this);
  this->warn = Warning(this);
  this->info = Info(this);
#ifdef LOG_MEM
  this->mem = Memory(this);
#endif
  // this->verbosity = 0;
  std::cout << "Logging to " + this->logFileLocation << std::endl;
}

/**
 * Overloaded Copy Constructor, needed to keep unified mutex locking
 */
ssrlcv::Logger::Logger(ssrlcv::Logger const &loggerCopy){
  //empty
  this->err << "ERROR: logger copy constructor was used, only one global instance of the logger is needed\n";
}

/**
 * Overloaded Assignment Operator, needed to keep unitied mutex locking
 */
ssrlcv::Logger &ssrlcv::Logger::operator=(ssrlcv::Logger const &loggerCopy){
  if (&loggerCopy != this) {
    // lock both objects
    std::unique_lock<std::mutex> lock_this(mtx, std::defer_lock);
    std::unique_lock<std::mutex> lock_copy(loggerCopy.mtx, std::defer_lock);
    // ensure no deadlock
    std::lock(lock_this, lock_copy);
    // default backgound states
    this->isLogging   = false;
    this->killLogging = false;
    this->logPath         = loggerCopy.logPath;
    this->logFileLocation = loggerCopy.logFileLocation;
    // setting up specific "stream" handlers
    this->err = Error(this);
    this->warn = Warning(this);
    this->info = Info(this);
  #ifdef LOG_MEM
    this->mem = Memory(this);
  #endif
    std::cout << "Logging to " + this->logFileLocation << std::endl;
  }
  return *this;
}

/**
 * Default destructor
 */
ssrlcv::Logger::~Logger(){
  this->err.logger = nullptr;
  this->warn.logger = nullptr;
  this->info.logger = nullptr;
#ifdef LOG_MEM
  this->mem.logger = nullptr;
#endif
  this->stopBackgroundLogging();
}




ssrlcv::Logger &ssrlcv::Logger::operator<<(const char *input){
  this->log(input);
  return *this;
}

ssrlcv::Logger::Error::Error(){
  this->logger = nullptr;
}

ssrlcv::Logger::Error::Error(ssrlcv::Logger *logger){
  this->logger = logger;
}

ssrlcv::Logger::Error &ssrlcv::Logger::Error::operator<<(const char *input){
#if LOG_LEVEL >= 1
  std::cout<<input<<std::endl;
  this->logger->logError(input);
#endif
  return *this;
}

ssrlcv::Logger::Error &ssrlcv::Logger::Error::operator<<(std::string input){
#if LOG_LEVEL >= 1
  std::cout<<input<<std::endl;
  this->logger->logError(input);
#endif
  return *this;
}

ssrlcv::Logger::Warning::Warning(){
  this->logger = nullptr;
}

ssrlcv::Logger::Warning::Warning(ssrlcv::Logger *logger){
  this->logger = logger;
}

ssrlcv::Logger::Warning &ssrlcv::Logger::Warning::operator<<(const char *input){
#if LOG_LEVEL >= 2
  this->logger->logWarning(input);
#endif
  return *this;
}

ssrlcv::Logger::Warning &ssrlcv::Logger::Warning::operator<<(std::string input){
#if LOG_LEVEL >= 2
  this->logger->logWarning(input);
#endif
  return *this;
}

ssrlcv::Logger::Info::Info(){
  this->logger = nullptr;
}

ssrlcv::Logger::Info::Info(ssrlcv::Logger *logger){
  this->logger = logger;
}

ssrlcv::Logger::Info &ssrlcv::Logger::Info::operator<<(const char *input){
#if LOG_LEVEL >= 3
  this->logger->logInfo(input);
#endif
  return *this;
}

ssrlcv::Logger::Info &ssrlcv::Logger::Info::operator<<(std::string input){
#if LOG_LEVEL >= 3
  this->logger->logInfo(input);
#endif
  return *this;
}


//
// Memory Logging
//

ssrlcv::Logger::Memory::Memory() {
  this->logger = nullptr;
}

ssrlcv::Logger::Memory::Memory(Logger* logger) {
  this->logger = logger;
#ifdef LOG_MEM
  // check if the log file exists
  this->memoryLogLocation = this->logger->logPath + "/memory.csv";
  std::ifstream exist(this->memoryLogLocation.c_str());
  if (!exist.good()){
    // make the file
    std::ofstream temp;
    temp.open(this->memoryLogLocation);
    temp.close();
  }
  // this->verbosity = 0;
  std::cout << "Logging memory to " + this->memoryLogLocation << std::endl;
#endif
}

void ssrlcv::Logger::Memory::logHostPinned(long change) {
#ifdef LOG_MEM
  this->host_pinned_mem += change;

  std::ofstream outstream;
  this->logger->mtx.lock();
  outstream.open(this->memoryLogLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->logger->getTime();
  // now print the real junk
  outstream << ",total," << this->device_mem + this->host_unpinned_mem + this->host_pinned_mem;
  outstream << ",host_unpinned," << this->host_unpinned_mem;
  outstream << ",host_pinned," << this->host_pinned_mem;
  outstream << ",device," << this->device_mem;
  outstream << std::endl;
  outstream.close();
  this->logger->mtx.unlock();
#endif
}

void ssrlcv::Logger::Memory::logHostUnpinned(long change) {
#ifdef LOG_MEM
  this->host_unpinned_mem += change;

  std::ofstream outstream;
  this->logger->mtx.lock();
  outstream.open(this->memoryLogLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->logger->getTime();
  // now print the real junk
  outstream << ",total," << this->device_mem + this->host_unpinned_mem + this->host_pinned_mem;
  outstream << ",host_unpinned," << this->host_unpinned_mem;
  outstream << ",host_pinned," << this->host_pinned_mem;
  outstream << ",device," << this->device_mem;
  outstream << std::endl;
  outstream.close();
  this->logger->mtx.unlock();
#endif
}

void ssrlcv::Logger::Memory::logDevice(long change) {
#ifdef LOG_MEM
  this->device_mem += change;

  std::ofstream outstream;
  this->logger->mtx.lock();
  outstream.open(this->memoryLogLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->logger->getTime();
  // now print the real junk
  outstream << ",total," << this->device_mem + this->host_unpinned_mem + this->host_pinned_mem;
  outstream << ",host_unpinned," << this->host_unpinned_mem;
  outstream << ",host_pinned," << this->host_pinned_mem;
  outstream << ",device," << this->device_mem;
  outstream << std::endl;
  outstream.close();
  this->logger->mtx.unlock();
#endif
}

// =============================================================================================================
//
// Direct Logging Methods
//
// =============================================================================================================

/**
 * write the input string to the log
 * @param input a string to write to the log
 */
void ssrlcv::Logger::log(const char* input){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",comment,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * write the input string to the log
 * @param input a string to write to the log
 */
void ssrlcv::Logger::log(std::string input){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",comment,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs a state with a state tag, it is expected that the programmer has a set of pre-defined states
 * @param state a string to be tagged as a state
 */
void ssrlcv::Logger::logState(const char* state){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",state,";

  // now print the real junk
  outstream << state << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs a state with a state tag, it is expected that the programmer has a set of pre-defined states
 * @param state a string to be tagged as a state
 */
void ssrlcv::Logger::logState(std::string state){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",state,";
  // now print the real junk
  outstream << state << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs a message with an error tag
 * @param input a string to write to the log
 */
void ssrlcv::Logger::logWarning(const char *input){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",warning,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs a message with an error tag
 * @param input a string to write to the log
 */
void ssrlcv::Logger::logInfo(std::string input){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",info,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs a message with an error tag
 * @param input a string to write to the log
 */
void ssrlcv::Logger::logInfo(const char *input){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",info,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs a message with an error tag
 * @param input a string to write to the log
 */
void ssrlcv::Logger::logWarning(std::string input){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",warning,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs a message with an error tag
 * @param input a string to write to the log
 */
void ssrlcv::Logger::logError(const char* input){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",error,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs a message with an error tag
 * @param input a string to write to the log
 */
void ssrlcv::Logger::logError(std::string input){
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << this->getTime() << ",error,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs the CPU names
 * for details see https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_tx2_32.html%23wwpID0E0ZS0HA
 */
void ssrlcv::Logger::logCPUnames(){

  std::string logline = "";

  // the paths to check the power states
  std::string startPath = "/sys/devices/system/cpu/cpu";
  std::string endPath   = "/cpuidle/state";

  // the id's of the Denver cores
  int Denver_IDs[2] = {1,2};      // 3 of each of these
  // the id's of the A57 cores
  int A57_IDs[4]    = {0,3,4,5};  // 2 of each of these

  // log the Denver Cores
  for (int i = 0; i < 2; i++){
    for (int j = 0; j < 3; j++){
      this->mtx.lock();
      std::ifstream infile(startPath + std::to_string(Denver_IDs[i]) + endPath + std::to_string(j) + "/name" );
      if (infile.is_open()){
        std::string line;
        while (getline(infile,line)){
          logline += line;
          logline += ",";
        }
        infile.close();
      } else {
        this->err << "ERROR: logger could not open CPU name \n";
      }
      this->mtx.unlock();
    }
  }

  // log the A57 Cores
  for (int i = 0; i < 4; i++){
    for (int j = 0; j < 2; j++){
      this->mtx.lock();
      std::ifstream infile(startPath + std::to_string(A57_IDs[i]) + endPath + std::to_string(j) + "/name" );
      if (infile.is_open()){
        std::string line;
        while (getline(infile,line)){
          logline += line;
          logline += ",";
        }
        infile.close();
      } else {
        this->err << "ERROR: logger could not open CPU name \n";
      }
      this->mtx.unlock();
    }
  }

  // log to the file
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << getTime() << ",";
  // now print the real junk
  outstream << logline << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs the system voltage
 * for details see https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_tx2_32.html%23wwpID0E05H0HA
 */
void ssrlcv::Logger::logVoltage(){

  std::string logline = "";

  // 0 and 1 are on the module, 2 and 3 are on the carrier board
  // at i2c address 0x40 and 0x41
  std::string monitor0 = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0";
  std::string monitor1 = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1";

  // read in VDD_SYS_GPU voltage
  this->mtx.lock();
  std::ifstream infile1(monitor0 + "/in_voltage0_input");
  if (infile1.is_open()){
    std::string line;
    while (getline(infile1,line)){
      logline += "VDD_SYS_GPU,";
      logline += line + ",";
    }
    infile1.close();
  } else {
    this->err << "ERROR: logger could not log voltage \n";
  }
  this->mtx.unlock();

  // read in VDD_SYS_GPU voltage
  this->mtx.lock();
  std::ifstream infile2(monitor0 + "/in_voltage1_input");
  if (infile2.is_open()){
    std::string line;
    while (getline(infile2,line)){
      logline += "VDD_SYS_SOC,";
      logline += line + ",";
    }
    infile2.close();
  } else {
    this->err << "ERROR: logger could not log voltage \n";
  }
  this->mtx.unlock();

  // read in VDD_IN voltage
  this->mtx.lock();
  std::ifstream infile3(monitor1 + "/in_voltage0_input");
  if (infile3.is_open()){
    std::string line;
    while (getline(infile3,line)){
      logline += "VDD_IN,";
      logline += line + ",";
    }
    infile3.close();
  } else {
    this->err << "ERROR: logger could not log voltage \n";
  }
  this->mtx.unlock();

  // read in VDD_SYS_GPU voltage
  this->mtx.lock();
  std::ifstream infile4(monitor1 + "/in_voltage1_input");
  if (infile4.is_open()){
    std::string line;
    while (getline(infile4,line)){
      logline += "VDD_SYS_CPU,";
      logline += line + ",";
    }
    infile4.close();
  } else {
    this->err << "ERROR: logger could not log voltage \n";
  }
  this->mtx.unlock();
  // log to the file
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << getTime() << ",";
  // now print the real junk
  outstream << logline << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs the system current
 * for details see https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_tx2_32.html%23wwpID0E05H0HA
 */
void ssrlcv::Logger::logCurrent(){

  std::string logline = "";

  // 0 and 1 are on the module, 2 and 3 are on the carrier board
  // at i2c address 0x40 and 0x41
  std::string monitor0 = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0";
  std::string monitor1 = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1";

  // read in VDD_SYS_GPU voltage
  this->mtx.lock();
  std::ifstream infile1(monitor0 + "/in_current0_input");
  if (infile1.is_open()){
    std::string line;
    while (getline(infile1,line)){
      logline += "CRR_SYS_GPU,";
      logline += line + ",";
    }
    infile1.close();
  } else {
    this->err << "ERROR: logger could not log current \n";
  }
  this->mtx.unlock();

  // read in VDD_SYS_GPU voltage
  this->mtx.lock();
  std::ifstream infile2(monitor0 + "/in_current1_input");
  if (infile2.is_open()){
    std::string line;
    while (getline(infile2,line)){
      logline += "CRR_SYS_SOC,";
      logline += line + ",";
    }
    infile2.close();
  } else {
    this->err << "ERROR: logger could not log current \n";
  }
  this->mtx.unlock();

  // read in VDD_IN voltage
  this->mtx.lock();
  std::ifstream infile3(monitor1 + "/in_current0_input");
  if (infile3.is_open()){
    std::string line;
    while (getline(infile3,line)){
      logline += "CRR_IN,";
      logline += line + ",";
    }
    infile3.close();
  } else {
    this->err << "ERROR: logger could not log current \n";
  }
  this->mtx.unlock();

  // read in VDD_SYS_GPU voltage
  this->mtx.lock();
  std::ifstream infile4(monitor1 + "/in_current1_input");
  if (infile4.is_open()){
    std::string line;
    while (getline(infile4,line)){
      logline += "CRR_SYS_CPU,";
      logline += line + ",";
    }
    infile4.close();
  } else {
    this->err << "ERROR: logger could not log current \n";
  }
  this->mtx.unlock();
  // log to the file
  std::ofstream outstream;
  this->mtx.lock();
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  outstream << std::fixed << std::setprecision(32) << getTime() << ",";
  // now print the real junk
  outstream << logline << std::endl;
  outstream.close();
  this->mtx.unlock();
}

/**
 * logs the system power
 * for details see https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_tx2_32.html%23wwpID0E05H0HA
 */
void ssrlcv::Logger::logPower(){

    std::string logline = "";

    // 0 and 1 are on the module, 2 and 3 are on the carrier board
    // at i2c address 0x40 and 0x41
    std::string monitor0 = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0";
    std::string monitor1 = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1";

    // read in VDD_SYS_GPU voltage
    this->mtx.lock();
    std::ifstream infile1(monitor0 + "/in_power0_input");
    if (infile1.is_open()){
      std::string line;
      while (getline(infile1,line)){
        logline += "PWR_SYS_GPU,";
        logline += line + ",";
      }
      infile1.close();
    } else {
      this->err << "ERROR: logger could not log power \n";
    }
    this->mtx.unlock();

    // read in VDD_SYS_GPU voltage
    this->mtx.lock();
    std::ifstream infile2(monitor0 + "/in_power1_input");
    if (infile2.is_open()){
      std::string line;
      while (getline(infile2,line)){
        logline += "PWR_SYS_SOC,";
        logline += line + ",";
      }
      infile2.close();
    } else {
      this->err << "ERROR: logger could not log power \n";
    }
    this->mtx.unlock();

    // read in VDD_IN voltage
    this->mtx.lock();
    std::ifstream infile3(monitor1 + "/in_power0_input");
    if (infile3.is_open()){
      std::string line;
      while (getline(infile3,line)){
        logline += "PWR_IN,";
        logline += line + ",";
      }
      infile3.close();
    } else {
      this->err << "ERROR: logger could not log power \n";
    }
    this->mtx.unlock();

    // read in VDD_SYS_GPU voltage
    this->mtx.lock();
    std::ifstream infile4(monitor1 + "/in_power1_input");
    if (infile4.is_open()){
      std::string line;
      while (getline(infile4,line)){
        logline += "PWR_SYS_CPU,";
        logline += line + ",";
      }
      infile4.close();
    } else {
      this->err << "ERROR: logger could not log power \n";
    }
    this->mtx.unlock();
    // log to the file
    std::ofstream outstream;
    this->mtx.lock();
    outstream.open(this->logFileLocation, std::ofstream::app);
    // ALWAYS LOG THE TIME!
    outstream << std::fixed << std::setprecision(32) << getTime() << ",";
    // now print the real junk
    outstream << logline << std::endl;
    outstream.close();
    this->mtx.unlock();
}

/**
 * starts logging in the backgound
 * for now this only logs Voltage, Power, and Current
 * @param rate is the number of seconds in between logging
 */
void ssrlcv::Logger::startBackgoundLogging(int rate){
  this->mtx.lock();
  this->logDelay = rate;
  this->mtx.unlock();
  this->mtx.lock();
  if (isLogging) {
    this->err << "ERROR: unable to start backgound logging because a background log is currently running. Stop it and restart it if so desired \n";
    this->mtx.unlock();
  } else {
    // spawn the new thread
    this->mtx.unlock();
    std::thread background_thread(&Logger::looper, this, rate);
    background_thread.detach();
    this->mtx.lock();
    this->isLogging = true;
    this->mtx.unlock();
  }
}

/**
 * stops the backgound if it is running
 */
void ssrlcv::Logger::stopBackgroundLogging(){
  // forces variables into this configeration
  this->mtx.lock();
  this->killLogging = true;
  this->mtx.unlock();
  this->mtx.lock();
  this->isLogging = false;
  this->mtx.unlock();
}

// =============================================================================================================
//
// Private Logging Methods
//
// =============================================================================================================

/**
 * Used my startBackgoundLogging to keep logging in the backgound
 * @param @param delay the time in seconds to wait in between logging
 */
void ssrlcv::Logger::looper(int delay){
  // loop and log!
  while(true){
    this->mtx.lock();
    bool b = this->killLogging;
    this->mtx.unlock();
    if (b){
      log("stoping logging thread safely");
      return;
    }

    // TODO could update logging paramters here by doing a mutex check of object variables which can change on the fly by the user

    // logVoltage();
    // logCurrent();
    // logPower();

    // wait to log again
    sleep(delay);
  }
}

/**
 * Used to return a high resolution timestamp for dense resolution timing of algorithms
 * this is not nessesary needed for a final logger and coult be substituted for a less
 * precise time measurement
 * @return timestamp the number of milliseconds sense January 1st 1970
 */
unsigned long ssrlcv::Logger::getTime(){
  std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
  return ms.count();
}




//
