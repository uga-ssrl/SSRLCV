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
    // ALWAYS LOG THE TIME!
    std::time_t t = std::time(nullptr);
    outstream << t << ",";
    // now print the real junk
    temp << "Log Generated," << this->logFileLocation << std::endl;
    temp.close();
  }
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
    std::time_t t = std::time(nullptr);
    outstream << t << ",";
    // now print the real junk
    temp << "Log Generated," << this->logFileLocation << std::endl;
    temp.close();
  }
}

/**
 * Default destructor
 */
ssrlcv::Logger::~Logger(){

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
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  std::time_t t = std::time(nullptr);
  outstream << t << ",comment,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
}

/**
 * write the input string to the log
 * @param input a string to write to the log
 */
void ssrlcv::Logger::log(std::string input){
  std::ofstream outstream;
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  std::time_t t = std::time(nullptr);
  outstream << t << ",comment,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
}

/**
 * logs a state with a state tag, it is expected that the programmer has a set of pre-defined states
 * @param state a string to be tagged as a state
 */
void log(const char* state){
  std::ofstream outstream;
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  std::time_t t = std::time(nullptr);
  outstream << t << ",state,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
}

/**
 * logs a state with a state tag, it is expected that the programmer has a set of pre-defined states
 * @param state a string to be tagged as a state
 */
void log(std::string state){
  std::ofstream outstream;
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  std::time_t t = std::time(nullptr);
  outstream << t << ",state,";
  // now print the real junk
  outstream << input << std::endl;
  outstream.close();
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
      std::ifstream infile(startPath + std::to_string(Denver_IDs[i]) + endPath + std::to_string(j) + "/name" );
      if (infile.is_open()){
        std::string line;
        while (getline(infile,line)){
          logline += line;
          logline += ",";
        }
        infile.close();
      } else {
        std::cerr << "ERROR: logger could not open CPU name " << std::endl;
      }
    }
  }

  // log the A57 Cores
  for (int i = 0; i < 4; i++){
    for (int j = 0; j < 2; j++){
      std::ifstream infile(startPath + std::to_string(A57_IDs[i]) + endPath + std::to_string(j) + "/name" );
      if (infile.is_open()){
        std::string line;
        while (getline(infile,line)){
          logline += line;
          logline += ",";
        }
        infile.close();
      } else {
        std::cerr << "ERROR: logger could not open CPU name " << std::endl;
      }
    }
  }

  // log to the file
  std::ofstream outstream;
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  std::time_t t = std::time(nullptr);
  outstream << t << ",";
  // now print the real junk
  outstream << logline << std::endl;
  outstream.close();
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
  std::ifstream infile1(monitor0 + "/in_voltage0_input");
  if (infile1.is_open()){
    std::string line;
    while (getline(infile1,line)){
      logline += "VDD_SYS_GPU,";
      logline += line + ",";
    }
    infile1.close();
  } else {
    std::cerr << "ERROR: logger could not log voltage " << std::endl;
  }

  // read in VDD_SYS_GPU voltage
  std::ifstream infile2(monitor0 + "/in_voltage1_input");
  if (infile2.is_open()){
    std::string line;
    while (getline(infile2,line)){
      logline += "VDD_SYS_SOC,";
      logline += line + ",";
    }
    infile2.close();
  } else {
    std::cerr << "ERROR: logger could not log voltage " << std::endl;
  }

  // read in VDD_IN voltage
  std::ifstream infile3(monitor1 + "/in_voltage0_input");
  if (infile3.is_open()){
    std::string line;
    while (getline(infile3,line)){
      logline += "VDD_IN,";
      logline += line + ",";
    }
    infile3.close();
  } else {
    std::cerr << "ERROR: logger could not log voltage " << std::endl;
  }

  // read in VDD_SYS_GPU voltage
  std::ifstream infile4(monitor1 + "/in_voltage1_input");
  if (infile4.is_open()){
    std::string line;
    while (getline(infile4,line)){
      logline += "VDD_SYS_CPU,";
      logline += line + ",";
    }
    infile4.close();
  } else {
    std::cerr << "ERROR: logger could not log voltage " << std::endl;
  }
  // log to the file
  std::ofstream outstream;
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  std::time_t t = std::time(nullptr);
  outstream << t << ",";
  // now print the real junk
  outstream << logline << std::endl;
  outstream.close();
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
  std::ifstream infile1(monitor0 + "/in_current0_input");
  if (infile1.is_open()){
    std::string line;
    while (getline(infile1,line)){
      logline += "CRR_SYS_GPU,";
      logline += line + ",";
    }
    infile1.close();
  } else {
    std::cerr << "ERROR: logger could not log current " << std::endl;
  }

  // read in VDD_SYS_GPU voltage
  std::ifstream infile2(monitor0 + "/in_current1_input");
  if (infile2.is_open()){
    std::string line;
    while (getline(infile2,line)){
      logline += "CRR_SYS_SOC,";
      logline += line + ",";
    }
    infile2.close();
  } else {
    std::cerr << "ERROR: logger could not log current " << std::endl;
  }

  // read in VDD_IN voltage
  std::ifstream infile3(monitor1 + "/in_current0_input");
  if (infile3.is_open()){
    std::string line;
    while (getline(infile3,line)){
      logline += "CRR_IN,";
      logline += line + ",";
    }
    infile3.close();
  } else {
    std::cerr << "ERROR: logger could not log current " << std::endl;
  }

  // read in VDD_SYS_GPU voltage
  std::ifstream infile4(monitor1 + "/in_current1_input");
  if (infile4.is_open()){
    std::string line;
    while (getline(infile4,line)){
      logline += "CRR_SYS_CPU,";
      logline += line + ",";
    }
    infile4.close();
  } else {
    std::cerr << "ERROR: logger could not log current " << std::endl;
  }
  // log to the file
  std::ofstream outstream;
  outstream.open(this->logFileLocation, std::ofstream::app);
  // ALWAYS LOG THE TIME!
  std::time_t t = std::time(nullptr);
  outstream << t << ",";
  // now print the real junk
  outstream << logline << std::endl;
  outstream.close();
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
    std::ifstream infile1(monitor0 + "/in_power0_input");
    if (infile1.is_open()){
      std::string line;
      while (getline(infile1,line)){
        logline += "PWR_SYS_GPU,";
        logline += line + ",";
      }
      infile1.close();
    } else {
      std::cerr << "ERROR: logger could not log power " << std::endl;
    }

    // read in VDD_SYS_GPU voltage
    std::ifstream infile2(monitor0 + "/in_power1_input");
    if (infile2.is_open()){
      std::string line;
      while (getline(infile2,line)){
        logline += "PWR_SYS_SOC,";
        logline += line + ",";
      }
      infile2.close();
    } else {
      std::cerr << "ERROR: logger could not log power " << std::endl;
    }

    // read in VDD_IN voltage
    std::ifstream infile3(monitor1 + "/in_power0_input");
    if (infile3.is_open()){
      std::string line;
      while (getline(infile3,line)){
        logline += "PWR_IN,";
        logline += line + ",";
      }
      infile3.close();
    } else {
      std::cerr << "ERROR: logger could not log power " << std::endl;
    }

    // read in VDD_SYS_GPU voltage
    std::ifstream infile4(monitor1 + "/in_power1_input");
    if (infile4.is_open()){
      std::string line;
      while (getline(infile4,line)){
        logline += "PWR_SYS_CPU,";
        logline += line + ",";
      }
      infile4.close();
    } else {
      std::cerr << "ERROR: logger could not log power " << std::endl;
    }
    // log to the file
    std::ofstream outstream;
    outstream.open(this->logFileLocation, std::ofstream::app);
    // ALWAYS LOG THE TIME!
    std::time_t t = std::time(nullptr);
    outstream << t << ",";
    // now print the real junk
    outstream << logline << std::endl;
    outstream.close();
}











//
