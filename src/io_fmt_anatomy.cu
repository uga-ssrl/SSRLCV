#include "io_fmt_anatomy.cuh"
#include <errno.h>


namespace ssrlcv { 
 namespace io { 
  namespace anatomy { 
    using namespace ssrlcv; 

    typedef ssrlcv::Feature<ssrlcv::SIFT_Descriptor> SiftFeature;

    std::shared_ptr<ssrlcv::Unity<SiftFeature>> readFeatures(std::istream & stream) { 
        const int LINE_SIZE = 1024; // It seems to me that most lines are about 350 chars
        const int DESC_SIZE = 128; // This is standard,,,, riiiiiiight?

        char * line = new char[LINE_SIZE];   
        std::stringstream * strings = nullptr; 

        std::vector<SiftFeature> datboi;

        float x, y, sigma, theta; 
        unsigned char descriptor[DESC_SIZE]; 

        while(! stream.eof()) {    
            stream.getline(line, LINE_SIZE); 
            if(stream.bad()) { 
                // I gotta fix this, I wasn't getting exceptions to play nicely with errno, and the C++ file stream model is hot garbage 
                int x = errno; 
                std::cout << "IO ERROR: " << strerror(x) << std::endl; 
                throw x; 
            }

            strings = new std::stringstream(line); 

            *strings >> x >> y >> sigma >> theta; 
            for(int i = 0; i < DESC_SIZE; i++) { 
                short x;
                *strings >> x;
                descriptor[i] = (unsigned char) x; 
            }

            float2 loc = { x, y };
            SIFT_Descriptor desc;
            desc.sigma = sigma;
            desc.theta = theta;
            std::memcpy(desc.values, descriptor, DESC_SIZE * sizeof(unsigned char));
            datboi.push_back(SiftFeature(loc, desc));

            delete strings; 
        }
        delete[] line; 

        size_t size = datboi.size(); 
        std::shared_ptr<ssrlcv::Unity<SiftFeature>> ret = std::make_shared<ssrlcv::Unity<SiftFeature>>(nullptr, size, cpu); 
        std::memcpy(ret->host.get(), datboi.data(), sizeof(SiftFeature) * size); 
        return ret; 
    }

    std::shared_ptr<ssrlcv::Unity<Match>> readMatches(std::istream & stream) { 
        stream.exceptions(std::ios::failbit);
        const int LINE_SIZE = 1024;
        char * line = new char[LINE_SIZE];   
        std::stringstream * strings = nullptr; 

        std::vector<Match> datboi;

        // s and t are sigma and theta
        float x1, y1, s1, t1, x2, y2, s2, t2; 

        while(! stream.eof()) {    
            stream.getline(line, LINE_SIZE); 
            strings = new std::stringstream(line); 

            *strings >> x1 >> y1 >> s1 >> t1 >> x2 >> y2 >> s2 >> t2; 
            KeyPoint kp1 = { -1, { x1, y1 } };
            KeyPoint kp2 = { -1, { x2, y2 } };
            datboi.push_back({ false, kp1, kp2 });

            delete strings; 
        }
        delete[] line; 

        size_t size = datboi.size(); 
        std::shared_ptr<ssrlcv::Unity<Match>> ret = std::make_shared<ssrlcv::Unity<Match>>(nullptr, size, cpu); 
        std::memcpy(ret->host.get(), datboi.data(), sizeof(Match) * size); 
        return ret; 

    }
  }
 }
}