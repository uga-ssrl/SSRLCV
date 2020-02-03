#include <cstdio>
#include <string> 
#include <iostream> 

int main(int argc, char ** argv) { 
	if(argc < 2) { 
		std::cerr << "Usage: binary-float [numbers]" << std::endl; 
		return 2; 
	}

	float f;
	for(int i = 1; i < argc; i++) {
		f = std::stof(argv[i], nullptr);
		fwrite((void *) &f, sizeof(f), 1, stdout); 
		fflush(stdout); 
	}
	return 0; 
}