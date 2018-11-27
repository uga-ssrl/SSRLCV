#include "reprojection.cuh"
//had to add these files to this file bc of linker errors`
//#include "cpuUtilities.cu"
//#include "gpuUtilities.cu"

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}

//global constants
const unsigned int   res  = 254;
const float          foc  = 0.160;
const float          fov  = (11.4212*PI/180);//0.0593412; //3.4 degrees to match the blender sim //0.8575553107; // 49.1343 degrees  // 0.785398163397; // 45 degrees
const float          dpix = (foc*tan(fov/2))/(res/2); //float          dpix = 0.00002831538; //(foc*tan(fov/2))/(res/2)

//============= DEVICE FUNCTIONS ================

__device__ int getIndex_gpu(int x, int y)
{
        return (3*x +y);
}
__device__ void multiply3x3x1_gpu(float A[9], float B[3], float (&C)[3])
{
  for (int r = 0; r < 3; ++r)
  {
    float val = 0;
    for (int c = 0; c < 3; ++c)
    {
      val += A[getIndex_gpu(r,c)] * B[c];
    }
    C[r] = val;
  }
}
__device__ void multiply3x3x1_gpu(float A[3][3], float B[3], float (&C)[3])
{
  for (int r = 0; r < 3; ++r)
  {
    float val = 0;
    for (int c = 0; c < 3; ++c)
    {
      val += A[r][c] * B[c];
    }
    C[r] = val;
  }
}
__device__ float dot_product_gpu(float a[3], float b[3])
{
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
__device__ float magnitude_gpu(float v[3])
{
  return sqrt(dot_product_gpu(v, v));
}
__device__ void normalize_gpu(float (&v)[3])
{
  float mag = magnitude_gpu(v);
  if(mag > 0)
        {
    v[0] = v[0]/mag;
    v[1] = v[1]/mag;
    v[2] = v[2]/mag;
  }
}
__device__ int getGlobalIdx_1D_1D()
{
        return blockIdx.x *blockDim.x + threadIdx.x;
}
__device__ void inverse3x3_gpu(float M[3][3], float (&Minv)[3][3])
{
  float d1 = M[1][1] * M[2][2] - M[2][1] * M[1][2];
  float d2 = M[1][0] * M[2][2] - M[1][2] * M[2][0];
  float d3 = M[1][0] * M[2][1] - M[1][1] * M[2][0];
  float det = M[0][0]*d1 - M[0][1]*d2 + M[0][2]*d3;
  if(det == 0)
        {
    // return pinv(M);
  }
  float invdet = 1/det;
  Minv[0][0] = d1*invdet;
  Minv[0][1] = (M[0][2]*M[2][1] - M[0][1]*M[2][2]) * invdet;
  Minv[0][2] = (M[0][1]*M[1][2] - M[0][2]*M[1][1]) * invdet;
  Minv[1][0] = -1 * d2 * invdet;
  Minv[1][1] = (M[0][0]*M[2][2] - M[0][2]*M[2][0]) * invdet;
  Minv[1][2] = (M[1][0]*M[0][2] - M[0][0]*M[1][2]) * invdet;
  Minv[2][0] = d3 * invdet;
  Minv[2][1] = (M[2][0]*M[0][1] - M[0][0]*M[2][1]) * invdet;
  Minv[2][2] = (M[0][0]*M[1][1] - M[1][0]*M[0][1]) * invdet;
}


__global__ void two_view_reproject(int numMatches, float4* matches, float cam1C[3], float cam1V[3],float cam2C[3], float cam2V[3], float K_inv[9], float rotationTranspose1[9], float rotationTranspose2[9], float3* points)
{

  if(!(getGlobalIdx_1D_1D()<numMatches))return;
	//check out globalID cheat sheet jackson gave you for this
	int matchIndex = getGlobalIdx_1D_1D(); //need to define once I calculate grid/block size
	//printf("thread index %d", getGlobalIdx_1D_1D());
	float4 match = matches[getGlobalIdx_1D_1D()];


	float pix1[3] =
	{
		match.x, match.y, 1
	};
	float pix2[3] =
	{
		match.z, match.w, 1
	};


	float inter1[3];
	float inter2[3];

	float temp[3];
	multiply3x3x1_gpu(K_inv, pix1, temp);
	multiply3x3x1_gpu(rotationTranspose1, temp, inter1);
	multiply3x3x1_gpu(K_inv, pix2, temp);
	multiply3x3x1_gpu(rotationTranspose2, temp, inter2);

	float worldP1[3] =
	{
		inter1[0]+cam1C[0], inter1[1]+cam1C[1], inter1[2]+cam1C[2]
	};

	float worldP2[3] =
	{
		inter2[0]+cam2C[0], inter2[1]+cam2C[1], inter2[2]+cam2C[2]
	};

	float v1[3] =
	{
		worldP1[0] - cam1C[0], worldP1[1] - cam1C[1], worldP1[2] - cam1C[2]
	};

	float v2[3] =
	{
		worldP2[0] - cam2C[0], worldP2[1] - cam2C[1], worldP2[2] - cam2C[2]
	};

	normalize_gpu(v1);
	normalize_gpu(v2);



	//match1 and match2?
	float M1[3][3] =
	{
		{ 1-(v1[0]*v1[0]), 0-(v1[0]*v1[1]), 0-(v1[0]*v1[2]) },
		{ 0-(v1[0]*v1[1]), 1-(v1[1]*v1[1]), 0-(v1[1]*v1[2]) },
		{ 0-(v1[0]*v1[2]), 0-(v1[1]*v1[2]), 1-(v1[2]*v1[2]) }
	};

	float M2[3][3] =
	{
		{ 1-(v2[0]*v2[0]), 0-(v2[0]*v2[1]), 0-(v2[0]*v2[2]) },
		{ 0-(v2[0]*v2[1]), 1-(v2[1]*v2[1]), 0-(v2[1]*v2[2]) },
		{ 0-(v2[0]*v2[2]), 0-(v2[1]*v2[2]), 1-(v2[2]*v2[2]) }
	};

	float q1[3];
	float q2[3];
	float Q[3];

	multiply3x3x1_gpu( M1, worldP1, q1);
	multiply3x3x1_gpu( M2, worldP2, q2);

	float M[3][3];
	float M_inv[3][3];

	for(int r = 0; r < 3; ++r)
	{
		for(int c = 0; c < 3; ++c)
		{
			M[r][c] = M1[r][c] + M2[r][c];
		}
		Q[r] = q1[r] + q2[r];
	}

	float solution[3];
	inverse3x3_gpu(M, M_inv);
	multiply3x3x1_gpu(M_inv, Q, solution);



  	points[matchIndex].x = solution[0];
  	points[matchIndex].y = solution[1];
  	points[matchIndex].z = solution[2];

}

//============= HOST FUNCTIONS ================

void transpose_cpu(float M[3][3], float (&M_t)[3][3])
{
  for(int r = 0; r < 3; ++r)
  {
    for(int c = 0; c < 3; ++c)
    {
      M_t[r][c] = M[c][r];
    }
  }
}
void inverse3x3_cpu(float M[3][3], float (&Minv)[3][3])
{
  float d1 = M[1][1] * M[2][2] - M[2][1] * M[1][2];
  float d2 = M[1][0] * M[2][2] - M[1][2] * M[2][0];
  float d3 = M[1][0] * M[2][1] - M[1][1] * M[2][0];
  float det = M[0][0]*d1 - M[0][1]*d2 + M[0][2]*d3;
  if(det == 0)
        {
    // return pinv(M);
  }
  float invdet = 1/det;
  Minv[0][0] = d1*invdet;
  Minv[0][1] = (M[0][2]*M[2][1] - M[0][1]*M[2][2]) * invdet;
  Minv[0][2] = (M[0][1]*M[1][2] - M[0][2]*M[1][1]) * invdet;
  Minv[1][0] = -1 * d2 * invdet;
  Minv[1][1] = (M[0][0]*M[2][2] - M[0][2]*M[2][0]) * invdet;
  Minv[1][2] = (M[1][0]*M[0][2] - M[0][0]*M[1][2]) * invdet;
  Minv[2][0] = d3 * invdet;
  Minv[2][1] = (M[2][0]*M[0][1] - M[0][0]*M[2][1]) * invdet;
  Minv[2][2] = (M[0][0]*M[1][1] - M[1][0]*M[0][1]) * invdet;
}
void multiply3x3_cpu(float A[3][3], float B[3][3], float (&C)[3][3])
{
  for(int r = 0; r < 3; ++r)
  {
    for(int c = 0; c < 3; ++c)
    {
      float entry = 0;
      for(int z = 0; z < 3; ++z)
      {
        entry += A[r][z]*B[z][c];
      }
      C[r][c] = entry;
    }
  }
}
void multiply3x3x1_cpu(float A[3][3], float B[3], float (&C)[3])
{
  for (int r = 0; r < 3; ++r)
  {
    float val = 0;
    for (int c = 0; c < 3; ++c)
    {
      val += A[r][c] * B[c];
    }
    C[r] = val;
  }
}
//using this to make the code prettier and not have a bunch of if statements
void debugPrint(std::string str)
{
        ConfigVals config(); //used for configuration
        if (true) //fix later
        {
                std::cout << str << std::endl;
        }
}
//takes in point cloud outputted from gpu and saves it in ply format
void savePly(PointCloud* pCloud)
{
        std::ofstream outputFile1("out/repro_output.ply");
        outputFile1 << "ply\nformat ascii 1.0\nelement vertex ";
        outputFile1 << pCloud->numPoints << "\n";
        outputFile1 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
        outputFile1 << "end_header\n";

        float3* currentPoint;
        for(int i = 0; i < pCloud->numPoints; i++)
        {
                currentPoint = &(pCloud->points[i]);
                outputFile1 << currentPoint->x << " " << currentPoint->y << " " << currentPoint->z << " " << 0 << " " << 254 << " " << 0 << "\n";
        }
        std::cout<<"out/repro_output.ply has been written"<<std::endl;
}
//used inside of loadMatchData to fill in match data for each line
 void parseMatchData(float4* &currentMatch, std::string line, int index)
{
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> data;
        while (getline(ss, token, ','))
        {
                debugPrint(token);
                data.push_back(token);
        }
        //debugPrint("ATTEMPTING TO RELLOCATE 4 ELEMENTS IN VECTOR LENGTH: " + data.size());
        //store match data
        currentMatch->x = stof(data[2]);
        currentMatch->y = stof(data[3]);
        currentMatch->z = stof(data[4]);
        currentMatch->w = stof(data[5]);
        //store match colors
        /*currentMatch->matchColorsL.x = stoi(data[6]);
        currentMatch->matchColorsL.y = stoi(data[7]);
        currentMatch->matchColorsL.z = stoi(data[8]);
        currentMatch->matchColorsR.x = stoi(data[9]);
        currentMatch->matchColorsR.y = stoi(data[10]);
        currentMatch->matchColorsR.z = stoi(data[11]);*/
}
//this function is used to load the feaure matches file and fill in the needed values for it
 void loadMatchData(FeatureMatches* &matches, std::string pathToMatches)
{
        matches = new FeatureMatches();
        std::ifstream inFile(pathToMatches);
        std::string line;
        bool first = 1;
        int index = 0;
        while (getline( inFile, line))
        {
		std::cout<<"entering load match loop\n";
                debugPrint(line);
                if (first)
                {
                        first = 0;
                        //allocate mem needed for matched point data
                        matches->numMatches = (std::stoi(line) + 1);
                        matches->matches = new float4[matches->numMatches];
                        debugPrint("DYNAMICALLY ALLOCATING ON HOST... \n");
                        std::cout<<"Matches found: "<<matches->numMatches<<"\n";
			//debugPrint("read: " << line << ", generating: " << matches->numMatches);
                        // TODO potentially average the colors instead of just picking the first image?
                }
                else
                {
                        float4* match = &(matches->matches[index]);
                        parseMatchData(match, line, index);
                        index++;
                }
		std::cout<<"index="<<index<<"\n";
        } //end while
	std::cout<<"made it out of load matches\n";
}
void parseCameraData(Camera* &currentCamera, std::string line)
{
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> data;
        while (getline(ss, token, ','))
        {
                debugPrint(token);
                data.push_back(token);
        }
        currentCamera->val1 = std::stof(data[1]);
        currentCamera->val2 = std::stof(data[2]);
        currentCamera->val3 = std::stof(data[3]);
        currentCamera->val4 = std::stof(data[4]);
        currentCamera->val5 = std::stof(data[5]);
        currentCamera->val6 = std::stof(data[6]);
}
void loadCameraData(CameraData* &cameras, std::string pathToCameras)
{
        ConfigVals config();
        cameras = new CameraData();
        cameras->numCameras = 2; //automate
        cameras->cameras = new Camera[2]; //automate

        std::ifstream inFile(pathToCameras);
        std::string line;
        int index = 0;
        while (getline(inFile, line))
        {
                debugPrint(line);
                Camera* currentCam = &(cameras->cameras[index]);
                parseCameraData(currentCam, line);
                index++;
        }
}

//input arguments, but if more are added this thing will shine
struct ArgParser
{
        std::string cameraPath;
        std::string matchesPath;

        ArgParser( int argc, char* argv[]);  //constructor
};

//arg parser methods
ArgParser::ArgParser(int argc, char* argv[])
{
        std::cout << "*===================* REPROJECTION *===================*" << std::endl;
        if (argc < 3)
        {
                std::cout << "ERROR: not enough arguments ... " << std::endl;
    std::cout << "USAGE: " << std::endl;
                std::cout << "./reprojection.x path/to/cameras.txt path/to/matches.txt" << std::endl;
                std::cout << "*=========================================================*" << std::endl;
                exit(-1); //goodbye cruel world
        }
        std::cout << "*                                                      *" << std::endl;
  std::cout << "*                     ~ UGA SSRL ~                     *" << std::endl;
  std::cout << "*        Multiview Onboard Computational Imager        *" << std::endl;
  std::cout << "*                                                      *" << std::endl;
        std::cout << "*=========================================================*" << std::endl;

        this->cameraPath = argv[1];
        this->matchesPath = argv[2];
}

//this method sets up the memory on the gpu, calls the kernel, then gets the result
void twoViewReprojection(FeatureMatches* fMatches, CameraData* cData, PointCloud* &pointCloud)
{
	//prepare to move input data
	const int FEATURE_DATA_BYTES = fMatches->numMatches * sizeof(float4);
	const int CAMERA_DATA_BYTES = 3*sizeof(float);
  const int MATRIX_DAYA_BYTES = 9*sizeof(float);

	//calculate output size
	const int POINT_CLOUD_SIZE = fMatches->numMatches;
	const int POINT_CLOUD_BYTES = (POINT_CLOUD_SIZE * sizeof(float3));


	//allocate mem for the point cloud output on the cpu
	pointCloud = new PointCloud();
	pointCloud->numPoints = fMatches->numMatches;
	pointCloud->points = new float3[POINT_CLOUD_SIZE];

	//initiliaze camera matrices
	Camera cam1 = cData->cameras[0];
	Camera cam2 = cData->cameras[1];
	float cam1C[3] =
	{
		cam1.val1, cam1.val2, cam1.val3
	};
	float cam1V[3] =
	{
		-1*cam1.val4, -1*cam1.val5, -1*cam1.val6
	};
	float cam2C[3] =
	{
		cam2.val1, cam2.val2, cam2.val3
	};
	float cam2V[3] =
  {
	  -1*cam2.val4, -1*cam2.val5, -1*cam2.val6
  };

	//other matrix data needed by all threads
	float K[3][3];  // intrinsic camera matrix
	float K_inv[3][3];  // inverse of K

	K[0][0] = foc/dpix;
  K[0][1] = 0;
  K[0][2] = (float)(res/2.0);
  K[1][0] = 0;
  K[1][1] = foc/dpix;
  K[1][2] = (float)(res/2.0);
  K[2][0] = 0;
  K[2][1] = 0;
  K[2][2] = 1;
	inverse3x3_cpu(K, K_inv);

	float x;
  float y;
  float z;
  // Rotate cam1V about the x axis
  x = cam1V[0];
  y = cam1V[1];
  z = cam1V[2];
	float angle1;  // angle between cam1V and x axis
	if (abs(z) < .00001)
	{
		if (y > 0)
		{
			angle1 = PI/2;
		}
		else
		{
			angle1 = -1*PI/2;
		}
	}
	else
	{
		angle1 = atan(y/z);
		if (z < 0 && y >= 0)
		{
			angle1 +=PI;
		}
		if (z < 0 && y < 0)
		{
			angle1 -= PI;
		}
	}
	float A1[3][3] =
	{
		{1, 0, 0},
		{0, cos(angle1), -sin(angle1)},
		{0, sin(angle1), cos(angle1)}
	};

	float temp[3];
	//apply transform matrix we just got
	multiply3x3x1_cpu(A1, cam1V, temp);

	//rotate around the y axis
	x = temp[0];
	y = temp[1];
	z = temp[2];
	float angle2;  // angle between temp and y axis
	if (abs(z) < .00001)
	{
		if (x <= 0)
		{
			angle2 = PI/2;
		}else
		{
			angle2 = -1*PI/2;
		}
	}else
	{
		angle2 = atan(-1*x / z);
		if(z < 0 && x < 0)
		{
			angle2 += PI;
		}
		if(z < 0 && x > 0)
		{
			angle2 -= PI;
		}
	}

	float B1[3][3] =
	{
		{cos(angle2), 0, sin(angle2)},
    {0, 1, 0},
    {-sin(angle2), 0, cos(angle2)}
	};

	float rotCam1[3];
	// apply transformation matrix B. store in rotcam1
  multiply3x3x1_cpu(B1, temp, rotCam1);

	float rotationMatrix1[3][3];
	float rotationTranspose1[3][3];

	//get rotation matrix as a single transform matrix
	multiply3x3_cpu(B1, A1, rotationMatrix1);
	transpose_cpu(rotationMatrix1, rotationTranspose1);
	multiply3x3x1_cpu(rotationTranspose1, rotCam1, temp); // temp should be original cam1C now

	// Rotate cam2V about the x axis
  x = cam2V[0];
  y = cam2V[1];
  z = cam2V[2];

  if(abs(z) < .00001)
	{
    if(y > 0)
		{
			angle1 = PI/2;
		} else
		{
		  angle1 = -1*PI/2;
		}
  } else
	{
    angle1 = atan(y / z);
    if(z<0 && y>=0)
		{
      angle1 += PI;
    }
    if(z<0 && y<0)
		{
      angle1 -= PI;
    }
  }
  float A2[3][3] =
	{
    {1, 0, 0},
    {0, cos(angle1), -sin(angle1)},
    {0, sin(angle1), cos(angle1)}
  };
  // apply transformation matrix A
  multiply3x3x1_cpu(A2, cam2V, temp);

  // Rotate about the y axis
  x = temp[0];
  y = temp[1];
  z = temp[2];
  if(abs(z) < .00001)
	{
    if(x <= 0){
			angle2 = PI/2;
		}else
		{
			angle2 = -1*PI/2;
		}
  } else
	{
    angle2 = atan(-1*x / z);
    if(z<0 && x<0)
		{
      angle2 += PI;
    }
    if(z<0 && x>0)
		{
      angle2 -= PI;
    }
  }
  float B2[3][3] =
	{
    {cos(angle2), 0, sin(angle2)},
    {0, 1, 0},
    {-sin(angle2), 0, cos(angle2)}
  };
  // apply transformation matrix B
	float rotCam2[3];
  multiply3x3x1_cpu(B2, temp, rotCam2);

	float rotationMatrix2[3][3];
	float rotationTranspose2[3][3];

  // Get rotation matrix as a single transformation matrix
  multiply3x3_cpu(B2, A2, rotationMatrix2);
  transpose_cpu(rotationMatrix2, rotationTranspose2);
  multiply3x3x1_cpu(rotationTranspose2, rotCam2, temp); // temp should be original cam2C now

	//linearize matrices
	//position in linear matrix = 3*x +y, [x][y]
	float K_inv_lin[9];
	K_inv_lin[0] = K_inv[0][0];
	K_inv_lin[1] = K_inv[0][1];
	K_inv_lin[2] = K_inv[0][2];
	K_inv_lin[3] = K_inv[1][0];
	K_inv_lin[4] = K_inv[1][1];
	K_inv_lin[5] = K_inv[1][2];
	K_inv_lin[6] = K_inv[2][0];
	K_inv_lin[7] = K_inv[2][1];
	K_inv_lin[8] = K_inv[2][2];

	float rotTran1_lin[9];
	rotTran1_lin[0] = rotationTranspose1[0][0];
	rotTran1_lin[1] = rotationTranspose1[0][1];
	rotTran1_lin[2] = rotationTranspose1[0][2];
	rotTran1_lin[3] = rotationTranspose1[1][0];
	rotTran1_lin[4] = rotationTranspose1[1][1];
	rotTran1_lin[5] = rotationTranspose1[1][2];
	rotTran1_lin[6] = rotationTranspose1[2][0];
	rotTran1_lin[7] = rotationTranspose1[2][1];
	rotTran1_lin[8] = rotationTranspose1[2][2];

	float rotTran2_lin[9];
	rotTran2_lin[0] = rotationTranspose2[0][0];
	rotTran2_lin[1] = rotationTranspose2[0][1];
	rotTran2_lin[2] = rotationTranspose2[0][2];
	rotTran2_lin[3] = rotationTranspose2[1][0];
	rotTran2_lin[4] = rotationTranspose2[1][1];
	rotTran2_lin[5] = rotationTranspose2[1][2];
	rotTran2_lin[6] = rotationTranspose2[2][0];
	rotTran2_lin[7] = rotationTranspose2[2][1];
	rotTran2_lin[8] = rotationTranspose2[2][2];

	//initialize point cloud data to 0
	//pointCloud->points = new  float3[POINT_CLOUD_SIZE];
	float3* currentPoint;
	for(int i = 0; i < POINT_CLOUD_SIZE; ++i)
	{
		currentPoint = &(pointCloud->points[i]);
    currentPoint->x = 0.0f;
		currentPoint->y = 0.0f;
		currentPoint->z = 0.0f;
  }

	//create pointers on the device. d_ indicates pointer to mem on device
	float4* d_in_matches; 		 //where feature matches data is stored
	float* d_in_cam1C; 		 //where camera data is stored
	float* d_in_cam1V; 		 //where camera data is stored
	float* d_in_cam2C; 		 //where camera data is stored
	float* d_in_cam2V; 		 //where camera data is stored
	float* d_in_k_inv;
	float* d_in_rotTran1;

	float* d_in_rotTran2;
	float3* d_out_pointCloud; //where point cloud output is stored

	//allocate the mem on the gpu
	cudaMalloc((void**) &d_in_matches, FEATURE_DATA_BYTES);
	cudaMalloc((void**) &d_in_cam1C, CAMERA_DATA_BYTES);
	cudaMalloc((void**) &d_in_cam1V, CAMERA_DATA_BYTES);
	cudaMalloc((void**) &d_in_cam2C, CAMERA_DATA_BYTES);
	cudaMalloc((void**) &d_in_cam2V, CAMERA_DATA_BYTES);
	cudaMalloc((void**) &d_in_k_inv, MATRIX_DAYA_BYTES);
	cudaMalloc((void**) &d_in_rotTran1, MATRIX_DAYA_BYTES);
	cudaMalloc((void**) &d_in_rotTran2, MATRIX_DAYA_BYTES);
	cudaMalloc((void**) &d_out_pointCloud, POINT_CLOUD_BYTES);

	//transfer input data to mem on the gpu
	cudaMemcpy(d_in_matches, fMatches->matches, FEATURE_DATA_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_cam1C, cam1C, CAMERA_DATA_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_cam1V, cam1V, CAMERA_DATA_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_cam2C, cam2C, CAMERA_DATA_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_cam2V, cam2V, CAMERA_DATA_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_k_inv, K_inv_lin, MATRIX_DAYA_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_rotTran1, rotTran1_lin, MATRIX_DAYA_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_rotTran2, rotTran2_lin, MATRIX_DAYA_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out_pointCloud, pointCloud->points, POINT_CLOUD_BYTES, cudaMemcpyHostToDevice);

	//block and thread count
	dim3 THREAD_COUNT = {512, 1, 1};
	dim3 BLOCK_COUNT = {(unsigned int)ceil((POINT_CLOUD_SIZE+512)/512),1, 1}; //(unsigned int)ceil(POINT_CLOUD_SIZE/512)

	//call kernel
	two_view_reproject<<<BLOCK_COUNT, THREAD_COUNT>>>(POINT_CLOUD_SIZE, d_in_matches, d_in_cam1C, d_in_cam1V, d_in_cam2C, d_in_cam2V, d_in_k_inv, d_in_rotTran1, d_in_rotTran2, d_out_pointCloud);

	//error check
	CudaCheckError();

	//get result
	cudaMemcpy(pointCloud->points, d_out_pointCloud, POINT_CLOUD_BYTES, cudaMemcpyDeviceToHost);

	pointCloud->numPoints = fMatches->numMatches;
	//free mem on gpu
	cudaFree(d_in_matches);
	cudaFree(d_in_cam1C);
	cudaFree(d_in_cam1V);
	cudaFree(d_in_cam2C);
	cudaFree(d_in_cam2V);
	cudaFree(d_out_pointCloud);
}
