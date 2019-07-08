// #include "MatchFactory.cuh"
//
//
// __constant__ int splineHelper[4][4] = {
//   {1,0,0,0},
//   {0,0,1,0},
//   {-3,3,-2,-1},
//   {2,-2,1,1}
// };
// __constant__ int splineHelperInv[4][4] = {
//   {1,0,-3,2},
//   {0,0,3,-2},
//   {0,1,-2,1},
//   {0,0,-1,1}
// };
//
// __device__ __host__ __forceinline__ float sum(const float3 &a){
//   return a.x + a.y + a.z;
// }
// __device__ __forceinline__ float square(const float &a){
//   return a*a;
// }
// __device__ __forceinline__ float calcElucid(const int2 &a, const int2 &b){
//   return sqrtf(dotProduct(a-b, a-b));
// }
// __device__ __forceinline__ float calcElucid_SIFTDescriptor(const unsigned char a[128], const unsigned char b[128]){
//   float dist = 0.0f;
//   for(int i = 0; i < 128; ++i){
//     dist += sqrtf(((float)(a[i] - b[i]))*((float)(a[i] - b[i])));
//   }
//   return dist;
// }
// __device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
//   float old;
//   old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
//     __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
//   return old;
// }
// __device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
//   float old;
//   old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
//     __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
//   return old;
// }
// __device__ __forceinline__ int findSubPixelContributer(const int2 &loc, const int &width){
//   return ((loc.y - 12)*(width - 24)) + (loc.x - 12);
// }
//
// /*
// matching
// */
// __global__ void matchFeaturesPairwiseBruteForce(int numFeaturesQuery, int numOrientationsQuery,
// int numFeaturesTarget, int numOrientationsTarget, SIFT_Descriptor* descriptorsQuery, SIFT_Feature* featuresQuery,
// SIFT_Descriptor* descriptorsTarget, SIFT_Feature* featuresTarget, Match* matches){
//   unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
//   int2 queryNumFeatures = {numFeaturesQuery, numOrientationsQuery};
//   int2 targetNumFeatures = {numFeaturesTarget, numOrientationsTarget};
//   if(blockId < queryNumFeatures.x*queryNumFeatures.y){
//     SIFT_Feature feature = featuresQuery[blockId/queryNumFeatures.y];
//     unsigned char descriptor[128] = {0};
//     for(int i = 0; i < 128; ++i){
//       descriptor[i] = descriptorsQuery[blockId].descriptor[i];
//     }
//     __shared__ int localMatch[1024];
//     __shared__ float localDist[1024];
//     localMatch[threadIdx.x] = -1;
//     localDist[threadIdx.x] = FLT_MAX;
//     __syncthreads();
//     float currentDist = 0.0f;
//     for(int f = threadIdx.x; f < targetNumFeatures.x*targetNumFeatures.y; f += 1024){
//       currentDist = 0.0f;
//       for(int i = 0; i < 128; ++i){
//         currentDist +=  square(((float)descriptor[i])-((float)descriptorsTarget[f].descriptor[i]));
//       }
//       if(localDist[threadIdx.x] > currentDist){
//         localDist[threadIdx.x] = currentDist;
//         localMatch[threadIdx.x] = f/targetNumFeatures.y;
//       }
//     }
//     __syncthreads();
//     if(threadIdx.x != 0) return;
//     currentDist = FLT_MAX;
//     int matchIndex = -1;
//     for(int i = 0; i < 1024; ++i){
//       if(currentDist > localDist[i]){
//         currentDist = localDist[i];
//         matchIndex = localMatch[i];
//       }
//     }
//     Match match;
//     match.features[0] = feature;
//     match.features[1] = featuresTarget[matchIndex];
//     match.distance = currentDist;
//     matches[blockId] = match;
//   }
// }
// __global__ void matchFeaturesPairwiseConstrained(int numFeaturesQuery, int numOrientationsQuery,
// int numFeaturesTarget, int numOrientationsTarget, SIFT_Descriptor* descriptorsQuery, SIFT_Feature* featuresQuery,
// SIFT_Descriptor* descriptorsTarget, SIFT_Feature* featuresTarget, Match* matches, float epsilon, float3* fundamental){
//   unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
//   int2 queryNumFeatures = {numFeaturesQuery, numOrientationsQuery};
//   int2 targetNumFeatures = {numFeaturesTarget, numOrientationsTarget};
//   float regEpsilon = epsilon;
//   if(blockId < queryNumFeatures.x*queryNumFeatures.y){
//     SIFT_Feature feature = featuresQuery[blockId/queryNumFeatures.y];
//     unsigned char descriptor[128] = {0};
//     for(int i = 0; i < 128; ++i){
//       descriptor[i] = descriptorsQuery[blockId].descriptor[i];
//     }
//     __shared__ int localMatch[1024];
//     __shared__ float localDist[1024];
//     localMatch[threadIdx.x] = -1;
//     localDist[threadIdx.x] = FLT_MAX;
//     __syncthreads();
//     float currentDist = 0.0f;
//
//     float3 epipolar = {0.0f,0.0f,0.0f};
//     epipolar.x = (fundamental[0].x*feature.loc.x) + (fundamental[0].y*feature.loc.y) + fundamental[0].z;
//     epipolar.y = (fundamental[1].x*feature.loc.x) + (fundamental[1].y*feature.loc.y) + fundamental[1].z;
//     epipolar.z = (fundamental[2].x*feature.loc.x) + (fundamental[2].y*feature.loc.y) + fundamental[2].z;
//
//     float p = 0.0f;
//
//     SIFT_Feature currentFeature;
//
//     for(int f = threadIdx.x; f < targetNumFeatures.x*targetNumFeatures.y; f += 1024){
//
//       currentFeature = featuresTarget[f/targetNumFeatures.y];
//       //ax + by + c = 0
//       p = -1*((epipolar.x*currentFeature.loc.x) + epipolar.z)/epipolar.y;
//       if(abs(currentFeature.loc.y - p) >= regEpsilon) continue;
//       currentDist = 0.0f;
//       for(int i = 0; i < 128; ++i){
//         currentDist +=  square(((float)descriptor[i])-((float)descriptorsTarget[f].descriptor[i]));
//       }
//       if(localDist[threadIdx.x] > currentDist){
//         localDist[threadIdx.x] = currentDist;
//         localMatch[threadIdx.x] = f/targetNumFeatures.y;
//       }
//     }
//     __syncthreads();
//     if(threadIdx.x != 0) return;
//     currentDist = FLT_MAX;
//     int matchIndex = -1;
//     for(int i = 0; i < 1024; ++i){
//       if(currentDist > localDist[i]){
//         currentDist = localDist[i];
//         matchIndex = localMatch[i];
//       }
//     }
//     Match match;
//     match.features[0] = feature;
//     match.features[1] = featuresTarget[matchIndex];
//     match.distance = currentDist;
//     matches[blockId] = match;
//   }
// }
//
//
// /*
// subpixel stuff
// */
// //TODO overload this kernel for different types of descriptors
//
// //NOTE THIS MIGHT ONLY WORK FOR DENSE SIFT
// __global__ void initializeSubPixels(Image_Descriptor query, Image_Descriptor target, unsigned long numMatches, Match* matches, SubPixelMatch* subPixelMatches, SubpixelM7x7* subPixelDescriptors,
// SIFT_Descriptor* queryDescriptors, int numFeaturesQuery, int numDescriptorsPerFeatureQuery, SIFT_Descriptor* targetDescriptors, int numFeaturesTarget, int numDescriptorsPerFeatureTarget){
//   unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
//   if(blockId < numMatches){
//     __shared__ SubpixelM7x7 subDescriptor;
//     SubPixelMatch subPixelMatch;
//     Match match = matches[blockId];
//     subPixelMatch.features[0] = match.features[0];
//     subPixelMatch.features[1] = match.features[1];
//     subPixelMatch.distance = match.distance;
//     int featuresInTarget = numFeaturesTarget;
//     int featuresInQuery = numFeaturesQuery;
//
//     //this now needs to be actual indices to contributers
//     int2 contrib = {((int)threadIdx.x) - 4, ((int)threadIdx.y) - 4};
//     int contribQuery = findSubPixelContributer(match.features[0].loc + contrib, query.size.x);
//     int contribTarget = findSubPixelContributer(match.features[1].loc + contrib, target.size.x);
//     int numOrientQuery = numDescriptorsPerFeatureQuery;
//     int numOrientTarget = numDescriptorsPerFeatureTarget;
//
//     float temp = 0.0f;
//     int pairedMatchIndex = findSubPixelContributer(match.features[1].loc, target.size.x);
//
//     for(int i = 0; i < numOrientQuery; ++i){
//       for(int j = 0; j < numOrientTarget; ++j){
//         if(contribTarget >= 0 && contribTarget < featuresInTarget){
//           temp = calcElucid_SIFTDescriptor(queryDescriptors[blockId*numDescriptorsPerFeatureQuery + i].descriptor, targetDescriptors[contribTarget*numDescriptorsPerFeatureTarget + j].descriptor);
//         }
//         if(subDescriptor.M1[threadIdx.x][threadIdx.y] > temp){
//           subDescriptor.M1[threadIdx.x][threadIdx.y] = temp;
//         }
//         if(contribQuery >= 0 && contribQuery < featuresInQuery){
//           temp = calcElucid_SIFTDescriptor(queryDescriptors[contribQuery*numDescriptorsPerFeatureQuery + i].descriptor, targetDescriptors[pairedMatchIndex*numDescriptorsPerFeatureTarget + j].descriptor);
//         }
//         if(subDescriptor.M2[threadIdx.x][threadIdx.y] > temp){
//           subDescriptor.M2[threadIdx.x][threadIdx.y] = temp;
//         }
//       }
//     }
//     if(threadIdx.x == 0 && threadIdx.y == 0){
//       subPixelMatch.subLocations[0] = {0.0f,0.0f};
//       subPixelMatch.subLocations[1] = {0.0f,0.0f};
//       subPixelMatches[blockId] = subPixelMatch;
//       subPixelDescriptors[blockId] = subDescriptor;
//     }
//   }
// }
//
// __global__ void fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, Spline* splines){
//   unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
//   if(blockId < numMatches*2){
//     float descriptor[9][9];
//     for(int x = 0; x < 9; ++x){
//       for(int y = 0; y < 9; ++y){
//         descriptor[x][y] = (blockId%2 == 0) ? subPixelDescriptors[blockId/2].M1[x][y] : subPixelDescriptors[blockId/2].M2[x][y];
//       }
//     }
//
//     __shared__ Spline spline;
//     int2 corner = {
//       ((int)threadIdx.z)%2,
//       ((int)threadIdx.z)/2
//     };
//     int2 contributer = {
//       ((int)threadIdx.x) + 2 + corner.x,
//       ((int)threadIdx.y) + 2 + corner.y
//     };
//     float4 localCoeff;
//     localCoeff.x = descriptor[contributer.x][contributer.y];
//     localCoeff.y = descriptor[contributer.x + 1][contributer.y] - descriptor[contributer.x - 1][contributer.y];
//     localCoeff.z = descriptor[contributer.x][contributer.y + 1] - descriptor[contributer.x][contributer.y - 1];
//     localCoeff.w = descriptor[contributer.x + 1][contributer.y + 1] - descriptor[contributer.x - 1][contributer.y - 1];
//
//     spline.coeff[threadIdx.x][threadIdx.y][corner.x][corner.y] = localCoeff.x;
//     spline.coeff[threadIdx.x][threadIdx.y][corner.x][corner.y + 2] = localCoeff.y;
//     spline.coeff[threadIdx.x][threadIdx.y][corner.x + 2][corner.y] = localCoeff.z;
//     spline.coeff[threadIdx.x][threadIdx.y][corner.x + 2][corner.y + 2] = localCoeff.z;
//
//     // Multiplying matrix a and b and storing in array mult.
//     if(threadIdx.z != 0) return;
//     float mult[4][4] = {0.0f};
//     for(int i = 0; i < 4; ++i){
//       for(int j = 0; j < 4; ++j){
//         for(int c = 0; c < 4; ++c){
//           mult[i][j] += splineHelper[i][c]*spline.coeff[threadIdx.x][threadIdx.y][c][j];
//         }
//       }
//     }
//     for(int i = 0; i < 4; ++i){
//       for(int j = 0; j < 4; ++j){
//         spline.coeff[threadIdx.x][threadIdx.y][i][j] = 0.0f;
//       }
//     }
//     for(int i = 0; i < 4; ++i){
//       for(int j = 0; j < 4; ++j){
//         for(int c = 0; c < 4; ++c){
//           spline.coeff[threadIdx.x][threadIdx.y][i][j] += mult[i][c]*splineHelperInv[c][j];
//         }
//       }
//     }
//
//     __syncthreads();
//     splines[blockId] = spline;
//   }
// }
// __global__ void determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, SubPixelMatch* subPixelMatches, Spline* splines){
//   unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
//   if(blockId < numMatches*2){
//     __shared__ float minimum;
//     minimum = FLT_MAX;
//     __syncthreads();
//     float localCoeff[4][4];
//     for(int i = 0; i < 4; ++i){
//       for(int j = 0; j < 4; ++j){
//         localCoeff[i][j] = splines[blockId].coeff[threadIdx.x][threadIdx.y][i][j];
//       }
//     }
//     float value = 0.0f;
//     float localMin = FLT_MAX;
//     float2 localSubLoc = {0.0f,0.0f};
//     for(float x = -1.0f; x <= 1.0f; x+=increment){
//       for(float y = -1.0f; y <= 1.0f; y+=increment){
//         value = 0.0f;
//         for(int i = 0; i < 4; ++i){
//           for(int j = 0; j < 4; ++j){
//             value += (localCoeff[i][j]*powf(x,i)*powf(y,j));
//           }
//         }
//         if(value < localMin){
//           localMin = value;
//           localSubLoc = {x,y};
//         }
//       }
//     }
//     atomicMinFloat(&minimum, localMin);
//     __syncthreads();
//     if(localMin == minimum){
//       if(blockId%2 == 0) subPixelMatches[blockId/2].subLocations[0]  = localSubLoc + subPixelMatches[blockId/2].features[0].loc;
//       else subPixelMatches[blockId/2].subLocations[1] = localSubLoc + subPixelMatches[blockId/2].features[1].loc;
//     }
//     else return;
//   }
// }
//
// /*
// MATCH REFINEMENT
// */
//
// __global__ void refineWCutoffRatio(int numMatches, Match* matches, int* matchCounter, float2 minMax, float cutoffRatio){
//   int globalId = blockIdx.x*blockDim.x + threadIdx.x;
//   if(globalId < numMatches){
//     float2 regMinMax = minMax;
//     if((matches[globalId].distance - regMinMax.x)/(regMinMax.y-regMinMax.x) < cutoffRatio){
//       matchCounter[globalId] = 1;
//     }
//     else{
//       matchCounter[globalId] = 0;
//     }
//   }
// }
// __global__ void refineWCutoffRatio(int numMatches, SubPixelMatch* matches, int* matchCounter, float2 minMax, float cutoffRatio){
//   int globalId = blockIdx.x*blockDim.x + threadIdx.x;
//   if(globalId < numMatches){
//     float2 regMinMax = minMax;
//     if((matches[globalId].distance - regMinMax.x)/(regMinMax.y-regMinMax.x) < cutoffRatio){
//       matchCounter[globalId] = 1;
//     }
//     else{
//       matchCounter[globalId] = 0;
//     }
//   }
// }
// __global__ void copyMatches(int numMatches, int* matchCounter, Match* minimizedMatches, Match* matches){
//   int globalId = blockIdx.x*blockDim.x + threadIdx.x;
//   if(globalId < numMatches){
//     int counterVal = matchCounter[globalId];
//     if(counterVal != 0 && counterVal > matchCounter[globalId - 1]){
//       minimizedMatches[counterVal - 1] = matches[globalId];
//     }
//   }
// }
// __global__ void copyMatches(int numMatches, int* matchCounter, SubPixelMatch* minimizedMatches, SubPixelMatch* matches){
//   int globalId = blockIdx.x*blockDim.x + threadIdx.x;
//   if(globalId < numMatches){
//     int counterVal = matchCounter[globalId];
//     if(counterVal != 0 && counterVal > matchCounter[globalId - 1]){
//       minimizedMatches[counterVal - 1] = matches[globalId];
//     }
//   }
// }
//
// /*
// fundamental matrix stuff
// */
// float3 multiply3x3x1(const float3 A[3], const float3 &B) {
//   return {sum(A[0]*B),sum(A[1]*B),sum(A[2]*B)};
// }
// void multiply3x3(const float3 A[3], const float3 B[3], float3 *C) {
//   float3 bX = {B[0].x,B[1].x,B[2].x};
//   float3 bY = {B[0].y,B[1].y,B[2].y};
//   float3 bZ = {B[0].z,B[1].z,B[2].z};
//   C[0] = {sum(A[0]*bX),sum(A[0]*bY),sum(A[0]*bZ)};
//   C[1] = {sum(A[1]*bX),sum(A[1]*bY),sum(A[1]*bZ)};
//   C[2] = {sum(A[2]*bX),sum(A[2]*bY),sum(A[2]*bZ)};
// }
// void transpose3x3(const float3 M[3], float3 (&M_T)[3]) {
//   M_T[0] = {M[0].x,M[1].x,M[2].x};
//   M_T[1] = {M[0].y,M[1].y,M[2].y};
//   M_T[2] = {M[0].z,M[1].z,M[2].z};
// }
// void inverse3x3(float3 M[3], float3 (&Minv)[3]) {
//   float d1 = M[1].y * M[2].z - M[2].y * M[1].z;
//   float d2 = M[1].x * M[2].z - M[1].z * M[2].x;
//   float d3 = M[1].x * M[2].y - M[1].y * M[2].x;
//   float det = M[0].x*d1 - M[0].y*d2 + M[0].z*d3;
//   if(det == 0) {
//     // return pinv(M);
//   }
//   float invdet = 1/det;
//   Minv[0].x = d1*invdet;
//   Minv[0].y = (M[0].z*M[2].y - M[0].y*M[2].z) * invdet;
//   Minv[0].z = (M[0].y*M[1].z - M[0].z*M[1].y) * invdet;
//   Minv[1].x = -1 * d2 * invdet;
//   Minv[1].y = (M[0].x*M[2].z - M[0].z*M[2].x) * invdet;
//   Minv[1].z = (M[1].x*M[0].z - M[0].x*M[1].z) * invdet;
//   Minv[2].x = d3 * invdet;
//   Minv[2].y = (M[2].x*M[0].y - M[0].x*M[2].y) * invdet;
//   Minv[2].z = (M[0].x*M[1].y - M[1].x*M[0].y) * invdet;
// }
// void calcFundamentalMatrix_2View(Image_Descriptor query, Image_Descriptor target, float3 *F){
//   if(query.fov != target.fov || query.foc != target.foc){
//     std::cout<<"ERROR calculating fundamental matrix for 2view needs to bet taken with same camera (foc&fov are same)"<<std::endl;
//     exit(-1);
//   }
//   float angle1;
//   if(abs(query.cam_vec.z) < .00001) {
//     if(query.cam_vec.y > 0)  angle1 = PI/2;
//     else       angle1 = -1*PI/2;
//   }
//   else {
//     angle1 = atan(query.cam_vec.y / query.cam_vec.z);
//     if(query.cam_vec.z<0 && query.cam_vec.y>=0) {
//       angle1 += PI;
//     }
//     if(query.cam_vec.z<0 && query.cam_vec.y<0) {
//       angle1 -= PI;
//     }
//   }
//   float3 A1[3] = {
//     {1, 0, 0},
//     {0, cos(angle1), -sin(angle1)},
//     {0, sin(angle1), cos(angle1)}
//   };
//
//   float3 temp = multiply3x3x1(A1, query.cam_vec);
//
//   float angle2 = 0.0f;
//   if(abs(temp.z) < .00001) {
//     if(temp.x <= 0)  angle1 = PI/2;
//     else       angle1 = -1*PI/2;
//   }
//   else {
//     angle2 = atan(-1*temp.x / temp.z);
//     if(temp.z<0 && temp.x<0) {
//       angle1 += PI;
//     }
//     if(temp.z<0 && temp.x>0) {
//       angle2 -= PI;
//     }
//   }
//   float3 B1[3] = {
//     {cos(angle2), 0, sin(angle2)},
//     {0, 1, 0},
//     {-sin(angle2), 0, cos(angle2)}
//   };
//
//   float3 temp2 = multiply3x3x1(B1, temp);
//   float3 rot1[3];
//   multiply3x3(B1, A1, rot1);
//   float3 rot1Transpose[3];
//   transpose3x3(rot1,rot1Transpose);
//   temp = multiply3x3x1(rot1Transpose, temp2);
//
//   angle1 = 0.0f;
//   if(abs(target.cam_vec.z) < .00001) {
//     if(target.cam_vec.y > 0)  angle1 = PI/2;
//     else       angle1 = -1*PI/2;
//   }
//   else {
//     angle1 = atan(target.cam_vec.y / target.cam_vec.z);
//     if(target.cam_vec.z<0 && target.cam_vec.y>=0) {
//       angle1 += PI;
//     }
//     if(target.cam_vec.z<0 && target.cam_vec.y<0) {
//       angle1 -= PI;
//     }
//   }
//   float3 A2[3] = {
//     {1, 0, 0},
//     {0, cos(angle1), -sin(angle1)},
//     {0, sin(angle1), cos(angle1)}
//   };
//   temp2 = multiply3x3x1(A2, target.cam_vec);
//
//   angle2 = 0.0f;
//   if(abs(temp2.z) < .00001) {
//     if(temp2.x <= 0)  angle1 = PI/2;
//     else       angle1 = -1*PI/2;
//   }
//   else {
//     angle2 = atan(-1*temp2.x / temp2.z);
//     if(temp2.z<0 && temp2.x<0) {
//       angle1 += PI;
//     }
//     if(temp2.z<0 && temp2.x>0) {
//       angle2 -= PI;
//     }
//   }
//   float3 B2[3] = {
//     {cos(angle2), 0, sin(angle2)},
//     {0, 1, 0},
//     {-sin(angle2), 0, cos(angle2)}
//   };
//
//   temp = multiply3x3x1(B2, temp2);
//
//   float3 rot2[3];
//   multiply3x3(B2, A2, rot2);
//   float3 rot2Transpose[3];
//   transpose3x3(rot2, rot2Transpose);
//
//   temp2 = multiply3x3x1(rot2Transpose, temp);
//
//   float2 dpix = {query.foc*tan(query.fov/2)/(query.size.x/2),
//     query.foc*tan(query.fov/2)/(query.size.y/2)};
//
//   float3 K[3] = {
//     {query.foc/dpix.x, 0, ((float)query.size.x)/2.0f},
//     {0, query.foc/dpix.y, ((float)query.size.y)/2.0f},
//     {0, 0, 1}
//   };
//   float3 K_inv[3];
//   inverse3x3(K,K_inv);
//   float3 K_invTranspose[3];
//   transpose3x3(K_inv,K_invTranspose);
//
//   float3 R[3];
//   multiply3x3(rot2Transpose, rot1, R);
//   float3 S[3] = {
//     {0, query.cam_pos.z - target.cam_pos.z, target.cam_pos.y - query.cam_pos.y},
//     {query.cam_pos.z - target.cam_pos.z,0, query.cam_pos.x - target.cam_pos.x},
//     {query.cam_pos.y - target.cam_pos.y, target.cam_pos.x - query.cam_pos.x, 0}
//   };
//   float3 E[3];;
//   multiply3x3(R,S,E);
//   float3 tempF[3];
//   multiply3x3(K_invTranspose, E,tempF);
//   multiply3x3(tempF, K_inv, F);
//   std::cout << std::endl <<"between image "<<query.id<<" and "<<target.id
//   <<" the final fundamental matrix result is: " << std::endl;
//   for(int r = 0; r < 3; ++r) {
//     std::cout << F[r].x << "  " << F[r].y << " "<<  F[r].z << std::endl;
//   }
//   std::cout<<std::endl;
// }
//
// MatchFactory::MatchFactory(){
//   this->numImages = 0;
//   this->cutoffRatio = 0.0f;
// }
//
// void MatchFactory::setCutOffRatio(float cutoffRatio){
//   this->cutoffRatio = cutoffRatio;
// }
//
// //TODO consider using a defualt cutoff if not set
// void MatchFactory::refineMatches(MatchSet* matchSet){
//   if(this->cutoffRatio == 0.0f){
//     std::cout<<"ERROR not cutoff ratio set for refinement"<<std::endl;
//     exit(-1);
//   }
//   Match* matches = nullptr;
//   Match* matches_device = nullptr;
//   if(matchSet->memoryState == gpu){
//     matches = new Match[matchSet->numMatches];
//     CudaSafeCall(cudaMemcpy(matches, matchSet->matches, matchSet->numMatches*sizeof(Match),cudaMemcpyDeviceToHost));
//     matches_device = matchSet->matches;
//   }
//   else{
//     CudaSafeCall(cudaMalloc((void**)&matches_device, matchSet->numMatches*sizeof(Match)));
//     CudaSafeCall(cudaMemcpy(matches_device, matchSet->matches, matchSet->numMatches*sizeof(Match),cudaMemcpyHostToDevice));
//     matches = matchSet->matches;
//   }
//   float max = 0.0f;
//   float min = FLT_MAX;
//   for(int i = 0; i < matchSet->numMatches; ++i){
//     if(matches[i].distance < min) min = matches[i].distance;
//     if(matches[i].distance > max) max = matches[i].distance;
//   }
//
//   delete[] matches;
//
//   printf("max dist = %f || min dist = %f\n",max,min);
//   int* matchCounter_device = nullptr;
//   CudaSafeCall(cudaMalloc((void**)&matchCounter_device, matchSet->numMatches*sizeof(int)));
//
//   dim3 grid = {1,1,1};
//   dim3 block = {1,1,1};
//   getFlatGridBlock((unsigned long) matchSet->numMatches, grid, block);
//   refineWCutoffRatio<<<grid,block>>>(matchSet->numMatches, matches_device, matchCounter_device, {min, max}, this->cutoffRatio);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//
//   thrust::device_ptr<int> sum(matchCounter_device);
//   thrust::inclusive_scan(sum, sum + matchSet->numMatches, sum);
//   unsigned long beforeCompaction = matchSet->numMatches;
//   CudaSafeCall(cudaMemcpy(&(matchSet->numMatches),matchCounter_device + (beforeCompaction - 1), sizeof(int), cudaMemcpyDeviceToHost));
//
//   Match* minimizedMatches_device = nullptr;
//   CudaSafeCall(cudaMalloc((void**)&minimizedMatches_device, matchSet->numMatches*sizeof(Match)));
//
//   copyMatches<<<grid,block>>>(beforeCompaction, matchCounter_device, minimizedMatches_device, matches_device);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//
//   // thrust::device_ptr<Match> arrayToCompact(matches_device);
//   // thrust::device_ptr<Match> arrayOut(minimizedMatches_device);
//   // thrust::copy_if(arrayToCompact, arrayToCompact + beforeCompaction, arrayOut, match_above_cutoff());
//   // CudaCheckError();
//
//   CudaSafeCall(cudaFree(matchCounter_device));
//   CudaSafeCall(cudaFree(matches_device));
//
//   printf("numMatches after eliminating base on %f cutoffRatio = %d (was %d)\n",this->cutoffRatio,matchSet->numMatches,beforeCompaction);
//
//   if(matchSet->memoryState == gpu){
//     matchSet->matches = minimizedMatches_device;
//   }
//   else if(matchSet->memoryState == cpu){
//     matchSet->matches = new Match[matchSet->numMatches];
//     CudaSafeCall(cudaMemcpy(matchSet->matches, minimizedMatches_device, matchSet->numMatches*sizeof(Match),cudaMemcpyDeviceToHost));
//     CudaSafeCall(cudaFree(minimizedMatches_device));
//   }
// }
// void MatchFactory::refineMatches(SubPixelMatchSet* matchSet){
//   if(this->cutoffRatio == 0.0f){
//     std::cout<<"ERROR not cutoff ratio set for refinement"<<std::endl;
//     exit(-1);
//   }
//   SubPixelMatch* matches = nullptr;
//   SubPixelMatch* matches_device = nullptr;
//   if(matchSet->memoryState == gpu){
//     matches = new SubPixelMatch[matchSet->numMatches];
//     CudaSafeCall(cudaMemcpy(matches, matchSet->matches, matchSet->numMatches*sizeof(SubPixelMatch),cudaMemcpyDeviceToHost));
//     matches_device = matchSet->matches;
//   }
//   else{
//     CudaSafeCall(cudaMalloc((void**)&matches_device, matchSet->numMatches*sizeof(SubPixelMatch)));
//     CudaSafeCall(cudaMemcpy(matches_device, matchSet->matches, matchSet->numMatches*sizeof(SubPixelMatch),cudaMemcpyHostToDevice));
//     matches = matchSet->matches;
//   }
//   float max = 0.0f;
//   float min = FLT_MAX;
//   for(int i = 0; i < matchSet->numMatches; ++i){
//     if(matches[i].distance < min) min = matches[i].distance;
//     if(matches[i].distance > max) max = matches[i].distance;
//   }
//
//   delete[] matches;
//
//   printf("max dist = %f || min dist = %f\n",max,min);
//   int* matchCounter_device = nullptr;
//   CudaSafeCall(cudaMalloc((void**)&matchCounter_device, matchSet->numMatches*sizeof(int)));
//
//   dim3 grid = {1,1,1};
//   dim3 block = {1,1,1};
//   getFlatGridBlock((unsigned long) matchSet->numMatches, grid, block);
//   refineWCutoffRatio<<<grid,block>>>(matchSet->numMatches, matches_device, matchCounter_device, {min, max}, this->cutoffRatio);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//
//   thrust::device_ptr<int> sum(matchCounter_device);
//   thrust::inclusive_scan(sum, sum + matchSet->numMatches, sum);
//   unsigned long beforeCompaction = matchSet->numMatches;
//   CudaSafeCall(cudaMemcpy(&(matchSet->numMatches),matchCounter_device + (beforeCompaction - 1), sizeof(int), cudaMemcpyDeviceToHost));
//
//   SubPixelMatch* minimizedMatches_device = nullptr;
//   CudaSafeCall(cudaMalloc((void**)&minimizedMatches_device, matchSet->numMatches*sizeof(SubPixelMatch)));
//
//   copyMatches<<<grid,block>>>(beforeCompaction, matchCounter_device, minimizedMatches_device, matches_device);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//
//   CudaSafeCall(cudaFree(matchCounter_device));
//   CudaSafeCall(cudaFree(matches_device));
//
//   // thrust::device_ptr<SubPixelMatch> arrayToCompact(matches_device);
//   // thrust::device_ptr<SubPixelMatch> arrayOut(minimizedMatches_device);
//   // thrust::copy_if(arrayToCompact, arrayToCompact + beforeCompaction, arrayOut, match_above_cutoff());
//   // CudaCheckError();
//
//   printf("numMatches after eliminating base on %f cutoffRatio = %d (was %d)\n",this->cutoffRatio,matchSet->numMatches,beforeCompaction);
//
//   if(matchSet->memoryState == gpu){
//     matchSet->matches = minimizedMatches_device;
//   }
//   else if(matchSet->memoryState == cpu){
//     matchSet->matches = new SubPixelMatch[matchSet->numMatches];
//     CudaSafeCall(cudaMemcpy(matchSet->matches, minimizedMatches_device, matchSet->numMatches*sizeof(SubPixelMatch),cudaMemcpyDeviceToHost));
//     CudaSafeCall(cudaFree(minimizedMatches_device));
//   }
// }
//
// void MatchFactory::generateMatchesPairwiseBruteForce(Image* query, Image* target, MatchSet* &matchSet, MemoryState return_state){
//   if(return_state != cpu && return_state != gpu){
//     std::cout<<"ERROR can only return return_state 1=host 2=device"<<std::endl;
//     exit(-1);
//   }
//   if(query->numDescriptorsPerFeature == 0 || target->numDescriptorsPerFeature == 0){
//     std::cout<<"ERROR must have a numDescriptorsPerFeature of more than 0 - no descriptors"<<std::endl;
//     exit(-1);
//   }
//
//   SIFT_Feature* queryFeatures_device;
//   SIFT_Feature* targetFeatures_device;
//   SIFT_Descriptor* queryDescriptors_device;
//   SIFT_Descriptor* targetDescriptors_device;
//
//   if(query->arrayStates[1] != gpu && query->arrayStates[1] != both){
//     if(query->arrayStates[1] != cpu){
//       std::cout<<"ERROR query does not have features"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&queryFeatures_device, query->numFeatures*sizeof(SIFT_Feature)));
//     CudaSafeCall(cudaMemcpy(queryFeatures_device, query->features, query->numFeatures*sizeof(SIFT_Feature), cudaMemcpyHostToDevice));
//   }
//   else queryFeatures_device = query->features_device;
//
//   if(target->arrayStates[1] != gpu && target->arrayStates[1] != both){
//     if(target->arrayStates[1] != cpu){
//       std::cout<<"ERROR target does not have features"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&targetFeatures_device, target->numFeatures*sizeof(SIFT_Feature)));
//     CudaSafeCall(cudaMemcpy(targetFeatures_device, target->features, target->numFeatures*sizeof(SIFT_Feature), cudaMemcpyHostToDevice));
//   }
//   else targetFeatures_device = target->features_device;
//
//   if(query->arrayStates[2] != gpu && query->arrayStates[2] != both){
//     if(query->arrayStates[2] != cpu){
//       std::cout<<"ERROR query does not have descriptors"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&queryDescriptors_device, query->numDescriptorsPerFeature*query->numFeatures*sizeof(SIFT_Descriptor)));
//     CudaSafeCall(cudaMemcpy(queryDescriptors_device, query->featureDescriptors, query->numDescriptorsPerFeature*query->numFeatures*sizeof(SIFT_Descriptor), cudaMemcpyHostToDevice));
//   }
//   else queryDescriptors_device = query->featureDescriptors_device;
//
//   if(target->arrayStates[2] != gpu && target->arrayStates[2] != both){
//     if(target->arrayStates[2] != cpu){
//       std::cout<<"ERROR target does not have descriptors"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&targetDescriptors_device, target->numDescriptorsPerFeature*target->numFeatures*sizeof(SIFT_Descriptor)));
//     CudaSafeCall(cudaMemcpy(targetDescriptors_device, target->featureDescriptors, target->numDescriptorsPerFeature*target->numFeatures*sizeof(SIFT_Descriptor), cudaMemcpyHostToDevice));
//   }
//   else targetDescriptors_device = target->featureDescriptors_device;
//
//   unsigned int numPossibleMatches = query->numFeatures;
//
//   matchSet = new MatchSet();
//   matchSet->numMatches = numPossibleMatches;
//   matchSet->memoryState = return_state;
//   Match* matches_device = nullptr;
//   Match* matches = nullptr;
//   CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));
//   matches = new Match[numPossibleMatches];
//
//   dim3 grid = {1,1,1};
//   dim3 block = {1024,1,1};
//   getGrid(numPossibleMatches,grid);
//
//   clock_t timer = clock();
//
//   matchFeaturesPairwiseBruteForce<<<grid, block>>>(query->numFeatures,
//     query->numDescriptorsPerFeature, target->numFeatures,
//     target->numDescriptorsPerFeature, queryDescriptors_device, queryFeatures_device,
//     targetDescriptors_device, targetFeatures_device, matches_device);
//
//   cudaDeviceSynchronize();
//   CudaCheckError();
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//   switch(return_state){
//     case cpu:
//       CudaSafeCall(cudaMemcpy(matches, matches_device, numPossibleMatches*sizeof(sizeof(Match)), cudaMemcpyDeviceToHost));
//       CudaSafeCall(cudaFree(matches_device));
//       matchSet->matches = matches;
//       break;
//     case gpu:
//       delete[] matches;
//       matchSet->matches = matches_device;
//       break;
//     default:
//       std::cout<<"ERROR invalid only return return_state"<<std::endl;
//       exit(-1);
//   }
//   if(query->arrayStates[1] != gpu && query->arrayStates[1] != both){
//     CudaSafeCall(cudaFree(queryFeatures_device));
//   }
//   if(target->arrayStates[1] != gpu && target->arrayStates[1] != both){
//     CudaSafeCall(cudaFree(targetFeatures_device));
//   }
//   if(query->arrayStates[2] != gpu && query->arrayStates[2] != both){
//     CudaSafeCall(cudaFree(queryDescriptors_device));
//   }
//   if(target->arrayStates[2] != gpu && target->arrayStates[2] != both){
//     CudaSafeCall(cudaFree(targetDescriptors_device));
//   }
// }
// void MatchFactory::generateMatchesPairwiseConstrained(Image* query, Image* target, float epsilon, MatchSet* &matchSet, MemoryState return_state){
//   if(return_state != cpu && return_state != gpu){
//     std::cout<<"ERROR can only return return_state 1=host 2=device"<<std::endl;
//     exit(-1);
//   }
//   if(query->numDescriptorsPerFeature == 0 || target->numDescriptorsPerFeature == 0){
//     std::cout<<"ERROR must have a numDescriptorsPerFeature of more than 0 - no descriptors"<<std::endl;
//     exit(-1);
//   }
//
//   SIFT_Feature* queryFeatures_device;
//   SIFT_Feature* targetFeatures_device;
//   SIFT_Descriptor* queryDescriptors_device;
//   SIFT_Descriptor* targetDescriptors_device;
//
//   if(query->arrayStates[1] != gpu && query->arrayStates[1] != both){
//     if(query->arrayStates[1] != cpu){
//       std::cout<<"ERROR query does not have features"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&queryFeatures_device, query->numFeatures*sizeof(SIFT_Feature)));
//     CudaSafeCall(cudaMemcpy(queryFeatures_device, query->features, query->numFeatures*sizeof(SIFT_Feature), cudaMemcpyHostToDevice));
//   }
//   else queryFeatures_device = query->features_device;
//
//   if(target->arrayStates[1] != gpu && target->arrayStates[1] != both){
//     if(target->arrayStates[1] != cpu){
//       std::cout<<"ERROR target does not have features"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&targetFeatures_device, target->numFeatures*sizeof(SIFT_Feature)));
//     CudaSafeCall(cudaMemcpy(targetFeatures_device, target->features, target->numFeatures*sizeof(SIFT_Feature), cudaMemcpyHostToDevice));
//   }
//   else targetFeatures_device = target->features_device;
//
//   if(query->arrayStates[2] != gpu && query->arrayStates[2] != both){
//     if(query->arrayStates[2] != cpu){
//       std::cout<<"ERROR query does not have descriptors"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&queryDescriptors_device, query->numDescriptorsPerFeature*query->numFeatures*sizeof(SIFT_Descriptor)));
//     CudaSafeCall(cudaMemcpy(queryDescriptors_device, query->featureDescriptors, query->numDescriptorsPerFeature*query->numFeatures*sizeof(SIFT_Descriptor), cudaMemcpyHostToDevice));
//   }
//   else queryDescriptors_device = query->featureDescriptors_device;
//
//   if(target->arrayStates[2] != gpu && target->arrayStates[2] != both){
//     if(target->arrayStates[2] != cpu){
//       std::cout<<"ERROR target does not have descriptors"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&targetDescriptors_device, target->numDescriptorsPerFeature*target->numFeatures*sizeof(SIFT_Descriptor)));
//     CudaSafeCall(cudaMemcpy(targetDescriptors_device, target->featureDescriptors, target->numDescriptorsPerFeature*target->numFeatures*sizeof(SIFT_Descriptor), cudaMemcpyHostToDevice));
//   }
//   else targetDescriptors_device = target->featureDescriptors_device;
//
//   unsigned int numPossibleMatches = query->numFeatures;
//
//   matchSet = new MatchSet();
//   matchSet->numMatches = numPossibleMatches;
//   matchSet->memoryState = return_state;
//   Match* matches_device = nullptr;
//   Match* matches = nullptr;
//   CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));
//   matches = new Match[numPossibleMatches];
//
//   dim3 grid = {1,1,1};
//   dim3 block = {1024,1,1};
//   getGrid(numPossibleMatches,grid);
//
//   clock_t timer = clock();
//   float3* fundamental = new float3[3];
//   calcFundamentalMatrix_2View(query->descriptor, target->descriptor, fundamental);
//
//   float3* fundamental_device;
//   CudaSafeCall(cudaMalloc((void**)&fundamental_device, 3*sizeof(float3)));
//   CudaSafeCall(cudaMemcpy(fundamental_device, fundamental, 3*sizeof(float3), cudaMemcpyHostToDevice));
//
//   matchFeaturesPairwiseConstrained<<<grid, block>>>(query->numFeatures,
//     query->numDescriptorsPerFeature, target->numFeatures, target->numDescriptorsPerFeature,
//     queryDescriptors_device, queryFeatures_device,
//     targetDescriptors_device, targetFeatures_device, matches_device, epsilon, fundamental_device);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//
//   CudaSafeCall(cudaFree(fundamental_device));
//
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//
//   switch(return_state){
//     case cpu:
//       CudaSafeCall(cudaMemcpy(matches, matches_device, numPossibleMatches*sizeof(sizeof(Match)), cudaMemcpyDeviceToHost));
//       CudaSafeCall(cudaFree(matches_device));
//       matchSet->matches = matches;
//       break;
//     case gpu:
//       delete[] matches;
//       matchSet->matches = matches_device;
//       break;
//     default:
//       std::cout<<"ERROR invalid only return return_state"<<std::endl;
//       exit(-1);
//   }
//   if(query->arrayStates[1] != gpu && query->arrayStates[1] != both){
//     CudaSafeCall(cudaFree(queryFeatures_device));
//   }
//   if(target->arrayStates[1] != gpu && target->arrayStates[1] != both){
//     CudaSafeCall(cudaFree(targetFeatures_device));
//   }
//   if(query->arrayStates[2] != gpu && query->arrayStates[2] != both){
//     CudaSafeCall(cudaFree(queryDescriptors_device));
//   }
//   if(target->arrayStates[2] != gpu && target->arrayStates[2] != both){
//     CudaSafeCall(cudaFree(targetDescriptors_device));
//   }
// }
//
// void MatchFactory::generateSubPixelMatchesPairwiseBruteForce(Image* query, Image* target, SubPixelMatchSet* &matchSet, MemoryState return_state){
//   SIFT_Descriptor* queryDescriptors_device;
//   SIFT_Descriptor* targetDescriptors_device;
//
//   if(query->arrayStates[2] != gpu && query->arrayStates[2] != both){
//     if(query->arrayStates[2] != cpu){
//       std::cout<<"ERROR query does not have descriptors"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&queryDescriptors_device, query->numDescriptorsPerFeature*query->numFeatures*sizeof(SIFT_Descriptor)));
//     CudaSafeCall(cudaMemcpy(queryDescriptors_device, query->featureDescriptors, query->numDescriptorsPerFeature*query->numFeatures*sizeof(SIFT_Descriptor), cudaMemcpyHostToDevice));
//   }
//   else queryDescriptors_device = query->featureDescriptors_device;
//
//   if(target->arrayStates[2] != gpu && target->arrayStates[2] != both){
//     if(target->arrayStates[2] != cpu){
//       std::cout<<"ERROR target does not have descriptors"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&targetDescriptors_device, target->numDescriptorsPerFeature*target->numFeatures*sizeof(SIFT_Descriptor)));
//     CudaSafeCall(cudaMemcpy(targetDescriptors_device, target->featureDescriptors, target->numDescriptorsPerFeature*target->numFeatures*sizeof(SIFT_Descriptor), cudaMemcpyHostToDevice));
//   }
//   else targetDescriptors_device = target->featureDescriptors_device;
//
//   MatchSet* nonSubSet = nullptr;
//   this->generateMatchesPairwiseBruteForce(query, target, nonSubSet, gpu);
//
//   int numMatches = nonSubSet->numMatches;
//
//   SubPixelMatch* subPixelMatches_device;
//   CudaSafeCall(cudaMalloc((void**)&subPixelMatches_device, numMatches*sizeof(SubPixelMatch)));
//
//   SubpixelM7x7* subDescriptors_device;
//   CudaSafeCall(cudaMalloc((void**)&subDescriptors_device, numMatches*sizeof(SubpixelM7x7)));
//
//   //TODO figure out if you want subPixelMatches on device before subpixelLocation determination
//
//   dim3 grid = {1,1,1};
//   dim3 block = {9,9,1};
//   getGrid(numMatches, grid);
//   std::cout<<"initializing subPixelMatches..."<<std::endl;
//   clock_t timer = clock();
//   initializeSubPixels<<<grid, block>>>(query->descriptor, target->descriptor, numMatches, nonSubSet->matches, subPixelMatches_device,
//     subDescriptors_device, queryDescriptors_device, query->numFeatures, query->numDescriptorsPerFeature,
//     targetDescriptors_device, target->numFeatures, target->numDescriptorsPerFeature);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//
//   delete nonSubSet;
//
//   if(query->arrayStates[2] != gpu && query->arrayStates[2] != both){
//     CudaSafeCall(cudaFree(queryDescriptors_device));
//   }
//   if(target->arrayStates[2] != gpu && target->arrayStates[2] != both){
//     CudaSafeCall(cudaFree(targetDescriptors_device));
//   }
//
//   Spline* splines_device;
//   CudaSafeCall(cudaMalloc((void**)&splines_device, numMatches*2*sizeof(Spline)));
//
//   grid = {1,1,1};
//   block = {6,6,4};
//   getGrid(numMatches*2, grid);
//
//   std::cout<<"filling bicubic splines..."<<std::endl;
//   timer = clock();
//   fillSplines<<<grid,block>>>(numMatches, subDescriptors_device, splines_device);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//   CudaSafeCall(cudaFree(subDescriptors_device));
//
//   std::cout<<"determining subpixel locations..."<<std::endl;
//   timer = clock();
//   determineSubPixelLocationsBruteForce<<<grid,block>>>(0.1, numMatches, subPixelMatches_device, splines_device);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//   CudaSafeCall(cudaFree(splines_device));
//
//   SubPixelMatch* subPixelMatches;
//   matchSet = new SubPixelMatchSet();
//   matchSet->numMatches = numMatches;
//   matchSet->memoryState = return_state;
//   switch(return_state){
//     case cpu:
//       subPixelMatches = new SubPixelMatch[numMatches];
//       CudaSafeCall(cudaMemcpy(subPixelMatches, subPixelMatches_device, numMatches*sizeof(SubPixelMatch), cudaMemcpyDeviceToHost));
//       CudaSafeCall(cudaFree(subPixelMatches_device));
//       break;
//     case gpu:
//       subPixelMatches = subPixelMatches_device;
//       break;
//     default:
//       std::cout<<"ERROR invalid return_state"<<std::endl;
//       exit(-1);
//   }
//   matchSet->matches = (return_state == cpu) ? subPixelMatches : subPixelMatches_device;
// }
// void MatchFactory::generateSubPixelMatchesPairwiseConstrained(Image* query, Image* target, float epsilon, SubPixelMatchSet* &matchSet, MemoryState return_state){
//   SIFT_Descriptor* queryDescriptors_device;
//   SIFT_Descriptor* targetDescriptors_device;
//
//   if(query->arrayStates[2] != gpu && query->arrayStates[2] != both){
//     if(query->arrayStates[2] != cpu){
//       std::cout<<"ERROR query does not have descriptors"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&queryDescriptors_device, query->numDescriptorsPerFeature*query->numFeatures*sizeof(SIFT_Descriptor)));
//     CudaSafeCall(cudaMemcpy(queryDescriptors_device, query->featureDescriptors, query->numDescriptorsPerFeature*query->numFeatures*sizeof(SIFT_Descriptor), cudaMemcpyHostToDevice));
//   }
//   else queryDescriptors_device = query->featureDescriptors_device;
//
//   if(target->arrayStates[2] != gpu && target->arrayStates[2] != both){
//     if(target->arrayStates[2] != cpu){
//       std::cout<<"ERROR target does not have descriptors"<<std::endl;
//       exit(-1);
//     }
//     CudaSafeCall(cudaMalloc((void**)&targetDescriptors_device, target->numDescriptorsPerFeature*target->numFeatures*sizeof(SIFT_Descriptor)));
//     CudaSafeCall(cudaMemcpy(targetDescriptors_device, target->featureDescriptors, target->numDescriptorsPerFeature*target->numFeatures*sizeof(SIFT_Descriptor), cudaMemcpyHostToDevice));
//   }
//   else targetDescriptors_device = target->featureDescriptors_device;
//
//   MatchSet* nonSubSet  = nullptr;
//   this->generateMatchesPairwiseConstrained(query, target, epsilon, nonSubSet, gpu);
//
//   int numMatches = nonSubSet->numMatches;
//
//   SubPixelMatch* subPixelMatches_device;
//   CudaSafeCall(cudaMalloc((void**)&subPixelMatches_device, numMatches*sizeof(SubPixelMatch)));
//
//   SubpixelM7x7* subDescriptors_device;
//   CudaSafeCall(cudaMalloc((void**)&subDescriptors_device, numMatches*sizeof(SubpixelM7x7)));
//
//   dim3 grid = {1,1,1};
//   dim3 block = {9,9,1};
//   getGrid(numMatches, grid);
//   std::cout<<"initializing subPixelMatches..."<<std::endl;
//   clock_t timer = clock();
//   initializeSubPixels<<<grid, block>>>(query->descriptor, target->descriptor, numMatches, nonSubSet->matches, subPixelMatches_device,
//     subDescriptors_device, queryDescriptors_device, query->numFeatures, query->numDescriptorsPerFeature,
//     targetDescriptors_device, target->numFeatures, target->numDescriptorsPerFeature);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//
//   delete nonSubSet;
//
//   if(query->arrayStates[2] != gpu && query->arrayStates[2] != both){
//     CudaSafeCall(cudaFree(queryDescriptors_device));
//   }
//   if(target->arrayStates[2] != gpu && target->arrayStates[2] != both){
//     CudaSafeCall(cudaFree(targetDescriptors_device));
//   }
//
//   Spline* splines_device;
//   CudaSafeCall(cudaMalloc((void**)&splines_device, numMatches*2*sizeof(Spline)));
//
//   grid = {1,1,1};
//   block = {6,6,4};
//   getGrid(numMatches*2, grid);
//
//   std::cout<<"filling bicubic splines..."<<std::endl;
//   timer = clock();
//   fillSplines<<<grid,block>>>(numMatches, subDescriptors_device, splines_device);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//   CudaSafeCall(cudaFree(subDescriptors_device));
//
//   std::cout<<"determining subpixel locations..."<<std::endl;
//   timer = clock();
//   determineSubPixelLocationsBruteForce<<<grid,block>>>(0.1, numMatches, subPixelMatches_device, splines_device);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//   CudaSafeCall(cudaFree(splines_device));
//
//   matchSet = new SubPixelMatchSet();
//   SubPixelMatch* subPixelMatches;
//   matchSet->memoryState = return_state;
//   matchSet->numMatches = numMatches;
//   switch(return_state){
//     case cpu:
//       subPixelMatches = new SubPixelMatch[numMatches];
//       CudaSafeCall(cudaMemcpy(subPixelMatches, subPixelMatches_device, numMatches*sizeof(SubPixelMatch), cudaMemcpyDeviceToHost));
//       CudaSafeCall(cudaFree(subPixelMatches_device));
//       break;
//     case gpu:
//       subPixelMatches = subPixelMatches_device;
//       break;
//     default:
//       std::cout<<"ERROR invalid return_state"<<std::endl;
//       exit(-1);
//   }
//   matchSet->matches = (return_state == cpu) ? subPixelMatches : subPixelMatches_device;
// }
