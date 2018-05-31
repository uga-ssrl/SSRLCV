#include "common_includes.h"
#include "octree.cuh"
#include "poisson.cuh"
using namespace std;

//TODO across octree and poisson determine if you really need to instantiate all device array values

int main(int argc, char *argv[]){
  try{
    if(argc == 2){
      string filePath = argv[1];
      clock_t totalTimer = clock();
      clock_t partialTimer = clock();

      //if we want further depth than 10 our nodeKeys will need to then be long or long long
      int depth = 10;
      Octree octree = Octree(filePath, depth);
      /*
      KEEP IN MIND THAT NORMALS ARE CURRENTLY READ FROM A PLY AND ARE INWARD FACING
      THIS MEANS THAT NORMALS INSTANTIATION WILL NEED TO BE REMOVED FROM Octree::parsePLY
      AND COLOR WILL BE READ IN ITS PLACE
      */

      octree.init_octree_gpu();
      octree.generateKeys();
      octree.sortByKey();
      octree.compactData();
      octree.fillUniqueNodesAtFinestLevel();
      octree.createFinalNodeArray();
      octree.freePrereqArrays();

      octree.fillLUTs();
      //octree.printLUTs();
      octree.fillNeighborhoods();

      octree.checkForGeneralNodeErrors();

      octree.computeVertexArray();
      octree.computeEdgeArray();
      octree.computeFaceArray();

      partialTimer = clock() - partialTimer;
      printf("\nOCTREE BUILD TOOK %f seconds.\n\n",((float) partialTimer)/CLOCKS_PER_SEC);
      partialTimer = clock();
      /*
      OCTREE HAS BEEN GENERATED NOW ONTO NORMAL COMPUTATION
      //TODO implement this as right now it is read in through the ply
      */

      //octree.computeNormals();

      /*
      RECONTRUCTION PREP HAS COMPLETED NOW ONTO POISSON RECONSTRUCTION
      */
      //TODO figure out if you want to free octree device data during poisson or leave until full delete octree
      Poisson poisson = Poisson(&octree);

      poisson.computeLUTs();
      poisson.computeDivergenceVector();
      //poisson.computeImplicitFunction();
      //poisson.marchingCubes();
      //poisson.isosurfaceExtraction();

      //cudaDeviceReset();
      partialTimer = clock() - partialTimer;
      printf("POISSON RECONSTRUCTION TOOK %f seconds.\n\n",((float) partialTimer)/CLOCKS_PER_SEC);

      totalTimer = clock() - totalTimer;
      printf("TOTAL TIME = %f seconds.\n\n",((float) totalTimer)/CLOCKS_PER_SEC);

      return 0;
    }
    else{
      cout<<"LACK OF PLY INPUT...goodbye"<<endl;
      exit(1);
    }
  }
  catch (const std::exception &e){
      std::cerr << "Caught exception: " << e.what() << '\n';
      std::exit(1);
  }
  catch (...){
      std::cerr << "Caught unknown exception\n";
      std::exit(1);
  }

}
