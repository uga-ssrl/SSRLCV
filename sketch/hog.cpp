/* SOURCES:
 * * Tomasi - https://www2.cs.duke.edu/courses/fall15/compsci527/notes/hog.pdf
 */

#include <iostream>
#include <math.h>

 // Utilities

void assert(bool assertion){ if(!assertion) std::cout << "ASSERTION FAILED" << std::endl; exit(1); }

typedef struct { double x; double y; } Vector;
typedef struct { double ang; double mag; } Polar;

const double TO_DEG 			= 57.29577;		// 180/pi
const double PI 				=  3.14159;
const double SPC				= 0.001; 		// "Small positive constant"


// Constants & parameters
// Defined as in Tomasi's paper
const int CELL_SIZE			= 8; // "C"
const int HIST_BINS			= 9; // "B"


//
// -- -- --
//

int main(int argc, char ** argv)
{
	//TODO: Input

	//TODO: Allocation here
	// Global allocations
	int XX, YY;							// Input size in X and Y in pixels
	int ** 			Image; 				// Image function			--		(x,y) ∈ Window		|-->	[0,255]
	Polar ** 		Gradient;			// Gradient 				--		(x,y) ∈ Window		|-->	θ ∈[0,2π] , r
	double ** 		CellHistograms;		// [0, numCells) |--> [0, HIST_BINS) |--> double
	double ** 		BlockHistograms;	// [0, numBlock) |--> [0, HIST_BINS) |--> double

	double * 		FeatureOutput;


	// --
	// Peripheral computations
	// --

	// Gotta be divisible by the cell size !!!!
	assert(XX % CELL_SIZE == 0);
	assert(YY % CELL_SIZE == 0); 

	int XC 			= XX / CELL_SIZE;  int YC = YY / CELL_SIZE;		// Number of cells 	in X and Y
	int XB 			= (XC - 1); int YB = (YC - 1);					// Number of blocks in X and Y
	int numCells 	= XC * YC;										// Total number of cells
	int numBlocks 	= XB * YB;										// Total number of blocks

	int FeatureOutputSize = numBlocks * 4 * HIST_BINS; // For each bin in each block.  There are 4 cells per block.


	// --
	// HOG Computations
	// --

	// -----------
	// Step 1: Compute gradients
	// Approximate the gradient.  Tomasi does this by central differences, for reasons enumerated in his paper
	//TODO: Address boundary conditions here.  Out of range for x ∈{0,imgX} , same with y
	for(int x = 0; x < XX; x++) { for(int y = 0; y < YY; y++) {

		static float dx = Image[x+1][y] - Image[x-1][y];
		static float dy = Image[x][y+1] - Image[x][y-1];

		//CUDA: g->mag computation, and atan2
		Polar * g = &(Gradient[x][y]);
		g->mag = sqrt(dx*dx + dy*dy);
		g->ang = fmod(atan2(dx,dy), PI) * TO_DEG; 
		// MOCI GANG MOCI GANG MOCI GANG MOCI GANG MOCI GANG MOCI GANG MOCI GANG

	} }

	// -----------
	// Step 2: Compute cell orientation histograms
	// Split the image up into "cells", CELL_SIZE x CELL_SIZE windows, of which there are (XC * YC = numCells).  
	for(int i = 0; i < numCells; i++) 
	{
		static const float width = 180 / HIST_BINS; // "w"

		// For each cell
		int cx = i % XC;									// x index of cell
		int cy = (i - cx) / XC;								// y index of cell
		int wx = cx * CELL_SIZE, wy = cy * CELL_SIZE;		// x and y coordinates of cell windows


		for(int x = wx; x < wx + CELL_SIZE; x++) { for(int y = wy; y < wy + CELL_SIZE; y++ ) {

			Polar * g = &(Gradient[x][y]);

			int bin1 		= (int) floor((g->ang / width) - 0.5 ) %  (int) HIST_BINS;	// "j"			-- Bin index
			double ctr1 	= width * (bin1 + 1 + 0.5);									// "c_(j+1)"	-- Center, of bin j+1
			double vot1 	= g->mag * (( ctr1 - g->ang ) / width);						// "v"			-- Bin vote

			int bin2		= (bin1 + 1) %  HIST_BINS;									// "j+1"		-- Bin index
			double ctr2 	= width * (bin1 + 0.5);										// "c_j"		-- Center, of bin j
			double vot2		= g->mag * (( g->ang - ctr2 ) / width);						// "v_(j+1)"	-- Bin vote

			CellHistograms[i][bin1] = vot1; 
			CellHistograms[i][bin2] = vot2;

		} }
	}

	// -------------
	// Step 3: Block normalization
	// Combine each group of four neighboring cells into overlapping blocks, of which there are (XB * YB = numBlock).
	// For each block, concatenate the histograms of the cells within into a single histogram.  Normalize this
	#define FOREACH_BIN(j) for(int j = 0; j < HIST_BINS; j++)
	for(int i = 0; i < numBlocks; i++)
	{
		// For each block; h1-4 are the histograms of the cells the blocks include
		double * block = BlockHistograms[i];
		int b = floor(i / XB);
		double * h1 = CellHistograms[b+i],  * h2 = CellHistograms[b+i+1], * h3 = CellHistograms[b+i+XC], * h4 = CellHistograms[b+i+XC+1];

		// Concatenate & normalize
		//CUDA: There is a norm function in CuBLAS, but it does not include the small positive constant below.  What do?
		double norm = 0.0;
		FOREACH_BIN(j)
		{
			double d = h1[j] + h2[j] + h3[j] + h4[j];
			block[j] = d;
			norm += (d * d);
		}
		norm = sqrt(norm + SPC); // "Small positive constant" to prevent division by zero, as in Tomasi's paper
		FOREACH_BIN(j) block[j] /= norm;
	}
	#undef FOREACH_BIN

	// -------------
	// Step 4: Vector normalization
	// This happens according to a pretty bizarre scheme, which is, of course, defined in Tomasi's paper.
	// His arguments, as they are with all of these things, are entirely empirical.

	#define FOREACH_BIN { int i = -1; for(int blk = 0; blk < numBlocks; blk++){ for(int bin = 0; bin < HIST_BINS; bin++){ i++; 

	{
		// Construct and normalize it the first time
		double norm = 0.0;
		FOREACH_BIN {
			double d = BlockHistograms[blk][bin];
			FeatureOutput[i] = d; 
			norm += (d*d);
		} } } }
		FOREACH_BIN { FeatureOutput[i] /= norm; } } } } 
	}

	// Clip everything to a threshold of 0.2
	// This ensures that "very large gradients do not have too much influence"
	FOREACH_BIN { double d = FeatureOutput[i]; FeatureOutput[i] = (0.2 < d ? 0.2 : d); } } } }

	{
		// Normalize it again
		double norm = 0.0;
		FOREACH_BIN { norm += (FeatureOutput[i]); } } } }
		FOREACH_BIN { FeatureOutput[i] /= norm; } } } } 
	}


	// -----
	// DONE!
	// -----

	//TODO: Output

	return 0;
}