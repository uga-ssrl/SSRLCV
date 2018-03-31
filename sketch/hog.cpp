/* SOURCES:
 * * Tomasi - https://www2.cs.duke.edu/courses/fall15/compsci527/notes/hog.pdf
 */

#include <iostream>
#include <math.h>

//TEMP
double norm(double x, double y){ return sqrt(x * x + y * y); } // Defined in CUDA

 // Utilities

typedef struct { double x; double y; } Vector;
typedef struct { double ang; double mag; } Polar;

#define FOREACH_X for(int x = 0; x < imgX; x++)
#define FOREACH_Y for(int y = 0; y < imgY; y++)
#define FOREACH_X_IN(a,b) for(int x = a; x < b; x++)
#define FOREACH_Y_IN(a,b) for(int y = a; y < b; y++)

const double TO_DEG 			= 57.29577;	// 180/pi
const double PI 				=  3.14159;


// Constants & parameters
// Defined as in Tomasi's paper
const int HIST_WINDOW_SIZE		= 8; // "C"
const int HIST_BINS				= 9; // "B"

// Globals.  Replace this however you want to
//TODO: Allocation here
int 			imgX, imgY; 		// Size of image			--		Let Window = { (x,y) | x ∈[0,imgX) ∧ y ∈[0,imgY) }
int ** 			I; 					// Image function			--		(x,y) ∈ Window		|-->	[0,255]
Polar ** 		G;					// Gradient 				--		(x,y) ∈ Window		|-->	θ ∈[0,2π] , r


//
// -- -- --
//


//
// Approximate the gradient.  Tomasi does this by central differences, for reasons enumerated in his paper
//
void ComputeGradient()
{
	//TODO: Address boundary conditions here.  Out of range for x ∈{0,imgX} , same with y
	for(x = 0; x < imgX; x++) { for(y = 0; y < imgY; y++) {

		static float dx = I[x+1][y] - I[x-1][y];
		static float dy = I[x][y+1] - I[x][y-1];

		//CUDA: norm and atan2.  mod as well?
		Polar * g = &(G[x][y]);
		g->mag = norm(dx, dy);						
		g->ang = fmod(atan2(dx,dy), PI) * TO_DEG; 
		// MOCI GANG MOCI GANG MOCI GANG MOCI GANG MOCI GANG MOCI GANG MOCI GANG

	} }
}

/** Compute the cell orientation histogram over a square window @(wx, wy) with size HIST_WINDOW_SIZE.
 * This is the first histogram generation step.
 *
 * @param result Histogram array: int [0,HIST_BINS) |--> double 
 * @param wx Window x coordinate
 * @param wy Window y coordinate
 */
void ComputeCellOrientationHistogram(double * result, double wx, double wy)
{
	static const float width = 180 / HIST_BINS; // "w"

	for(x = wx; x < wx + HIST_WINDOW_SIZE; x++) { for(y = wy; y < wy + HIST_WINDOW_SIZE; y++ ) {

		Polar * g = &(G[x][y]);
		int bin = fmod(floor((g->ang / width) - 0.5 ),  HIST_BINS);			// "j"
		double center = width * (bin + 0.5);								// "c_i"
		double vote = g.mag * (( center -> g.ang ) / width);				// "v"

		hist[bin] = vote; 

	} }
}


int main(int argc, char ** argv)
{
	//TODO: Input 

	//TODO: Allocation here
	int 		numKeypoints;	// 
	int * 		KeypointsX;		// [0, numKeypoints) |--> int
	int *		KeypointsY;		// [0, numKeypoints) |--> int
	double **	Histograms;		// [0, numKeypoints) |--> [0, HIST_BINS) |--> double;


	ComputeGradient();	
	for(int i = 0, i < numKeypoints; i++)  ComputeCellOrientationHistogram( Histograms[i],   KeypointsX[i], KeypointsY[i]   );

	//TODO: Output

	return 0;
}