//======================================
// UGA SSRL Reprojection
// Author: Caleb Adams
// Contact: CalebAshmoreAdams@gmail.com
//======================================

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

// == GLOBAL VARIABLES ==
bool verbose        = 1;
bool debug          = 1;

unsigned short match_count;
// ======================

void load_matches(){
  ifstream infile("data/carl_pan_matches.txt");
  string line;
  unsigned short c = 0;
  bool first = 1;
  while (getline(infile, line)){
      istringstream iss(line);
      if (debug) cout << line << endl;
      if (first){
        first = 0;
      }

      c++;
  }
  match_count = c-1;
  if (verbose) cout << "Loaded: " << match_count << " matches" << endl;
}

void write_ply(){

}

int euclid_dist(float x1,float y1,float z1,float x2,float y2,float z2){
  return 0;
}



int main(){
  cout << " == REPROJECTION == " << endl;

  load_matches();

  return 0;
}
