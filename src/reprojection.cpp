//======================================
// UGA SSRL Reprojection
// Author: Caleb Adams
// Contact: CalebAshmoreAdams@gmail.com
//======================================

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// == GLOBAL VARIABLES == //
bool           verbose = 1;
bool           debug   = 0;

unsigned short match_count;
unsigned short camera_count;

unsigned short cameras_l = 0;
unsigned short matches_l = 0;

unsigned int   res  = 1024;
float          dpix = 0.00002831538; //(foc*tan(fov/2))/(res/2)
float          foc  = 0.035;
float          fov  = 0.785398163397; // 45 degrees

vector< vector<string> > matches;
vector< vector<string> > cameras;
vector< vector<string> > projections;
vector< vector<float> >  points;
vector< vector<float> >  matchesr3;
vector< vector<int> >    colors;
// ====================== //

//
// parses comma delemeted string
//
void parse_comma_delem(string str, unsigned short flag){
  istringstream ss(str);
  string token;
  vector<string> v;
  while(getline(ss, token, ',')) {
    if (debug) cout << token << endl;
    v.push_back(token);
  }
  switch (flag){
    case 1: // matches
      matches.push_back(v);
      matches_l++;
      break;
    case 2: // cameras
      cameras.push_back(v);
      cameras_l++;
      break;
    default:
      break;
  }
}

//
// loads matches from a match.txt file
//
void load_matches(){
  ifstream infile("data/spongebob_pan_matches.txt");
  string line;
  unsigned short c = 0;
  bool first = 1;
  while (getline(infile, line)){
      istringstream iss(line);
      if (debug) cout << line << endl;
      if (first){
        first = 0;
      } else {
        parse_comma_delem(line, 1);
        c++;
      }
  }
  match_count = c-1;
  if (verbose or debug) cout << "Loaded: " << match_count << " matches" << endl;
}

//
// loads cameras from a camera.txt file
//
void load_cameras(){
  ifstream infile("data/spongebob_pan_cameras.txt");
  string line;
  unsigned short c = 0;
  while (getline(infile, line)){
      istringstream iss(line);
      if (debug) cout << line << endl;
      parse_comma_delem(line, 2);
      c++;
  }
  camera_count = c;
  if (verbose or debug) cout << "Loaded: " << camera_count << " cameras" << endl;
}

//
// This is used for linear approximation
// TODO develop a better way to do linear approximation
//
float euclid_dist(float p1[3], float p2[3]){
  return sqrt(((p2[0] - p1[0])*(p2[0] - p1[0])) + ((p2[1] - p1[1])*(p2[1] - p1[1])) + ((p2[2] - p1[2])*(p2[2] - p1[2])));
}

//
// loads cameras from a camera.txt file
// this is currently a 2-view system
//
void two_view_reproject(){
  // get the data that we want to compute
  cout << "2-view trianulating... " << endl;
  for(int i = 0; i < matches.size(); i++){
    int   image1     = stoi(matches[i][0].substr(0,4));
    int   image2     = stoi(matches[i][1].substr(0,4));
    float camera1[6] = {stof(cameras[image1-1][1]),stof(cameras[image1-1][2]),stof(cameras[image1-1][3]),stof(cameras[image1-1][4]),stof(cameras[image1-1][5]),stof(cameras[image1-1][6])};
    float camera2[6] = {stof(cameras[image2-1][1]),stof(cameras[image2-1][2]),stof(cameras[image2-1][3]),stof(cameras[image2-1][4]),stof(cameras[image2-1][5]),stof(cameras[image2-1][6])};
    // TODO rotate the projection so that the camera unit vector is normal to it
    float x1 = dpix * (stof(matches[i][2]) - res/2.0);
    float y1 = dpix * (stof(matches[i][3]) - res/2.0);
    float x2 = dpix * (stof(matches[i][4]) - res/2.0);
    float y2 = dpix * (stof(matches[i][5]) - res/2.0);
    float points1[6] = {camera1[0],camera1[1],camera1[2],x1 + camera1[0],y1 + camera1[1],foc + camera1[2]};
    vector<float> r31;
    r31.push_back(points1[3]);
    r31.push_back(points1[4]);
    r31.push_back(points1[5]);
    matchesr3.push_back(r31);
    float points2[6] = {camera2[0],camera2[1],camera2[2],x2 + camera2[0],y2 + camera2[1],foc + camera2[2]};
    vector<float> r32;
    r32.push_back(points2[3]);
    r32.push_back(points2[4]);
    r32.push_back(points2[5]);
    matchesr3.push_back(r32);
    int   rgb[3]     = {stoi(matches[i][6]),stoi(matches[i][7]),stoi(matches[i][8])};
    // find the vectors
    float v1[3]      = {points1[3] - points1[0],points1[4] - points1[1],points1[5] - points1[2]};
    float v2[3]      = {points2[3] - points2[0],points2[4] - points2[1],points2[5] - points2[2]};
    // prepare for the linear approximation
    float smallest = numeric_limits<float>::max();
    float p1[3] = {0.0,0.0,0.0};
    float p2[3] = {0.0,0.0,0.0};
    for (float i = 5.0; i < 500.0; i += 0.001){
      // get the points on the lines
      p1[0]  = points1[0] + v1[0]*i;
      p1[1]  = points1[1] + v1[1]*i;
      p1[2]  = points1[2] + v1[2]*i;
      p2[0]  = points2[0] + v2[0]*i;
      p2[1]  = points2[1] + v2[1]*i;
      p2[2]  = points2[2] + v2[2]*i;
      // update the estimate
      float dist = euclid_dist(p1,p2);
      if (dist < smallest) smallest = dist;
      else break;
    }
    // store the result if it sasifies the boundary conditions
    if (p1[2] > 1.4 && p1[2] < 2.7){
      vector<float> v;
      vector<int>   c;
      v.push_back(p1[0]);
      v.push_back(p1[1]);
      v.push_back(p1[2]);
      c.push_back(rgb[0]);
      c.push_back(rgb[1]);
      c.push_back(rgb[2]);
      if (debug) cout << p1[0] << "," << p1[1] << "," << p1[2] << endl;
      points.push_back(v);
      colors.push_back(c);
    }
  }
  cout << "Generated: " << points.size() << " valid points" << endl;
}

void save_ply(){
  ofstream outputFile1("output.ply");
  outputFile1 << "ply\nformat ascii 1.0\nelement vertex ";
  outputFile1 << points.size() << "\n";
  outputFile1 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
  outputFile1 << "end_header\n";
  for(int i = 0; i < points.size(); i++){
    outputFile1 << points[i][0] << " " << points[i][1] << " " << points[i][2] << " " << colors[i][0] << " " << colors[i][1] << " " << colors[i][2] << "\n";
  }
  ofstream outputFile2("cameras.ply");
  outputFile2 << "ply\nformat ascii 1.0\nelement vertex ";
  outputFile2 << cameras.size() << "\n";
  outputFile2 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
  outputFile2 << "end_header\n";
  for(int i = 0; i < cameras.size(); i++){
    outputFile2 << cameras[i][1] << " " << cameras[i][2] << " " << cameras[i][3] << " 255 0 0\n";
  }
  ofstream outputFile3("matches.ply");
  outputFile3 << "ply\nformat ascii 1.0\nelement vertex ";
  outputFile3 << matchesr3.size() << "\n";
  outputFile3 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
  outputFile3 << "end_header\n";
  for(int i = 0; i < matchesr3.size(); i++){
    outputFile3 << matchesr3[i][0] << " " << matchesr3[i][1] << " " << matchesr3[i][2] << " 0 2551 0\n";
  }
}

//
// This is the main method
//
int main(){
  cout << " == REPROJECTION == " << endl;

  load_matches();
  load_cameras();
  two_view_reproject();
  save_ply();

  return 0;
}
