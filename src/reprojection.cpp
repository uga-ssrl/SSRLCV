//======================================//
// UGA SSRL Reprojection                //
// Author: Caleb Adams                  //
// Contact: CalebAshmoreAdams@gmail.com //
//======================================//

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
bool           debug   = 1;
bool           simple  = 0;

string cameras_path;
string matches_path;

unsigned short match_count;
unsigned short camera_count;

// TODO (some of) this stuff should be set by camera calibration
unsigned int   res  = 1024;
float          dpix = 0.00002831538; //(foc*tan(fov/2))/(res/2)
float          foc  = 0.035;
float          fov  = 0.785398163397; // 45 degrees
float          PI   = 3.1415926535;

// for debugging
float          max_angle = -1000.0;
float          min_angle =  1000.0;

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
      break;
    case 2: // cameras
      cameras.push_back(v);
      break;
    default:
      break;
  }
}

//
// loads matches from a match.txt file
//
void load_matches(){
  ifstream infile(matches_path);
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
  ifstream infile(cameras_path);
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
// This is used to rotate the projection in the z axis
//
vector<float> rotate_projection_x(float x, float y, float z, float r){
  vector<float> v;
  // count clockwize rotatoin around x axis
  float x_n = x;
  float y_n = cos(r)*y + -1*sin(r)*z;
  float z_n = sin(r)*y + cos(r)*z;
  v.push_back(x_n);
  v.push_back(y_n);
  v.push_back(z_n);
  return v;
}

//
// TODO make this
// This is used to rotate the projection in the Y axis
//
vector<float> rotate_projection_y(float x, float y, float z, float r){
  vector<float> v;
  // count clockwize rotatoin around x axis
  // float x_n = cos(r)*x + -1*sin(r)*y;
  // float y_n = sin(r)*x + cos(r)*y;
  // float z_n = z;
  // v.push_back(x_n);
  // v.push_back(y_n);
  // v.push_back(z_n);
  return v;
}

//
// This is used to rotate the projection in the z axis
//
vector<float> rotate_projection_z(float x, float y, float z, float r){
  vector<float> v;
  // count clockwize rotatoin around z axis
  float x_n = cos(r)*x + -1*sin(r)*y;
  float y_n = sin(r)*x + cos(r)*y;
  float z_n = z;
  v.push_back(x_n);
  v.push_back(y_n);
  v.push_back(z_n);
  return v;
}

//
// This is used to get the angle
// if flag == 0 // find angle on x
// if flag == 1 // find angle on y
// if flag == 2 // find angle on z
// v is the input vector
// we assume we compare the the vector: [1.0, 0.0, 0.0]
//
float get_angle(float cam[6], int flag){
  // TODO add other methods of the same name
  // to make this work for a more general case
  float w[3] = {1.0,0.0,0.0};
  float v[3] = {cam[3],cam[4],cam[5]};
  // find the dot product
  float dot_v_w = v[0]*w[0] + v[1]*w[1] + v[2]*w[2];
  // calculate magnitude for v
  float v_mag = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  // make the fraction:
  float fract = (dot_v_w)/(v_mag);
  // find the angle
  float angle = acos(fract);
  // check to see if outside of second quad
  if (cam[4] < 0.0) angle = 2 * PI - angle;
  if (angle > max_angle) max_angle = angle;
  if (angle < min_angle) min_angle = angle;
  return angle;
}

//
// loads cameras from a camera.txt file
// this is currently a 2-view system
//
void two_view_reproject(){
  // get the data that we want to compute
  cout << "2-view trianulating... " << endl;
  int length = matches.size();
  if (simple) length = 5000; // limit the number of points to 5k
  for(int i = 0; i < length; i++){
    int   image1     = stoi(matches[i][0].substr(0,4));
    int   image2     = stoi(matches[i][1].substr(0,4));
    float camera1[6] = {stof(cameras[image1-1][1]),stof(cameras[image1-1][2]),stof(cameras[image1-1][3]),stof(cameras[image1-1][4]),stof(cameras[image1-1][5]),stof(cameras[image1-1][6])};
    float camera2[6] = {stof(cameras[image2-1][1]),stof(cameras[image2-1][2]),stof(cameras[image2-1][3]),stof(cameras[image2-1][4]),stof(cameras[image2-1][5]),stof(cameras[image2-1][6])};
    // scale the projection's coordinates
    float x1 = dpix * (stof(matches[i][2]) - res/2.0);
    float y1 = dpix * (stof(matches[i][3]) - res/2.0);
    float x2 = dpix * (stof(matches[i][4]) - res/2.0);
    float y2 = dpix * (stof(matches[i][5]) - res/2.0);

    // NOTE FROM HERE ON THERE ARE THINGS for the rotation

    // rotate the coords to be correct
    if (debug){
      cout << "camera1 unit vector: " << camera1[3] << "," << camera1[4] << "," << camera1[5] << endl;
      cout << "rotation1: " << get_angle(camera1, 0) << endl;
      cout << "camera2 unit vector: " << camera2[3] << "," << camera2[4] << "," << camera2[5] << endl;
      cout << "rotation2: " << get_angle(camera2, 0) << endl;
    }

    // get the needed rotation
    float r1 = get_angle(camera1, 0);
    float r2 = get_angle(camera2, 0);

    // for some reason it was not in the right plane?
    vector<float> kp1 = rotate_projection_x(x1,y1,0.0,PI/2);
    vector<float> kp2 = rotate_projection_x(x2,y2,0.0,PI/2);

    // adjust the kp's location
    kp1[0] = camera1[0] - (kp1[0] + (camera1[3] * foc));
    kp1[1] = camera1[1] - (kp1[1] + (camera1[4] * foc));

    kp2[0] = camera2[0] - (kp2[0] + (camera2[3] * foc));
    kp2[1] = camera2[1] - (kp2[1] + (camera2[4] * foc));

    // rotate it the right amount
    // vector<float> kp1 = rotate_projection_z(temp1[0],temp1[1],0.0,r1);
    // vector<float> kp2 = rotate_projection_z(temp2[0],temp2[1],0.0,r2);

    // rotate the projections coordinates
    // vector<float> proj1 = rotate_projection_z(x1,y1,camera1[3],camera1[4],camera1[5]);
    // vector<float> proj2 = rotate_projection_z(x2,y2,camera2[3],camera2[4],camera2[5]);
    // store the final camera - projection match
    // TODO I have the unit vectors stored wrong I think. x should be z... not sure what happened
    // float points1[6] = {camera1[0],camera1[1],camera1[2],kp1[0],kp1[1],foc + kp1[2]};
    // float points2[6] = {camera2[0],camera2[1],camera2[2],kp2[0],kp2[1],foc + kp2[2]};

    float points1[6] = {camera1[0],camera1[1],camera1[2],kp1[0],kp1[1],kp1[2]};
    float points2[6] = {camera2[0],camera2[1],camera2[2],kp2[0],kp2[1],kp2[2]};
    int   rgb[3]     = {stoi(matches[i][6]),stoi(matches[i][7]),stoi(matches[i][8])};

    // NOTE This is the pan-view way to do this

    // float points1[6] = {camera1[0],camera1[1],camera1[2],x1 + camera1[0],y1 + camera1[1],foc + camera1[2]};
    // float points2[6] = {camera2[0],camera2[1],camera2[2],x2 + camera2[0],y2 + camera2[1],foc + camera2[2]};
    // int   rgb[3]     = {stoi(matches[i][6]),stoi(matches[i][7]),stoi(matches[i][8])};

    // END NOTE

    // this is just for storing the projections for a ply file later
    vector<float> r32;
    r32.push_back(points2[3]);
    r32.push_back(points2[4]);
    r32.push_back(points2[5]);
    matchesr3.push_back(r32);
    vector<float> r31;
    r31.push_back(points1[3]);
    r31.push_back(points1[4]);
    r31.push_back(points1[5]);
    matchesr3.push_back(r31);
    // find the vectors
    float v1[3]      = {points1[3] - points1[0],points1[4] - points1[1],points1[5] - points1[2]};
    float v2[3]      = {points2[3] - points2[0],points2[4] - points2[1],points2[5] - points2[2]};
    // prepare for the linear approximation
    float smallest = numeric_limits<float>::max();
    float p1[3] = {0.0,0.0,0.0};
    float p2[3] = {0.0,0.0,0.0};
    for (float i = 5.0; i < 100.0; i += 0.001){
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
    if (verbose) cout << (((((float)i))/((float)length)) * 100.0) << " \%" << endl;
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
int main(int argc, char* argv[]){
  cout << "*===================* REPROJECTION *===================*" << endl;
  if (argc < 3){
    cout << "not enough arguments ... " << endl;
    cout << "USAGE: " << endl;
    cout << "./reprojection.x path/to/cameras.txt path/to/matches.txt" << endl;
    cout << "*======================================================*" << endl;
    return 0; // end it all
  } else {
    cout << "*                                                      *" << endl;
    cout << "*                     ~ UGA SSRL ~                     *" << endl;
    cout << "*        Multiview Onboard Computational Imager        *" << endl;
    cout << "*                                                      *" << endl;
  }
  cout << "*======================================================*" << endl;

  cameras_path = argv[1];
  matches_path = argv[2];

  load_matches();
  load_cameras();
  two_view_reproject();
  save_ply();

  if (verbose) cout << "done!\nresults saved to output.ply" << endl;
  if (debug) cout << "max angle: " << max_angle << " | min angle: " << min_angle << endl;

  return 0;
}
