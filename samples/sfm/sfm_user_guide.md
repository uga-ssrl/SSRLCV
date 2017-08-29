Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

Structure from Motion Simplest Pipeline Sample
@brief Structure from Motion simplest pipeline sample for "parking" use case

## Introduction ##

`nvx_sample_sfm` shows the simplest Structure from Motion pipeline which
reconstructs on-the-fly (30 fps) sparse 3D scenes from each couple of
sequential frames.

The VisionWorks Structure from Motion (SFM) sample demonstrates how
you can implement the simplest SFM pipeline using VisionWorks SFM library
building blocks (primitives). The pipeline is not intended to produce high
quality construction, but is designed to form a real-time
sample, with which you can extend and build custom SFM pipelines.

The implemented sample pipeline works under the following assumptions:

* mostly static scene outside
* low speed motion (e.g., parking scenario)
* textured ground-plane (with enough tracking features)

Consequently, the provided sample does not cover all of the use cases as other more complicated
SLAM or SFM algorithms. It is not aimed to solve SLAM tasks or to get a complete 3D-recovered
map of the surrounding environment with precise odometry. At the current stage, it can be
considered as a starting point for more sophisticated SFM algorithms. For example, in the
provided sample, recovered 3D-structure is used as an input for high-level processing
(parking spot detection).

The sample uses:

* Harris feature detector to get the list of features in two consecutive frames
that have different camera poses.
* Sparse pyramidal Lucas-Kanade optical flow to find the locations of
corresponding corners between two frames.
* Feature-based algorithm (find fundamental matrix and decompose fundamental matrix)
to estimate camera motion.
* Triangulation to reconstruct the scene and get a 3D-point cloud.
* Scale-resolving algorithm based on ground plane estimation (optional).

## Screen output##

This sample provides screen output in two modes; you can switch between them by pressing the
`P` key:
 - **2D mode**---In this mode, input video sequence is shown on the screen with
 projected 3D points.
 - **point_cloud mode**---In this mode, only recovered 3D-point cloud is shown
 and you can handle the camera by pressing the `W`, `S`, `A`, `D`, `-`, `=`
 buttons where `W` / `S` adjusts pitch, `A` / `D` adjusts yaw, and `-` / `=` adjusts zoom.

 In visualization of the 3D-point cloud, color and size differences are used
 to represent depth difference (green and small for distant points, red and
 big for close ones). In both modes, you can optionally show detected fences
 by pressing the `F` button. Fences are rendered as transparent walls with
 different colors. The color coding is the same as in the point cloud
 visualization; it depends on the distance between certain fence planes and the
 camera position (green for distant, red for close).

## Installation and Building ##

`nvx_sample_sfm` is installed in the following directory:

    /usr/share/visionworks-sfm/sources/samples/sfm

For the steps to build sample applications, see the see: nvx_module_samples section for your OS.

## Executing the Structure From Motion Sample ##

    ./nvx_sample_sfm [options]

@note The V4L platform has a permissions issue. The hardware decoder is used and the sample must be
executed with super user permissions, i.e., with `sudo`.

### Command Line Options ###

This topic provides a list of supported options and the values they consume.

#### -f, \--fullPipeline ####
- Parameter: [true]
- Description: Enables scale-resolving algorithm  based on ground plane estimation.
Without this option, scale information is extracted from an external source file with
ground truth. Here in the sample, we use smoothed coordinates for obtaining scale
information.

#### \--mask ####
- Parameter: [path to image]
- Description: Specifies an optional mask to filter out features. This must be
a grayscale image that is the same size as the input source. The sample uses
features only in the non-zero regions on the mask.

#### -h, \--help ####
- Parameter: [true]
- Description: Prints the help message.

#### -n, \--noLoop ####
- Parameter: [true]
- Description: Runs sample without loop.

### Operational Keys ###
- Use `ESC` to close the sample.
- Use `W`, `A`, `S`, `D` to rotate camera in `point_cloud` mode.
- Use `-`, `=` to zoom in point `cloud_mode`.
- Use `P` to switch between modes.
- Use `F` to show/hide fences.
- Use `G` to show/hide ground plane. This key is enabled only with
`-f, --fullPipeline` option.
- Use `Space` to pause/resume the sample.

## Related Papers ##
* **Multiple View Geometry in Computer Vision**, Richard Hartley and Andrew
Zisserman, 2003
*  **Robust Scale Estimation in Real-Time Monocular SFM for**
**Autonomous Driving**, Song, Chandraker, 2014

##  Input Data ##
- config `*.ini` file (default: `sfm_config.ini`), its URI:
'/path_to_sfm_samples_sources/VisionWorks-SFM-Ver-Samples/data/sfm/sfm_config.ini'
   It has the following structure:
  - \b harris_k
    - Parameter: [floating-point >= 0]
    - Description: Harris Corner Detector k parameter,
    values in the range [0.04, 0.15] have been reported as feasible in
    literature.

  - \b harris_thresh
    - Parameter: [floating-point >= 0]
    - Description: Harris Corner Detector corner
    strength threshold.

  - \b harris_cell_size
    - Parameter: [integer < input image dimensions]
    - Description: Size of cells for
    cell-based non-max suppression in Harris Corner Detector.

  - \b pyr_levels
    - Parameter: [integer in range [1..8]]
    - Description: Number of levels for Gaussian
    pyramid in Lucas-Kanade Optical Flow algorithm.

  - \b lk_num_iters
    - Parameter: [integer > 0]
    - Description: Number of iterations in Lucas-Kanade
    Optical Flow algorithm.

  - \b lk_win_size
    - Parameter: [integer in range [3..32]]
    - Description: Window size in Lucas-Kanade
    Optical Flow algorithm.

  - \b seed
    - Parameter: [integer >= 0]
    - Description: Seed for RNG used in Find Fundamental Matrix
    procedure.

  - \b samples
    - Parameter: [integer > 0]
    - Description: Number of samples for RNG used in Find
    Fundamental Matrix procedure.

  - \b errThreshold
    - Parameter: [floating-point >= 0]
    - Description: Error threshold in the unit of
    pixels in Find Fundamental Matrix procedure.

  - \b minPixelDis
    - Parameter: [floating-point >= 0]
    - Description: Minimal pixel distance for a valid
    triangulation in Decompose Fundamental Matrix procedure.

  - \b maxPixelDis
    - Parameter: [floating-point >= 0]
    - Description: Maximal pixel distance for a valid
    triangulation in Decompose Fundamental Matrix procedure.

  - \b medianFlowThreshold
    - Parameter: [floating-point >= 0]
    - Description: Threshold for median flow norm to
    identify special cases with zero camera motion in order
    to simplify the Find Fundamental Matrix procedure.

  - \b camModelOpt
    - Parameter: [0]
    - Description: camera model, only pinhole-camera model is supported,
    parameters below - camera intrinsics for pinhole-camera model in OpenCV
    format (see details [here](http://docs.opencv.
    org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html))

    - \b pFx
      - Parameter: [floating-point >= 0]
      - Description: Focal length along x-axis.

    - \b pFy
      - Parameter: [floating-point >= 0]
      - Description: Focal length along y-axis.

    - \b pCx
      - Parameter: [floating-point >= 0]
      - Description: Principal point x-coordinate.

    - \b pCy
      - Parameter: [floating-point >= 0]
      - Description: Principal point y-coordinate.

    - \b pK1
      - Parameter: [floating-point >= 0]
      - Description: 1st radial distortion coefficient.

    - \b pK2
      - Parameter: [floating-point >= 0]
      - Description: 2nd radial distortion coefficient.

    - \b pK3
      - Parameter: [floating-point >= 0]
      - Description: 3rd radial distortion coefficient.

    - \b pP1
      - Parameter: [floating-point >= 0]
      - Description: 1st tangential distortion coefficient.

    - \b pP2
      - Parameter: [floating-point >= 0]
      - Description: 2nd tangential distortion coefficient.

- video sequence with "parking" use case (default: `parking_sfm.mp4`), its URI:
`/path_to_sfm_samples_sources/VisionWorks-SFM-Ver-Samples/data/sfm/parking_sfm.mp4'
- [optional] external scale data for input video sequence (e.g., IMU data) when a pipeline with external scale information is used. It can be organized as follows:
    * `*.txt` file with IMU data itself (default: `imu_data.txt`), its URI:
    `/path_to_sfm_samples_sources/VisionWorks-SFM-Ver-Samples/data/sfm/imu_data.txt'
    * `*.txt` file that has matching between time stamps and frame number in video
    sequence (default: `images_timestamps.txt`), its URI:
    `/path_to_sfm_samples_sources/VisionWorks-SFM-Ver-Samples/data/sfm/images_timestamps.txt'.
    Each stamp record corresponds to each frame in sequence starting from the beginning.
    Spherical linear interpolation is used for better matching.

    records in "imu_data.txt":

        stamp:
          secs: 1411078514
          nsecs: 861046227
        smoothed_coordinates_ned:
          x: -2.97453767856
          y: 1.25072643177
          z: -9.27311098299
        ...

    records in "images_timestamps.txt":

        stamp: 1411078514.889800486
        stamp: 1411078514.969731379
        ...

## Ground Plane Estimation ##
There is one user-node that was not initially a part of the traditional pipeline.
It is aimed on resolving the scale-ambiguity problem and is used only in a full pipeline.
The approach implemented in this node is based on ground
plane estimation via homography calculation. Then distance from the camera
location to the ground plane is used to determine the scale factor of the whole scene.
For a successful result, it demands presence of features on asphalt.

## Point Cloud Post Processing ##
The fence detection algorithm implements a simple heuristic that uses 3D information
to determine occupied and free spaces in a parking lot use case. The algorithm finds
occupied and free spaces by pushing a series of line segments away from the camera
position until a "threshold" value is met. This algorithm must be configured properly
so that the empty space between two cars is identified regardless of the density of
the point-clouds and their arrangement. You may need to configure the length of the line
segments when applying fence detection algorithm to a different setup. Each line
segment is designated by slope and intersection to y-axis parameters.

In some cases, a large number of feature points at both ends of the free space combined
with few sparse features in the unoccupied region may lead the heuristic to **incorrectly
miss free space**. You must tune and customize the algorithm using the contextual
information available from your setup to get the best results.

## General Notes for Good Results ##
* For better results, provide the file with scale information. If the format of
the file with scale information differs from the one used in the sample, you
must adapt sample code using the `ReaderBase` class for extracting scale
information.
* Use a **high framerate** of at least 30 fps (depending on the movement speed).
Our experiments used between 30 and 60 fps.
* We use an image resolution of **1024x1024**; significantly higher or lower
resolutions may require some hard-coded parameters related to visualization
and fence detection to be adapted in the sample code.
* The SFM approach requires **sufficient camera translation**: Rotating the camera
without translating it at the same time will not work. Generally, sideways
motion ("parking" use case) is best, depending on the field of view of your
camera. Forwards / backwards motion is equally good.

## Note on Visualization ##
Since the renderer's camera Field Of View (FOV) is calculated by default in order to
correspond with the FOV of the camera used for capturing video, minor scale
changes are not seen in **point_cloud mode** until you start pressing `=` to zoom
point cloud out. When you press `=`, the renderer's camera FOV is changed and you may
see minor scale changes ("shaking" point cloud).

## Note on Handling Use Case with Zero Camera Motion ##
The general fundamental matrix evaluation procedure falls short in a specific use case when
there's no camera motion (i.e., the car doesn't move). To detect such cases, we use a rather
simple approach: median flow is calculated using the evaluated motion vectors and its norm
is compared with a threshold. If it's below the selected threshold, then Find Fundamental
Matrix primitive returns zero matrix. This, in turn, gives a hint to Decompose Fundamental
Matrix primitive. It returns an identity matrix as rotation and zero vector as translation
in this case.

