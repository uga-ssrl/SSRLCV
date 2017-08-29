/*
# Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __NVX_UTILS_HPP__
#define __NVX_UTILS_HPP__

#include <vector>
#include <Eigen/Dense>

#include <NVX/nvx.h>
#include "OVX/FrameSourceOVX.hpp"
#include "OVX/RenderOVX.hpp"

#include "SfM.hpp"

//row-major storage order for compatibility with vx_matrix
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

// Fence detector with Kalman filter
class FenceDetectorWithKF
{
public:
    FenceDetectorWithKF();

    void getFencePlaneVertices(vx_array point_cloud, vx_array fence_plane_vertices);
    void reset();

private:
    std::vector<Eigen::Vector3f> getFencePlanes(vx_array point_cloud);
    void processLineSeg(std::vector<Eigen::Vector2f> &line_z, float z_max, std::vector<int> &num_center_support);
    void getLineZfromKF(MatrixXf_rm &x_km_km, std::vector<Eigen::Vector2f> &line_x, std::vector<Eigen::Vector2f> &line_z, float z_max);
    void findPointIndices(std::vector<Eigen::Vector3f> &cloud, size_t num_of_points, float x_min, float x_max, float z_min, float z_max,
                          std::vector<int> &indices, int &num_center);

    MatrixXf_rm x_km_km_;
    MatrixXf_rm P_km_km_;

    int dim_;
};

struct EventData
{
    EventData(): shouldStop(false), showPointCloud(false), showFences(false), showGP(false), pause(false) {}
    bool shouldStop;
    bool showPointCloud;
    bool showFences;
    bool showGP;
    bool pause;
};

class GroundPlaneSmoother
{
public:

    GroundPlaneSmoother(int size);

    float getSmoothedY(vx_matrix gp, float x, float z);

private:

    std::vector<float> pool_;
    int size_;
    int iter_;
    bool init_;
};

bool isPointValid(const nvx_point3f_t &pt);

void eventCallback(void* eventData, vx_char key, vx_uint32, vx_uint32);

bool read(const std::string &nf, nvx::SfM::SfMParams &config, std::string &message);

void filterPoints(vx_array in, vx_array out);

std::string createInfo(bool fullPipeline, double proc_ms, double total_ms, const EventData &eventData);

#endif
