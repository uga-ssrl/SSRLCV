/*
# Copyright (c) 2014-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <cmath>
#include <iomanip>

#include "NVX/Application.hpp"
#include "OVX/UtilityOVX.hpp"
#include "NVX/ConfigParser.hpp"

#include "utils.hpp"
#include "polyfit.hpp"

void eventCallback(void* eventData, vx_char key, vx_uint32, vx_uint32)
{
    EventData* data = static_cast<EventData*>(eventData);

    if (key == 27)
    {
        data->shouldStop = true;
    }
    else if (key == 'p')
    {
        data->showPointCloud = !data->showPointCloud;
    }
    else if (key == 'f')
    {
        data->showFences = !data->showFences;
    }
    else if (key == 'g')
    {
        data->showGP = !data->showGP;
    }
    else if (key == 32)
    {
        data->pause = !data->pause;
    }
}

bool isPointValid(const nvx_point3f_t &pt)
{
    //TwoViewTraingulation node uses -10000.f values for coordinates as indicator of invalid point
    return !(fabs(pt.x + 10000.f) < 10e-6f ||
             fabs(pt.y + 10000.f) < 10e-6f ||
             fabs(pt.z + 10000.f) < 10e-6f);
}

bool checkParams(const nvx::SfM::SfMParams &params, std::string & message)
{
    if (params.minPixelDis >= params.maxPixelDis)
    {
        message = "Inconsistent minPixelDis and maxPixelDis values, maxPixelDis should be greater than minPixelDis";
    }

    return message.empty();
}

bool read(const std::string &nf, nvx::SfM::SfMParams &config, std::string &message)
{
    std::unique_ptr<nvxio::ConfigParser> ftparser(nvxio::createConfigParser());
    ftparser->addParameter("pyr_levels",nvxio::OptionHandler::unsignedInteger(&config.pyr_levels,
             nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(8u)));
    ftparser->addParameter("lk_win_size",nvxio::OptionHandler::unsignedInteger(&config.lk_win_size,
             nvxio::ranges::atLeast(3u) & nvxio::ranges::atMost(32u)));
    ftparser->addParameter("lk_num_iters",nvxio::OptionHandler::unsignedInteger(&config.lk_num_iters,
             nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(100u)));
    ftparser->addParameter("harris_k",nvxio::OptionHandler::real(&config.harris_k,
             nvxio::ranges::moreThan(0.0f)));
    ftparser->addParameter("harris_thresh",nvxio::OptionHandler::real(&config.harris_thresh,
             nvxio::ranges::moreThan(0.0f)));
    ftparser->addParameter("harris_cell_size",nvxio::OptionHandler::unsignedInteger(&config.harris_cell_size,
             nvxio::ranges::atLeast(1u)));
    ftparser->addParameter("seed",nvxio::OptionHandler::unsignedInteger(&config.seed,
             nvxio::ranges::all<unsigned int>()));
    ftparser->addParameter("samples",nvxio::OptionHandler::unsignedInteger(&config.samples,
             nvxio::ranges::moreThan(0u)));
    ftparser->addParameter("errThreshold",nvxio::OptionHandler::real(&config.errorThreshold,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("medianFlowThreshold",nvxio::OptionHandler::real(&config.medianFlowThreshold,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("minPixelDis",nvxio::OptionHandler::real(&config.minPixelDis,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("maxPixelDis",nvxio::OptionHandler::real(&config.maxPixelDis,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("camModelOpt",nvxio::OptionHandler::unsignedInteger(&config.camModelOpt,
             nvxio::ranges::atLeast(0u) & nvxio::ranges::atMost(0u)));
    ftparser->addParameter("pFx",nvxio::OptionHandler::real(&config.pFx,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("pFy",nvxio::OptionHandler::real(&config.pFy,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("pCx",nvxio::OptionHandler::real(&config.pCx,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("pCy",nvxio::OptionHandler::real(&config.pCy,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("pK1",nvxio::OptionHandler::real(&config.pK1,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("pK2",nvxio::OptionHandler::real(&config.pK2,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("pP1",nvxio::OptionHandler::real(&config.pP1,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("pP2",nvxio::OptionHandler::real(&config.pP2,
             nvxio::ranges::atLeast(0.0f)));
    ftparser->addParameter("pK3",nvxio::OptionHandler::real(&config.pK3,
             nvxio::ranges::atLeast(0.0f)));

    message = ftparser->parse(nf);

    if (!message.empty())
    {
        return false;
    }

    return checkParams(config, message);
}

void FenceDetectorWithKF::findPointIndices(std::vector<Eigen::Vector3f> &cloud, size_t num_of_points, float x_min, float x_max, float z_min, float z_max, std::vector<int> &indices,
                                           int &num_center)
{
    int num_left = 0, num_right = 0;
    for (size_t i = 0; i < num_of_points; i++)
    {
        Eigen::Vector3f v = cloud[i];
        if (v(0) > x_min && v(0) <= x_max && v(2) > z_min && v(2) <= z_max)
        {
            indices.push_back((int)i);
        }
    }
    for (size_t i = 0; i < indices.size(); i++)
    {
        Eigen::Vector3f v = cloud[indices[i]];
        if (v(0) < x_min + 0.6f)
            num_left++;
        if (v(0) > x_max - 0.6f)
            num_right++;
    }
    num_center = static_cast<int>(indices.size()-num_left-num_right);
}

void FenceDetectorWithKF::processLineSeg(std::vector<Eigen::Vector2f> &line_z, float z_max, std::vector<int> &num_center_support)
{
    int num_seg = static_cast<int>(line_z.size());
    int si, si_start, si_end;


    // if the first segment is open and the second segment is blocked, change the first segment to be a blocked segment
    if (fabs(line_z[0](0) - z_max) < 1e-4f && fabs(line_z[0](1) - z_max) < 1e-4f && line_z[1](0) < z_max && line_z[1](1) < z_max)
    {
        line_z[0](0) = line_z[0](0) - line_z[0](1) + line_z[1](0);
        line_z[0](1) = line_z[1](0);
    }

    // if the last segment is open and the second last segment is blocked, change the last segment to be a blocked segment
    if (fabs(line_z[num_seg-1](0) - z_max) < 1e-4f && fabs(line_z[num_seg-1](1) - z_max) < 1e-4f && line_z[num_seg-2](0) < z_max && line_z[num_seg-2](1) < z_max)
    {
        line_z[num_seg-1](1) = line_z[num_seg-1](1) - line_z[num_seg-1](0) + line_z[num_seg-2](1);
        line_z[num_seg-1](0) = line_z[num_seg-2](1);
    }

    // search for the first block segment
    si = 0;
    while(fabs(line_z[si](0) - z_max) < 1e-4f && fabs(line_z[si](1) - z_max) < 1e-4f && si < num_seg-1)
    {
        si++;
    }
    si_start = si;

    while(line_z[si](0) < z_max && line_z[si](1) < z_max && si < num_seg-1)
    {
        si++;
    }
    si_end = si-1;


    int max_num_center_support = num_center_support[si_start];
    int in = si_start;
    for (int i = si_start; i <= si_end; i++)
    {
        if (num_center_support[i] > max_num_center_support)
        {
            max_num_center_support = num_center_support[i];
            in = i;
        }
    }

    float z_seg1 = std::min(line_z[in](0), line_z[in](1));

    for (int i = in; i >= si_start; i--)
    {
        if (fabs(std::min(line_z[i](0), line_z[i](1))-z_seg1) > 0.2f)
        {
            line_z[i](0) = line_z[i](0) - line_z[i](1) + line_z[i+1](0);
            line_z[i](1) = line_z[i+1](0);
        }

    }

    for (int i = in; i <= si_end; i++)
    {
        if (fabs(std::min(line_z[i](0), line_z[i](1))-z_seg1) > 0.2f)
        {
            line_z[i](1) = line_z[i](1) - line_z[i](0) + line_z[i-1](1);
            line_z[i](0) = line_z[i-1](1);
        }
    }

    // search for the second block segment
    while(fabs(line_z[si](0) - z_max) < 1e-4f && fabs( line_z[si](1) - z_max) < 1e-4f && si < num_seg - 1)
    {
        si++;
    }
    si_start = si;

    while(line_z[si](0) < z_max && line_z[si](1) < z_max && si < num_seg - 1)
    {
        si++;
    }
    si_end = si - 1;

    max_num_center_support = num_center_support[si_start];
    in = si_start;
    for (int i = si_start; i <= si_end; i++)
    {
        if (num_center_support[i] > max_num_center_support)
        {
            max_num_center_support = num_center_support[i];
            in = i;
        }
    }

    float z_seg2 = std::min(line_z[in](0), line_z[in](1));

    for (int i = in; i >= si_start; i--)
    {
        if (fabs(std::min(line_z[i](0), line_z[i](1)) - z_seg2) > 0.2f)
        {
            line_z[i](0) = line_z[i](0) - line_z[i](1) + line_z[i+1](0);
            line_z[i](1) = line_z[i+1](0);
        }
    }

    for (int i = in; i <= si_end; i++)
    {
        if (fabs(std::min(line_z[i](0), line_z[i](1)) - z_seg2) > 0.2f)
        {
            line_z[i](1) = line_z[i](1) - line_z[i](0) + line_z[i-1](1);
            line_z[i](0) = line_z[i-1](1);
        }
    }

    // search for the third segment
    while(fabs(line_z[si](0) - z_max) < 1e-4f && fabs(line_z[si](1) - z_max) < 1e-4f && si < num_seg - 1)
    {
        si++;
    }
    si_start = si;

    while(line_z[si](0) < z_max && line_z[si](1) < z_max && si < num_seg - 1)
    {
        si++;
    }
    si_end = si - 1;

    max_num_center_support = num_center_support[si_start];
    in = si_start;
    for (int i = si_start; i <= si_end; i++)
    {
        if (num_center_support[i] > max_num_center_support)
        {
            max_num_center_support = num_center_support[i];
            in = i;
        }
    }

    float z_seg3 = std::min(line_z[in](0), line_z[in](1));

    for (int i = in; i >= si_start; i--)
    {
        if (fabs(std::min(line_z[i](0), line_z[i](1))-z_seg3) > 0.2f)
        {
            line_z[i](0) = line_z[i](0) - line_z[i](1) + line_z[i+1](0);
            line_z[i](1) = line_z[i+1](0);
        }

    }

    for (int i = in; i <= si_end; i++)
    {
        if (fabs(std::min(line_z[i](0), line_z[i](1)) - z_seg3) > 0.2f)
        {
            line_z[i](1) = line_z[i](1) - line_z[i](0) + line_z[i-1](1);
            line_z[i](0) = line_z[i-1](1);
        }
    }

    // if an open segment appears in the middle of the two blocked segment, connect the two blocked segments
    for (int i = 1; i < num_seg - 1; i++)
    {
        if (fabs(line_z[i](0) - z_max) < 1e-4f && fabs(line_z[i](1) - z_max) < 1e-4f && line_z[i-1](0) < z_max && line_z[i-1](1) < z_max
                && line_z[i+1](0) < z_max && line_z[i+1](1) < z_max)
        {
            line_z[i](0) = line_z[i-1](1);
            line_z[i](1) = line_z[i+1](0);
        }
    }

    // if two open segments appear in the middle of the two blocked segment, connect the two blocked segments
    for (int i = 1; i < num_seg - 2; i++)
    {
        if (fabs(line_z[i](0) - z_max) < 1e-4f && fabs(line_z[i](1) - z_max) < 1e-4f && fabs(line_z[i+1](0) - z_max) < 1e-4f && fabs(line_z[i+1](1) - z_max) < 1e-4f
                && line_z[i-1](0) < z_max && line_z[i-1](1) < z_max && line_z[i+2](0) < z_max && line_z[i+2](1) < z_max)
        {
            line_z[i](0) = line_z[i-1](1);
            line_z[i+1](1) = line_z[i+2](0);
            line_z[i](1) = 0.5f * (line_z[i](0) + line_z[i+1](1));
            line_z[i+1](0) = 0.5f * (line_z[i](0) + line_z[i+1](1));
        }
    }
}

void FenceDetectorWithKF::getLineZfromKF(MatrixXf_rm &x_km_km, std::vector<Eigen::Vector2f> &line_x, std::vector<Eigen::Vector2f> &line_z, float z_max)
{
    size_t size = line_x.size();
    for (size_t i = 0; i < size; i++)
    {
        if (fabs(line_z[i](0) - z_max) < 1e-4f && fabs(line_z[i](1) - z_max) < 1e-4f)
        {
            line_z[i](0) = z_max;
            line_z[i](1) = z_max;
        }
        else
        {
            line_z[i](0) = line_x[i](0) * x_km_km(2*i, 0) + x_km_km(2*i+1, 0);
            line_z[i](1) = line_x[i](1) * x_km_km(2*i, 0) + x_km_km(2*i+1, 0);
        }
    }
}

FenceDetectorWithKF::FenceDetectorWithKF()
{
    dim_ = 40;

    reset();
}

void FenceDetectorWithKF::reset()
{
    x_km_km_ = MatrixXf_rm::Zero(dim_, 1);
    P_km_km_ = MatrixXf_rm::Identity(dim_, dim_) * 10000.0f;
}

std::vector<Eigen::Vector3f> FenceDetectorWithKF::getFencePlanes(vx_array point_cloud)
{
    std::vector<Eigen::Vector3f> p3_line;

    vx_size num_of_points;
    vxQueryArray(point_cloud, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_of_points, sizeof(num_of_points));

    std::vector<Eigen::Vector3f> cloud;
    if (num_of_points > 0)
    {
        vx_size stride = 0;
        void *ptr = NULL;
        vxAccessArrayRange(point_cloud, 0, num_of_points, &stride, &ptr, VX_READ_ONLY);
        for (size_t i = 0; i < num_of_points; i++)
        {
            nvx_point3f_t pt = vxArrayItem(nvx_point3f_t, ptr, i, stride);
            if (isPointValid(pt))
            {
                cloud.push_back(Eigen::Vector3f(pt.x, -pt.y, pt.z)); // y coordinate is reversed
            }

        }
        vxCommitArrayRange(point_cloud, 0, num_of_points, ptr);
    }
    else
    {
        return p3_line;
    }

    // filter out the y<-0.9
    for (size_t i=0; i<cloud.size(); i++)
    {
        Eigen::Vector3f v = cloud[i];
        if (v(1) < -0.9f)
        {
            cloud[i] = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
        }
    }
    num_of_points = cloud.size();

    MatrixXf_rm F = Eigen::MatrixXf::Identity(dim_, dim_);
    MatrixXf_rm H = Eigen::MatrixXf::Identity(dim_, dim_);
    MatrixXf_rm Q = Eigen::MatrixXf::Identity(dim_, dim_);
    MatrixXf_rm R = Eigen::MatrixXf::Identity(dim_, dim_);
    MatrixXf_rm z_k(dim_, 1);

    // KF prediction (future work: combine the physical car motion model in KF framework)
    MatrixXf_rm x_k_km = F * x_km_km_;
    MatrixXf_rm P_k_km = F * P_km_km_*F.transpose() + Q;

    int z_k_count = 0;

    // construct line segments
    // case1: output a fitted line if there are enough supported features
    // case2: output a flat line for open space
    std::vector<int> indices;
    std::vector<Eigen::Vector2f> p2;
    std::vector<Eigen::Vector2f> line_x;
    std::vector<Eigen::Vector2f> line_z;
    std::vector<int> num_center_support;
    float x_start = -2.0f, x_interval = 0.2f, x_end = 2.0f;
    float z_start = 2.0f, z_interval = 0.4f, z_end = 5.2f;
    while(x_start < x_end)
    {
        z_start = 2.0f;
        while(z_start < z_end)
        {
            indices.clear();
            p2.clear();
            int num_center = 0;
            findPointIndices(cloud, num_of_points, x_start - 0.8f, x_start + 0.8f, z_start - 0.5f, z_start + 0.5f, indices,
                             num_center);

            for (size_t i = 0; i < indices.size(); i++)
            {
                p2.push_back(Eigen::Vector2f(cloud[indices[i]](0), cloud[indices[i]](2)));
            }

            if (p2.size() > 3 && num_center > 2)
            {
                std::vector<float> x;
                std::vector<float> y;
                for (size_t i = 0; i < p2.size(); i++)
                {
                    x.push_back(p2[i](0));
                    y.push_back(p2[i](1));
                }
                std::vector<float> coeffs = mathalgo::polyfit(x, y, 1);
                float a = coeffs[1];
                float b = coeffs[0];
                line_x.push_back(Eigen::Vector2f(x_start, x_start+x_interval));
                line_z.push_back(Eigen::Vector2f(x_start*a + b, (x_start + x_interval)*a + b));
                num_center_support.push_back(num_center);

                z_k(z_k_count * 2, 0) = a;
                z_k(z_k_count * 2 + 1, 0) = b;
                z_k_count++;

                break;
            }
            else
            {
                if (fabs(z_start + z_interval-z_end) < 1e-2f)
                {
                    float a = 0;
                    float b = z_start + z_interval;

                    line_x.push_back(Eigen::Vector2f(x_start, x_start + x_interval));
                    line_z.push_back(Eigen::Vector2f(x_start*a + b, (x_start + x_interval)*a + b));
                    num_center_support.push_back(num_center);

                    z_k(z_k_count * 2, 0) = a;
                    z_k(z_k_count * 2 + 1, 0) = b;
                    z_k_count++;

                    break;
                }
            }

            z_start += z_interval;
        }

        x_start += x_interval;
    }

    // KF update
    MatrixXf_rm y_k = z_k - H * x_k_km;
    MatrixXf_rm S_k = H * P_k_km * H.transpose() + R;
    MatrixXf_rm K_k = P_k_km * H.transpose() * S_k.inverse();

    x_km_km_ = x_k_km + K_k * y_k; // for the next state
    P_km_km_ = (MatrixXf_rm::Identity(dim_, dim_) - K_k * H) * P_k_km; // for the next state

    // convert KF state into line_z
    getLineZfromKF(x_km_km_, line_x, line_z, z_end);

    // process the line segments
    processLineSeg(line_z, z_end, num_center_support);

    // expected ground plane location, should be calculated
    float start_y_coord = 0.5f;
    float end_y_coord = 1.4f;

    for (size_t i=0; i<line_x.size(); i++)
    {
        p3_line.push_back(Eigen::Vector3f(line_x[i](0), end_y_coord, line_z[i](0)));
        p3_line.push_back(Eigen::Vector3f(line_x[i](0), start_y_coord, line_z[i](0)));
        p3_line.push_back(Eigen::Vector3f(line_x[i](1), start_y_coord, line_z[i](1)));
        p3_line.push_back(Eigen::Vector3f(line_x[i](1), end_y_coord, line_z[i](1)));
    }
    return p3_line;
}

void FenceDetectorWithKF::getFencePlaneVertices(vx_array point_cloud, vx_array fence_plane_vertices)
{
    NVXIO_SAFE_CALL( vxTruncateArray(fence_plane_vertices, 0) );

    std::vector<Eigen::Vector3f> plane_points = getFencePlanes(point_cloud);

    if (plane_points.size() % 4 == 0)
    {
        for(size_t i = 0; i < plane_points.size(); i++)
        {
            nvx_point3f_t vx_pt;
            vx_pt.x = plane_points[i](0);
            vx_pt.y = plane_points[i](1);
            vx_pt.z = plane_points[i](2);

            vxAddArrayItems(fence_plane_vertices, 1, &vx_pt, sizeof(vx_pt));
        }
    }
}


void filterPoints(vx_array in, vx_array out)
{
    vxTruncateArray(out, 0);

    vx_size size = 0;
    NVXIO_SAFE_CALL( vxQueryArray(in, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size)) );

    if (size > 0)
    {
        void *in_ptr = 0;
        vx_size in_stride = 0;

        NVXIO_SAFE_CALL( vxAccessArrayRange(in, 0, size, &in_stride, &in_ptr, VX_READ_ONLY) );

        for (vx_size i = 0; i < size; ++i)
        {
            nvx_point3f_t pt = vxArrayItem(nvx_point3f_t, in_ptr, i, in_stride);

            if (isPointValid(pt))
            {
                NVXIO_SAFE_CALL( vxAddArrayItems(out, 1, &pt, sizeof(pt)) );
            }
        }

        NVXIO_SAFE_CALL( vxCommitArrayRange(in, 0, 0, in_ptr) );
    }
}


std::string createInfo(bool fullPipeline, double proc_ms, double total_ms, const EventData &eventData)
{
    std::ostringstream txt;
    txt << std::fixed << std::setprecision(1);
    txt << "Algorithm: " << proc_ms << " ms / " << 1000.0f/ proc_ms << " FPS" << std::endl;
    txt << "Display: " << total_ms  << " ms / " << 1000.0f / total_ms << " FPS" << std::endl;

    txt << std::setprecision(6);
    txt.unsetf(std::ios_base::floatfield);
    txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;

    if (eventData.showPointCloud)
    {
        txt << "W / S - pitch" << std::endl;
        txt << "A / D - yaw" << std::endl;
        txt << "- / = - zoom" << std::endl;
        txt << "P - show frames" << std::endl;
    }
    else
    {
        txt << "P - show point cloud" << std::endl;
    }

    txt << "F - show fences" << std::endl;

    if (fullPipeline)
        txt << "G - show ground plane" << std::endl;

    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - quit the demo" << std::endl;

    return txt.str();
}


GroundPlaneSmoother::GroundPlaneSmoother(int size):
    pool_(size), size_(size), iter_(0), init_(false) {}

float GroundPlaneSmoother::getSmoothedY(vx_matrix gp, float x, float z)
{
    float gpData[3];
    vxReadMatrix(gp, gpData);

    float y = (1 - gpData[0]*x - gpData[2]*z) / gpData[1];

    if (!init_)
    {
        for(int i=0; i<size_; ++i)
            pool_[i] = y;

        init_ = true;
    }

    iter_ = iter_ % size_;
    pool_[iter_] = y;

    ++iter_;

    float aver = 0;
    for(int i=0; i<size_; ++i)
        aver += pool_[i];

    aver /= size_;

    return aver;
}
