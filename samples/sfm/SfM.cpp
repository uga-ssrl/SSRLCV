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


#include <climits>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "OVX/UtilityOVX.hpp"
#include "NVX/Application.hpp"

#include "SfM.hpp"
#include "utils.hpp"

#include <NVX/sfm/sfm.h>
#include <cuda_runtime.h>


//row-major storage order for compatibility with vx_matrix
typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3f_rm;

// Constant for user kernels
enum {
    // Library ID
    USER_LIBRARY = 0x1,
    // Kernel ID
    USER_KERNEL_SCALE_CLOUD = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x0,
    USER_KERNEL_FILTER_OUTLIERS = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x1,
    USER_KERNEL_FIND_GROUND_PLANE = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x2
};

static vx_status registerScaleCloudKernel(vx_context context);
static vx_node scaleCloudNode(vx_graph graph, vx_matrix gp, vx_array cloud, vx_float32 expectedGpHeight, vx_array scaledCloud, vx_matrix scaledGP);
static vx_status registerFindGroundPlaneKernel(vx_context context);
static vx_node findGroundPlaneNode(vx_graph graph,  vx_array points2d1,
                                   vx_array points2d2,
                                   vx_array triangulatedPoints3d,
                                   vx_uint32 gpHeightThreshold,
                                   vx_float32 gpReprojThreshold,
                                   vx_matrix gpNormal,
                                   vx_matrix gp);

namespace
{
    //
    // Base class for reading
    //
    class ReaderBase
    {
    public:

        bool open(const std::string &file)
        {
            if (reader_.is_open())
                reader_.close();

            reader_.open(file.c_str());
            if (!reader_.is_open())
            {
                return false;
            }

            return true;
        }

        virtual ~ReaderBase() {}

    protected:

        std::ifstream reader_;

        bool goToNode(const std::string &node)
        {
            std::string str;
            while(true)
            {
                if (reader_.eof())
                {
                    return false;
                }

                reader_ >> str;

                if (str.find(node) != std::string::npos)
                    return true;
            }
        }

        template<class T> bool get(const std::string &key, T &result)
        {
            std::string str;
            while(true)
            {
                if (reader_.eof())
                {
                    return false;
                }

                reader_ >> str;

                if (str.find(key) != std::string::npos)
                    break;
            }

            reader_ >> str;

            std::istringstream ss(str);
            if ( !(ss >> result) )
                return false;

            return true;
        }
    };

#define READER_SAFE_CALL(operation) \
    do \
    { \
        bool status = (operation); \
        if (status != true) \
        { \
            return false; \
        } \
    } while (0)

    //
    // Class for IMU data reading
    //
    class ImuDataReader: public ReaderBase
    {
        template<class T> bool getValue(const std::string &key, T &result)
        {
            std::string str;
            reader_ >> str;
            if (str == key+":")
            {
                reader_ >> result;
                return true;
            }
            else
            {
                return false;
            }
        }

    public:
        struct Data
        {
            Eigen::Vector4f orientation;
            Eigen::Vector3f coord;

            vx_int64 t;

            Data(): t(0) {}
        };

        bool read(Data &data)
        {
            if (reader_.is_open())
            {
                READER_SAFE_CALL( goToNode("stamp") );
                READER_SAFE_CALL( getValue<vx_int64>("secs", data.t) );

                vx_int64 ns;
                READER_SAFE_CALL( getValue<vx_int64>("nsecs", ns) );
                data.t = data.t*1000000000 + ns;

                READER_SAFE_CALL( goToNode("orientation") );
                READER_SAFE_CALL( getValue<float>("x", data.orientation(0)) );
                READER_SAFE_CALL( getValue<float>("y", data.orientation(1)) );
                READER_SAFE_CALL( getValue<float>("z", data.orientation(2)) );
                READER_SAFE_CALL( getValue<float>("w", data.orientation(3)) );

                READER_SAFE_CALL( goToNode("smoothed_coordinates_ned") );
                READER_SAFE_CALL( getValue<float>("x", data.coord(0)) );
                READER_SAFE_CALL( getValue<float>("y", data.coord(1)) );
                READER_SAFE_CALL( getValue<float>("z", data.coord(2)) );

                return true;
            }
            else
            {
                return false;
            }
        }
    };

    vx_int64 convertSecStrToNanosecInt64(const std::string &str)
    {
        size_t ind = str.find(".");

        std::string sec = str.substr(0, ind);
        std::string ns = str.substr(ind+1, str.size());

        std::istringstream secVal(sec);
        std::istringstream nsVal(ns);

        vx_int64 t1, t2, t;
        secVal >> t1;
        nsVal >> t2;

        t = t1 * 1000000000 + t2;

        return t;
    }

    //
    // Class for image data reading
    //
    class ImageDataReader: public ReaderBase
    {
    public:
        struct Data
        {
            vx_int64 t;

            Data(): t(0) {}
        };

        bool read(Data &data)
        {
            if (reader_.is_open())
            {
                std::string strTime;
                READER_SAFE_CALL( get<std::string>("stamp", strTime) );
                data.t = convertSecStrToNanosecInt64(strTime);

                return true;
            }
            else
            {
                return false;
            }
        }
    };

    //
    // Spherical linear interpolation of quaternions
    //
    void SLERP(const Eigen::Vector4f &q1, const Eigen::Vector4f &q2, float t, Eigen::Vector4f &q)
    {

        assert(t >= 0 && t <= 1);

        Eigen::Vector4f temp(q2);

        float cosTheta = q1.dot(q2);

        if( cosTheta < 0.0f )
        {
            cosTheta = - cosTheta;
            temp = - temp;
        }

        float ratioA, ratioB;
        if( ( 1.0f - cosTheta ) > 1e-6f )
        {
            float theta = acos( cosTheta );
            float invSin = 1.0f / sin( theta );
            ratioA = sin( ( 1.0f - t ) * theta ) * invSin;
            ratioB = sin( t * theta ) * invSin;
        }
        else
        {
            ratioA = 1.0f - t;
            ratioB = t;
        }

        q = ratioA * q1 + ratioB * temp;
    }

    //
    // Class that allows to get IMU data for arbitrary timestamp using interpolation
    //
    class IMUDataInterpolator
    {
    public:

        bool init(const std::string &imuDataFile)
        {
            if (!imuReader_.open(imuDataFile))
            {
                return false;
            }

            bool res = imuReader_.read(prevData_);

            return res;
        }

        bool interpolate(vx_int64 frameT, ImuDataReader::Data &inter)
        {
            ImuDataReader::Data data;
            while(true)
            {
                if (!imuReader_.read(data))
                    return false;

                if (prevData_.t <= frameT && data.t > frameT)
                    break;

                prevData_ = data;
            }

            inter.t = frameT;


            float t = float(frameT - prevData_.t) / (data.t - prevData_.t);
            SLERP(prevData_.orientation, data.orientation, t, inter.orientation);
            inter.coord = prevData_.coord + (data.coord - prevData_.coord) * t;

            return true;
        }

    private:

        ImuDataReader imuReader_;
        ImuDataReader::Data prevData_;
    };

    //
    // SfM based on Harris features + PyrLK optical flow + RANSAC + Triangulation
    //

    class SfMHarrisPyrLK : public nvx::SfM
    {
    public:
        SfMHarrisPyrLK(vx_context context, const SfMParams& params);
        ~SfMHarrisPyrLK();

        vx_status init(vx_image firstFrame, vx_image mask, const std::string &imuDataFile, const std::string &frameDataFile);
        vx_status track(vx_image newFrame, vx_image mask = 0);

        vx_array getPrevFeatures() const;
        vx_array getCurrFeatures() const;
        vx_array getPointCloud() const;
        vx_matrix getRotation() const;
        vx_matrix getTranslation() const;
        vx_matrix getGroundPlane() const;

        void printPerfs() const;

    private:
        void createDataObjects();

        void processFirstFrame(vx_image frame, vx_image mask);
        void createMainGraph(vx_image frame, vx_image mask);

        void release();

        SfMParams params_;

        vx_context context_;

        // Format for current frames
        vx_df_image format_;
        vx_uint32 width_;
        vx_uint32 height_;

        // Pyramids for two successive frames
        vx_delay pyr_delay_;

        // Points to track for two successive frames
        vx_delay pts_delay_;

        // Camera parameters
        vx_matrix lensIntrinsics_;
        vx_matrix lensDistortion_;

        // Rotation and translation matrix
        vx_matrix fundMat_;
        vx_matrix rotM_;
        vx_matrix transM_;

        vx_matrix groundPlane_;
        vx_matrix scaledGroundPlane_;
        vx_matrix gpNormal_;

        // Tracked points
        vx_array kp_tracked_list_;
        vx_array kp_filtered_list_;

        // Triangulated point clouds
        vx_array pointClouds_;
        vx_array prevPointClouds_;

        // Randomly generated indices
        vx_array indices_;

        // RNG state
        nvx_sfm_random_state state_;

        // Scale from external data
        vx_scalar scale_;

        // OpenVX graph and nodes (used to print performance results)
        vx_graph main_graph_;
        vx_node cvt_color_node_;
        vx_node pyr_node_;
        vx_node opt_flow_node_;
        vx_node feature_track_node_;
        vx_node find_fundamental_mat_node_;
        vx_node decompose_fundamental_node_;
        vx_node two_view_triangulation_node_;
        vx_node find_ground_plane_node_;
        vx_node calc_scale_node_;

        vx_uint32 gpHeightThreshold_;
        vx_float32 gpReprojThreshold_;
        vx_float32 expectedGpHeight;


        bool useExternalScaleData_;
        bool usePrevPointClouds_;

        ImageDataReader imageDataReader_;
        IMUDataInterpolator IMUDataInterpolator_;

        ImuDataReader::Data prevImuData_;

        const vx_size pointsCapacity_;

        static const vx_uint32 PTS_IN_RANSAC = 7;

    };

    SfMHarrisPyrLK::SfMHarrisPyrLK(vx_context context, const SfMParams& params) :
        params_(params), pointsCapacity_(2000)
    {
        context_ = context;

        format_ = VX_DF_IMAGE_VIRT;
        width_ = 0;
        height_ = 0;

        pyr_delay_ = 0;

        pts_delay_ = 0;

        lensIntrinsics_ = 0;
        lensDistortion_ = 0;

        fundMat_ = 0;
        rotM_ = 0;
        transM_ = 0;
        groundPlane_ = 0;
        scaledGroundPlane_ = 0;
        gpNormal_ = 0;

        kp_tracked_list_ = 0;
        kp_filtered_list_ = 0;

        pointClouds_ = 0;
        prevPointClouds_ = 0;

        indices_ = 0;

        state_ = 0;

        scale_ = 0;

        main_graph_ = 0;

        cvt_color_node_ = 0;
        pyr_node_ = 0;
        opt_flow_node_ = 0;
        feature_track_node_ = 0;
        find_fundamental_mat_node_ = 0;
        two_view_triangulation_node_ = 0;
        decompose_fundamental_node_ = 0;
        find_ground_plane_node_ = 0;
        calc_scale_node_ = 0;

        gpHeightThreshold_ = 0;
        gpReprojThreshold_ = 0.5f;
        expectedGpHeight = 1.0f;

        useExternalScaleData_ = false;
        usePrevPointClouds_ = false;

    }

    SfMHarrisPyrLK::~SfMHarrisPyrLK()
    {
        release();
    }

    vx_status SfMHarrisPyrLK::init(vx_image firstFrame, vx_image mask, const std::string &imuDataFile, const std::string &frameDataFile)
    {
        if (imuDataFile.empty() && frameDataFile.empty())
        {
            useExternalScaleData_ = false;
        }
        else
        {
            // open files that contain IMU and frame data
            if ( !(imuDataFile.empty() || frameDataFile.empty()) )
            {
                if ( !(IMUDataInterpolator_.init(imuDataFile) && imageDataReader_.open(frameDataFile)) )
                {
                    vxAddLogEntry((vx_reference)context_,
                                  VX_FAILURE,
                                  "cannot open IMU or/and frame data files");
                    return VX_FAILURE;
                }
                else
                {
                    useExternalScaleData_ = true;
                }
            }
            else
            {
                vxAddLogEntry((vx_reference)context_,
                              VX_FAILURE,
                              "both of IMU and frame data files should be provided");
                return VX_FAILURE;
            }
        }

        // Check input format
        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        NVXIO_ASSERT(format == VX_DF_IMAGE_RGBX);

        if (mask)
        {
            vx_df_image mask_format = VX_DF_IMAGE_VIRT;
            vx_uint32 mask_width = 0;
            vx_uint32 mask_height = 0;

            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_FORMAT, &mask_format, sizeof(mask_format)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_WIDTH, &mask_width, sizeof(mask_width)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_HEIGHT, &mask_height, sizeof(mask_height)) );

            NVXIO_ASSERT(mask_format == VX_DF_IMAGE_U8);
            NVXIO_ASSERT(mask_width == width);
            NVXIO_ASSERT(mask_height == height);
        }

        if (useExternalScaleData_)
        {
            // get IMU data and timestamp for the first frame
            ImageDataReader::Data imageData;
            if (!imageDataReader_.read(imageData))
            {

                vxAddLogEntry((vx_reference)context_, VX_FAILURE,
                              "cannot read frame data file");
                return VX_FAILURE;
            }
            if (!IMUDataInterpolator_.interpolate(imageData.t, prevImuData_))
            {
                vxAddLogEntry((vx_reference)context_, VX_FAILURE,
                              "cannot read IMU data file");
                return VX_FAILURE;
            }
        }

        // Re-create graph if the input size was changed

        if (width != width_ || height != height_)
        {
            release();

            format_ = format;
            width_ = width;
            height_ = height;

            createDataObjects();

            // assume that ground plane is located in the bottom of the frame and
            // its points have vertical coordinates bigger than gpHeightThresh
            gpHeightThreshold_ = 3 * height / 4;

            // assume that the normal to the ground plane is vertical
            float gpNormalData[] = {0, 1, 0};
            vxWriteMatrix(gpNormal_, gpNormalData);

            //assume that expected height from camera to ground plane is equal to 1.5 m
            expectedGpHeight = 1.5f;

            createMainGraph(firstFrame, mask);
        }

        // Process first frame

        processFirstFrame(firstFrame, mask);

        return VX_SUCCESS;
    }

    vx_status SfMHarrisPyrLK::track(vx_image newFrame, vx_image mask)
    {
        // Age delays
        NVXIO_SAFE_CALL( vxAgeDelay(pyr_delay_) );
        NVXIO_SAFE_CALL( vxAgeDelay(pts_delay_) );

        // Check input format

        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        NVXIO_ASSERT(format == format_);
        NVXIO_ASSERT(width == width_);
        NVXIO_ASSERT(height == height_);

        if (mask)
        {
            vx_df_image mask_format = VX_DF_IMAGE_VIRT;
            vx_uint32 mask_width = 0;
            vx_uint32 mask_height = 0;

            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_FORMAT, &mask_format, sizeof(mask_format)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_WIDTH, &mask_width, sizeof(mask_width)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_HEIGHT, &mask_height, sizeof(mask_height)) );

            NVXIO_ASSERT(mask_format == VX_DF_IMAGE_U8);
            NVXIO_ASSERT(mask_width == width_);
            NVXIO_ASSERT(mask_height == height_);

            NVXIO_SAFE_CALL( vxSetParameterByIndex(feature_track_node_, 2, (vx_reference)mask) );
        }

        // Process graph

        NVXIO_SAFE_CALL( vxSetParameterByIndex(cvt_color_node_, 0, (vx_reference)newFrame) );

        vx_size num_items = 0;
        vxQueryArray( (vx_array)vxGetReferenceFromDelay(pts_delay_, -1), VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(num_items));

        if(num_items >= PTS_IN_RANSAC)
        {
            nvxSfmGenerateRandomSamples(state_, indices_, num_items, PTS_IN_RANSAC);
            vxSetParameterByIndex(find_fundamental_mat_node_, 6, (vx_reference)indices_);
        }

        if (useExternalScaleData_)
        {
            ImageDataReader::Data imageData;
            ImuDataReader::Data imuData;

            if (!imageDataReader_.read(imageData))
            {
                vxAddLogEntry((vx_reference)context_, VX_FAILURE,
                              "cannot read frame data file");
                return VX_FAILURE;
            }
            if (!IMUDataInterpolator_.interpolate(imageData.t, imuData))
            {
                vxAddLogEntry((vx_reference)context_, VX_FAILURE,
                              "cannot read IMU data file");
                return VX_FAILURE;
            }

            Eigen::Vector3f translation = imuData.coord - prevImuData_.coord;

            prevImuData_ = imuData;

            float scale = translation.norm();

            vxWriteScalarValue(scale_, &scale);

            NVXIO_SAFE_CALL( vxSetParameterByIndex(two_view_triangulation_node_, 4, (vx_reference)scale_) );
        }


        NVXIO_SAFE_CALL( vxProcessGraph(main_graph_) );

        float rotM_arr[9] = {0.0f};
        float transM_arr[3] = {0.0f};

        NVXIO_SAFE_CALL( vxReadMatrix(rotM_, rotM_arr) );
        NVXIO_SAFE_CALL( vxReadMatrix(transM_, transM_arr) );

        bool eyeR = true;
        bool zeroT = true;

        for (uint i = 0; i < 9; i++)
        {
            //diagonal elements
            if (i == 0 || i == 4 || i == 8)
            {
                if (rotM_arr[i] != 1.0f)
                {
                    eyeR = false;
                    break;
                }
            }
            else
            {
                if (rotM_arr[i] != 0.0f)
                {
                    eyeR = false;
                    break;
                }
            }
        }

        for (uint i = 0; i < 3; i++)
        {
            if (transM_arr[i] != 0.0f)
            {
                zeroT = false;
                break;
            }
        }

        if (eyeR && zeroT)
        {
            usePrevPointClouds_ = true;
        }
        else
        {
            usePrevPointClouds_ = false;

            vx_enum itemType = 0;
            NVXIO_SAFE_CALL( vxQueryArray(pointClouds_, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemType, sizeof(itemType)) );

            vx_size itemSize = 0;
            NVXIO_SAFE_CALL( vxQueryArray(pointClouds_, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemSize, sizeof(itemSize)) );

            vx_size pointCloudsCount = 0;
            NVXIO_SAFE_CALL( vxQueryArray(pointClouds_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &pointCloudsCount, sizeof(pointCloudsCount)));


            NVXIO_SAFE_CALL( vxTruncateArray(prevPointClouds_, 0) );

            if (pointCloudsCount > 0)
            {
                void* pPointClouds = NULL;
                vx_size pointCloudsStride = 0;
                NVXIO_SAFE_CALL( vxAccessArrayRange(pointClouds_, 0, pointCloudsCount, &pointCloudsStride, &pPointClouds, NVX_READ_ONLY_CUDA) );

                std::vector<vx_uint8> prev_point_clouds_vec(itemSize * pointCloudsCount);
                NVXIO_SAFE_CALL( vxAddArrayItems(prevPointClouds_, pointCloudsCount, &prev_point_clouds_vec[0], itemSize) );

                void* pPrevPointClouds = NULL;
                vx_size prevPointCloudsStride = 0;
                NVXIO_SAFE_CALL( vxAccessArrayRange(prevPointClouds_, 0, pointCloudsCount, &prevPointCloudsStride, &pPrevPointClouds, NVX_WRITE_ONLY_CUDA) );

                NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(pPrevPointClouds, pPointClouds, pointCloudsCount * itemSize, cudaMemcpyDeviceToDevice, NULL) );

                NVXIO_SAFE_CALL( vxCommitArrayRange(pointClouds_, 0, pointCloudsCount, pPointClouds) );
                NVXIO_SAFE_CALL( vxCommitArrayRange(prevPointClouds_, 0, pointCloudsCount, pPrevPointClouds) );
            }

        }

        return VX_SUCCESS;
    }

    vx_array SfMHarrisPyrLK::getPrevFeatures() const
    {
        return (vx_array)vxGetReferenceFromDelay(pts_delay_, -1);
    }

    vx_array SfMHarrisPyrLK::getCurrFeatures() const
    {
        return kp_tracked_list_;
    }

    vx_array SfMHarrisPyrLK::getPointCloud() const
    {
        if (usePrevPointClouds_)
            return prevPointClouds_;
        else
            return pointClouds_;
    }

    vx_matrix SfMHarrisPyrLK::getRotation() const
    {
        return rotM_;
    }

    vx_matrix SfMHarrisPyrLK::getTranslation() const
    {
        return transM_;
    }

    vx_matrix SfMHarrisPyrLK::getGroundPlane() const
    {
        if (useExternalScaleData_)
            return groundPlane_;
        else
            return scaledGroundPlane_;
    }

    void SfMHarrisPyrLK::printPerfs() const
    {
        vx_perf_t perf;

        NVXIO_SAFE_CALL( vxQueryGraph(main_graph_, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

        ovxio::printPerf(cvt_color_node_, "Color Convert");
        ovxio::printPerf(pyr_node_, "Pyramid");
        ovxio::printPerf(opt_flow_node_, "Optical Flow");
        ovxio::printPerf(feature_track_node_, "Feature Track");
        ovxio::printPerf(find_fundamental_mat_node_, "Find Fundamental Mat");
        ovxio::printPerf(decompose_fundamental_node_, "Decompose Fundamental Mat");
        ovxio::printPerf(two_view_triangulation_node_, "Two View Triangulation");

        if (!useExternalScaleData_)
        {
            ovxio::printPerf(find_ground_plane_node_, "Find Ground Plane");
            ovxio::printPerf(calc_scale_node_, "Scale Calculate");
        }

        std::cout << std::endl;
    }

    void SfMHarrisPyrLK::release()
    {
        format_ = VX_DF_IMAGE_VIRT;
        width_ = 0;
        height_ = 0;

        vxReleaseDelay(&pyr_delay_);
        vxReleaseDelay(&pts_delay_);
        vxReleaseMatrix(&lensIntrinsics_);
        vxReleaseMatrix(&lensDistortion_);
        vxReleaseMatrix(&rotM_);
        vxReleaseMatrix(&transM_);
        vxReleaseMatrix(&fundMat_);
        vxReleaseMatrix(&gpNormal_);
        vxReleaseMatrix(&groundPlane_);
        vxReleaseMatrix(&scaledGroundPlane_);
        vxReleaseArray(&kp_tracked_list_);
        vxReleaseArray(&kp_filtered_list_);
        vxReleaseArray(&pointClouds_);
        vxReleaseArray(&prevPointClouds_);
        vxReleaseArray(&indices_);
        nvxSfmReleaseRandomState(&state_);
        vxReleaseScalar(&scale_);

        vxReleaseNode(&cvt_color_node_);
        vxReleaseNode(&pyr_node_);
        vxReleaseNode(&opt_flow_node_);
        vxReleaseNode(&feature_track_node_);
        vxReleaseNode(&find_fundamental_mat_node_);
        vxReleaseNode(&decompose_fundamental_node_);
        vxReleaseNode(&two_view_triangulation_node_);
        vxReleaseNode(&find_ground_plane_node_);
        vxReleaseNode(&calc_scale_node_);

        vxReleaseGraph(&main_graph_);
    }

    void SfMHarrisPyrLK::createDataObjects()
    {
        // Image pyramids for two successive frames are necessary for the computation.
        // A delay object with 2 slots is created for this purpose
        vx_pyramid pyr_exemplar = vxCreatePyramid(context_, params_.pyr_levels, VX_SCALE_PYRAMID_HALF, width_, height_, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(pyr_exemplar);
        pyr_delay_ = vxCreateDelay(context_, (vx_reference)pyr_exemplar, 2);
        NVXIO_CHECK_REFERENCE(pyr_delay_);
        vxReleasePyramid(&pyr_exemplar);

        // Input points to track need to kept for two successive frames.
        // A delay object with 2 slots is created for this purpose
        vx_array pts_exemplar = vxCreateArray(context_, NVX_TYPE_POINT2F, pointsCapacity_);
        NVXIO_CHECK_REFERENCE(pts_exemplar);
        pts_delay_ = vxCreateDelay(context_, (vx_reference)pts_exemplar, 2);
        NVXIO_CHECK_REFERENCE(pts_delay_);
        vxReleaseArray(&pts_exemplar);

        lensIntrinsics_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
        NVXIO_CHECK_REFERENCE(lensIntrinsics_);
        float pinholeIntrinsics[9] = {params_.pFx, 0.0f, params_.pCx, 0.0f, params_.pFy, params_.pCy, 0.0f, 0.0f, 1.0f};
        vxWriteMatrix(lensIntrinsics_, pinholeIntrinsics);

        lensDistortion_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 5, 1);
        NVXIO_CHECK_REFERENCE(lensDistortion_);
        float pinholeDistortion[5] = {params_.pK1, params_.pK2, params_.pP1, params_.pP2, params_.pK3};
        vxWriteMatrix(lensDistortion_, pinholeDistortion);

        fundMat_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
        NVXIO_CHECK_REFERENCE(fundMat_);

        rotM_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
        NVXIO_CHECK_REFERENCE(rotM_);

        transM_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 1);
        NVXIO_CHECK_REFERENCE(transM_);

        scaledGroundPlane_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 1);
        NVXIO_CHECK_REFERENCE(scaledGroundPlane_);

        groundPlane_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 1);
        NVXIO_CHECK_REFERENCE(groundPlane_);

        gpNormal_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 1);
        NVXIO_CHECK_REFERENCE(gpNormal_);

        kp_tracked_list_ = vxCreateArray(context_, NVX_TYPE_POINT2F, pointsCapacity_);
        NVXIO_CHECK_REFERENCE(kp_tracked_list_);

        kp_filtered_list_ = vxCreateArray(context_, NVX_TYPE_POINT2F, pointsCapacity_);
        NVXIO_CHECK_REFERENCE(kp_filtered_list_);

        pointClouds_ = vxCreateArray(context_, NVX_TYPE_POINT3F, pointsCapacity_);
        NVXIO_CHECK_REFERENCE(pointClouds_);

        prevPointClouds_ = vxCreateArray(context_, NVX_TYPE_POINT3F, pointsCapacity_);
        NVXIO_CHECK_REFERENCE(prevPointClouds_);

        indices_ = vxCreateArray(context_, VX_TYPE_UINT32, params_.samples * PTS_IN_RANSAC);
        NVXIO_CHECK_REFERENCE(indices_);

        std::vector<vx_uint32> indices_v(params_.samples * PTS_IN_RANSAC);
        vxAddArrayItems(indices_, indices_v.size(), &indices_v[0], sizeof(vx_uint32));

        nvxSfmCreateRandomState(context_, &state_, params_.samples, params_.seed);

        vx_float32 scale = 1.0f;
        scale_ = vxCreateScalar(context_, VX_TYPE_FLOAT32, &scale);
    }

    void SfMHarrisPyrLK::processFirstFrame(vx_image frame, vx_image mask)
    {
        vx_image frameGray = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(frameGray);

        NVXIO_SAFE_CALL( vxuColorConvert(context_, frame, frameGray) );
        NVXIO_SAFE_CALL( vxuGaussianPyramid(context_, frameGray, (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0)) );
        NVXIO_SAFE_CALL( nvxuHarrisTrack(context_, frameGray, (vx_array)vxGetReferenceFromDelay(pts_delay_, 0), mask,
                                         0, params_.harris_k, params_.harris_thresh, params_.harris_cell_size, NULL) );

        vxReleaseImage(&frameGray);
    }

    void SfMHarrisPyrLK::createMainGraph(vx_image frame, vx_image mask)
    {
        main_graph_ = vxCreateGraph(context_);
        NVXIO_CHECK_REFERENCE(main_graph_);

        // Intermediate images. Both images are created as 'virtual' in order to inform the OpenVX
        // framework that the application will never access their content.
        vx_image frameGray = vxCreateVirtualImage(main_graph_, width_, height_, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(frameGray);

        // RGB to Y conversion nodes
        cvt_color_node_ = vxColorConvertNode(main_graph_, frame, frameGray);
        NVXIO_CHECK_REFERENCE(cvt_color_node_);

        // Pyramid image node
        pyr_node_ = vxGaussianPyramidNode(main_graph_, frameGray, (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0));
        NVXIO_CHECK_REFERENCE(pyr_node_);

        // Lucas-Kanade optical flow node
        // Note: keypoints of the previous frame are also given as 'new points estimates'
        vx_float32 lk_epsilon = 0.01f;
        vx_scalar s_lk_epsilon = vxCreateScalar(context_, VX_TYPE_FLOAT32, &lk_epsilon);
        NVXIO_CHECK_REFERENCE(s_lk_epsilon);

        vx_scalar s_lk_num_iters = vxCreateScalar(context_, VX_TYPE_UINT32, &params_.lk_num_iters);
        NVXIO_CHECK_REFERENCE(s_lk_num_iters);

        vx_bool lk_use_init_est = vx_false_e;
        vx_scalar s_lk_use_init_est = vxCreateScalar(context_, VX_TYPE_BOOL, &lk_use_init_est);
        NVXIO_CHECK_REFERENCE(s_lk_use_init_est);

        opt_flow_node_ = vxOpticalFlowPyrLKNode(main_graph_,
                                                (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1), (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0),
                                                (vx_array)vxGetReferenceFromDelay(pts_delay_, -1), (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),
                                                kp_tracked_list_, VX_TERM_CRITERIA_BOTH, s_lk_epsilon, s_lk_num_iters, s_lk_use_init_est, params_.lk_win_size);
        NVXIO_CHECK_REFERENCE(opt_flow_node_);


        vx_array filteredPoints0 = vxCreateVirtualArray(main_graph_, NVX_TYPE_POINT2F, pointsCapacity_);
        NVXIO_CHECK_REFERENCE(filteredPoints0);
        vx_array filteredPoints1 = vxCreateVirtualArray(main_graph_, NVX_TYPE_POINT2F, pointsCapacity_);
        NVXIO_CHECK_REFERENCE(filteredPoints1);

        if (!useExternalScaleData_)
        {
            find_fundamental_mat_node_ = nvxSfmFindFundamentalMatNode(main_graph_,
                                                               (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),
                                                               kp_tracked_list_, kp_filtered_list_, fundMat_,
                                                               params_.errorThreshold, params_.samples, indices_, params_.medianFlowThreshold);
            NVXIO_CHECK_REFERENCE(find_fundamental_mat_node_);

            decompose_fundamental_node_ = nvxSfmDecomposeFundamentalMatNode(
                        main_graph_,
                        (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),
                        kp_filtered_list_, lensIntrinsics_, fundMat_, rotM_, transM_,
                        params_.minPixelDis, params_.maxPixelDis);
            NVXIO_CHECK_REFERENCE(decompose_fundamental_node_);

            vx_array pointCloudsInt = vxCreateVirtualArray(main_graph_, NVX_TYPE_POINT3F, pointsCapacity_);
            NVXIO_CHECK_REFERENCE(pointCloudsInt);

            // Triangulation node
            two_view_triangulation_node_ = nvxSfmTwoViewTriangulationNode(
                        main_graph_, (vx_array)vxGetReferenceFromDelay(pts_delay_, -1), kp_filtered_list_,
                        rotM_, transM_, 1.0f, params_.minPixelDis, params_.maxPixelDis,
                        lensIntrinsics_, lensDistortion_,
                        pointCloudsInt, params_.camModelOpt);
            NVXIO_CHECK_REFERENCE(two_view_triangulation_node_);

            // FindGroundPlane node
            registerFindGroundPlaneKernel(context_);
            find_ground_plane_node_ = findGroundPlaneNode(main_graph_, (vx_array)vxGetReferenceFromDelay(pts_delay_, -1), kp_filtered_list_, pointCloudsInt,
                                                             gpHeightThreshold_, gpReprojThreshold_, gpNormal_, groundPlane_);
            NVXIO_CHECK_REFERENCE(find_ground_plane_node_);

            registerScaleCloudKernel(context_);
            // scale node

            calc_scale_node_ = scaleCloudNode(main_graph_, groundPlane_, pointCloudsInt, expectedGpHeight, pointClouds_, scaledGroundPlane_);
            NVXIO_CHECK_REFERENCE(calc_scale_node_);

            vxReleaseArray(&pointCloudsInt);
        }
        else
        {
            find_fundamental_mat_node_ = nvxSfmFindFundamentalMatNode(main_graph_,
                                                               (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),
                                                               kp_tracked_list_, kp_filtered_list_, fundMat_,
                                                               params_.errorThreshold, params_.samples, indices_, params_.medianFlowThreshold);
            NVXIO_CHECK_REFERENCE(find_fundamental_mat_node_);

            decompose_fundamental_node_ = nvxSfmDecomposeFundamentalMatNode(
                        main_graph_,
                        (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),
                        kp_filtered_list_, lensIntrinsics_, fundMat_, rotM_, transM_,
                        params_.minPixelDis, params_.maxPixelDis);
            NVXIO_CHECK_REFERENCE(decompose_fundamental_node_);

            // Triangulation node
            two_view_triangulation_node_ = nvxSfmTwoViewTriangulationNode(
                        main_graph_, (vx_array)vxGetReferenceFromDelay(pts_delay_, -1), kp_filtered_list_,
                        rotM_, transM_, 1.0f, params_.minPixelDis, params_.maxPixelDis,
                        lensIntrinsics_, lensDistortion_,
                        pointClouds_, params_.camModelOpt);
            NVXIO_CHECK_REFERENCE(two_view_triangulation_node_);
        }

        // Extended Harris corner node
        feature_track_node_ = nvxHarrisTrackNode(main_graph_, frameGray, (vx_array)vxGetReferenceFromDelay(pts_delay_, 0), mask,
                                                 kp_tracked_list_, params_.harris_k, params_.harris_thresh, params_.harris_cell_size, NULL);
        NVXIO_CHECK_REFERENCE(feature_track_node_);

        NVXIO_SAFE_CALL( vxVerifyGraph(main_graph_) );

        vxReleaseScalar(&s_lk_epsilon);
        vxReleaseScalar(&s_lk_num_iters);
        vxReleaseScalar(&s_lk_use_init_est);
        vxReleaseImage(&frameGray);
        vxReleaseArray(&filteredPoints0);
        vxReleaseArray(&filteredPoints1);
    }
}

nvx::SfM::SfMParams::SfMParams()
{
    // Set parameters to some default values

    // pyramid level
    pyr_levels = 6;

    // parameters for harris_track node
    harris_k = 0.04f;
    harris_thresh = 100.0f;
    harris_cell_size = 18;

    // parameters for optical flow node
    lk_num_iters = 5;
    lk_win_size = 10;

    // parameters for find fundamental mat node
    seed = 1234;
    samples = 1200;
    errorThreshold = 0.5f;
    medianFlowThreshold = 0.5f;

    // pinhole camera intrinsics
    pFx = 723.106140f;
    pFy = 723.781128f;
    pCx = 512.f;
    pCy = 512.f;

    // pinhole camera distortion
    pK1 = 0.0f;
    pK2 = 0.0f;
    pP1 = 0.0f;
    pP2 = 0.0f;
    pK3 = 0.0f;

    // parameters for triangulation
    minPixelDis = 1.0;
    maxPixelDis = 15.0;

    camModelOpt = NVX_SFM_CAMERA_PINHOLE;
}

nvx::SfM* nvx::SfM::createSfM(vx_context context, const SfMParams& params)
{
    return new SfMHarrisPyrLK(context, params);
}

// Scale Cloud kernel implementation
static vx_status VX_CALLBACK scalecloud_kernel(vx_node /*node*/, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 5)
        return VX_FAILURE;

    vx_status status = VX_SUCCESS;

    vx_matrix gp = (vx_matrix)parameters[0];
    vx_array vxCloud = (vx_array)parameters[1];
    vx_scalar s_expectedGpHeight = (vx_scalar)parameters[2];
    vx_array scaledCloud = (vx_array)parameters[3];
    vx_matrix scaledGP = (vx_matrix)parameters[4];


    status |= vxTruncateArray(scaledCloud, 0);

    vx_float32 gpData[3];
    status |= vxReadMatrix(gp, gpData);
    Eigen::Vector3f plane(gpData[0],gpData[1], gpData[2]);


    vx_float32 expectedGpHeight = 1.0f;
    status |= vxReadScalarValue(s_expectedGpHeight, &expectedGpHeight);

    vx_float32 scale = 1.0f;
    if (plane !=  Eigen::Vector3f(0,0,0))
    {
        // distance to the ground plane
        float gpHeight = 1.f / sqrt(plane(0)*plane(0) + plane(1)*plane(1) + plane(2)*plane(2));
        scale = expectedGpHeight / gpHeight;
    }

    void *ptr = 0;
    vx_size size = 0, stride = 0;

    status |= vxQueryArray(vxCloud, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size));

    if (size > 0)
    {
        status |= vxAccessArrayRange(vxCloud, 0, size, &stride, &ptr, VX_READ_ONLY);

        for (vx_size i = 0; i < size; ++i)
        {
            nvx_point3f_t pt = vxArrayItem(nvx_point3f_t, ptr, i, stride);
            if ( isPointValid(pt) )
            {
                pt.x *= scale;
                pt.y *= scale;
                pt.z *= scale;
            }

            vxAddArrayItems(scaledCloud, 1, &pt, sizeof(pt));
        }
        status |= vxCommitArrayRange(vxCloud, 0, 0, ptr);
    }

    for(int i=0; i<3; ++i)
        gpData[i] /= scale;

    vxWriteMatrix(scaledGP, gpData);

    return status;
}

// Input validator
static vx_status VX_CALLBACK scalecloud_input_validate(vx_node node, vx_uint32 index)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;

    if (index == 0)
    {
        vx_parameter param = vxGetParameterByIndex(node, index);

        if (param)
        {
            vx_matrix gp;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &gp, sizeof(gp));
            if (gp)
            {
                vx_enum dataType = 0;
                vx_size rows = 0ul, cols = 0ul;
                vxQueryMatrix(gp, VX_MATRIX_ATTRIBUTE_TYPE, &dataType, sizeof(dataType));
                vxQueryMatrix(gp, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows));
                vxQueryMatrix(gp, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols));
                if ( (dataType == VX_TYPE_FLOAT32) && (cols == 3) && (rows == 1))
                {
                    status = VX_SUCCESS;
                }
                else
                {
                    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                                  "invalid type/size of gp matrix (expected type: VX_TYPE_FLOAT32, expected cols: 3, expected rows 1");
                }
                vxReleaseMatrix(&gp);
            }
            vxReleaseParameter(&param);
        }
    }
    else if (index == 1)
    {
        vx_parameter param = vxGetParameterByIndex(node, index);

        if (param)
        {
            vx_array cloud = 0;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &cloud, sizeof(cloud));

            if (cloud)
            {
                vx_enum type = 0;
                vxQueryArray(cloud, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type));

                if (type == NVX_TYPE_POINT3F)
                {
                    status = VX_SUCCESS;
                }
                else
                {
                    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                                  "invalid type of cloud array (expected type: NVX_TYPE_POINT3F");
                }

                vxReleaseArray(&cloud);
            }
            vxReleaseParameter(&param);
        }
    }
    else if (index == 2)
    {
        vx_parameter param = vxGetParameterByIndex(node, index);

        if (param)
        {
            vx_scalar expectedGpHeight = 0;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &expectedGpHeight, sizeof(expectedGpHeight));

            if (expectedGpHeight)
            {
                vx_enum type = 0;
                vxQueryScalar(expectedGpHeight, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type));

                if (type == VX_TYPE_FLOAT32)
                {
                    status = VX_SUCCESS;
                }
                else
                {
                    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                                  "invalid type of expectedGpHeight scalar (expected type: VX_TYPE_FLOAT32");
                }

                vxReleaseScalar(&expectedGpHeight);
            }
            vxReleaseParameter(&param);
        }
    }

    return status;
}

// Output validator
static vx_status VX_CALLBACK scalecloud_output_validate(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    vx_status status = VX_FAILURE;
    if (index == 3)
    {
        vx_parameter param = vxGetParameterByIndex(node, 1);
        if (param)
        {
            vx_array inputPoints = 0;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputPoints, sizeof(inputPoints));
            if (inputPoints)
            {
                vx_enum type = 0;
                vxQueryArray(inputPoints, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type));
                vx_size capacity = 0;
                vxQueryArray(inputPoints, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity));

                vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type));
                vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity));

                vxReleaseArray(&inputPoints);
                status = VX_SUCCESS;
            }
            vxReleaseParameter(&param);
        }
    }
    else if (index == 4)
    {
        vx_parameter param = vxGetParameterByIndex(node, 0);
        if (param)
        {
            vx_matrix gp = 0;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &gp, sizeof(gp));
            if (gp)
            {
                vx_enum type = 0;
                vxQueryMatrix(gp, VX_MATRIX_ATTRIBUTE_TYPE, &type, sizeof(type));
                vx_size cols = 0, rows = 0;
                vxQueryMatrix(gp, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols));
                vxQueryMatrix(gp, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows));

                vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_TYPE, &type, sizeof(type));
                vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols));
                vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows));

                vxReleaseMatrix(&gp);
                status = VX_SUCCESS;
            }
            vxReleaseParameter(&param);
        }
    }


    return status;
}

// Register user defined kernel in OpenVX context
vx_status registerScaleCloudKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_kernel kernel = vxAddKernel(context, const_cast<vx_char*>("user.kernel.scale_cloud"),
                                   USER_KERNEL_SCALE_CLOUD,
                                   scalecloud_kernel,
                                   5,
                                   scalecloud_input_validate,
                                   scalecloud_output_validate,
                                   NULL,
                                   NULL
                                   );

    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] failed to create ScaleCloud Kernel", __FUNCTION__, __LINE__);
        return status;
    }

    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // ground plane
    status |= vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // cloud
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED); // expected height to ground plane
    status |= vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED); // scaled cloud
    status |= vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // scaled ground plane

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] failed to initialize ScaleCloud Kernel parameters", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    status = vxFinalizeKernel(kernel);
    vxReleaseKernel(&kernel);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] failed to finalize ScaleCloud Kernel", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    return status;
}

vx_node scaleCloudNode(vx_graph graph, vx_matrix gp, vx_array cloud, vx_float32 expectedGpHeight, vx_array scaledCloud, vx_matrix scaledGP)
{
    vx_node node = NULL;

    vx_kernel kernel = vxGetKernelByEnum(vxGetContext((vx_reference)graph), USER_KERNEL_SCALE_CLOUD);

    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);

        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vx_scalar s_expectedGpHeight= vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &expectedGpHeight);
            vxSetParameterByIndex(node, 0, (vx_reference)gp);
            vxSetParameterByIndex(node, 1, (vx_reference)cloud);
            vxSetParameterByIndex(node, 2, (vx_reference)s_expectedGpHeight);
            vxSetParameterByIndex(node, 3, (vx_reference)scaledCloud);
            vxSetParameterByIndex(node, 4, (vx_reference)scaledGP);
            vxReleaseScalar(&s_expectedGpHeight);
        }
    }

    return node;
}

static Eigen::Vector3f fitPlane(std::vector<Eigen::Vector3f> &points, const Eigen::Vector3f &normal = Eigen::Vector3f::Zero())
{
    if ( normal == Eigen::Vector3f::Zero() )
    {
        Eigen::Map< MatrixXf_rm> A(&points[0](0), points.size(), 3);;
        Eigen::VectorXf b = Eigen::VectorXf::Ones(points.size());

        Eigen::Vector3f plane =  A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        return plane;
    }
    else
    {
        float sum = 0;
        float sum2 = 0;
        for (size_t i = 0; i < points.size(); i++)
        {
            sum += (normal(0)*points[i](0) + normal(1)*points[i](1) + normal(2)*points[i](2));
            sum2 += (normal(0)*points[i](0) + normal(1)*points[i](1) + normal(2)*points[i](2)) *
                    (normal(0)*points[i](0) + normal(1)*points[i](1) + normal(2)*points[i](2));
        }

        float scale = sum / sum2;
        Eigen::Vector3f plane = normal * scale;

        return plane;
    }
}

static vx_status VX_CALLBACK findgroundplane_kernel(vx_node /*node*/, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 7)
        return VX_FAILURE;

    vx_status status = VX_SUCCESS;

    vx_context context = vxGetContext(parameters[0]);

    vx_array p1 = (vx_array)parameters[0];
    vx_array p2 = (vx_array)parameters[1];
    vx_array cloud = (vx_array)parameters[2];
    vx_scalar gpHeightThreshold = (vx_scalar)parameters[3];
    vx_scalar gpReprojThreshold = (vx_scalar)parameters[4];
    vx_matrix gpNormal = (vx_matrix)parameters[5];
    vx_matrix gp = (vx_matrix)parameters[6];

    vx_size numP1 = 0, numP2 = 0, numP3D = 0;
    status |= vxQueryArray(p1, VX_ARRAY_ATTRIBUTE_NUMITEMS, &numP1, sizeof(numP1));
    status |= vxQueryArray(p2, VX_ARRAY_ATTRIBUTE_NUMITEMS, &numP2, sizeof(numP2));
    status |= vxQueryArray(cloud, VX_ARRAY_ATTRIBUTE_NUMITEMS, &numP3D, sizeof(numP3D));


    Eigen::Vector3f plane = Eigen::Vector3f::Zero();
    const vx_size minNumPointsForH = 4;

    if ((numP1 != numP2) || (numP1 != numP3D))
    {
        vxWriteMatrix(gp, plane.data());
        vxAddLogEntry((vx_reference)context, VX_FAILURE,
                      "input arrays have different sizes "
                      "(" VX_FMT_SIZE " for points2d1, " VX_FMT_SIZE " for points2d2, " VX_FMT_SIZE " for triangulatedPoints3d)",
                      numP1, numP2, numP3D);
        return VX_FAILURE;
    }


    if (numP1 < minNumPointsForH) //in this case homagraphy cannot be estimated
    {
        vxWriteMatrix(gp, plane.data());
        vxAddLogEntry((vx_reference)context, VX_FAILURE,
                      "not enough points for Ground Plane estimation"
                      " (" VX_FMT_SIZE " is needed, but only " VX_FMT_SIZE " was supplied)",
                      minNumPointsForH, numP1);
        return status;
    }

    int heightThresh = 0;
    if (gpHeightThreshold)
    {
        vxReadScalarValue(gpHeightThreshold, &heightThresh);
    }

    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    if (gpNormal)
    {
        float normalData[3];
        vxReadMatrix(gpNormal, normalData);
        normal(0) = normalData[0];
        normal(1) = normalData[1];
        normal(2) = normalData[2];
    }

    void *ptr1 = 0, *ptr2 = 0;
    vx_size stride1 = 0, stride2 = 0;
    status |= vxAccessArrayRange(p1, 0, numP1, &stride1, &ptr1, VX_READ_ONLY);
    status |= vxAccessArrayRange(p2, 0, numP1, &stride2, &ptr2, VX_READ_ONLY);

    vx_array bottomPoints1 = vxCreateArray(context, NVX_TYPE_POINT2F, numP1);
    vx_array bottomPoints2 = vxCreateArray(context, NVX_TYPE_POINT2F, numP1);

    std::vector<int> bottomPointsIndices;
    for(vx_size i = 0; i < numP1; ++i)
    {
        nvx_point2f_t& pt1 = vxArrayItem(nvx_point2f_t, ptr1, i, stride1);
        nvx_point2f_t& pt2 = vxArrayItem(nvx_point2f_t, ptr2, i, stride2);

        if (pt1.x >= 0 && pt1.y > heightThresh && pt2.x >= 0 && pt2.y > heightThresh )
        {
            bottomPointsIndices.push_back(static_cast<int>(i));

            vxAddArrayItems(bottomPoints1, 1, &pt1, sizeof(pt1));
            vxAddArrayItems(bottomPoints2, 1, &pt2, sizeof(pt2));
        }
    }
    status |= vxCommitArrayRange(p1, 0, 0, ptr1);
    status |= vxCommitArrayRange(p2, 0, 0, ptr2);

    vx_size numOfBottomPoints = 0;
    status |= vxQueryArray(bottomPoints1, VX_ARRAY_ATTRIBUTE_NUMITEMS, &numOfBottomPoints, sizeof(numOfBottomPoints));

    if (numOfBottomPoints < minNumPointsForH) //in this case homagraphy cannot be estimated
    {
        vxWriteMatrix(gp, plane.data());

        vxReleaseArray(&bottomPoints1);
        vxReleaseArray(&bottomPoints2);

        vxAddLogEntry((vx_reference)context, VX_FAILURE,
                      "not enough features on ground plane for Ground Plane estimation"
                      " (" VX_FMT_SIZE " is needed, but only " VX_FMT_SIZE " was on the ground plane)",
                      minNumPointsForH, numOfBottomPoints);
        return status;
    }

    vx_matrix HMat = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);

    float HThresh = 0.5f;
    if (gpReprojThreshold)
    {
        vxReadScalarValue(gpReprojThreshold, &HThresh);
    }

    vx_enum HMethod = NVX_FIND_HOMOGRAPHY_METHOD_RANSAC;
    vx_array HMask = vxCreateArray(context, VX_TYPE_UINT8, numP1);

    vx_node find_homography_node = NULL;
    vx_graph graph = vxCreateGraph(context);
    find_homography_node = nvxFindHomographyNode(graph, bottomPoints1, bottomPoints2,
                                                 HMat, HMethod, HThresh,
                                                 2000, 10, 0.995f, 0.45f, HMask);
    status |= vxVerifyGraph(graph);
    status |= vxProcessGraph(graph);
    status |= vxReleaseNode(&find_homography_node);
    status |= vxReleaseGraph(&graph);

    void *ptrMask = 0;
    vx_size sizeMask = 0, strideMask = 0;
    status |= vxQueryArray(HMask, VX_ARRAY_ATTRIBUTE_NUMITEMS, &sizeMask, sizeof(sizeMask));
    status |= vxAccessArrayRange(HMask, 0, sizeMask, &strideMask, &ptrMask, VX_READ_ONLY);

    void *ptrCloud = 0;
    vx_size sizeCloud = 0, strideCloud = 0;
    status |= vxQueryArray(cloud, VX_ARRAY_ATTRIBUTE_NUMITEMS, &sizeCloud, sizeof(sizeCloud));
    status |= vxAccessArrayRange(cloud, 0, sizeCloud, &strideCloud, &ptrCloud, VX_READ_ONLY);

    std::vector<Eigen::Vector3f> gpCloud;
    for(vx_size i=0; i<sizeMask; ++i)
    {
        vx_uint8 m = vxArrayItem(vx_uint8, ptrMask, i, strideMask);
        if (m)
        {
            nvx_point3f_t& pt = vxArrayItem(nvx_point3f_t, ptrCloud, bottomPointsIndices[i], strideCloud);
            if (isPointValid(pt))
            {
                gpCloud.push_back(Eigen::Vector3f(pt.x,pt.y,pt.z));
            }
        }
    }
    status |= vxCommitArrayRange(HMask, 0, 0, ptrMask);
    status |= vxCommitArrayRange(cloud, 0, 0, ptrCloud);

    vxReleaseArray(&bottomPoints1);
    vxReleaseArray(&bottomPoints2);
    vxReleaseArray(&HMask);
    vxReleaseMatrix(&HMat);

    const vx_size numNumPointsForGP = 3;
    if (gpCloud.size() < numNumPointsForGP)
    {
        vxWriteMatrix(gp, plane.data());
        vxAddLogEntry((vx_reference)context, VX_FAILURE,
                      "not enough triangulated points for Ground Plane estimation"
                      " (" VX_FMT_SIZE " is needed, but only " VX_FMT_SIZE " was found)",
                      numNumPointsForGP, gpCloud.size());

        return status;
    }

    plane = fitPlane(gpCloud, normal);
    vxWriteMatrix(gp, plane.data());

    return status;
}

static vx_status VX_CALLBACK findgroundplane_input_validate(vx_node node, vx_uint32 index)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;

    if (index == 0)
    {
        vx_parameter param = vxGetParameterByIndex(node, index);

        if (param)
        {
            vx_array p1 = 0;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &p1, sizeof(p1));

            if (p1)
            {
                vx_enum type = 0;
                vxQueryArray(p1, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type));

                if (type == NVX_TYPE_POINT2F)
                {
                    status = VX_SUCCESS;
                }
                else
                {
                    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                                  "invalid type of points2d1 array (expected type: NVX_TYPE_POINT2F");
                }

                vxReleaseArray(&p1);
            }
            vxReleaseParameter(&param);
        }
    }
    else if (index == 1)
    {
        vx_parameter param[2] = { vxGetParameterByIndex(node, 0), vxGetParameterByIndex(node, 1), };

        if (param[0] && param[1])
        {
            vx_array p1(0), p2(0);

            vxQueryParameter(param[0], VX_PARAMETER_ATTRIBUTE_REF, &p1, sizeof(p1));
            vxQueryParameter(param[1], VX_PARAMETER_ATTRIBUTE_REF, &p2, sizeof(p2));

            if (p1 && p2)
            {
                vx_enum p1Type(0), p2Type(0);
                vxQueryArray(p1, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &p1Type, sizeof(p1Type));
                vxQueryArray(p2, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &p2Type, sizeof(p2Type));

                vx_size capOfP1(0), capOfP2(0);
                vxQueryArray(p1, VX_ARRAY_ATTRIBUTE_CAPACITY, &capOfP1, sizeof(capOfP1));
                vxQueryArray(p2, VX_ARRAY_ATTRIBUTE_CAPACITY, &capOfP2, sizeof(capOfP2));

                if (p2Type == p1Type)
                {
                    if (capOfP1 == capOfP2)
                    {
                        status = VX_SUCCESS;
                    }
                    else
                    {

                        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS, "2d features arrays have different capacities"
                                                                                          "(" VX_FMT_SIZE " for points2d1, " VX_FMT_SIZE " for points2d1",
                                                                                          capOfP1, capOfP2);
                    }
                }
                else
                {
                    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                                  "input arrays have different types (expected type: NVX_TYPE_POINT2F)");
                }

                vxReleaseArray(&p1);
                vxReleaseArray(&p2);
            }
            vxReleaseParameter(&param[0]);
            vxReleaseParameter(&param[1]);
        }
    }
    else if (index == 2)
    {
        vx_parameter param[2] = { vxGetParameterByIndex(node, 0), vxGetParameterByIndex(node, 2), };

        if (param[0] && param[1])
        {
            vx_array p1(0), p3D(0);

            vxQueryParameter(param[0], VX_PARAMETER_ATTRIBUTE_REF, &p1, sizeof(p1));
            vxQueryParameter(param[1], VX_PARAMETER_ATTRIBUTE_REF, &p3D, sizeof(p3D));

            if (p1 && p3D)
            {
                vx_enum p3DType(0);
                vxQueryArray(p3D, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &p3DType, sizeof(p3DType));

                vx_size capOfP1(0), capOfP3D(0);
                vxQueryArray(p1, VX_ARRAY_ATTRIBUTE_CAPACITY, &capOfP1, sizeof(capOfP1));
                vxQueryArray(p3D, VX_ARRAY_ATTRIBUTE_CAPACITY, &capOfP3D, sizeof(capOfP3D));

                if (p3DType == NVX_TYPE_POINT3F)
                {
                    if (capOfP1 == capOfP3D)
                    {
                        status = VX_SUCCESS;
                    }
                    else
                    {
                        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS, "2d features and 3d points arrays have different capacities"
                                                                                          "(" VX_FMT_SIZE " for points2d1 and points2d2, " VX_FMT_SIZE " for triangulatedPoints3d",
                                                                                          capOfP1, capOfP3D);
                    }
                }
                else
                {
                    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                                  "invalid type of points2d3 array (expected type: NVX_TYPE_POINT3F");
                }

                vxReleaseArray(&p1);
                vxReleaseArray(&p3D);
            }
            vxReleaseParameter(&param[0]);
            vxReleaseParameter(&param[1]);
        }
    }
    else if (index == 3)
    {
        vx_parameter param = vxGetParameterByIndex(node, index);
        if (param)
        {
            vx_scalar scalar = 0;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar));
            if (scalar)
            {
                vx_enum scalarType = 0;
                vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &scalarType, sizeof(scalarType));

                if (scalarType == VX_TYPE_UINT32)
                {
                    status = VX_SUCCESS;
                }
                else
                {
                    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                                  "invalid type of gpHeightThreshold scalar (expected type: VX_TYPE_UINT32");
                }

                vxReleaseScalar(&scalar);
            }
            vxReleaseParameter(&param);
        }
    }
    else if (index == 4)
    {
        vx_parameter param = vxGetParameterByIndex(node, index);
        if (param)
        {
            vx_scalar scalar = 0;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar));
            if (scalar)
            {
                vx_enum scalarType = 0;
                vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &scalarType, sizeof(scalarType));

                if (scalarType == VX_TYPE_FLOAT32)
                {
                    status = VX_SUCCESS;
                }
                else
                {
                    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                                  "invalid type of gpReprojThreshold scalar (expected type: VX_TYPE_FLOAT32");
                }

                vxReleaseScalar(&scalar);
            }
            vxReleaseParameter(&param);
        }
    }
    else if (index == 5)
    {
        vx_parameter param = vxGetParameterByIndex(node, index);
        if (param)
        {
            vx_matrix gpNormal;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &gpNormal, sizeof(gpNormal));
            if (gpNormal)
            {
                vx_enum dataType = 0;
                vx_size rows = 0ul, cols = 0ul;
                vxQueryMatrix(gpNormal, VX_MATRIX_ATTRIBUTE_TYPE, &dataType, sizeof(dataType));
                vxQueryMatrix(gpNormal, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows));
                vxQueryMatrix(gpNormal, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols));
                if ( (dataType == VX_TYPE_FLOAT32) && (cols == 3) && (rows == 1))
                {
                    status = VX_SUCCESS;
                }
                else
                {
                    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                                  "invalid type/size of gpNormal matrix (expected type: VX_TYPE_FLOAT32, expected cols: 3, expected rows 1");
                }
                vxReleaseMatrix(&gpNormal);
            }
            vxReleaseParameter(&param);
        }
    }

    return status;
}

static vx_status VX_CALLBACK findgroundplane_output_validate(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;

    if (index == 6)
    {
        vx_parameter param = vxGetParameterByIndex(node, index);
        if (param)
        {
            vx_matrix matrix;
            vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &matrix, sizeof(matrix));

            if (matrix)
            {
                vx_size cols = 3;
                vx_size rows = 1;
                vx_enum type = VX_TYPE_FLOAT32;
                vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols));
                vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows));
                vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_TYPE, &type, sizeof(type));

                status = VX_SUCCESS;

                vxReleaseMatrix(&matrix);
            }
            vxReleaseParameter(&param);
        }
    }

    return status;
}

vx_status registerFindGroundPlaneKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_kernel kernel = vxAddKernel(context, const_cast<vx_char*>("user.kernel.find_ground_plane"),
                                   USER_KERNEL_FIND_GROUND_PLANE,
                                   findgroundplane_kernel,
                                   7,
                                   findgroundplane_input_validate,
                                   findgroundplane_output_validate,
                                   NULL,
                                   NULL
                                   );

    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] failed to create FindGroundPlane Kernel", __FUNCTION__, __LINE__);
        return status;
    }

    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // points2d1
    status |= vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);  // points2d2
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED); // triangulatedPoints3d
    status |= vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL); // gpHeightThreshold
    status |= vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL); // gpReprojThreshold
    status |= vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_OPTIONAL); // gpNormal
    status |= vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // gp

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] failed to initialize FindGroundPlane Kernel parameters", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    status = vxFinalizeKernel(kernel);
    vxReleaseKernel(&kernel);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] failed to finalize FindGroundPlane Kernel", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    return status;
}

vx_node findGroundPlaneNode(vx_graph graph,
                               vx_array points2d1,
                               vx_array points2d2,
                               vx_array triangulatedPoints3d,
                               vx_uint32 gpHeightThreshold,
                               vx_float32 gpReprojThreshold,
                               vx_matrix gpNormal,
                               vx_matrix gp)
{
    vx_node node = NULL;

    vx_kernel kernel = vxGetKernelByEnum(vxGetContext((vx_reference)graph), USER_KERNEL_FIND_GROUND_PLANE);
    vx_scalar s_gpHeightThreshold = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &gpHeightThreshold);
    vx_scalar s_reprojThreshold = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &gpReprojThreshold);

    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);

        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)points2d1);
            vxSetParameterByIndex(node, 1, (vx_reference)points2d2);
            vxSetParameterByIndex(node, 2, (vx_reference)triangulatedPoints3d);
            vxSetParameterByIndex(node, 3, (vx_reference)s_gpHeightThreshold);
            vxSetParameterByIndex(node, 4, (vx_reference)s_reprojThreshold);
            vxSetParameterByIndex(node, 5, (vx_reference)gpNormal);
            vxSetParameterByIndex(node, 6, (vx_reference)gp);
        }
    }

    vxReleaseScalar(&s_gpHeightThreshold);
    vxReleaseScalar(&s_reprojThreshold);

    return node;
}
