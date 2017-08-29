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

#ifndef __NVX_SFM_HPP__
#define __NVX_SFM_HPP__

#include <VX/vx.h>
#include <string>

namespace nvx
{
    class SfM
    {
    public:
        struct SfMParams
        {
            // pyramid level
            vx_uint32 pyr_levels;

            // parameters for harris_track node
            vx_float32 harris_k;
            vx_float32 harris_thresh;
            vx_uint32 harris_cell_size;

            // parameters for optical flow node
            vx_uint32 lk_num_iters;
            vx_uint32 lk_win_size;

            // parameters for find fundamental mat node
            vx_uint32 seed;
            vx_uint32 samples;
            vx_float32 errorThreshold;
            vx_float32 medianFlowThreshold;

            // pinhole camera intrinsics
            vx_float32 pFx;
            vx_float32 pFy;
            vx_float32 pCx;
            vx_float32 pCy;

            // pinhole camera distortion
            vx_float32 pK1;
            vx_float32 pK2;
            vx_float32 pP1;
            vx_float32 pP2;
            vx_float32 pK3;

            // parameters for triangulation
            vx_float32 minPixelDis;
            vx_float32 maxPixelDis;

            // camera model options: 0 - pinhole, 1 - fisheye (Scaramuzza)
            vx_uint32 camModelOpt;

            SfMParams();
        };


        static SfM* createSfM(vx_context context, const SfMParams& params = SfMParams());

        virtual ~SfM() {}

        virtual vx_status init(vx_image firstFrame, vx_image mask = 0, const std::string &imuDataFile = "", const std::string &frameDataFile = "") = 0;
        virtual vx_status track(vx_image newFrame, vx_image mask = 0) = 0;

        // get list of tracked features on previous frame
        virtual vx_array getPrevFeatures() const = 0;

        // get list of tracked features on current frame
        virtual vx_array getCurrFeatures() const = 0;

        virtual vx_array getPointCloud() const = 0;
        virtual vx_matrix getRotation() const = 0;
        virtual vx_matrix getTranslation() const = 0;
        virtual vx_matrix getGroundPlane() const = 0;

        virtual void printPerfs() const = 0;
    };
}

#endif
