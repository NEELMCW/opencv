#include "hip/hip_runtime.h"
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_CUDAFILTERS

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        /////////////////////////////////////////// Corner Harris /////////////////////////////////////////////////

#ifdef HIP_TO_DO
        texture<float, hipTextureType2D, hipReadModeElementType> harrisDxTex(0, hipFilterModePoint, hipAddressModeClamp);
        texture<float, hipTextureType2D, hipReadModeElementType> harrisDyTex(0, hipFilterModePoint, hipAddressModeClamp);

        __global__ void cornerHarris_kernel(const int block_size, const float k, PtrStepSzf dst)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                float a = 0.f;
                float b = 0.f;
                float c = 0.f;

                const int ibegin = y - (block_size / 2);
                const int jbegin = x - (block_size / 2);
                const int iend = ibegin + block_size;
                const int jend = jbegin + block_size;

                for (int i = ibegin; i < iend; ++i)
                {
                    for (int j = jbegin; j < jend; ++j)
                    {
                        float dx = tex2D(harrisDxTex, j, i);
                        float dy = tex2D(harrisDyTex, j, i);

                        a += dx * dx;
                        b += dx * dy;
                        c += dy * dy;
                    }
                }

                dst(y, x) = a * c - b * b - k * (a + c) * (a + c);
            }
        }

        template <typename BR, typename BC>
        __global__ void cornerHarris_kernel(const int block_size, const float k, PtrStepSzf dst, const BR border_row, const BC border_col)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                float a = 0.f;
                float b = 0.f;
                float c = 0.f;

                const int ibegin = y - (block_size / 2);
                const int jbegin = x - (block_size / 2);
                const int iend = ibegin + block_size;
                const int jend = jbegin + block_size;

                for (int i = ibegin; i < iend; ++i)
                {
                    const int y = border_col.idx_row(i);

                    for (int j = jbegin; j < jend; ++j)
                    {
                        const int x = border_row.idx_col(j);

                        float dx = tex2D(harrisDxTex, x, y);
                        float dy = tex2D(harrisDyTex, x, y);

                        a += dx * dx;
                        b += dx * dy;
                        c += dy * dy;
                    }
                }

                dst(y, x) = a * c - b * b - k * (a + c) * (a + c);
            }
        }
#endif //HIP_TO_DO

        void cornerHarris_gpu(int block_size, float k, PtrStepSzf Dx, PtrStepSzf Dy, PtrStepSzf dst, int border_type, hipStream_t stream)
        {
#ifdef HIP_TO_DO
            dim3 block(32, 8);
            dim3 grid(divUp(Dx.cols, block.x), divUp(Dx.rows, block.y));

            bindTexture(&harrisDxTex, Dx);
            bindTexture(&harrisDyTex, Dy);

            switch (border_type)
            {
            case BORDER_REFLECT101:
                hipLaunchKernelGGL((cornerHarris_kernel), dim3(grid), dim3(block), 0, stream, block_size, k, dst, BrdRowReflect101<void>(Dx.cols), BrdColReflect101<void>(Dx.rows));
                break;

            case BORDER_REFLECT:
                hipLaunchKernelGGL((cornerHarris_kernel), dim3(grid), dim3(block), 0, stream, block_size, k, dst, BrdRowReflect<void>(Dx.cols), BrdColReflect<void>(Dx.rows));
                break;

            case BORDER_REPLICATE:
                hipLaunchKernelGGL((cornerHarris_kernel), dim3(grid), dim3(block), 0, stream, block_size, k, dst);
                break;
            }

            cudaSafeCall( hipGetLastError() );

            if (stream == 0)
                cudaSafeCall( hipDeviceSynchronize() );
#endif //HIP_TO_DO

        }

        /////////////////////////////////////////// Corner Min Eigen Val /////////////////////////////////////////////////

#ifdef HIP_TO_DO
        texture<float, hipTextureType2D, hipReadModeElementType> minEigenValDxTex(0, hipFilterModePoint, hipAddressModeClamp);
        texture<float, hipTextureType2D, hipReadModeElementType> minEigenValDyTex(0, hipFilterModePoint, hipAddressModeClamp);
#endif //HIP_TO_DO

        __global__ void cornerMinEigenVal_kernel(const int block_size, PtrStepSzf dst)
        {
#ifdef HIP_TO_DO
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                float a = 0.f;
                float b = 0.f;
                float c = 0.f;

                const int ibegin = y - (block_size / 2);
                const int jbegin = x - (block_size / 2);
                const int iend = ibegin + block_size;
                const int jend = jbegin + block_size;

                for (int i = ibegin; i < iend; ++i)
                {
                    for (int j = jbegin; j < jend; ++j)
                    {
                        float dx = tex2D(minEigenValDxTex, j, i);
                        float dy = tex2D(minEigenValDyTex, j, i);

                        a += dx * dx;
                        b += dx * dy;
                        c += dy * dy;
                    }
                }

                a *= 0.5f;
                c *= 0.5f;

                dst(y, x) = (a + c) - sqrtf((a - c) * (a - c) + b * b);
            }
#endif //HIP_TO_DO

        }


        template <typename BR, typename BC>
        __global__ void cornerMinEigenVal_kernel(const int block_size, PtrStepSzf dst, const BR border_row, const BC border_col)
        {
#ifdef HIP_TO_DO
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                float a = 0.f;
                float b = 0.f;
                float c = 0.f;

                const int ibegin = y - (block_size / 2);
                const int jbegin = x - (block_size / 2);
                const int iend = ibegin + block_size;
                const int jend = jbegin + block_size;

                for (int i = ibegin; i < iend; ++i)
                {
                    int y = border_col.idx_row(i);

                    for (int j = jbegin; j < jend; ++j)
                    {
                        int x = border_row.idx_col(j);

                        float dx = tex2D(minEigenValDxTex, x, y);
                        float dy = tex2D(minEigenValDyTex, x, y);

                        a += dx * dx;
                        b += dx * dy;
                        c += dy * dy;
                    }
                }

                a *= 0.5f;
                c *= 0.5f;

                dst(y, x) = (a + c) - sqrtf((a - c) * (a - c) + b * b);
            }
#endif //HIP_TO_DO

        }

        void cornerMinEigenVal_gpu(int block_size, PtrStepSzf Dx, PtrStepSzf Dy, PtrStepSzf dst, int border_type, hipStream_t stream)
        {
#ifdef HIP_TO_DO
            dim3 block(32, 8);
            dim3 grid(divUp(Dx.cols, block.x), divUp(Dx.rows, block.y));

            bindTexture(&minEigenValDxTex, Dx);
            bindTexture(&minEigenValDyTex, Dy);

            switch (border_type)
            {
            case BORDER_REFLECT101:
                hipLaunchKernelGGL((cornerMinEigenVal_kernel), dim3(grid), dim3(block), 0, stream, block_size, dst, BrdRowReflect101<void>(Dx.cols), BrdColReflect101<void>(Dx.rows));
                break;

            case BORDER_REFLECT:
                hipLaunchKernelGGL((cornerMinEigenVal_kernel), dim3(grid), dim3(block), 0, stream, block_size, dst, BrdRowReflect<void>(Dx.cols), BrdColReflect<void>(Dx.rows));
                break;

            case BORDER_REPLICATE:
                hipLaunchKernelGGL((cornerMinEigenVal_kernel), dim3(grid), dim3(block), 0, stream, block_size, dst);
                break;
            }

            cudaSafeCall( hipGetLastError() );

            if (stream == 0)
                cudaSafeCall(hipDeviceSynchronize());
#endif //HIP_TO_DO
                
        }
    }
}}}

#endif // HAVE_OPENCV_CUDAFILTERS

#endif // CUDA_DISABLER
