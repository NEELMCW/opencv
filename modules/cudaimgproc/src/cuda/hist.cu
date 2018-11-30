
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

#include "hip/hip_runtime.h"

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/transform.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace hist
{
    __global__ void histogram256Kernel(const uchar* src, int cols, int rows, size_t step, int* hist)
    {
        __shared__ int shist[256];

        const int y = blockIdx.x * blockDim.y + threadIdx.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        shist[tid] = 0;
        __syncthreads();

        if (y < rows)
        {
            const unsigned int* rowPtr = (const unsigned int*) (src + y * step);

            const int cols_4 = cols / 4;
            for (int x = threadIdx.x; x < cols_4; x += blockDim.x)
            {
                unsigned int data = rowPtr[x];

                Emulation::smem::atomicAdd(&shist[(data >>  0) & 0xFFU], 1);
                Emulation::smem::atomicAdd(&shist[(data >>  8) & 0xFFU], 1);
                Emulation::smem::atomicAdd(&shist[(data >> 16) & 0xFFU], 1);
                Emulation::smem::atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
            }

            if (cols % 4 != 0 && threadIdx.x == 0)
            {
                for (int x = cols_4 * 4; x < cols; ++x)
                {
                    unsigned int data = ((const uchar*)rowPtr)[x];
                    Emulation::smem::atomicAdd(&shist[data], 1);
                }
            }
        }

        __syncthreads();

        const int histVal = shist[tid];
        if (histVal > 0)
            ::atomicAdd(hist + tid, histVal);
    }

    void histogram256(PtrStepSzb src, int* hist, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.rows, block.y));

        hipLaunchKernelGGL((histogram256Kernel), dim3(grid), dim3(block), 0, stream, src.data, src.cols, src.rows, src.step, hist);
        cudaSafeCall( hipGetLastError() );

        if (stream == 0)
            cudaSafeCall( hipDeviceSynchronize() );
    }

    __global__ void histogram256Kernel(const uchar* src, int cols, int rows, size_t srcStep, const uchar* mask, size_t maskStep, int* hist)
    {
        __shared__ int shist[256];

        const int y = blockIdx.x * blockDim.y + threadIdx.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        shist[tid] = 0;
        __syncthreads();

        if (y < rows)
        {
            const unsigned int* rowPtr = (const unsigned int*) (src + y * srcStep);
            const unsigned int* maskRowPtr = (const unsigned int*) (mask + y * maskStep);

            const int cols_4 = cols / 4;
            for (int x = threadIdx.x; x < cols_4; x += blockDim.x)
            {
                unsigned int data = rowPtr[x];
                unsigned int m = maskRowPtr[x];

                if ((m >>  0) & 0xFFU)
                    Emulation::smem::atomicAdd(&shist[(data >>  0) & 0xFFU], 1);

                if ((m >>  8) & 0xFFU)
                    Emulation::smem::atomicAdd(&shist[(data >>  8) & 0xFFU], 1);

                if ((m >>  16) & 0xFFU)
                    Emulation::smem::atomicAdd(&shist[(data >> 16) & 0xFFU], 1);

                if ((m >>  24) & 0xFFU)
                    Emulation::smem::atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
            }

            if (cols % 4 != 0 && threadIdx.x == 0)
            {
                for (int x = cols_4 * 4; x < cols; ++x)
                {
                    unsigned int data = ((const uchar*)rowPtr)[x];
                    unsigned int m = ((const uchar*)maskRowPtr)[x];

                    if (m)
                        Emulation::smem::atomicAdd(&shist[data], 1);
                }
            }
        }

        __syncthreads();

        const int histVal = shist[tid];
        if (histVal > 0)
            ::atomicAdd(hist + tid, histVal);
    }

    void histogram256(PtrStepSzb src, PtrStepSzb mask, int* hist, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.rows, block.y));

        hipLaunchKernelGGL((histogram256Kernel), dim3(grid), dim3(block), 0, stream, (const uchar*)src.data, (int)src.cols, (int)src.rows, (size_t)src.step, (const uchar*)mask.data, (size_t)mask.step, (int*)hist);
        cudaSafeCall( hipGetLastError() );

        if (stream == 0)
            cudaSafeCall( hipDeviceSynchronize() );
    }
}

/////////////////////////////////////////////////////////////////////////

namespace hist
{
    __device__ __forceinline__ void histEvenInc(int* shist, uint data, int binSize, int lowerLevel, int upperLevel)
    {
        if (data >= lowerLevel && data <= upperLevel)
        {
            const uint ind = (data - lowerLevel) / binSize;
            Emulation::smem::atomicAdd(shist + ind, 1);
        }
    }

    __global__ void histEven8u(const uchar* src, const size_t step, const int rows, const int cols,
                               int* hist, const int binCount, const int binSize, const int lowerLevel, const int upperLevel)
    {
        HIP_DYNAMIC_SHARED( int, shist)

        const int y = blockIdx.x * blockDim.y + threadIdx.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        if (tid < binCount)
            shist[tid] = 0;

        __syncthreads();

        if (y < rows)
        {
            const uchar* rowPtr = src + y * step;
            const uint* rowPtr4 = (uint*) rowPtr;

            const int cols_4 = cols / 4;
            for (int x = threadIdx.x; x < cols_4; x += blockDim.x)
            {
                const uint data = rowPtr4[x];

                histEvenInc(shist, (data >>  0) & 0xFFU, binSize, lowerLevel, upperLevel);
                histEvenInc(shist, (data >>  8) & 0xFFU, binSize, lowerLevel, upperLevel);
                histEvenInc(shist, (data >> 16) & 0xFFU, binSize, lowerLevel, upperLevel);
                histEvenInc(shist, (data >> 24) & 0xFFU, binSize, lowerLevel, upperLevel);
            }

            if (cols % 4 != 0 && threadIdx.x == 0)
            {
                for (int x = cols_4 * 4; x < cols; ++x)
                {
                    const uchar data = rowPtr[x];
                    histEvenInc(shist, data, binSize, lowerLevel, upperLevel);
                }
            }
        }

        __syncthreads();

        if (tid < binCount)
        {
            const int histVal = shist[tid];

            if (histVal > 0)
                ::atomicAdd(hist + tid, histVal);
        }
    }

    void histEven8u(PtrStepSzb src, int* hist, int binCount, int lowerLevel, int upperLevel, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.rows, block.y));

        const int binSize = divUp(upperLevel - lowerLevel, binCount);

        const size_t smem_size = binCount * sizeof(int);

        hipLaunchKernelGGL((histEven8u), dim3(grid), dim3(block), smem_size, stream, (const uchar*)src.data, (const size_t)src.step, (const int)src.rows,(const int)src.cols,(int*) hist, (const int)binCount,(const int) binSize, (const int)lowerLevel, (const int)upperLevel);
        cudaSafeCall( hipGetLastError() );
        
        if (stream == 0)
            cudaSafeCall( hipDeviceSynchronize() );
    }
}

/////////////////////////////////////////////////////////////////////////

namespace hist
{
    __constant__ int c_lut[256];

    struct EqualizeHist : unary_function<uchar, uchar>
    {
        float scale;

        __host__ EqualizeHist(float _scale) : scale(_scale) {}

        __device__ __forceinline__ uchar operator ()(uchar val) const
        {
            const int lut = c_lut[val];
            return __float2int_rn(scale * lut);
        }
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits<hist::EqualizeHist> : DefaultTransformFunctorTraits<hist::EqualizeHist>
    {
        enum { smart_shift = 4 };
    };
}}}

namespace hist
{
    void equalizeHist(PtrStepSzb src, PtrStepSzb dst, const int* lut, hipStream_t stream)
    {
        if (stream == 0)
            cudaSafeCall( hipMemcpyToSymbol(c_lut, lut, 256 * sizeof(int), 0, hipMemcpyDeviceToDevice) );
        else {
            cudaSafeCall( hipMemcpyToSymbolAsync(c_lut, lut, 256 * sizeof(int), 0, hipMemcpyDeviceToDevice, stream) );
        }

        const float scale = 255.0f / (src.cols * src.rows);

        device::transform(src, dst, EqualizeHist(scale), WithOutMask(), stream);
    }
}

#endif /* CUDA_DISABLER */
