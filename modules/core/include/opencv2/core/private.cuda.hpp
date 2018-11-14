/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_CORE_PRIVATE_CUDA_HPP
#define OPENCV_CORE_PRIVATE_CUDA_HPP

#ifndef __OPENCV_BUILD
#  error this is a private header which should not be used from outside of the OpenCV library
#endif

#include "cvconfig.h"

#include "opencv2/core/cvdef.h"
#include "opencv2/core/base.hpp"

#include "opencv2/core/cuda.hpp"

#ifdef HAVE_HIP
#  include <cuda.h>
#  include <hip/hip_runtime.h>
#  if defined(__CUDACC_VER_MAJOR__) && (8 <= __CUDACC_VER_MAJOR__)
#    if defined (__GNUC__) && !defined(__HIPCC__)
#     pragma GCC diagnostic push
#     pragma GCC diagnostic ignored "-Wstrict-aliasing"
#     include <hip/hip_fp16.h>
#     pragma GCC diagnostic pop
#    else
#     include <cuda_fp16.h>
#    endif
#  endif // defined(__CUDACC_VER_MAJOR__) && (8 <= __CUDACC_VER_MAJOR__)

#ifdef NPP_ENABLE
#  include <npp.h>
#endif //NPP_ENABLE

#  include "opencv2/core/cuda_stream_accessor.hpp"
#  include "opencv2/core/cuda/common.hpp"

#ifdef NPP_ENABLE
#  define NPP_VERSION (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD)
#endif //NPP_ENABLE

#  define CUDART_MINIMUM_REQUIRED_VERSION 6050

#  if (CUDART_VERSION < CUDART_MINIMUM_REQUIRED_VERSION)
#    error "Insufficient Cuda Runtime library version, please update it."
#  endif

#  if defined(CUDA_ARCH_BIN_OR_PTX_10)
#    error "OpenCV CUDA module doesn't support NVIDIA compute capability 1.0"
#  endif
#endif

//! @cond IGNORED

namespace cv { namespace cuda {

#ifdef NPP_ENABLE
    CV_EXPORTS cv::String getNppErrorMessage(int code);
#endif //NPP_ENABLE

    CV_EXPORTS cv::String getCudaDriverApiErrorMessage(int code);

    CV_EXPORTS GpuMat getInputMat(InputArray _src, Stream& stream);

    CV_EXPORTS GpuMat getOutputMat(OutputArray _dst, int rows, int cols, int type, Stream& stream);
    static inline GpuMat getOutputMat(OutputArray _dst, Size size, int type, Stream& stream)
    {
        return getOutputMat(_dst, size.height, size.width, type, stream);
    }

    CV_EXPORTS void syncOutput(const GpuMat& dst, OutputArray _dst, Stream& stream);
}}

#ifndef HAVE_HIP

static inline CV_NORETURN void throw_no_cuda() { CV_Error(cv::Error::GpuNotSupported, "The library is compiled without CUDA support"); }

#else // HAVE_HIP

#ifdef NPP_ENABLE
#define nppSafeSetStream(oldStream, newStream) { if(oldStream != newStream) { hipStreamSynchronize(oldStream); nppSetStream(newStream); } }
#endif //NPP_ENABLE

static inline CV_NORETURN void throw_no_cuda() { CV_Error(cv::Error::StsNotImplemented, "The called functionality is disabled for current build or platform"); }

namespace cv { namespace cuda
{
#ifdef NPP_ENABLE
    static inline void checkNppError(int code, const char* file, const int line, const char* func)
    {
        if (code < 0)
            cv::error(cv::Error::GpuApiCallError, getNppErrorMessage(code), func, file, line);
    }
#endif //NPP_ENABLE

    static inline void checkCudaDriverApiError(int code, const char* file, const int line, const char* func)
    {
        if (code != CUDA_SUCCESS)
            cv::error(cv::Error::GpuApiCallError, getCudaDriverApiErrorMessage(code), func, file, line);
    }

#ifdef NPP_ENABLE
    template<int n> struct NPPTypeTraits;
    template<> struct NPPTypeTraits<CV_8U>  { typedef Npp8u npp_type; };
    template<> struct NPPTypeTraits<CV_8S>  { typedef Npp8s npp_type; };
    template<> struct NPPTypeTraits<CV_16U> { typedef Npp16u npp_type; };
    template<> struct NPPTypeTraits<CV_16S> { typedef Npp16s npp_type; };
    template<> struct NPPTypeTraits<CV_32S> { typedef Npp32s npp_type; };
    template<> struct NPPTypeTraits<CV_32F> { typedef Npp32f npp_type; };
    template<> struct NPPTypeTraits<CV_64F> { typedef Npp64f npp_type; };

    class NppStreamHandler
    {
    public:
        inline explicit NppStreamHandler(Stream& newStream)
        {
            oldStream = nppGetStream();
            nppSafeSetStream(oldStream, StreamAccessor::getStream(newStream));
        }

        inline explicit NppStreamHandler(hipStream_t newStream)
        {
            oldStream = nppGetStream();
            nppSafeSetStream(oldStream, newStream);
        }

        inline ~NppStreamHandler()
        {
            nppSafeSetStream(nppGetStream(), oldStream);
        }

    private:
        hipStream_t oldStream;
    };
#endif //NPP_ENABLE

}}

#ifdef NPP_ENABLE
#define nppSafeCall(expr)  cv::cuda::checkNppError(expr, __FILE__, __LINE__, CV_Func)
#endif //NPP_ENABLE

#define cuSafeCall(expr)  cv::cuda::checkCudaDriverApiError(expr, __FILE__, __LINE__, CV_Func)

#endif // HAVE_HIP

//! @endcond

#endif // OPENCV_CORE_PRIVATE_CUDA_HPP
