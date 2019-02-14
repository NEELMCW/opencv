set(OPENCL_FOUND ON CACHE BOOL "OpenCL library is found")
if(APPLE)
  set(OPENCL_LIBRARY "-framework OpenCL" CACHE STRING "OpenCL library")
  set(OPENCL_INCLUDE_DIR "" CACHE PATH "OpenCL include directory")
else(APPLE)
  set(OPENCL_LIBRARY "" CACHE STRING "OpenCL library")
  set(OPENCL_INCLUDE_DIR "${OpenCV_SOURCE_DIR}/3rdparty/include/opencl/1.2" CACHE PATH "OpenCL include directory")
  ocv_install_3rdparty_licenses(opencl-headers "${OpenCV_SOURCE_DIR}/3rdparty/include/opencl/LICENSE.txt")
endif(APPLE)
mark_as_advanced(OPENCL_INCLUDE_DIR OPENCL_LIBRARY)

if(OPENCL_FOUND)
  if(OPENCL_LIBRARY)
    set(HAVE_OPENCL_STATIC ON)
    set(OPENCL_LIBRARIES "${OPENCL_LIBRARY}")
  else()
    set(HAVE_OPENCL_STATIC OFF)
  endif()

  if(NOT HAVE_OPENCL_STATIC)
    try_compile(__VALID_OPENCL
      "${OpenCV_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/opencl.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OPENCL_INCLUDE_DIR}"
      OUTPUT_VARIABLE TRY_OUT
      )
    if(NOT TRY_OUT MATCHES "OpenCL is valid")
      message(WARNING "Can't use OpenCL")
      return()
    endif()
  endif()

  set(HAVE_OPENCL 1)

  if(WITH_OPENCL_SVM)
    set(HAVE_OPENCL_SVM 1)
  endif()

  set(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR})

  if(WITH_OPENCLFFT)
    find_path(CLFFT_ROOT_DIR
              NAMES include/clFFT.h
              PATHS ENV CLFFT_PATH ENV ProgramFiles
              PATH_SUFFIXES clAmdFft AMD/clAmdFft
              DOC "AMD FFT root directory"
              NO_DEFAULT_PATH)

    find_path(CLFFT_INCLUDE_DIR
              NAMES clAmdFft.h
              HINTS ${CLFFT_ROOT_DIR}
              PATH_SUFFIXES include
              DOC "clAmdFft include directory")

    if(CLFFT_INCLUDE_DIR)
      set(HAVE_CLFFT 1)
      list(APPEND OPENCL_INCLUDE_DIRS "${CLFFT_INCLUDE_DIR}")
    endif()
  endif()

  if(WITH_OPENCLBLAS)
    find_path(CLBLAS_ROOT_DIR
              NAMES include/clBLAS.h
              PATHS ENV CLBLAS_PATH ENV ProgramFiles
              PATH_SUFFIXES clAmdBlas AMD/clAmdBlas
              DOC "AMD FFT root directory"
              NO_DEFAULT_PATH)

    find_path(CLBLAS_INCLUDE_DIR
              NAMES clAmdBlas.h
              HINTS ${CLBLAS_ROOT_DIR}
              PATH_SUFFIXES include
              DOC "clAmdFft include directory")

    if(CLBLAS_INCLUDE_DIR)
      set(HAVE_CLBLAS 1)
      list(APPEND OPENCL_INCLUDE_DIRS "${CLBLAS_INCLUDE_DIR}")
    endif()
  endif()
endif()

