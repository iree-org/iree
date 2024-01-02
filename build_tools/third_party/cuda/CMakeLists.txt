# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(fetch_cuda_toolkit)
  set(_DOWNLOAD_SCRIPT_PATH "${IREE_SOURCE_DIR}/third_party/nvidia_sdk_download/fetch_cuda_toolkit.py")
  message(STATUS "Checking and downloading CUDA SDK toolkit components")
  execute_process(COMMAND ${Python3_EXECUTABLE}
    "${_DOWNLOAD_SCRIPT_PATH}" "${CMAKE_CURRENT_BINARY_DIR}"
    RESULT_VARIABLE _EXEC_RESULT
    OUTPUT_VARIABLE _ACTUAL_DOWNLOAD_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(_EXEC_RESULT AND NOT _EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "Error fetching CUDA toolkit")
  endif()
  set(CUDAToolkit_ROOT ${_ACTUAL_DOWNLOAD_PATH} PARENT_SCOPE)
endfunction()

if(DEFINED ENV{IREE_CUDA_DEPS_DIR})
  # We define the magic IREE_CUDA_DEPS_DIR env var in our CI docker images if we
  # have a stripped down CUDA toolkit suitable for compiling available. We
  # trigger on this below as a fallback for locating headers and libdevice
  # files. See build_tools/docker/context/fetch_cuda_deps.sh for what this
  # includes (it does not include enough for the normal CMake toolkit search
  # to succeed).
  set(CUDAToolkit_ROOT "$ENV{IREE_CUDA_DEPS_DIR}")
  message(STATUS "Using CUDA minimal deps for CI using IREE_CUDA_DEPS = ${CUDAToolkit_ROOT}")
else()
  # Attempt to deduce a CUDAToolkit_ROOT (possibly downloading) if not
  # explicitly set.
  if(NOT CUDAToolkit_ROOT)
    if(CUDAToolkit_FOUND)
      # Found on the system somewhere, no need to install our own copy.
      cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH CUDAToolkit_ROOT)
      set(CUDAToolkit_ROOT "${CUDAToolkit_ROOT}" PARENT_SCOPE)
      message(STATUS "Using found CUDA toolkit: ${CUDAToolkit_ROOT}")
    else()
      # Download a copy of the CUDA toolkit into the build directory if needed.
      fetch_cuda_toolkit()
      if(CUDAToolkit_ROOT)
        # Download succeeded.
        message(STATUS "Using downloaded CUDA toolkit: ${CUDAToolkit_ROOT}")
        set(CUDAToolkit_ROOT "${CUDAToolkit_ROOT}" PARENT_SCOPE)
        # For some reason having the BIN_DIR set wrong can cause mayhem. Just make
        # sure it is right.
        set(CUDAToolkit_BIN_DIR "${CUDAToolkit_ROOT}/bin" PARENT_SCOPE)
      else()
        message(SEND_ERROR "Failed to download a CUDA toolkit. Check the logs and/or set CUDAToolkit_ROOT to an existing installation.")
      endif()
    endif()
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

# Locate the libdevice file.
if(EXISTS "${IREE_CUDA_LIBDEVICE_PATH}")
  # Explicitly provided: do nothing.
elseif(CUDAToolkit_FOUND AND CUDAToolkit_LIBRARY_ROOT)
  # Note that the variable CUDAToolkit_LIBRARY_ROOT keys off of the presence
  # of version.txt, which was changed to version.json in recent releases
  # and thwarts the search.
  set(IREE_CUDA_LIBDEVICE_PATH "${CUDAToolkit_LIBRARY_ROOT}/nvvm/libdevice/libdevice.10.bc")
elseif(CUDAToolkit_FOUND AND CUDAToolkit_BIN_DIR)
  # Back-track from the bin dir as a fallback.
  set(IREE_CUDA_LIBDEVICE_PATH "${CUDAToolkit_BIN_DIR}/../nvvm/libdevice/libdevice.10.bc")
elseif(CUDAToolkit_ROOT)
  # Sometimes the CUDA toolkit doesn't detect... because, you know. Computers
  # are hard and such. In this case, if the user went to the trouble to
  # tell us where it is, we have enough information.
  set(IREE_CUDA_LIBDEVICE_PATH "${CUDAToolkit_ROOT}/nvvm/libdevice/libdevice.10.bc")
elseif(IREE_CUDA_DOWNLOAD_LIBDEVICE_PATH)
  message(STATUS "Using downloaded CUDA libdevice")
  set(IREE_CUDA_LIBDEVICE_PATH "${IREE_CUDA_DOWNLOAD_LIBDEVICE_PATH}")
else()
  message(FATAL_ERROR "Building with CUDA enabled requires either a CUDA SDK (consult CMake docs for your version: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html) or an explicit path to libdevice (set with -DIREE_CUDA_LIBDEVICE_PATH=/path/to/libdevice.10.bc)")
endif()

if(EXISTS "${IREE_CUDA_LIBDEVICE_PATH}")
  message(STATUS "Using CUDA libdevice: ${IREE_CUDA_LIBDEVICE_PATH}")
else()
  message(SEND_ERROR "Cannot find CUDA libdevice file (${IREE_CUDA_LIBDEVICE_PATH}). Either configure your CUDA SDK such that it can be found or specify explicitly via -DIREE_CUDA_LIBDEVICE_PATH=/path/to/libdevice.10.bc")
endif()

# Locate runtime components.
if(CUDAToolkit_FOUND)
  message(STATUS "Using CUDA INCLUDE_DIRS from found SDK: ${CUDAToolkit_INCLUDE_DIRS}")
elseif(CUDAToolkit_ROOT)
  # See note above about computers being hard.
  # We make minimal use of CUDA for the runtime and really just need cuda.h
  # presently. So let's make a guess at that.
  set(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_ROOT}/include")
  if(EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cuda.h")
    message(STATUS "Using CUDA INCLUDE_DIRS from CUDAToolkit_ROOT: ${CUDAToolkit_INCLUDE_DIRS}")
  else()
    message(SEND_ERROR "Using explicitly specified CUDAToolkit_ROOT, could not find cuda.h at: ${CUDAToolkit_INCLUDE_DIRS}")
  endif()
elseif(IREE_CUDA_DOWNLOAD_INCLUDE_PATH)
  message(STATUS "Using downloaded CUDA includes")
  set(CUDAToolkit_INCLUDE_DIRS "${IREE_CUDA_DOWNLOAD_INCLUDE_PATH}")
else()
  message(SEND_ERROR "Cannot build IREE with CUDA enabled because a CUDA SDK was not found. Consult CMake docs for your version: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html")
endif()

################################################################################
# Targets that IREE depends on which encapsulate access to CUDA.
################################################################################

# We don't install any headers because their only use is in built code
# (not public).
set(IREE_HDRS_ROOT_PATH OFF)
# Considered part of the runtime. See runtime/src/CMakeLists.txt.
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT IREEDevLibraries)
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET Runtime)

iree_c_embed_data(
  PACKAGE
    iree_cuda
  NAME
    libdevice_embedded
  SRCS
    "${IREE_CUDA_LIBDEVICE_PATH}"
  C_FILE_OUTPUT
    "iree_cuda/libdevice_embedded.c"
  H_FILE_OUTPUT
    "iree_cuda/libdevice_embedded.h"
  INCLUDES
    # Allows an include like "iree_cuda/libdevice_embedded.h"
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  FLATTEN
  PUBLIC
)

# Make include dirs only for the BUILD_INTERFACE
set(_include_dirs ${CUDAToolkit_INCLUDE_DIRS})
list(TRANSFORM _include_dirs PREPEND "$<BUILD_INTERFACE:")
list(TRANSFORM _include_dirs APPEND ">")

iree_cc_library(
  PACKAGE
    iree_cuda
  NAME
    headers
  INCLUDES
    ${_include_dirs}
)
