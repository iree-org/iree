# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(fetch_cuda_toolkit)
  # Parameters to the download script.
  # Look for an appropriate redistrib_*.json here to verify:
  #   https://developer.download.nvidia.com/compute/cuda/redist/
  set(_VERSION "11.6.2")
  set(_PRODUCT "cuda")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_OS "linux")
  elseif(WIN32)
    set(_OS "windows")
  else()
    message(SEND_ERROR "Unsupported OS environment. Must be Windows or Linux.")
    return()
  endif()
  # CUDA is only supported on Linux/Windows where x64 is the only arch for now.
  set(_ARCH "x86_64")

  set(_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/${_VERSION}")
  set(_DOWNLOAD_SCRIPT_URL "https://raw.githubusercontent.com/NVIDIA/build-system-archive-import-examples/44dfb51fad75a8a2f1044a4fe221aba70571b86f/parse_redist.py")
  set(_DOWNLOAD_SCRIPT_PATH "${_TARGET_DIR}/parse_redist.py")

  # Only download if haven't already.
  # This will produce a unified directory tree under:
  #   flat/$OS-$ARCH
  set(_ARCH_DIR "${_TARGET_DIR}/${_OS}-${_ARCH}")
  set(_TOUCH_FILE "${_TARGET_DIR}/cuda_toolkit.downloaded")

  if(NOT EXISTS "${_TOUCH_FILE}")
    # The parse_redist.py script requires the Python requests module, which
    # is not yet installed by default. Check for it.
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import requests"
      RESULT_VARIABLE _PY_MODULES_EXIST_CODE
      OUTPUT_QUIET
    )
    if(NOT ${_PY_MODULES_EXIST_CODE} EQUAL 0)
      message(SEND_ERROR "CUDA auto-download requires Python packages that do not exist on your system. Recommend running: \n  ${Python3_EXECUTABLE} -m pip install requests")
      return()
    endif()

    # Components that we need to fetch.
    set(_COMPONENTS_TO_FETCH "")
    list(APPEND _COMPONENTS_TO_FETCH "cuda_nvcc")
    list(APPEND _COMPONENTS_TO_FETCH "cuda_cudart")

    message(STATUS "Extracting CUDA Toolkit to ${_TARGET_DIR}")
    file(MAKE_DIRECTORY ${_TARGET_DIR})

    # First fetch the download script to its own directory.
    file(DOWNLOAD ${_DOWNLOAD_SCRIPT_URL} ${_DOWNLOAD_SCRIPT_PATH})

    # Then use the download script to fetch and flatten each component we want
    # into the target dir.
    foreach(COMPONENT ${_COMPONENTS_TO_FETCH})
      message(STATUS "Downloading component ${COMPONENT}")
      execute_process(COMMAND ${Python3_EXECUTABLE} "${_DOWNLOAD_SCRIPT_PATH}"
        --label "${_VERSION}"
        --product "${_PRODUCT}"
        --os "${_OS}"
        --arch "${_ARCH}"
        --component "${COMPONENT}"
        --output "${_TARGET_DIR}")
    endforeach()
  endif()

  if(NOT EXISTS "${_ARCH_DIR}")
    message(FATAL_ERROR "Download did not produce expected source dir: ${_ARCH_DIR}")
    return()
  endif()

  file(TOUCH "${_TOUCH_FILE}")
  set(CUDAToolkit_ROOT "${_ARCH_DIR}" PARENT_SCOPE)
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
    # Download the CUDA toolkit if needed.
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
    "${CMAKE_CURRENT_BINARY_DIR}"
  FLATTEN
  PUBLIC
)

iree_cc_library(
  PACKAGE
    iree_cuda
  NAME
    headers
  INCLUDES
    ${CUDAToolkit_INCLUDE_DIRS}
)
