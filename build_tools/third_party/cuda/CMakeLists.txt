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
  if(UNIX)
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
    # Components that we need to fetch.
    set(_COMPONENTS_TO_FETCH "")
    list(APPEND _COMPONENTS_TO_FETCH "cuda_nvcc")
    list(APPEND _COMPONENTS_TO_FETCH "cuda_cudart")

    message(STATUS "Extracting to ${_TARGET_DIR}")
    file(MAKE_DIRECTORY ${_TARGET_DIR})

    # First fetch the download script to the tmp dir.
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

fetch_cuda_toolkit()
message(STATUS "Using downloaded CUDA toolkit: ${CUDAToolkit_ROOT}")
set(CUDAToolkit_ROOT "${CUDAToolkit_ROOT}" PARENT_SCOPE)
# For some reason having the BIN_DIR set wrong can cause mayhem. Just make
# sure it is right.
set(CUDAToolkit_BIN_DIR "${CUDAToolkit_ROOT}/bin" PARENT_SCOPE)
