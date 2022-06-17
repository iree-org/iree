# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(_DOWNLOAD_SCRIPT_URL "https://raw.githubusercontent.com/NVIDIA/build-system-archive-import-examples/44dfb51fad75a8a2f1044a4fe221aba70571b86f/parse_redist.py")
set(_DOWNLOAD_DIR ${_TARGET_DIR}/download)
set(_DOWNLOAD_SCRIPT_PATH ${_DOWNLOAD_DIR}/parse_redist.py)

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

# Components that we need to fetch.
set(_COMPONENTS_FO_FETCH "")
list(APPEND _COMPONENTS_FO_FETCH "cuda_nvcc")
list(APPEND _COMPONENTS_FO_FETCH "cuda_cudart")

# Paths within the arch specific installation that we want to retain.
set(_RETAIN_PATHS "")
list(APPEND _RETAIN_PATHS "LICENSE")
list(APPEND _RETAIN_PATHS "nvvm/libdevice/libdevice.10.bc")
list(APPEND _RETAIN_PATHS "include/cuda.h")

message(STATUS "Extracting to ${_TARGET_DIR}")
file(MAKE_DIRECTORY ${_DOWNLOAD_DIR})

# First fetch the download script to the tmp dir.
file(DOWNLOAD ${_DOWNLOAD_SCRIPT_URL} ${_DOWNLOAD_SCRIPT_PATH})

# Then use the download script to fetch and flatten each component we want
# into the tmp dir.
# This will produce a unified directory tree under:
#   flat/$OS-$ARCH
set(SRC_DIR "${_DOWNLOAD_DIR}/${_OS}-${_ARCH}")
foreach(COMPONENT ${_COMPONENTS_FO_FETCH})
  message(STATUS "Downloading component ${COMPONENT}")
  execute_process(COMMAND ${Python3_EXECUTABLE} "${_DOWNLOAD_SCRIPT_PATH}"
    --label "${_VERSION}"
    --product "${_PRODUCT}"
    --os "${_OS}"
    --arch "${_ARCH}"
    --component "${COMPONENT}"
    --output "${_DOWNLOAD_DIR}")
endforeach()

if(NOT EXISTS "${SRC_DIR}")
  message(FATAL_ERROR "Download did not produce expected source dir: ${SRC_DIR}")
  return()
endif()

foreach(REL_PATH ${_RETAIN_PATHS})
  set(SRC_FILE "${SRC_DIR}/${REL_PATH}")
  set(TARGET_FILE "${_TARGET_DIR}/${REL_PATH}")
  message(STATUS "Copy ${SRC_FILE} -> ${TARGET_FILE}")
  file(COPY ${SRC_FILE} DESTINATION ${TARGET_FILE})
endforeach()

# Delete tmp directory.
file(REMOVE_RECURSE ${_DOWNLOAD_DIR})

# Set vars for downloaded cuda deps
set(IREE_CUDA_DOWNLOAD_LIBDEVICE_PATH
    "${_TARGET_DIR}/nvvm/libdevice/libdevice.10.bc/libdevice.10.bc" PARENT_SCOPE)
set(IREE_CUDA_DOWNLOAD_INCLUDE_PATH "${_TARGET_DIR}/include" PARENT_SCOPE)
