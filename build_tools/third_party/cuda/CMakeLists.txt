# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(DOWNLOAD_SCRIPT_URL "https://raw.githubusercontent.com/NVIDIA/build-system-archive-import-examples/44dfb51fad75a8a2f1044a4fe221aba70571b86f/parse_redist.py")
set(DOWNLOAD_DIR ${TARGET_DIR}/download)
set(DOWNLOAD_SCRIPT_PATH ${DOWNLOAD_DIR}/parse_redist.py)

# Parameters to the download script.
# Look for an appropriate redistrib_*.json here to verify:
#   https://developer.download.nvidia.com/compute/cuda/redist/
set(VERSION "11.6.2")
set(PRODUCT "cuda")
if(UNIX)
  set(OS "linux")
elseif(WIN32)
  set(OS "windows")
else()
  message(SEND_ERROR "Unsupported OS environment. Must be Windows or Linux.")
  return()
endif()
set(ARCH "x86_64")

# Components that we need to fetch.
unset(COMPONENTS)
list(APPEND COMPONENTS "cuda_nvcc")
list(APPEND COMPONENTS "cuda_cudart")

# Paths within the arch specific installation that we want to retain.
unset(RETAIN_PATHS)
list(APPEND RETAIN_PATHS "LICENSE")
list(APPEND RETAIN_PATHS "nvvm/libdevice/libdevice.10.bc")
list(APPEND RETAIN_PATHS "include/cuda.h")

message(NOTICE "Extracting to ${TARGET_DIR}")
file(MAKE_DIRECTORY ${DOWNLOAD_DIR})

# First fetch the download script to the tmp dir.
file(DOWNLOAD ${DOWNLOAD_SCRIPT_URL} ${DOWNLOAD_SCRIPT_PATH})

# Then use the download script to fetch and flatten each component we want
# into the tmp dir.
# This will produce a unified directory tree under:
#   flat/$OS-$ARCH
set(SRC_DIR "${DOWNLOAD_DIR}/${OS}-${ARCH}")
foreach(COMPONENT ${COMPONENTS})
  message(NOTICE "Downloading component ${COMPONENT}")
  execute_process(COMMAND python3 "${DOWNLOAD_SCRIPT_PATH}"
    --label "${VERSION}"
    --product "${PRODUCT}"
    --os "${OS}"
    --arch "${ARCH}"
    --component "${COMPONENT}"
    --output "${DOWNLOAD_DIR}")
endforeach()

if(NOT EXISTS "${SRC_DIR}")
  message(SEND_ERROR "Download did not produce expected source dir: ${SRC_DIR}")
  return()
endif()

foreach(REL_PATH ${RETAIN_PATHS})
  set(SRC_FILE "${SRC_DIR}/${REL_PATH}")
  set(TARGET_FILE "${TARGET_DIR}/${REL_PATH}")
  message(NOTICE "Copy ${SRC_FILE} -> ${TARGET_FILE}")
  file(COPY ${SRC_FILE} DESTINATION ${TARGET_FILE})
endforeach()

# Delete tmp directory.
file(REMOVE_RECURSE ${DOWNLOAD_DIR})

# Set vars for downloaded cuda deps
set(IREE_CUDA_DOWNLOAD_LIBDEVICE_PATH "${TARGET_DIR}/nvvm/libdevice/libdevice.10.bc" PARENT_SCOPE)
set(IREE_CUDA_DOWNLOAD_INCLUDE_PATH "${TARGET_DIR}/include" PARENT_SCOPE)
