# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Provides the hip_api_headers::headers CMake target wrapping HIP API headers.
# HIP headers are either found via find_package(HIP) on system installs or
# from the bundled third_party/hip-build-deps submodule.

if(NOT DEFINED HIP_API_HEADERS_ROOT)
  if(IREE_USE_SYSTEM_DEPS)
    find_package(HIP CONFIG REQUIRED)
    set(HIP_API_HEADERS_ROOT "${HIP_INCLUDE_DIRS}")
  else()
    set(HIP_API_HEADERS_ROOT "${IREE_SOURCE_DIR}/third_party/hip-build-deps/include")
  endif()
endif()

if(NOT EXISTS "${HIP_API_HEADERS_ROOT}/hip/hip_version.h")
  message(SEND_ERROR "Could not find HIP headers at: ${HIP_API_HEADERS_ROOT}")
endif()

# We don't install any headers because their only use is in built code
# (not public).
set(IREE_HDRS_ROOT_PATH OFF)
# Considered part of the runtime. See runtime/src/CMakeLists.txt.
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT IREEDevLibraries)
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET Runtime)

# Make include dirs only for the BUILD_INTERFACE.
set(_include_dirs "${HIP_API_HEADERS_ROOT}")
list(TRANSFORM _include_dirs PREPEND "$<BUILD_INTERFACE:")
list(TRANSFORM _include_dirs APPEND ">")

iree_cc_library(
  PACKAGE
    hip_api_headers
  NAME
    headers
  INCLUDES
    ${_include_dirs}
)
