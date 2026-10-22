# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Locates the Qualcomm Hexagon SDK and defines imported targets for it.
#
# Used by the experimental Hexagon HAL driver
# (experimental/runtime/src/iree/hal/drivers/hexagon). The SDK location is
# taken from the HEXAGON_SDK_ROOT CMake or environment variable.
#
# Must be kept in sync with build_tools/bazel/hexagon_sdk_repo.bzl.

if(TARGET hexagon_sdk::qaic)
  return()
endif()

set(HEXAGON_SDK_ROOT "$ENV{HEXAGON_SDK_ROOT}" CACHE PATH
    "Path to the Qualcomm Hexagon SDK")
if(NOT HEXAGON_SDK_ROOT OR NOT EXISTS "${HEXAGON_SDK_ROOT}/incs/remote.h")
  message(FATAL_ERROR
    "The Hexagon HAL driver requires the Qualcomm Hexagon SDK. Set "
    "HEXAGON_SDK_ROOT to the SDK directory (got '${HEXAGON_SDK_ROOT}'). The "
    "SDK can be downloaded from https://github.com/snapdragon-toolchain/hexagon-sdk/releases")
endif()

# The compiler toolchain lives in tools/HEXAGON_Tools/<version>/Tools, where the
# version differs between SDK releases (e.g. 19.0.04 in 6.4.0.2 and 21.0.04 in
# 6.7.0.0). Pick the newest version if there are several.
file(GLOB _HEXAGON_TOOLS_VERSIONS LIST_DIRECTORIES true
     RELATIVE "${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools"
     "${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/*")
if(NOT _HEXAGON_TOOLS_VERSIONS)
  message(FATAL_ERROR
    "No compiler toolchain found in ${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools")
endif()
list(SORT _HEXAGON_TOOLS_VERSIONS COMPARE NATURAL ORDER DESCENDING)
list(GET _HEXAGON_TOOLS_VERSIONS 0 _HEXAGON_TOOLS_VERSION)
set(IREE_HEXAGON_TOOLS_DIR
    "${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/${_HEXAGON_TOOLS_VERSION}/Tools"
    CACHE INTERNAL "Hexagon compiler toolchain directory in the Hexagon SDK")

# RPC interface compiler.
add_executable(hexagon_sdk::qaic IMPORTED GLOBAL)
set_target_properties(hexagon_sdk::qaic PROPERTIES
  IMPORTED_LOCATION "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/bin/qaic"
)

# FastRPC headers and client library for Android aarch64. The library itself is
# provided by the device at runtime.
add_library(hexagon_sdk::fastrpc_headers INTERFACE IMPORTED GLOBAL)
set_target_properties(hexagon_sdk::fastrpc_headers PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES
    "${HEXAGON_SDK_ROOT}/incs;${HEXAGON_SDK_ROOT}/incs/stddef;${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc"
)

add_library(hexagon_sdk::cdsprpc_android_aarch64 SHARED IMPORTED GLOBAL)
set_target_properties(hexagon_sdk::cdsprpc_android_aarch64 PROPERTIES
  IMPORTED_LOCATION
    "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so"
  INTERFACE_LINK_LIBRARIES hexagon_sdk::fastrpc_headers
)
