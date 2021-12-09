# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT (${IREE_HAL_WEBGPU_PLATFORM} STREQUAL "wgpu-native"))
  return()
endif()

include(FetchContent)

FetchContent_Declare(
  wgpu-native
  GIT_REPOSITORY https://github.com/gfx-rs/wgpu-native.git
  GIT_TAG a2e81dc1271b876ebf0c0d35857040838ed58297 # 2021-12-04
)
FetchContent_MakeAvailable(wgpu-native)
FetchContent_GetProperties(wgpu-native SOURCE_DIR WGPU_NATIVE_SOURCE)

FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/AndrewGaspar/corrosion.git
  GIT_TAG f679545a63a8b214a415e086f910126ab66714fa # 2021-12-02
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH ${WGPU_NATIVE_SOURCE}/Cargo.toml)

set(LIBCLANG_PATH "D:\\Tools\\LLVM\\lib")
set_property(
  TARGET wgpu-native APPEND
  PROPERTY CORROSION_ENVIRONMENT_VARIABLES
  "LIBCLANG_PATH=${LIBCLANG_PATH}"
)

target_include_directories(wgpu-native-static SYSTEM INTERFACE
  ${WGPU_NATIVE_SOURCE}/ffi
)
if(MSVC)
  add_definitions(-DWGPU_TARGET=WGPU_TARGET_WINDOWS)
  target_link_libraries(wgpu-native-static INTERFACE
    "userenv"
    "ws2_32"
    "Dwmapi"
    "dbghelp"
    "d3dcompiler"
    "D3D12"
    "D3D11"
    "DXGI"
    "setupapi"
    "bcrypt"
  )
elseif(APPLE)
  add_definitions(-DWGPU_TARGET=WGPU_TARGET_MACOS)
  target_link_libraries(wgpu-native-static INTERFACE
    "-framework Cocoa"
    "-framework CoreVideo"
    "-framework IOKit"
    "-framework QuartzCore"
  )
else()
  add_definitions(-DWGPU_TARGET=WGPU_TARGET_LINUX)
endif()
