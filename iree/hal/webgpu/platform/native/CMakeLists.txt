# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    native
  HDRS
    "native_driver.h"
  SRCS
    "native_driver.c"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::tracing
    iree::hal
    iree::hal::webgpu::platform
  PUBLIC
)

if(${IREE_HAL_WEBGPU_PLATFORM} STREQUAL "dawn")
  target_compile_definitions(
    iree_hal_webgpu_platform_native_native
    PUBLIC
      "IREE_HAL_WEBGPU_PLATFORM_DAWN=1"
  )
  target_sources(
    native
    PRIVATE
      "dawn_util.c"
  )
  # DO NOT SUBMIT
  # target_link_libraries(
  #   native
  #   PUBLIC
  #     dawn
  # )
elseif(${IREE_HAL_WEBGPU_PLATFORM} STREQUAL "wgpu-native")
  target_compile_definitions(
    iree_hal_webgpu_platform_native_native
    PUBLIC
      "IREE_HAL_WEBGPU_PLATFORM_WGPU_NATIVE=1"
  )
  target_sources(
    iree_hal_webgpu_platform_native_native
    PRIVATE
      "wgpu_util.c"
  )
  target_link_libraries(
    iree_hal_webgpu_platform_native_native
    PUBLIC
      wgpu-native-static
  )
endif()
