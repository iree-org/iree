# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    registration
  HDRS
    "driver_module.h"
  DEPS
    iree::base
    iree::base::internal::flags
    iree::hal
    iree::hal::drivers::webgpu
  DEFINES
    "IREE_HAL_HAVE_WEBGPU_DRIVER_MODULE=1"
  PUBLIC
)

if(${EMSCRIPTEN})
  target_sources(
    iree_hal_drivers_webgpu_registration_registration
    PUBLIC
      "driver_module_emscripten.c"
  )
  target_link_libraries(
    iree_hal_drivers_webgpu_registration_registration
    INTERFACE
      iree::hal::drivers::webgpu::platform::emscripten
  )
else()
  target_sources(
    iree_hal_drivers_webgpu_registration_registration
    PUBLIC
      "driver_module_native.c"
  )
  target_link_libraries(
    iree_hal_drivers_webgpu_registration_registration
    INTERFACE
      iree::hal::drivers::webgpu::platform::native
  )
endif()
