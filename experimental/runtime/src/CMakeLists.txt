# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Experimental runtime code mirroring the runtime/src/ layout. Package names
# start here, e.g. iree/hal/drivers/hexagon/ -> iree::hal::drivers::hexagon.
#
# Included via the external HAL driver registration of the experimental Hexagon
# HAL driver in the root CMakeLists.txt (-DIREE_EXTERNAL_HAL_DRIVERS=hexagon).

iree_setup_c_src_root(
  DEFAULT_EXPORT_SET Runtime
  DEFAULT_INSTALL_COMPONENT IREEDevLibraries-Runtime
  IMPLICIT_DEFS_TARGET iree_experimental_runtime_defs
)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Android" OR NOT IREE_ARCH STREQUAL "arm_64")
  message(FATAL_ERROR
    "The experimental Hexagon HAL driver requires an Android arm64 build; see "
    "experimental/runtime/src/iree/hal/drivers/hexagon/README.md")
endif()
include(iree_hexagon_sdk)
add_subdirectory(iree/hal/drivers/hexagon)
