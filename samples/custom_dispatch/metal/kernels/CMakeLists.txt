# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HAL_DRIVER_METAL)
  return()
endif()

if(NOT IREE_HOST_BIN_DIR AND NOT IREE_TARGET_BACKEND_METAL_SPIRV)
  return()
endif()

find_program(XCRUN xcrun)
if(NOT XCRUN)
  message(STATUS "IREE custom_dispatch/metal/kernels ignored -- xcrun not found")
  return()
endif()

# The precompiled form: kernels.metal -> kernels.air -> kernels.metallib.
# example_msl.mlir uses kernels.metal itself, which iree-compile compiles.
add_custom_command(
  OUTPUT kernels.air
  DEPENDS kernels.metal
  COMMAND ${XCRUN} -sdk macosx metal
      -c ${CMAKE_CURRENT_SOURCE_DIR}/kernels.metal
      -o kernels.air
  VERBATIM
)
add_custom_command(
  OUTPUT kernels.metallib
  DEPENDS kernels.air
  COMMAND ${XCRUN} -sdk macosx metallib
      kernels.air
      -o kernels.metallib
  VERBATIM
)
add_custom_target(iree_samples_custom_dispatch_metal_kernels_metallib DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib
)
add_dependencies(iree-sample-deps iree_samples_custom_dispatch_metal_kernels_metallib)

iree_lit_test_suite(
  NAME
    example
  SRCS
    "example.mlir"
    "example_msl.mlir"
  DATA
    iree_samples_custom_dispatch_metal_kernels_metallib
    iree::samples::custom_dispatch::metal::kernels::kernels.metal
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=metal"
    "hostonly"
)
