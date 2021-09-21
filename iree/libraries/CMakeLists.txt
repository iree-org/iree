# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This package defines "mondo" static and shared libraries corresponding
# to various slices of IREE's codebase (intended for external consumption).
# Generally, libraries are aggregated per-directory.
#
# Note that if depending on these from CMake (i.e. for demos/samples/etc),
# then it is important to not *also* link against the individual libraries
# in the main source tree, as doing so will produce duplicate symbols. These
# libraries represent the public interface for third parties to depend on
# IREE. When using static libraries from outside of CMake, users must take
# care to link against these libraries and their dependencies in the correct
# order. As an example:
#
#   -liree_runtime_core_drivers -liree_runtime_base -lflatcc -lcpuinfo -ldl
#
# See the DEPS sections of each.
#
# Also, when used from within CMake, compile definitions (include directories,
# defines, etc) are automatically managed. From the outside, these must
# be specified.

################################################################################
# iree_runtime_base library
# Contains the core runtime and supporting code. Does not contain any drivers.
################################################################################

iree_aggregate_library(
  NAME
    runtime_base
  OUTPUT_NAME
    iree_runtime_base
  PACKAGES
    iree_base
    iree_base_internal
    iree_hal
    iree_hal_local
    iree_hal_local_elf
    iree_hal_local_loaders
    iree_hal_utils
    iree_modules_check
    iree_modules_hal
    iree_runtime
    iree_task
    iree_vm
  DEPS
    ::flatcc
    cpuinfo
    ${CMAKE_DL_LIBS}
)

iree_aggregate_library(
  NAME
    flatcc
  OUTPUT_NAME
    iree_flatcc
  PACKAGES
    flatcc
)

################################################################################
# iree_runtime_core_drivers library
# Runtime drivers that are built in to the core distribution.
################################################################################

# Setup packages for enabled core drivers.
set(_enabled_driver_packages)

if(IREE_HAL_DRIVER_CUDA)
  list(APPEND _enabled_driver_packages
    iree_hal_cuda
    iree_hal_cuda_registration
  )
endif()

if(IREE_HAL_DRIVER_DYLIB)
  list(APPEND _enabled_driver_packages
    iree_hal_dylib
    iree_hal_dylib_registration
  )
endif()

if(IREE_HAL_DRIVER_VULKAN)
  list(APPEND _enabled_driver_packages
    iree_hal_vulkan
    iree_hal_vulkan_registration
    iree_hal_vulkan_util
  )
endif()

if(IREE_HAL_DRIVER_VMVX)
  list(APPEND _enabled_driver_packages
    iree_hal_vmvx
    iree_hal_vmvx_registration
    iree_modules_vmvx
  )
endif()

iree_aggregate_library(
  NAME
    runtime_core_drivers
  OUTPUT_NAME
    iree_runtime_core_drivers
  PACKAGES
    iree_hal_drivers  # Registers all
    ${_enabled_driver_packages}
  DEPS
    ::runtime_base
)
