# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeDependentOption)

#-------------------------------------------------------------------------------
# Core options that may need to be configured in a containing scope when
# including IREE as a sub-project (otherwise, they are just inline in the
# main CMakeLists.txt).
#-------------------------------------------------------------------------------

macro(iree_set_compiler_cmake_options)
  # These options are also set up in the main CMakeLists (for readability),
  # but are included here for maximum compatibility with CMP0126 for the
  # external case (i.e. need to ensure that a cache var is defined before
  # use as a local).
  option(IREE_BUILD_COMPILER "Builds the IREE compiler." ON)
  option(IREE_BUILD_PYTHON_BINDINGS "Builds the IREE python bindings" OFF)

  option(IREE_TARGET_BACKEND_DEFAULTS
         "Sets the default value for all compiler target backends" ON)

  # The VMVX backend is always enabled.
  cmake_dependent_option(IREE_TARGET_BACKEND_VMVX "Enables the 'vmvx' compiler target backend" ON ${IREE_BUILD_COMPILER} OFF)

  # Supported default target backends.
  cmake_dependent_option(IREE_TARGET_BACKEND_DYLIB_LLVM_AOT "Enables the 'dylib-llvm-aot' compiler target backend" ${IREE_TARGET_BACKEND_DEFAULTS} ${IREE_BUILD_COMPILER} OFF)
  cmake_dependent_option(IREE_TARGET_BACKEND_METAL_SPIRV "Enables the 'metal-spirv' compiler target backend" ${IREE_TARGET_BACKEND_DEFAULTS} ${IREE_BUILD_COMPILER} OFF)
  cmake_dependent_option(IREE_TARGET_BACKEND_WASM_LLVM_AOT "Enables the 'wasm-llvm-aot' compiler target backend" ${IREE_TARGET_BACKEND_DEFAULTS} ${IREE_BUILD_COMPILER} OFF)
  cmake_dependent_option(IREE_TARGET_BACKEND_VULKAN_SPIRV "Enables the 'vulkan-spirv' compiler target backend" ${IREE_TARGET_BACKEND_DEFAULTS} ${IREE_BUILD_COMPILER} OFF)

  # Non-default target backends either have additional dependencies or are
  # experimental/niche in some fashion.
  cmake_dependent_option(IREE_TARGET_BACKEND_CUDA "Enables the 'cuda' compiler target backend" OFF ${IREE_BUILD_COMPILER} OFF)
  cmake_dependent_option(IREE_TARGET_BACKEND_ROCM "Enables the 'rocm' compiler target backend" OFF ${IREE_BUILD_COMPILER} OFF)
  # Disable WebGPU by default - it has complex deps and is under development.
  cmake_dependent_option(IREE_TARGET_BACKEND_WEBGPU "Enables the 'webgpu' compiler target backend" OFF ${IREE_BUILD_COMPILER} OFF)
endmacro()
