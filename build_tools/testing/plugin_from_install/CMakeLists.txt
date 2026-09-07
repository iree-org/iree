# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A plugin built as one in another repository would be: IREE comes from an
# install tree and nothing here reads the IREE sources.
#
# IREE does not install LLVM/MLIR C++ headers; use the host compiler's build.
cmake_minimum_required(VERSION 3.21)
# Reuse the host toolchain without importing its project-wide compile flags.
if(NOT IREE_HOST_BINARY_DIR)
  message(FATAL_ERROR "IREE_HOST_BINARY_DIR must name the host compiler build")
endif()
set(_toolchain_vars CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_TOOLCHAIN_FILE
  CMAKE_SYSROOT CMAKE_OSX_SYSROOT CMAKE_OSX_ARCHITECTURES
  CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_C_COMPILER_TARGET CMAKE_CXX_COMPILER_TARGET
  CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN)
load_cache("${IREE_HOST_BINARY_DIR}" READ_WITH_PREFIX _host_ ${_toolchain_vars})
foreach(_var IN LISTS _toolchain_vars)
  if(DEFINED _host_${_var})
    set(${_var} "${_host_${_var}}")
  endif()
endforeach()
project(iree_plugin_from_install CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The llvm/mlir headers need it.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(IREECompiler REQUIRED)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "IREE ABI prefix: ${IREE_COMPILER_ABI_PREFIX}")

# Exercise a separate helper archive, which must also be listed for renaming.
add_library(registration STATIC "plugin.cpp")
add_library(helper STATIC "helper.cpp")
option(TEST_UPDATED_HELPER "Introduce a new MLIR reference in the helper" OFF)
if(TEST_UPDATED_HELPER)
  target_compile_definitions(helper PRIVATE TEST_UPDATED_HELPER=1)
endif()
foreach(_lib registration helper)
  set_target_properties(${_lib} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(${_lib} PRIVATE iree_compiler_PluginAPI_build_options)
  target_include_directories(${_lib} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
endforeach()

iree_compiler_register_dynamic_plugin(
  PLUGIN_ID install_tree_probe
  TARGET registration
  EXTRA_ARCHIVES $<TARGET_FILE:helper>
)
