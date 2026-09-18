# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A plugin built the way one in another repository would be: IREE comes from an
# install tree through find_package, and nothing here reads the IREE sources.
#
# LLVM and MLIR come from wherever the caller points them, because IREE installs
# no llvm/mlir headers and a plugin compiling MLIR C++ needs the ones the
# compiler was built from.
cmake_minimum_required(VERSION 3.21)
project(iree_plugin_from_install CXX C)

# What IREE and the llvm/mlir headers are built as; a compiler defaulting
# lower fails inside those headers rather than in the plugin.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(IREECompiler REQUIRED)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "IREE ABI prefix: ${IREE_COMPILER_ABI_PREFIX}")

# Two libraries, which is what any real plugin is: the second has to be named
# so it is renamed alongside the first.
add_library(registration STATIC "plugin.cpp")
add_library(helper STATIC "helper.cpp")
foreach(_lib registration helper)
  set_target_properties(${_lib} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(${_lib} PRIVATE iree_compiler_PluginAPI_headers)
  target_include_directories(${_lib} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  # Match the compiler this loads into; MLIR's own flags do not carry over.
  target_compile_options(${_lib} PRIVATE -fno-rtti -fno-exceptions)
endforeach()

iree_compiler_register_dynamic_plugin(
  PLUGIN_ID install_tree_probe
  TARGET registration
  EXTRA_ARCHIVES $<TARGET_FILE:helper>
)
