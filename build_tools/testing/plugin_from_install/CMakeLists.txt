# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A plugin built as one in another repository would be: IREE comes from an
# install tree and nothing here reads the IREE sources.
#
# IREE installs no llvm/mlir headers, so the caller points at the compiler's.
cmake_minimum_required(VERSION 3.21)
project(iree_plugin_from_install CXX C)

# The llvm/mlir headers need it.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(IREECompiler REQUIRED)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "IREE ABI prefix: ${IREE_COMPILER_ABI_PREFIX}")

# Two libraries, as any real plugin; the second must be named to be renamed too.
add_library(registration STATIC "plugin.cpp")
add_library(helper STATIC "helper.cpp")
option(TEST_UPDATED_HELPER "Introduce a new MLIR reference in the helper" OFF)
if(TEST_UPDATED_HELPER)
  target_compile_definitions(helper PRIVATE TEST_UPDATED_HELPER=1)
endif()
foreach(_lib registration helper)
  set_target_properties(${_lib} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(${_lib} PRIVATE iree_compiler_PluginAPI_headers)
  target_include_directories(${_lib} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  # Match the host compiler; MLIR's flags do not carry over.
  target_compile_options(${_lib} PRIVATE -fno-rtti -fno-exceptions)
endforeach()

iree_compiler_register_dynamic_plugin(
  PLUGIN_ID install_tree_probe
  TARGET registration
  EXTRA_ARCHIVES $<TARGET_FILE:helper>
)
