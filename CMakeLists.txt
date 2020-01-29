# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.12)
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree CXX C)
set(IREE_IDE_FOLDER IREE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(IREE_ENABLE_DEBUG "Enables debugging of the VM." ON)
option(IREE_ENABLE_LLVM "Enables LLVM dependencies." ON)
option(IREE_ENABLE_TRACING "Enables WTF tracing." OFF)

option(IREE_BUILD_COMPILER "Builds the IREE compiler." ON)
option(IREE_BUILD_TESTS "Builds IREE unit tests." ON)
option(IREE_BUILD_SAMPLES "Builds IREE sample projects." ON)
option(IREE_BUILD_DEBUGGER "Builds the IREE debugger app." OFF)
option(IREE_BUILD_PYTHON_BINDINGS "Builds the IREE python bindings" OFF)

if(${IREE_BUILD_SAMPLES})
  set(IREE_BUILD_COMPILER ON CACHE BOOL "Build the IREE compiler for sample projects." FORCE)
endif()

if(${IREE_BUILD_COMPILER})
  set(IREE_ENABLE_LLVM ON CACHE BOOL "Enable LLVM dependencies if the IREE compiler is build." FORCE)
endif()

#-------------------------------------------------------------------------------
# IREE-specific CMake configuration
#-------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
  ${CMAKE_CURRENT_LIST_DIR}/bindings/python/build_tools/cmake/
  ${CMAKE_CURRENT_LIST_DIR}/third_party/abseil-cpp/absl/copts/
)

include(iree_macros)
include(iree_copts)
include(iree_cc_binary)
include(iree_cc_library)
include(iree_cc_test)
include(iree_tablegen_library)
include(iree_cc_embed_data)
include(iree_bytecode_module)
include(iree_glslang)
include(iree_glsl_vulkan)
include(iree_spirv_kernel_cc_library)
include(iree_pybind_cc_library)
include(iree_glob_lit_tests)
include(iree_lit_test)
include(iree_alwayslink)

string(JOIN " " CMAKE_CXX_FLAGS ${IREE_DEFAULT_COPTS})

set(CMAKE_CXX_FLAGS_FASTBUILD "-gmlt" CACHE STRING "Flags used by the C++ compiler during fast builds." FORCE)
set(CMAKE_C_FLAGS_FASTBUILD "-gmlt" CACHE STRING "Flags used by the C compiler during fast builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_FASTBUILD "-Wl,-S" CACHE STRING "Flags used for linking binaries during fast builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_FASTBUILD "-Wl,-S" CACHE STRING "Flags used by the shared libraries linker binaries during fast builds." FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_FASTBUILD
  CMAKE_C_FLAGS_FASTBUILD
  CMAKE_EXE_LINKER_FLAGS_FASTBUILD
  CMAKE_SHARED_LINKER_FLAGS_FASTBUILD
)

set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#-------------------------------------------------------------------------------
# Third-party dependencies
#-------------------------------------------------------------------------------

if(${IREE_BUILD_COMPILER})
  # Compiler dependencies require Python.
  # Find version 3+ first, so CMake doesn't find a lower version like 2.7 first.
  find_package(PythonInterp 3 REQUIRED)
  find_package(PythonLibs 3 REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/third_party/flatbuffers/CMake/
)

include(external_cc_library)
include(flatbuffer_cc_library)

add_subdirectory(build_tools/third_party/ruy EXCLUDE_FROM_ALL)

add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
add_subdirectory(third_party/abseil-cpp EXCLUDE_FROM_ALL)
add_subdirectory(third_party/flatbuffers EXCLUDE_FROM_ALL)
add_subdirectory(third_party/vulkan_headers EXCLUDE_FROM_ALL)

if(${IREE_ENABLE_LLVM})
  # If CMAKE_BUILD_TYPE is FastBuild, set to Debug for llvm
  set(_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
  string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
  if(NOT uppercase_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELEASE|RELWITHDEBINFO|MINSIZEREL)$")
    set(CMAKE_BUILD_TYPE "Debug")
  endif()

  add_subdirectory(third_party/llvm-project/llvm EXCLUDE_FROM_ALL)
  set_alwayslink_mlir_libs()

  # Reset CMAKE_BUILD_TYPE to its previous setting
  set(CMAKE_BUILD_TYPE "${_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)

  include(external_tablegen_library)
endif()

if(${IREE_BUILD_COMPILER})
  add_subdirectory(build_tools/third_party/tensorflow/tensorflow/compiler/mlir/xla EXCLUDE_FROM_ALL)
  set_alwayslink_tensorflow_libs()
endif()

if(${IREE_BUILD_DEBUGGER} OR ${IREE_BUILD_SAMPLES})
  add_subdirectory(third_party/sdl2 EXCLUDE_FROM_ALL)
  add_subdirectory(build_tools/third_party/dear_imgui EXCLUDE_FROM_ALL)
endif()

if(${IREE_BUILD_TESTS})
  add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
  enable_testing(iree)
endif()

if(${IREE_BUILD_PYTHON_BINDINGS})
  add_subdirectory(third_party/pybind11 EXCLUDE_FROM_ALL)
endif()

#-------------------------------------------------------------------------------
# IREE top-level libraries
#-------------------------------------------------------------------------------

add_subdirectory(build_tools/embed_data/)

add_subdirectory(iree/base)
add_subdirectory(iree/hal)
add_subdirectory(iree/modules)
add_subdirectory(iree/schemas)
add_subdirectory(iree/testing)

if(${IREE_ENABLE_LLVM})
  # The VM requires LLVM to build its op definitions.
  add_subdirectory(iree/vm)
endif()

if(${IREE_BUILD_COMPILER})
  add_subdirectory(third_party/glslang EXCLUDE_FROM_ALL)
  add_subdirectory(iree/compiler)
elseif(${IREE_ENABLE_LLVM})
  # If not building the compiler, tablegen is still needed
  # to generate vm ops so deep include it only.
  add_subdirectory(iree/compiler/Dialect/VM/Tools)
endif()

if(${IREE_BUILD_PYTHON_BINDINGS})
  add_subdirectory(bindings/python)
endif()

add_subdirectory(iree/tools)

if(${IREE_BUILD_SAMPLES})
  add_subdirectory(iree/samples)
endif()
