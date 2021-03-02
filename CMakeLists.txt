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

cmake_minimum_required(VERSION 3.13.4)
# Allow target_link_libraries() from other directories (since 3.13):
#   https://cmake.org/cmake/help/v3.13/policy/CMP0079.html
if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(iree CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(IREE_IDE_FOLDER IREE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# Project component configuration
#-------------------------------------------------------------------------------

option(IREE_ENABLE_RUNTIME_TRACING "Enables instrumented runtime tracing." OFF)
option(IREE_ENABLE_MLIR "Enables MLIR/LLVM dependencies." ON)
option(IREE_ENABLE_EMITC "Enables MLIR EmitC dependencies." OFF)

option(IREE_BUILD_COMPILER "Builds the IREE compiler." ON)
option(IREE_BUILD_TESTS "Builds IREE unit tests." ON)
option(IREE_BUILD_DOCS "Builds IREE docs." OFF)
option(IREE_BUILD_SAMPLES "Builds IREE sample projects." ON)
option(IREE_BUILD_DEBUGGER "Builds the IREE debugger app." OFF)

option(IREE_BUILD_TENSORFLOW_COMPILER "Builds TensorFlow compiler frontend." OFF)
option(IREE_BUILD_TFLITE_COMPILER "Builds the TFLite compiler frontend." OFF)
option(IREE_BUILD_XLA_COMPILER "Builds TensorFlow XLA compiler frontend." OFF)

set(IREE_HAL_DRIVERS_TO_BUILD "all"
  CACHE STRING "Semicolon-separated list of HAL drivers to build, or \"all\".")
set(IREE_TARGET_BACKENDS_TO_BUILD "all"
  CACHE STRING "Semicolon-separated list of target backends to build, or \"all\".")

# Properties controlling version and naming of release artifacts.
set(IREE_RELEASE_PACKAGE_SUFFIX "-dev" CACHE STRING "Suffix to append to distributed package names")
set(IREE_RELEASE_VERSION "0.1a1" CACHE STRING "Version to embed in distributed packages")
set(IREE_RELEASE_REVISION "HEAD" CACHE STRING "Version control revision information to embed in distributed packages")

# Master enable for tensorflow build support.
# Note that this is a normal CMake variable used to gate build features (not
# a cache variable that is user-settable).
set(IREE_ENABLE_TENSORFLOW OFF)
if(${IREE_BUILD_TENSORFLOW_COMPILER} OR
   ${IREE_BUILD_TFLITE_COMPILER} OR
   ${IREE_BUILD_XLA_COMPILER})
  set(IREE_ENABLE_TENSORFLOW ON)
endif()

option(IREE_BUILD_BINDINGS_TFLITE "Builds the IREE TFLite C API compatibility shim" ON)

# Default python bindings to enabled for some features.
if(${IREE_ENABLE_TENSORFLOW})
  option(IREE_BUILD_PYTHON_BINDINGS "Builds the IREE python bindings" ON)
else()
  option(IREE_BUILD_PYTHON_BINDINGS "Builds the IREE python bindings" OFF)
endif()

#-------------------------------------------------------------------------------
# Experimental project flags
#-------------------------------------------------------------------------------

option(IREE_BUILD_EXPERIMENTAL_MODEL_BUILDER "Builds experimental Model-Builder sub-project." OFF)
option(IREE_BUILD_EXPERIMENTAL_REMOTING "Builds experimental remoting support." OFF)
option(IREE_BUILD_EXPERIMENTAL_JAVA_BINDINGS "Builds the experimental java bindings." OFF)

#-------------------------------------------------------------------------------
# Derived flags based on primary options
#-------------------------------------------------------------------------------

if(${IREE_BUILD_SAMPLES} OR ${IREE_BUILD_EXPERIMENTAL_MODEL_BUILDER})
  set(IREE_BUILD_COMPILER ON CACHE BOOL "Build the IREE compiler for sample projects." FORCE)
endif()

if(${IREE_BUILD_COMPILER})
  set(IREE_ENABLE_MLIR ON CACHE BOOL "Enable LLVM dependencies if the IREE compiler is build." FORCE)
endif()

if(${IREE_ENABLE_MLIR})
  set(IREE_MLIR_DEP_MODE "BUNDLED" CACHE STRING "One of BUNDLED (default), DISABLED, INSTALLED")
endif()

if(${IREE_ENABLE_EMITC})
  if(NOT ${IREE_ENABLE_MLIR})
    message(FATAL_ERROR "Enabling EmitC requires setting IREE_ENABLE_MLIR to ON.")
  endif()

  string(TOUPPER "${IREE_MLIR_DEP_MODE}" uppercase_IREE_MLIR_DEP_MODE)
  if(NOT uppercase_IREE_MLIR_DEP_MODE MATCHES "^(BUNDLED|INSTALLED)$")
    message(FATAL_ERROR "Enabling EmitC requires IREE_MLIR_DEP_MODE set to BUNDELED or INSTALLED.")
  endif()
endif()

#-------------------------------------------------------------------------------
# Target and backend configuration
#-------------------------------------------------------------------------------

# List of all HAL drivers to be built by default:
set(IREE_ALL_HAL_DRIVERS
  Cuda
  DyLib
  VMLA
  Vulkan
)

if(IREE_HAL_DRIVERS_TO_BUILD STREQUAL "all")
  set(IREE_HAL_DRIVERS_TO_BUILD ${IREE_ALL_HAL_DRIVERS})

  # For Apple platforms we need to use Metal instead of Vulkan.
  if(APPLE)
    list(REMOVE_ITEM IREE_HAL_DRIVERS_TO_BUILD Vulkan)
  endif()
  # Remove Cuda from Android and Apple platforms.
  if(ANDROID OR APPLE)
    list(REMOVE_ITEM IREE_HAL_DRIVERS_TO_BUILD Cuda)
  endif()
endif()
message(STATUS "Building HAL drivers: ${IREE_HAL_DRIVERS_TO_BUILD}")

# Default every IREE_HAL_DRIVER_* to OFF
foreach(_backend ${IREE_ALL_HAL_DRIVERS})
  string(TOUPPER "${_backend}" uppercase_backend)
  set(IREE_HAL_DRIVER_${uppercase_backend} OFF CACHE BOOL "" FORCE)
endforeach()

# Set IREE_HAL_DRIVER_* based on configuration
foreach(_backend ${IREE_HAL_DRIVERS_TO_BUILD})
  string(TOUPPER "${_backend}" uppercase_backend)
  string(REPLACE "\"" "" uppercase_backend ${uppercase_backend})
  set(IREE_HAL_DRIVER_${uppercase_backend} ON CACHE BOOL "" FORCE)
endforeach()

# List of all target backends to be built by default:
set(IREE_ALL_TARGET_BACKENDS
  CUDA
  DYLIB-LLVM-AOT
  Metal-SPIRV
  Vulkan-SPIRV
  VMLA
)

if( IREE_TARGET_BACKENDS_TO_BUILD STREQUAL "all" )
  set( IREE_TARGET_BACKENDS_TO_BUILD ${IREE_ALL_TARGET_BACKENDS} )
endif()
message(STATUS "Building target backends: ${IREE_TARGET_BACKENDS_TO_BUILD}")

# Default every IREE_TARGET_BACKEND_* to OFF
foreach(_backend ${IREE_ALL_TARGET_BACKENDS})
  string(TOUPPER "${_backend}" uppercase_backend)
  set(IREE_TARGET_BACKEND_${uppercase_backend} OFF CACHE BOOL "" FORCE)
endforeach()

# Set IREE_TARGET_BACKEND_* based on configuration
foreach(_backend ${IREE_TARGET_BACKENDS_TO_BUILD})
  string(TOUPPER "${_backend}" uppercase_backend)
  string(REPLACE "\"" "" uppercase_backend ${uppercase_backend})
  set(IREE_TARGET_BACKEND_${uppercase_backend} ON CACHE BOOL "" FORCE)
endforeach()

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
  ${CMAKE_CURRENT_LIST_DIR}/bindings/python/build_tools/cmake/
  ${CMAKE_CURRENT_LIST_DIR}/third_party/abseil-cpp/absl/copts/
)

#-------------------------------------------------------------------------------
# IREE compilation toolchain configuration
#-------------------------------------------------------------------------------

# Enable using lld as the linker for C/C++ targets. This affects IREE and all
# dependency projects.
option(IREE_ENABLE_LLD "Use lld when linking" OFF)
option(IREE_ENABLE_ASAN "Enable address sanitizer" OFF)
option(IREE_ENABLE_MSAN "Enable memory sanitizer" OFF)
option(IREE_ENABLE_TSAN "Enable thread sanitizer" OFF)
option(IREE_ENABLE_CCACHE "Use ccache if installed to speed up rebuilds." OFF)

if(${IREE_ENABLE_CCACHE})
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
  endif()
endif()

#-------------------------------------------------------------------------------
# IREE utility definitions
#-------------------------------------------------------------------------------

include(iree_macros)
include(iree_copts)
include(sanitizers)
include(iree_cc_binary)
include(iree_cc_library)
include(iree_cc_test)
include(iree_tablegen_library)
include(iree_tablegen_doc)
include(iree_cc_embed_data)
include(iree_bytecode_module)
include(iree_python)
include(iree_lit_test)
include(iree_add_all_subdirs)
include(iree_check_test)
include(iree_run_binary_test)

set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#-------------------------------------------------------------------------------
# IREE compilation flags
#-------------------------------------------------------------------------------

iree_append_list_to_string(CMAKE_C_FLAGS_DEBUG ${IREE_C_FLAGS_DEBUG_LIST})
iree_append_list_to_string(CMAKE_CXX_FLAGS_DEBUG ${IREE_CXX_FLAGS_DEBUG_LIST})

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

include(iree_setup_toolchain)

#-------------------------------------------------------------------------------
# Configure python early if there are any features that need it.
# Note that doing this early ensures that dependencies that make incidental
# use of Python (such as LLVM) resolve the same version.
#-------------------------------------------------------------------------------

if(${IREE_BUILD_COMPILER} OR
   ${IREE_BUILD_PYTHON_BINDINGS})
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM Dependency
# We treat the LLVM dependency specially because we support several different
# ways to use it:
#   - Bundled (default): a source dependency directly on the
#     third_party/llvm-project submodule.
#   - External: An external (source or installed) dependency on LLVM.
#   - Provided: When IREE is used as a sub-project, it is assumed that the LLVM
#     dependency is added prior to including this configuration.
#-------------------------------------------------------------------------------

# Disable LLVM's warnings.
set(LLVM_ENABLE_ASSERTIONS OFF CACHE BOOL "don't use global flags /facepalm")
set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "don't use global flags /facepalm")

# Adds bundled projects that must be included after the LLVM directory has
# been added and within the scope of its settings (i.e. build type override,
# etc).
function(add_bundled_mlir_dependent_projects)
  if(${IREE_ENABLE_EMITC})
    add_subdirectory(third_party/mlir-emitc EXCLUDE_FROM_ALL)
  endif()

  if(${IREE_BUILD_COMPILER})
    add_subdirectory(third_party/mlir-hlo EXCLUDE_FROM_ALL)
  endif()
endfunction()

function(add_iree_mlir_src_dep llvm_monorepo_path)
  # Stash cmake build type in case LLVM messes with it.
  set(_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")

  # experimental model builder uses vulkan runner.
  if(${IREE_BUILD_EXPERIMENTAL_MODEL_BUILDER})
    set(MLIR_VULKAN_RUNNER_ENABLED ON)
  endif()

  add_subdirectory("${llvm_monorepo_path}/llvm" "third_party/llvm-project/llvm" EXCLUDE_FROM_ALL)

  # Reset CMAKE_BUILD_TYPE to its previous setting.
  set(CMAKE_BUILD_TYPE "${_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endfunction()

if(${IREE_ENABLE_MLIR})
  if(${IREE_MLIR_DEP_MODE} STREQUAL "DISABLED")
    message(STATUS "Not adding MLIR/LLVM dep due to IREE_MLIR_DEP_MODE=DISABLED")
  elseif(${IREE_MLIR_DEP_MODE} STREQUAL "BUNDLED")
    message(STATUS "Adding bundled LLVM source dependency")
    add_iree_mlir_src_dep("third_party/llvm-project")

    # Extend module path to allow submodules to use LLVM and MLIR CMake modules.
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/lib/cmake/llvm/")

    # Add the bundled include directories for cmake files looking for them.
    list(APPEND LLVM_INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/llvm/include
      ${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/include
    )
    list(APPEND MLIR_INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/mlir/include
      ${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include
    )

    # Avoid globally modifying paths by instead adding the include paths to the
    # rules that really should have them in the first place.
    target_include_directories(LLVMSupport PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/llvm/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/include>
    )
    target_include_directories(MLIRSupport PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/mlir/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include>
    )

    # Set build option to use MHLO alongside with bundled MLIR
    set(MHLO_BUILD_EMBEDDED ON)
  elseif(${IREE_MLIR_DEP_MODE} STREQUAL "INSTALLED")
    # Deps of installed MLIR/LLVM packages.
    find_package(ZLIB)  # See: https://reviews.llvm.org/D79219
    message(STATUS "Looking for installed MLIR/LLVM packages (configure with MLIR_DIR variable)")
    find_package(MLIR REQUIRED CONFIG)
    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)

    # Add include/link directories
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
    link_directories(${LLVM_BUILD_LIBRARY_DIR})
    add_definitions(${LLVM_DEFINITIONS})
  else()
    message(FATAL "Unsupported IREE_MLIR_DEP_MODE=${IREE_MLIR_DEP_MODE}")
  endif()

  add_bundled_mlir_dependent_projects()
endif()

#-------------------------------------------------------------------------------
# Python bindings.
#-------------------------------------------------------------------------------

if(${IREE_BUILD_PYTHON_BINDINGS})
  # Note: Optional because python libs can be manually specified.
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()

#-------------------------------------------------------------------------------
# Other dependencies.
#-------------------------------------------------------------------------------

include(external_cc_library)
include(flatbuffer_c_library)

add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
add_subdirectory(build_tools/third_party/cuda_headers EXCLUDE_FROM_ALL)
add_subdirectory(build_tools/third_party/flatcc EXCLUDE_FROM_ALL)
add_subdirectory(build_tools/third_party/half EXCLUDE_FROM_ALL)
add_subdirectory(build_tools/third_party/pffft EXCLUDE_FROM_ALL)
add_subdirectory(build_tools/third_party/renderdoc_api EXCLUDE_FROM_ALL)
add_subdirectory(build_tools/third_party/vulkan_memory_allocator EXCLUDE_FROM_ALL)

add_subdirectory(third_party/cpuinfo EXCLUDE_FROM_ALL)
add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
add_subdirectory(third_party/abseil-cpp EXCLUDE_FROM_ALL)
add_subdirectory(third_party/flatcc EXCLUDE_FROM_ALL)
add_subdirectory(third_party/vulkan_headers EXCLUDE_FROM_ALL)
# ruy must come after cpuinfo so it uses already-defined targets.
add_subdirectory(third_party/ruy EXCLUDE_FROM_ALL)

# TODO(scotttodd): Iterate some more and find a better place for this.
if (NOT CMAKE_CROSSCOMPILING)
  install(TARGETS iree-flatcc-cli
          COMPONENT iree-flatcc-cli
          RUNTIME DESTINATION bin)
endif()

if(${IREE_BUILD_COMPILER})
  add_subdirectory(build_tools/third_party/mlir-hlo EXCLUDE_FROM_ALL)
endif()

if(${IREE_BUILD_TESTS})
  enable_testing(iree)
endif()

if(${IREE_BUILD_PYTHON_BINDINGS})
  add_subdirectory(third_party/pybind11 EXCLUDE_FROM_ALL)
endif()

if(${IREE_TARGET_BACKEND_METAL-SPIRV})
  # SPIRV-Cross is needed to cross compile SPIR-V into MSL source code.
  add_subdirectory(third_party/spirv_cross EXCLUDE_FROM_ALL)
endif()

#-------------------------------------------------------------------------------
# IREE top-level targets
#-------------------------------------------------------------------------------

if(${IREE_BUILD_DOCS})
  # Add a top-level custom target to drive generating all documentation.
  # Register it to the default target given that IREE_BUILD_DOCS is explicitly
  # requested.
  add_custom_target(iree-doc ALL)
endif()

#-------------------------------------------------------------------------------
# IREE top-level libraries
#-------------------------------------------------------------------------------

add_subdirectory(build_tools/embed_data/)

add_subdirectory(iree/base)
add_subdirectory(iree/hal)
add_subdirectory(iree/modules)
add_subdirectory(iree/schemas)
add_subdirectory(iree/task)
add_subdirectory(iree/testing)
add_subdirectory(iree/test)
add_subdirectory(iree/vm)

if(${IREE_BUILD_COMPILER})
  add_subdirectory(iree/compiler)
endif()

add_subdirectory(iree/tools)

# Order constraint: The python bindings install tools targets from iree/tools
# and must come after it.
if(${IREE_BUILD_PYTHON_BINDINGS})
  add_subdirectory(bindings/python)
endif()

if(${IREE_BUILD_BINDINGS_TFLITE})
  add_subdirectory(bindings/tflite)
endif()

if(${IREE_BUILD_SAMPLES})
  add_subdirectory(iree/samples)
endif()

if(${IREE_BUILD_EXPERIMENTAL_MODEL_BUILDER})
  add_subdirectory(experimental/ModelBuilder)
endif()

if(${IREE_BUILD_EXPERIMENTAL_REMOTING})
  # NOTE: Currently liburing is only used by the experimental remoting
  # support, so keeping it scoped here. If this broadens, then include along
  # with other dependencies as normal.
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(STATUS "Enabling liburing")
    add_subdirectory(build_tools/third_party/liburing EXCLUDE_FROM_ALL)
  endif()
  add_subdirectory(experimental/remoting)
endif()

if(${IREE_BUILD_EXPERIMENTAL_JAVA_BINDINGS})
  add_subdirectory(experimental/bindings/java)
endif()

if(${IREE_ENABLE_TENSORFLOW})
  add_subdirectory(integrations/tensorflow)
endif()

set(IREE_PUBLIC_INCLUDE_DIRS "${IREE_COMMON_INCLUDE_DIRS}"
    CACHE INTERNAL "IREE: Include Directories" FORCE)
