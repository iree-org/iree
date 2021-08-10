# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.13.4)

# Do not set MSVC warning flags like /W3 by default (since 3.15):
#   https://cmake.org/cmake/help/v3.15/policy/CMP0092.html
if(POLICY CMP0092)
  cmake_policy(SET CMP0092 NEW)
endif()

# LLVM requires CMP0116 for tblgen: https://reviews.llvm.org/D101083
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(iree ASM C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(IREE_IDE_FOLDER IREE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(${MSVC})
  enable_language(ASM_MASM)
else()
  enable_language(ASM)
endif()

#-------------------------------------------------------------------------------
# Project component configuration
#-------------------------------------------------------------------------------

option(IREE_ENABLE_RUNTIME_TRACING "Enables instrumented runtime tracing." OFF)
option(IREE_ENABLE_COMPILER_TRACING "Enables instrumented compiler tracing." OFF)

option(IREE_BUILD_COMPILER "Builds the IREE compiler." ON)
option(IREE_BUILD_TESTS "Builds IREE unit tests." ON)
option(IREE_BUILD_BENCHMARKS "Builds IREE benchmark suites." OFF)
option(IREE_BUILD_DOCS "Builds IREE docs." OFF)
option(IREE_BUILD_SAMPLES "Builds IREE sample projects." ON)

option(IREE_BUILD_TENSORFLOW_ALL "Builds all TensorFlow compiler frontends." OFF)
option(IREE_BUILD_TENSORFLOW_COMPILER "Builds TensorFlow compiler frontend." OFF)
option(IREE_BUILD_TFLITE_COMPILER "Builds the TFLite compiler frontend." OFF)
option(IREE_BUILD_XLA_COMPILER "Builds TensorFlow XLA compiler frontend." OFF)
option(IREE_ENABLE_THREADING "Builds IREE in with thread library support." ON)

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
if(${IREE_BUILD_TENSORFLOW_ALL} OR
   ${IREE_BUILD_TENSORFLOW_COMPILER} OR
   ${IREE_BUILD_TFLITE_COMPILER} OR
   ${IREE_BUILD_XLA_COMPILER})
  set(IREE_ENABLE_TENSORFLOW ON)
endif()

option(IREE_BUILD_BINDINGS_TFLITE "Builds the IREE TFLite C API compatibility shim" ON)
option(IREE_BUILD_BINDINGS_TFLITE_JAVA "Builds the IREE TFLite Java bindings with the C API compatibility shim" ON)

# Default python bindings to enabled for some features.
if(${IREE_ENABLE_TENSORFLOW})
  option(IREE_BUILD_PYTHON_BINDINGS "Builds the IREE python bindings" ON)
else()
  option(IREE_BUILD_PYTHON_BINDINGS "Builds the IREE python bindings" OFF)
endif()

#-------------------------------------------------------------------------------
# Experimental project flags
#-------------------------------------------------------------------------------

option(IREE_BUILD_EXPERIMENTAL_REMOTING "Builds experimental remoting support." OFF)
option(IREE_BUILD_EXPERIMENTAL_JAVA_BINDINGS "Builds the experimental java bindings." OFF)
option(IREE_BUILD_EXPERIMENTAL_ROCM "Builds the experimental ROCM Backend." OFF)
option(IREE_ENABLE_NEW_INTEGRATION_TESTS "Enables new integration tests and disables old." OFF)

#-------------------------------------------------------------------------------
# Derived flags based on primary options
#-------------------------------------------------------------------------------

option(IREE_ENABLE_EMITC "Enables MLIR EmitC dependencies." ${IREE_BUILD_COMPILER})

#-------------------------------------------------------------------------------
# Target and backend configuration
#-------------------------------------------------------------------------------

# List of all HAL drivers to be built by default:
set(IREE_ALL_HAL_DRIVERS
  Cuda
  DyLib
  VMVX
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
  WASM-LLVM-AOT
  Metal-SPIRV
  ROCM
  Vulkan-SPIRV
  VMVX
)

if(${IREE_BUILD_COMPILER})
  if(IREE_TARGET_BACKENDS_TO_BUILD STREQUAL "all")
    set(IREE_TARGET_BACKENDS_TO_BUILD ${IREE_ALL_TARGET_BACKENDS})
  endif()
  message(STATUS "Building target backends: ${IREE_TARGET_BACKENDS_TO_BUILD}")
else()
  set(IREE_TARGET_BACKENDS_TO_BUILD "" CACHE STRING "" FORCE)
  message(STATUS "Compiler is disabled, building no target backends")
endif()

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
# IREE assertions
# We don't love the way this is done, but we have to line it up with how LLVM
# does it and not diverge, since all implementations and all header users must
# have the same definition of NDEBUG.
#
# LLVM defaults LLVM_ENABLE_ASSERTIONS to ON for Debug builds only but then
# conditions itself to only update flags if not building Debug. We just let
# IREE_ENABLE_ASSERTIONS be not conditioned on anything and only update the
# flags in appropriate build types.
#
# If IREE_ENABLE_ASSERTIONS is set ON manually, then
#   - NDEBUG must be undefined
#   - LLVM_ENABLE_ASSERTIONS is forced off in order to keep multiple parties
#     from mucking with globals.
#
# Since CMake forces NDEBUG for !Debug builds, some surgery needs to be done
# at the top level to avoid divergence.
#-------------------------------------------------------------------------------

option(IREE_ENABLE_ASSERTIONS "Force unset of NDEBUG compile option" OFF)

# Filter -DNDEBUG from CMAKE_CXX_FLAGS_* and CMAKE_C_FLAGS_* (if
# CMAKE_BUILD_TYPE is not Debug).
function(iree_fix_ndebug)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
  if(IREE_ENABLE_ASSERTIONS AND NOT "${uppercase_CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
      # Also remove /D NDEBUG to avoid MSVC warnings about conflicting defines.
      foreach (flags_var_to_scrub
              CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}
              CMAKE_C_FLAGS_${uppercase_CMAKE_BUILD_TYPE})
          set(original_flags "${${flags_var_to_scrub}}")
          string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
                  altered_flags "${original_flags}")
          if(NOT "${original_flags}" STREQUAL "${altered_flags}")
            message(STATUS
              "IREE_ENABLE_ASSERTIONS force disabled NDEBUG for ${flags_var_to_scrub}: '${original_flags}' -> '${altered_flags}'")
            set(${flags_var_to_scrub} "${altered_flags}" PARENT_SCOPE)
          endif()
      endforeach()

      # Make sure that LLVM doesn't add its own logic for assertion disabling.
      # We'd like to make sure that we are not dueling over globals.
      set(LLVM_ENABLE_ASSERTIONS OFF PARENT_SCOPE)
  endif()
endfunction()
iree_fix_ndebug()

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
include(iree_third_party_cmake_options)
include(iree_c_embed_data)
include(iree_bytecode_module)
include(iree_c_module)
include(iree_python)
include(iree_lit_test)
include(iree_add_all_subdirs)
include(iree_check_test)
include(iree_run_binary_test)
include(iree_mlir_benchmark_suite)

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
# Check if git submodules have been initialized.
# This will only run if python3 is available.
#-------------------------------------------------------------------------------

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} scripts/git/check_submodule_init.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE ret
  )
  if(NOT ret EQUAL "0")
    message(FATAL_ERROR "check_submodule_init.py failed, see the logs above")
  endif()
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM Dependency
#-------------------------------------------------------------------------------

if(NOT IREE_BUILD_COMPILER)
  message(STATUS "Not adding LLVM/MLIR because the configuration does not require it")
elseif(TARGET LLVMSupport)
  message(STATUS "Not adding IREE bundled LLVM because it has already been included")
  if(NOT TARGET MLIRIR)
    message(FATAL_ERROR "Detected externally provided LLVM project but could not find MLIR projects (is it enabled/installed?)")
  endif()
else()
  message(STATUS "Adding bundled LLVM source dependency")
  iree_set_llvm_cmake_options()

  # Disable LLVM's warnings.
  set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "don't use global flags /facepalm")

  # Stash cmake build type in case LLVM messes with it.
  set(_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")

  # Add default external projects.
  iree_add_llvm_external_project(mlir-iree-dialects MLIR_IREE_DIALECTS ${CMAKE_CURRENT_SOURCE_DIR}/llvm-external-projects/iree-dialects)
  iree_add_llvm_external_project(mlir-hlo MLIR_HLO ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlir-hlo)

  add_subdirectory("third_party/llvm-project/llvm" EXCLUDE_FROM_ALL)

  # Reset CMAKE_BUILD_TYPE to its previous setting.
  set(CMAKE_BUILD_TYPE "${_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)

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

  function(_hack_llvm_include_paths)
    set(_common_include_dirs
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/llvm/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/mlir/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir-hlo/include>
    )
    # Avoid globally modifying paths by instead adding the include paths to the
    # rules that really should have them in the first place.
    target_include_directories(LLVMSupport PUBLIC ${_common_include_dirs})
    target_include_directories(MLIRSupport PUBLIC ${_common_include_dirs})
  endfunction()
  _hack_llvm_include_paths()
endif()

#-------------------------------------------------------------------------------
# Python bindings.
#-------------------------------------------------------------------------------

if(IREE_BUILD_PYTHON_BINDINGS)
  # Note: Optional because python libs can be manually specified.
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()

#-------------------------------------------------------------------------------
# Other dependencies.
#-------------------------------------------------------------------------------

include(external_cc_library)
include(flatbuffer_c_library)

add_subdirectory(build_tools/third_party/cuda EXCLUDE_FROM_ALL)
add_subdirectory(build_tools/third_party/libyaml EXCLUDE_FROM_ALL)
add_subdirectory(build_tools/third_party/stblib EXCLUDE_FROM_ALL)
add_subdirectory(build_tools/third_party/vulkan_memory_allocator EXCLUDE_FROM_ALL)

iree_set_googletest_cmake_options()
add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)

if(IREE_ENABLE_THREADING)
  iree_set_benchmark_cmake_options()
  add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
  iree_set_cpuinfo_cmake_options()
  add_subdirectory(third_party/cpuinfo EXCLUDE_FROM_ALL)
endif()

iree_set_flatcc_cmake_options()
add_subdirectory(build_tools/third_party/flatcc EXCLUDE_FROM_ALL)
add_subdirectory(third_party/flatcc EXCLUDE_FROM_ALL)

add_subdirectory(third_party/vulkan_headers EXCLUDE_FROM_ALL)

# TODO(scotttodd): Iterate some more and find a better place for this.
if (NOT CMAKE_CROSSCOMPILING)
  install(TARGETS iree-flatcc-cli
          COMPONENT iree-flatcc-cli
          RUNTIME DESTINATION bin)
endif()

if(IREE_BUILD_COMPILER)
  add_subdirectory(build_tools/third_party/mlir-hlo EXCLUDE_FROM_ALL)
endif()

if(IREE_ENABLE_EMITC)
  add_subdirectory(build_tools/third_party/mlir-emitc EXCLUDE_FROM_ALL)
endif()

if(IREE_BUILD_TESTS)
  enable_testing(iree)
endif()

if(IREE_BUILD_PYTHON_BINDINGS)
  add_subdirectory(third_party/pybind11 EXCLUDE_FROM_ALL)
endif()

if(IREE_TARGET_BACKEND_METAL-SPIRV)
  iree_set_spirv_cross_cmake_options()
  # SPIRV-Cross is needed to cross compile SPIR-V into MSL source code.
  add_subdirectory(third_party/spirv_cross EXCLUDE_FROM_ALL)
endif()

#-------------------------------------------------------------------------------
# IREE top-level targets
#-------------------------------------------------------------------------------

if(${IREE_BUILD_BENCHMARKS})
  # Add a top-level custom target to drive generating benchmark suites.
  add_custom_target(iree-benchmark-suites)
endif()

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
add_subdirectory(iree/runtime)
add_subdirectory(iree/schemas)
add_subdirectory(iree/task)
add_subdirectory(iree/testing)
add_subdirectory(iree/test)
add_subdirectory(iree/vm)

if(${IREE_BUILD_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()

if(${IREE_BUILD_EXPERIMENTAL_ROCM})
  add_subdirectory(build_tools/third_party/rocm EXCLUDE_FROM_ALL)
  add_subdirectory(experimental/rocm)
endif()

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
