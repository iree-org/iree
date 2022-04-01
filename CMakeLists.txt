# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.17...3.22)

# LLVM requires CMP0116 for tblgen: https://reviews.llvm.org/D101083
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(IREE ASM C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(IREE_IDE_FOLDER IREE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(${MSVC})
  enable_language(ASM_MASM)
else()
  enable_language(ASM)
endif()

include(CMakeDependentOption)

#-------------------------------------------------------------------------------
# Project component configuration
# See also: build_tools/cmake/iree_cmake_options.cmake
#-------------------------------------------------------------------------------

option(IREE_ENABLE_RUNTIME_TRACING "Enables instrumented runtime tracing." OFF)
option(IREE_ENABLE_COMPILER_TRACING "Enables instrumented compiler tracing." OFF)
option(IREE_ENABLE_THREADING "Builds IREE in with thread library support." ON)

option(IREE_BUILD_COMPILER "Builds the IREE compiler." ON)
option(IREE_BUILD_TESTS "Builds IREE unit tests." ON)
option(IREE_BUILD_BENCHMARKS "Builds IREE benchmark suites." OFF)
option(IREE_BUILD_DOCS "Builds IREE docs." OFF)
option(IREE_BUILD_SAMPLES "Builds IREE sample projects." ON)
option(IREE_BUILD_TRACY "Builds tracy server tools." OFF)

option(IREE_BYTECODE_MODULE_FORCE_SYSTEM_DYLIB_LINKER "Use the system linker when generating IREE modules in tests/samples/benchmarks (useful for Tracy)." OFF)

# Properties controlling version and naming of release artifacts.
set(IREE_RELEASE_PACKAGE_SUFFIX "" CACHE STRING "Suffix to append to distributed package names")
set(IREE_RELEASE_VERSION "0.1a1" CACHE STRING "Version to embed in distributed packages")
set(IREE_RELEASE_REVISION "HEAD" CACHE STRING "Version control revision information to embed in distributed packages")

option(IREE_BUILD_BINDINGS_TFLITE "Builds the IREE TFLite C API compatibility shim" ON)
option(IREE_BUILD_BINDINGS_TFLITE_JAVA "Builds the IREE TFLite Java bindings with the C API compatibility shim" ON)

# Default python bindings to enabled for some features.
option(IREE_BUILD_PYTHON_BINDINGS "Builds the IREE python bindings" OFF)

#-------------------------------------------------------------------------------
# Experimental project flags
#-------------------------------------------------------------------------------

option(IREE_BUILD_EXPERIMENTAL_REMOTING "Builds experimental remoting support." OFF)
option(IREE_BUILD_EXPERIMENTAL_WEB_SAMPLES "Builds experimental web samples." OFF)
option(IREE_HAL_DRIVER_EXPERIMENTAL_ROCM "Builds the experimental ROCm Backend." OFF)

#-------------------------------------------------------------------------------
# Derived flags based on primary options
#-------------------------------------------------------------------------------

cmake_dependent_option(IREE_ENABLE_EMITC "Enables MLIR EmitC dependencies." ON ${IREE_BUILD_COMPILER} OFF)

#-------------------------------------------------------------------------------
# Runtime HAL Driver Options
# By default, all runtime drivers supported by the current platform which do
# not require external deps are enabled by default. This can be changed with:
#   -DIREE_HAL_DRIVER_DEFAULTS=OFF
#-------------------------------------------------------------------------------

option(IREE_HAL_DRIVER_DEFAULTS "Sets the default value for all runtime HAL drivers" ON)
# CUDA support must be explicitly enabled.
set(IREE_HAL_DRIVER_CUDA_DEFAULT OFF)

# Vulkan is not natively supported on Apple platforms.
# Metal should generally be used instead, though MoltenVK may also work.
if (APPLE)
  set(IREE_HAL_DRIVER_VULKAN_DEFAULT OFF)
else()
  set(IREE_HAL_DRIVER_VULKAN_DEFAULT ${IREE_HAL_DRIVER_DEFAULTS})
endif()

option(IREE_HAL_DRIVER_CUDA "Enables the 'cuda' runtime HAL driver" ${IREE_HAL_DRIVER_CUDA_DEFAULT})
option(IREE_HAL_DRIVER_DYLIB "Enables the 'dylib' runtime HAL driver" ${IREE_HAL_DRIVER_DEFAULTS})
option(IREE_HAL_DRIVER_DYLIB_SYNC "Enables the 'dylib-sync' runtime HAL driver" ${IREE_HAL_DRIVER_DEFAULTS})
option(IREE_HAL_DRIVER_VMVX "Enables the 'vmvx' runtime HAL driver" ${IREE_HAL_DRIVER_DEFAULTS})
option(IREE_HAL_DRIVER_VMVX_SYNC "Enables the 'vmvx-sync' runtime HAL driver" ${IREE_HAL_DRIVER_DEFAULTS})
option(IREE_HAL_DRIVER_VULKAN "Enables the 'vulkan' runtime HAL driver" ${IREE_HAL_DRIVER_VULKAN_DEFAULT})

message(STATUS "IREE runtime drivers:")
if(IREE_HAL_DRIVER_CUDA)
  message(STATUS "  - cuda")
endif()
if(IREE_HAL_DRIVER_DYLIB)
  message(STATUS "  - dylib")
endif()
if(IREE_HAL_DRIVER_DYLIB_SYNC)
  message(STATUS "  - dylib-sync")
endif()
if(IREE_HAL_DRIVER_VMVX)
  message(STATUS "  - vmvx")
endif()
if(IREE_HAL_DRIVER_VMVX_SYNC)
  message(STATUS "  - vmvx-sync")
endif()
if(IREE_HAL_DRIVER_VULKAN)
  message(STATUS "  - vulkan")
endif()

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

option(IREE_DEV_MODE "Configure settings to optimize for IREE development (as opposed to CI or release)" OFF)

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

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
  ${CMAKE_CURRENT_LIST_DIR}/bindings/python/build_tools/cmake/
)

include(iree_macros)
include(iree_copts)
include(sanitizers)
include(iree_cc_binary)
include(iree_cc_library)
include(iree_cc_test)
include(iree_cmake_options)
include(iree_external_cmake_options)
include(iree_tablegen_library)
include(iree_tablegen_doc)
include(iree_c_embed_data)
include(iree_bytecode_module)
include(iree_c_module)
include(iree_python)
include(iree_lit_test)
include(iree_add_all_subdirs)
include(iree_check_test)
include(iree_trace_runner_test)
include(iree_native_test)
include(iree_cc_binary_benchmark)
include(iree_benchmark_suite)
include(iree_hal_cts_test_suite)

set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#-------------------------------------------------------------------------------
# Core options initialization (for options defined in macros in
# iree_cmake_options.cmake).
#-------------------------------------------------------------------------------

iree_set_compiler_cmake_options()

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
# Python
# If building features that require Python development, find them early in
# one invocation (some CMake versions are sensitive to resolving out of order).
# Otherwise, for features that just require the interpreter, find that alone.
#-------------------------------------------------------------------------------

if(IREE_BUILD_PYTHON_BINDINGS)
  # After CMake 3.18, we are able to limit the scope of the search to just
  # Development.Module. Searching for Development will fail in situations where
  # the Python libraries are not available. When possible, limit to just
  # Development.Module.
  # See https://pybind11.readthedocs.io/en/stable/compiling.html#findpython-mode
  if(CMAKE_VERSION VERSION_LESS "3.18.0")
    message(WARNING
      "This version of CMake is not compatible with statically built Python "
      "installations. If Python fails to detect below this may apply to you. "
      "Recommend upgrading to at least CMake 3.18. "
      "Detected current version: ${CMAKE_VERSION}"
    )
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  else()
    # Configuring the Development.Module is flaky in multi-project setups.
    # "Bootstrapping" by first looking for the optional Development component
    # seems to be robust generally.
    # See: https://reviews.llvm.org/D118148
    find_package(Python3 COMPONENTS Interpreter Development)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  endif()
elseif(IREE_BUILD_COMPILER OR IREE_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

# Extended Python environment checks.
if(Python3_FOUND)
  iree_detect_pyyaml()
endif()

if(IREE_BUILD_TESTS AND NOT IREE_PYYAML_FOUND)
  message(WARNING "IREE's regression test suite requires PyYAML to run all tests. It is not installed, so some tests will be disabled.")
endif()

#-------------------------------------------------------------------------------
# Check if git submodules have been initialized.
# This will only run if python3 is available.
#-------------------------------------------------------------------------------

option(IREE_ERROR_ON_MISSING_SUBMODULES "Error if submodules have not been initialized." ON)

find_package(Python3 COMPONENTS Interpreter QUIET)
find_package(Git)
if(IREE_ERROR_ON_MISSING_SUBMODULES AND Python3_FOUND AND Git_FOUND)
  # Only check submodule status when the git commit changes.
  execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE SHORT_HASH_RESULT
    OUTPUT_VARIABLE SHORT_HASH)
  string(REGEX REPLACE "\n$" "" SHORT_HASH "${SHORT_HASH}")
  if(SHORT_HASH_RESULT EQUAL "0" AND NOT "${IREE_GIT_SHORT_HASH}" STREQUAL "${SHORT_HASH}")
    execute_process(
      COMMAND ${Python3_EXECUTABLE} scripts/git/check_submodule_init.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE SUBMODULE_INIT_RESULT
    )
    if(NOT SUBMODULE_INIT_RESULT EQUAL "0")
      message(FATAL_ERROR "check_submodule_init.py failed, see the logs above")
    else()
      set(IREE_GIT_SHORT_HASH "${SHORT_HASH}" CACHE STRING "" FORCE)
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# CUDA configuration for both the compiler and runtime.
# We do this at the top level so that we can fail fast and make global
# decisions that effect both compiler and runtime. It also helps with error
# messaging to do this all in one place, since we can provide very targeted
# advice.
#-------------------------------------------------------------------------------

set(IREE_CUDA_LIBDEVICE_PATH "" CACHE STRING "Absolute path to an appropriate libdevice.*.bc (needed to build the IREE cuda compiler target)")

# If any CUDA features are being built, try to locate a CUDA SDK. We will fall
# back to this as needed for specific features.
if(IREE_TARGET_BACKEND_CUDA OR IREE_HAL_DRIVER_CUDA)
  find_package(CUDAToolkit)
endif()

# If an explicit libdevice file was not specified, and the compiler backend
# is being built, probe for one.
if(IREE_TARGET_BACKEND_CUDA)
  if(IREE_CUDA_LIBDEVICE_PATH)
    # Explicitly provided: do nothing.
  elseif(CUDAToolkit_FOUND AND CUDAToolkit_LIBRARY_ROOT)
    # Note that the variable CUDAToolkit_LIBRARY_ROOT keys off of the presence
    # of version.txt, which was changed to version.json in recent releases
    # and thwarts the search.
    set(IREE_CUDA_LIBDEVICE_PATH "${CUDAToolkit_LIBRARY_ROOT}/nvvm/libdevice/libdevice.10.bc")
  elseif(CUDAToolkit_FOUND AND CUDAToolkit_BIN_DIR)
    # Back-track from the bin dir as a fallback.
    set(IREE_CUDA_LIBDEVICE_PATH "${CUDAToolkit_BIN_DIR}/../nvvm/libdevice/libdevice.10.bc")
  elseif(CUDAToolkit_ROOT)
    # Sometimes the CUDA toolkit doesn't detect... because, you know. Computers
    # are hard and such. In this case, if the user went to the trouble to
    # tell us where it is, we have enough information.
    set(IREE_CUDA_LIBDEVICE_PATH "${CUDAToolkit_ROOT}/nvvm/libdevice/libdevice.10.bc")
  else()
    message(FATAL_ERROR "Building with IREE_TARGET_BACKEND_CUDA requires either a CUDA SDK (consult CMake docs for your version: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html) or an explicit path to libdevice (set with -DIREE_CUDA_LIBDEVICE_PATH=/path/to/libdevice.10.bc)")
  endif()

  if(EXISTS "${IREE_CUDA_LIBDEVICE_PATH}")
    message(STATUS "Using CUDA libdevice: ${IREE_CUDA_LIBDEVICE_PATH}")
  else()
    message(SEND_ERROR "Cannot find CUDA libdevice file (${IREE_CUDA_LIBDEVICE_PATH}). Either configure your CUDA SDK such that it can be found or specify explicitly via -DIREE_CUDA_LIBDEVICE_PATH=/path/to/libdevice.10.bc")
  endif()
endif()

if(IREE_HAL_DRIVER_CUDA)
  if(CUDAToolkit_FOUND)
    message(STATUS "Using CUDA INCLUDE_DIRS from found SDK: ${CUDAToolkit_INCLUDE_DIRS}")
  elseif(CUDAToolkit_ROOT)
    # See note above about computers being hard.
    # We make minimal use of CUDA for the runtime and really just need cuda.h
    # presently. So let's make a guess at that.
    set(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_ROOT}/include")
    if(EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cuda.h")
      message(STATUS "Using CUDA INCLUDE_DIRS from CUDAToolkit_ROOT: ${CUDAToolkit_INCLUDE_DIRS}")
    else()
      message(SEND_ERROR "Using explicitly specified CUDAToolkit_ROOT, could not find cuda.h at: ${CUDAToolkit_INCLUDE_DIRS}")
    endif()
  else()
    message(SEND_ERROR "Cannot build IREE runtime CUDA components (-DIREE_HAL_DRIVER_CUDA=ON) because a CUDA SDK was not found. Consult CMake docs for your version: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html")
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

  # Enable MLIR Python bindings if IREE Python bindings enabled.
  if(IREE_BUILD_PYTHON_BINDINGS)
    set(MLIR_ENABLE_BINDINGS_PYTHON ON)
    set(MHLO_ENABLE_BINDINGS_PYTHON ON)
  endif()

  # Disable LLVM's warnings.
  set(LLVM_ENABLE_WARNINGS OFF)

  # Stash cmake build type in case LLVM messes with it.
  set(_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")

  # Add default external projects.
  iree_add_llvm_external_project(mlir-iree-dialects MLIR_IREE_DIALECTS ${CMAKE_CURRENT_SOURCE_DIR}/llvm-external-projects/iree-dialects)
  iree_add_llvm_external_project(mlir-hlo MLIR_HLO ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlir-hlo)
  iree_add_llvm_external_project(torch-mlir-dialects TORCH_MLIR_DIALECTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/torch-mlir-dialects)

  add_subdirectory("third_party/llvm-project/llvm" EXCLUDE_FROM_ALL)

  # Reset CMAKE_BUILD_TYPE to its previous setting.
  set(CMAKE_BUILD_TYPE "${_CMAKE_BUILD_TYPE}" )

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
# Other dependencies
#-------------------------------------------------------------------------------

include(external_cc_library)
include(flatbuffer_c_library)

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

add_subdirectory(build_tools/third_party/flatcc EXCLUDE_FROM_ALL)

add_subdirectory(third_party/vulkan_headers EXCLUDE_FROM_ALL)

# TODO(scotttodd): Iterate some more and find a better place for this.
if (NOT CMAKE_CROSSCOMPILING)
  install(
    TARGETS iree-flatcc-cli
    COMPONENT iree-flatcc-cli
    RUNTIME DESTINATION bin
  )
endif()

if(IREE_BUILD_COMPILER)
  add_subdirectory(build_tools/third_party/mlir-hlo EXCLUDE_FROM_ALL)
endif()

if(IREE_BUILD_TESTS)
  enable_testing(iree)
endif()

if(IREE_BUILD_PYTHON_BINDINGS)
  if(NOT TARGET pybind11::module)
    message(STATUS "Using bundled pybind11")
    add_subdirectory(third_party/pybind11 EXCLUDE_FROM_ALL)
  else()
    message(STATUS "Not including bundled pybind11 (already configured)")
  endif()
endif()

if(IREE_TARGET_BACKEND_METAL_SPIRV)
  iree_set_spirv_cross_cmake_options()
  # SPIRV-Cross is needed to cross compile SPIR-V into MSL source code.
  add_subdirectory(third_party/spirv_cross EXCLUDE_FROM_ALL)
endif()

if(IREE_TARGET_BACKEND_WEBGPU)
  # Tint is needed to compile SPIR-V into WGSL source code.
  # Tint also requires SPIRV-Tools, which requires SPIRV-Headers.
  iree_set_spirv_headers_cmake_options()
  add_subdirectory(third_party/spirv_headers EXCLUDE_FROM_ALL)
  add_subdirectory(build_tools/third_party/spirv-tools EXCLUDE_FROM_ALL)
  add_subdirectory(build_tools/third_party/tint EXCLUDE_FROM_ALL)
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

# Testing rules that require generation will add dependencies to this target.
# This allows them to be EXCLUDE_FROM_ALL but still invokable.
add_custom_target(iree-test-deps COMMENT "Building IREE test deps")

#-------------------------------------------------------------------------------
# IREE top-level libraries
#-------------------------------------------------------------------------------

add_subdirectory(build_tools/embed_data/)

add_subdirectory(iree/base)
add_subdirectory(iree/builtins)
add_subdirectory(iree/hal)
add_subdirectory(iree/modules)
add_subdirectory(iree/runtime)
add_subdirectory(iree/schemas)
add_subdirectory(iree/task)
add_subdirectory(iree/testing)
# Note: Test deps are not built as part of all (use the iree-test-deps target).
add_subdirectory(iree/test EXCLUDE_FROM_ALL)
add_subdirectory(iree/vm)

if(${IREE_BUILD_BENCHMARKS})
  find_program(IREE_IMPORT_TFLITE_PATH iree-import-tflite)
  if(IREE_IMPORT_TFLITE_PATH)
    message(STATUS "Found ${IREE_IMPORT_TFLITE_PATH} to generate benchmark artifacts")
  else()
    message(STATUS "iree-import-tflite not found. Some benchmarks may not configure")
  endif()
  add_subdirectory(benchmarks)
endif()

if(${IREE_HAL_DRIVER_EXPERIMENTAL_ROCM})
  add_subdirectory(build_tools/third_party/rocm EXCLUDE_FROM_ALL)
  add_subdirectory(experimental/rocm)
endif()

if(${IREE_BUILD_COMPILER})
  add_subdirectory(iree/compiler)
endif()

add_subdirectory(iree/tools)

if(IREE_BUILD_TRACY)
  if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(WARNING "Building Tracy (IREE_BUILD_TRACY) on non-Linux is unsupported and may fail below.")
  endif()
  add_subdirectory(build_tools/third_party/tracy ${CMAKE_CURRENT_BINARY_DIR}/tracy)
  if(NOT TARGET IREETracyCaptureServer)
    message(SEND_ERROR "Could not build Tracy. Either unset IREE_BUILD_TRACY or look for missing dependencies above and install them.")
  endif()
endif()

# Order constraint: The python bindings install tools targets from iree/tools
# and tracy, and must come after it.
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

if(${IREE_BUILD_EXPERIMENTAL_WEB_SAMPLES})
  add_subdirectory(experimental/web)
endif()

set(IREE_PUBLIC_INCLUDE_DIRS "${IREE_COMMON_INCLUDE_DIRS}"
    CACHE INTERNAL "IREE: Include Directories" FORCE)

# Include the iree-compiler-api sub-project. We do this for development
# and CI so that developers have access to the API and tools it provides
# (otherwise, they would need to build multiple top-level projects).
# However, logically, iree-compiler-api depends on iree, and for deployment
# is always built standalone, taking responsibility to include iree and LLVM
# as sub-projects.
# The dependency mode is controlled by the variables
# IREE_COMPILER_API_STANDALONE, which will be set if iree-compiler-api is
# top-level. Otherwise, we set IREE_COMPILER_API_SUB_PROJECT, indicating it
# is being embedded as a sub project.
if(IREE_BUILD_COMPILER AND IREE_BUILD_PYTHON_BINDINGS)
  if(NOT IREE_COMPILER_API_STANDALONE)
    message(STATUS "Including iree-compiler-api as a sub-project")
    set(IREE_COMPILER_API_SUB_PROJECT ON)
    add_subdirectory(llvm-external-projects/iree-compiler-api
        "${CMAKE_CURRENT_BINARY_DIR}/compiler-api")

    # Write out a .env file to make IDEs and developers happy.
    # Yes, we are writing this to the source dir. It is only for IDEs and if
    # it gets clobbered, it is fine.
    set(_pythonpath_env "PYTHONPATH=$<SHELL_PATH:${CMAKE_CURRENT_BINARY_DIR}/compiler-api/python_package;${CMAKE_CURRENT_BINARY_DIR}/bindings/python>\n")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/.env"
      CONTENT "${_pythonpath_env}"
    )
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/.env"
      CONTENT "${_pythonpath_env}"
    )
  endif()
endif()

#-------------------------------------------------------------------------------
# IREE benchmark tools
#-------------------------------------------------------------------------------

add_subdirectory(build_tools/benchmarks)
