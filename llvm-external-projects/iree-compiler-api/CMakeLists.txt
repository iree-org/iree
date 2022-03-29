# This is a standalone project for the creation of compiler API and tools
# packages. It is meant to be built directly and aggregates all other projects
# that make up the IREE compiler input surface area. As a standalone
# project intended for direct distribution, it hard-codes a number of important
# CMake settings for producing optimal binaries and packages.
cmake_minimum_required(VERSION 3.17...3.22)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# Honor all visibility presets (please, please, pretty-please).
# This needs to be set as a default because some IREE dependencies specify
# a CMake min version of 3.0, which causes them to set it locally to OLD.
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)

project(IREE_COMPILER_API LANGUAGES C CXX)

if(IREE_COMPILER_API_SUB_PROJECT)
  # Building as part of a top-level IREE project.
  message(STATUS "Building iree-compiler-api as part of IREE")
  set(LLVM_MAIN_SRC_DIR "${IREE_SOURCE_DIR}/third_party/llvm-project/llvm")
  set(LLVM_MAIN_BINARY_DIR "${IREE_BINARY_DIR}/third_party/llvm-project")
else()
  # Standalone build.
  message(STATUS "Building iree-compiler-api standalone")
  set(IREE_COMPILER_API_STANDALONE ON)
  set(IREE_BUILD_TESTS OFF)  # Conflicts with our tests if we are top-level.
  set(IREE_BUILD_SAMPLES OFF)
  set(LLVM_EXTERNAL_MLIR_IREE_DIALECTS_SOURCE_DIR "${IREE_COMPILER_API_SOURCE_DIR}/../iree-dialects")
  set(IREE_SOURCE_DIR "${IREE_COMPILER_API_SOURCE_DIR}/../..")
  set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/iree")
  set(LLVM_MAIN_SRC_DIR "${IREE_COMPILER_API_SOURCE_DIR}/../../third_party/llvm-project/llvm")
  set(LLVM_MAIN_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm")
  set(LLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR "${IREE_COMPILER_API_SOURCE_DIR}/../../third_party/mlir-hlo")
  set(LLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR "${IREE_COMPILER_API_SOURCE_DIR}/../../third_party/torch-mlir-dialects")
  # Resources generated on windows must have valid version numbers.
  # See set_windows_version_resource_properties.
  set(LLVM_VERSION_MAJOR 0)
  set(LLVM_VERSION_MINOR 0)
  set(LLVM_VERSION_PATCH 0)
  set(PACKAGE_VERSION 0)
  enable_testing()
endif()

message(STATUS "iree-compiler-api Directories:
     IREE_COMPILER_API_SOURCE_DIR = ${IREE_COMPILER_API_SOURCE_DIR}
     IREE_COMPILER_API_BINARY_DIR = ${IREE_COMPILER_API_BINARY_DIR}
     IREE_SOURCE_DIR = ${IREE_SOURCE_DIR}
     IREE_BINARY_DIR = ${IREE_BINARY_DIR}
     LLVM_MAIN_SRC_DIR = ${LLVM_MAIN_SRC_DIR}
     LLVM_MAIN_BINARY_DIR = ${LLVM_MAIN_BINARY_DIR}")

# LLVM dependent project directories.
set(LLD_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../lld)
set(LLD_MAIN_BINARY_DIR ${LLVM_MAIN_BINARY_DIR}/tools/lld)
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_MAIN_BINARY_DIR "${LLVM_MAIN_BINARY_DIR}/tools/mlir")

# Configuration includes.
include(${IREE_SOURCE_DIR}/build_tools/cmake/iree_cmake_options.cmake)
include(${IREE_SOURCE_DIR}/build_tools/cmake/iree_external_cmake_options.cmake)

# CMake settings.
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")

# Hack system includes to account for LLVM not doing includes right.
# TODO: Fix this upstream. Each of these system include hacks is broken in
# a different way, so there is not an easy local fix. They should be removed
# one be one until this project builds. Since this is the first time all of this
# has been mashed together, this is disappointing but not unexpected.
macro(add_system_include_hack include_dir)
  if("${include_dir}" STREQUAL "")
    message(SEND_ERROR "Cannot add empty include dir (will kill tablegen): ${include_dir}")
  endif()
  include_directories(SYSTEM "${include_dir}")
endmacro()

add_system_include_hack(${LLVM_MAIN_SRC_DIR}/include)
add_system_include_hack(${LLVM_MAIN_BINARY_DIR}/include)
add_system_include_hack(${LLD_MAIN_SRC_DIR}/include)
add_system_include_hack(${LLD_MAIN_BINARY_DIR}/include)
add_system_include_hack(${MLIR_MAIN_SRC_DIR}/include)
add_system_include_hack(${MLIR_MAIN_BINARY_DIR}/include)
add_system_include_hack(${IREE_SOURCE_DIR})
add_system_include_hack(${IREE_BINARY_DIR})
add_system_include_hack(${LLVM_EXTERNAL_MLIR_IREE_DIALECTS_SOURCE_DIR}/include)
add_system_include_hack(${LLVM_MAIN_BINARY_DIR}/tools/iree-dialects/include)
add_system_include_hack(${LLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR}/include)
add_system_include_hack(${LLVM_MAIN_BINARY_DIR}/tools/mlir-hlo/include)
add_system_include_hack(${LLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR}/include)
add_system_include_hack(${LLVM_MAIN_BINARY_DIR}/tools/torch-mlir-dialects/include)
add_system_include_hack(${IREE_COMPILER_API_SOURCE_DIR}/include)

function(iree_compiler_target_includes target)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${IREE_COMPILER_API_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${IREE_COMPILER_API_BINARY_DIR}/include>
  )
endfunction()

# Common CMake module paths.
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake/modules)

# Configure Python3 deps so that everyone downstream of here latches to the same
# thing.
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

# Include IREE and LLVM if building standalone.
if(IREE_COMPILER_API_STANDALONE)
  iree_set_compiler_cmake_options()
  iree_set_llvm_cmake_options()
  iree_add_llvm_external_project(mlir-iree-dialects MLIR_IREE_DIALECTS ${LLVM_EXTERNAL_MLIR_IREE_DIALECTS_SOURCE_DIR})
  iree_add_llvm_external_project(mlir-hlo MLIR_HLO ${LLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR})
  iree_add_llvm_external_project(torch-mlir-dialects TORCH_MLIR_DIALECTS ${LLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR})
  set(MLIR_ENABLE_BINDINGS_PYTHON ON)

  # Required mlir-hlo settings.
  # TODO: Consider removing this upstream and just using the main
  # MLIR_ENABLE_BINDINGS_PYTHON option.
  set(MHLO_ENABLE_BINDINGS_PYTHON ON)

  # Required IREE settings.
  set(IREE_BUILD_PYTHON_BINDINGS ON)

  # Include LLVM.
  message(STATUS "Configuring LLVM from (${LLVM_MAIN_SRC_DIR} into ${LLVM_MAIN_BINARY_DIR})...")
  add_subdirectory("${LLVM_MAIN_SRC_DIR}" "${LLVM_MAIN_BINARY_DIR}" EXCLUDE_FROM_ALL)

  # Include IREE.
  message(STATUS "Configuring IREE from (${IREE_SOURCE_DIR} into ${IREE_BINARY_DIR}")
  add_subdirectory("${IREE_SOURCE_DIR}" "${IREE_BINARY_DIR}" EXCLUDE_FROM_ALL)
endif()

# Sub-directories.
# Since building outside of the LLVM build system, setup options for local
# sources.
include(HandleLLVMOptions)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(unittests)
