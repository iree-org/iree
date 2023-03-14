if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
message(FATAL_ERROR
  "This project is intended to be built as part of LLVM via "
  "-DLLVM_EXTERNAL_PROJECTS=iree-dialects "
  "-DLLVM_EXTERNAL_IREE_DIALECTS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

option(MLIR_ENABLE_BINDINGS_PYTHON "Enables MLIR Python Bindings" OFF)

set(IREE_DIALECTS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_DIALECTS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Building iree-dialects project at ${IREE_DIALECTS_SOURCE_DIR} (into ${IREE_DIALECTS_BINARY_DIR})")

# TODO: Fix this upstream so that global include directories are not needed.
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)

# TODO: Needed for tablegen. Remove.
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_GENERATED_INCLUDE_DIR})
include_directories(SYSTEM ${IREE_DIALECTS_SOURCE_DIR}/include)

function(iree_dialects_target_includes target)
  set(_DIRS
    $<BUILD_INTERFACE:${MLIR_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${MLIR_GENERATED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${IREE_DIALECTS_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${IREE_DIALECTS_BINARY_DIR}/include>
  )
  # In LLVM parlance, the actual target may just be an interface and may not
  # be responsible for actually compiling anything. The corresponding obj.
  # target, when present, is just used for compilation and does not
  # contribute to the interface properties.
  # TODO: Normalize this upstream.
  target_include_directories(${target} PUBLIC ${_DIRS})
  if(TARGET obj.${target})
    target_include_directories(obj.${target} PRIVATE ${_DIRS})
  endif()
endfunction()

# Configure CMake and tablegen.
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

################################################################################
# Setup python.
################################################################################

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()
