if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
message(FATAL_ERROR
  "This project is intended to be built as part of LLVM via "
  "-DLLVM_EXTERNAL_PROJECTS=torch-mlir-dialects "
  "-DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

option(MLIR_ENABLE_BINDINGS_PYTHON "Enables MLIR Python Bindings" OFF)

set(TORCH_MLIR_DIALECTS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_MLIR_DIALECTS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Building torch-mlir-dialects project at ${TORCH_MLIR_DIALECTS_SOURCE_DIR} (into ${TORCH_MLIR_DIALECTS_BINARY_DIR})")

# TODO: Needed for tablegen. Remove.
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${TORCH_MLIR_DIALECTS_SOURCE_DIR}/include)

function(torch_mlir_dialects_target_includes target)
  set(_DIRS
    ${MLIR_INCLUDE_DIRS}
    ${TORCH_MLIR_DIALECTS_SOURCE_DIR}/include
    ${TORCH_MLIR_DIALECTS_BINARY_DIR}/include
  )
  # In LLVM parlance, the actual target may just be an interface and may not
  # be responsible for actually compiling anything. The corresponding obj.
  # target, when present, is just used for compilation and does not
  # contribute to the interface properties.
  # TODO: Normalize this upstream.
  set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${_DIRS}>)
  if(TARGET obj.${target})
    set_property(TARGET ${target} APPEND PROPERTY INCLUDE_DIRECTORIES
      $<BUILD_INTERFACE:${_DIRS}>)
  endif()
endfunction()

# Configure CMake and tablegen.
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(include)
add_subdirectory(lib)
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tools")
  add_subdirectory(tools)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test")
  add_subdirectory(test)
endif()
