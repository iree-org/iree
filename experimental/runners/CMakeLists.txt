cmake_minimum_required(VERSION 3.18)
project(linalg-opt CXX)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_DIR})
list(APPEND CMAKE_MODULE_PATH ${MLIR_DIR}/../llvm)
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
)

add_llvm_executable(linalg-opt
  mlir-opt.cpp
  LinalgTileToGeneric.cpp
  LinalgComprehensiveBufferizePass.cpp
  LinalgTensorCodegenStrategy.cpp
)
target_link_libraries(linalg-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(linalg-opt)
