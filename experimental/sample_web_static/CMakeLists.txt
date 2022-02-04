# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT EMSCRIPTEN)
  return()
endif()

set(_MNIST_OBJECT_NAME "iree_experimental_sample_web_static_mnist")
add_library(${_MNIST_OBJECT_NAME} STATIC ${CMAKE_CURRENT_BINARY_DIR}/mnist_static.o)
SET_TARGET_PROPERTIES(${_MNIST_OBJECT_NAME} PROPERTIES LINKER_LANGUAGE C)

#-------------------------------------------------------------------------------
# Sync
#-------------------------------------------------------------------------------

set(_NAME "iree_experimental_sample_web_static_sync")
add_executable(${_NAME} "")
target_include_directories(${_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_sources(${_NAME}
  PRIVATE
    main.c
    device_sync.c
    ${CMAKE_CURRENT_BINARY_DIR}/mnist_static.h
    ${CMAKE_CURRENT_BINARY_DIR}/mnist_bytecode.h
    ${CMAKE_CURRENT_BINARY_DIR}/mnist_bytecode.c
)
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "sample-web-static-sync")

# Note: we have to be very careful about dependencies here.
#
# The general purpose libraries link in multiple executable loaders and HAL
# drivers/devices, which include code not compatible with Emscripten.
target_link_libraries(${_NAME}
  ${_MNIST_OBJECT_NAME}
  iree_runtime_runtime
  iree_hal_local_loaders_static_library_loader
  iree_hal_local_sync_driver
)

target_link_options(${_NAME} PRIVATE
  # https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-ccall-cwrap
  "-sEXPORTED_FUNCTIONS=['_setup_sample', '_cleanup_sample', '_run_sample']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
  #
  "-sASSERTIONS=1"
  #
  # https://developer.chrome.com/blog/wasm-debugging-2020/
  "-g"
  "-gseparate-dwarf"
)

#-------------------------------------------------------------------------------
# Multithreaded
#-------------------------------------------------------------------------------

set(_NAME "iree_experimental_sample_web_static_multithreaded")
add_executable(${_NAME} "")
target_include_directories(${_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_sources(${_NAME}
  PRIVATE
    main.c
    device_multithreaded.c
    ${CMAKE_CURRENT_BINARY_DIR}/mnist_static.h
    ${CMAKE_CURRENT_BINARY_DIR}/mnist_bytecode.h
    ${CMAKE_CURRENT_BINARY_DIR}/mnist_bytecode.c
)
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "sample-web-static-multithreaded")

# Note: we have to be very careful about dependencies here.
#
# The general purpose libraries link in multiple executable loaders and HAL
# drivers/devices, which include code not compatible with Emscripten.
target_link_libraries(${_NAME}
  ${_MNIST_OBJECT_NAME}
  iree_runtime_runtime
  iree_hal_local_loaders_static_library_loader
  iree_hal_local_task_driver
  iree_task_api
)

target_link_options(${_NAME} PRIVATE
  # https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-ccall-cwrap
  "-sEXPORTED_FUNCTIONS=['_setup_sample', '_cleanup_sample', '_run_sample']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
  #
  "-sASSERTIONS=1"
  #
  # https://developer.chrome.com/blog/wasm-debugging-2020/
  "-g"
  "-gseparate-dwarf"
  #
  # https://emscripten.org/docs/porting/pthreads.html#compiling-with-pthreads-enabled
  "-pthread"
  # "-sINITIAL_MEMORY=67108864"  # 64MB
  "-sPTHREAD_POOL_SIZE=2"
  # https://emscripten.org/docs/porting/pthreads.html#additional-flags
  # "-sPROXY_TO_PTHREAD"
)
