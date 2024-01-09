# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT EMSCRIPTEN)
  return()
endif()

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/mnist_static.h")
  message(WARNING "Missing mnist_static.h, run ${CMAKE_CURRENT_SOURCE_DIR}/build_sample.sh to generate it")
  return()
endif()

set(_MNIST_OBJECT_NAME "iree_experimental_web_sample_static_mnist")
add_library(${_MNIST_OBJECT_NAME} STATIC ${CMAKE_CURRENT_BINARY_DIR}/mnist_static.o)
SET_TARGET_PROPERTIES(${_MNIST_OBJECT_NAME} PROPERTIES LINKER_LANGUAGE C)

#-------------------------------------------------------------------------------
# Sync
#-------------------------------------------------------------------------------

set(_NAME "iree_experimental_web_sample_static_sync")
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
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "web-sample-static-sync")

target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

# Note: we have to be very careful about dependencies here.
#
# The general purpose libraries link in multiple executable loaders and HAL
# drivers/devices, which include code not compatible with Emscripten.
target_link_libraries(${_NAME}
  ${_MNIST_OBJECT_NAME}
  iree_runtime_runtime
  iree_hal_hal
  iree_hal_local_loaders_static_library_loader
  iree_hal_drivers_local_sync_sync_driver
)

target_link_options(${_NAME} PRIVATE
  # https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-ccall-cwrap
  "-sEXPORTED_FUNCTIONS=['_setup_sample', '_cleanup_sample', '_run_sample', '_malloc']"
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

set(_NAME "iree_experimental_web_sample_static_multithreaded")
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
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "web-sample-static-multithreaded")

target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

# Note: we have to be very careful about dependencies here.
#
# The general purpose libraries link in multiple executable loaders and HAL
# drivers/devices, which include code not compatible with Emscripten.
target_link_libraries(${_NAME}
  ${_MNIST_OBJECT_NAME}
  iree_runtime_runtime
  iree_hal_hal
  iree_hal_local_loaders_static_library_loader
  iree_hal_drivers_local_task_task_driver
  iree_task_api
)

target_link_options(${_NAME} PRIVATE
  # https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-ccall-cwrap
  "-sEXPORTED_FUNCTIONS=['_setup_sample', '_cleanup_sample', '_run_sample', '_malloc']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
  #
  "-sASSERTIONS=1"
  #
  # https://developer.chrome.com/blog/wasm-debugging-2020/
  "-g"
  "-gseparate-dwarf"
  #
  # ------------------------------------------------------------------------- #
  # Multithreading with pthreads, built on Web Workers and SharedArrayBuffer.
  # Docs: https://emscripten.org/docs/porting/pthreads.html
  # IREE's use of threading is pretty restricted so we could eventually drop
  # Emscripten's pthreads implementation in favor of our own.
  #
  # Note: this -pthread flag also needs to be set in compile options.
  "-pthread"
  #
  # IREE creates long lived worker threads during device creation around system
  # startup time. No fixed size thread pool is necessary, so disable thread
  # pool size checks and do not specify the pool size.
  # By starting IREE from a Web Worker, we can avoid needing `PROXY_TO_PTHREAD`
  # and can block when interacting with threads.
  # Web Worker creation requires yielding back to the main browser event loop,
  # so splitting program execution across multiple JavaScript function calls
  # (e.g. `createDevice(); runInference();`) also helps avoid deadlocks.
  "-sPTHREAD_POOL_SIZE_STRICT=0"
  # "-sPTHREAD_POOL_SIZE=N"
  # "-sPROXY_TO_PTHREAD"
  #
  # Threads use memory, so using a larger pool of memory or allowing memory
  # growth may be necessary when using `emscripten_num_logical_cores()`
  # or `navigator.hardwareConcurrency` to pick the worker thread count.
  # IREE is pretty good about not allocating outside of startup, so concerns
  # about this causing slow access to memory *may* not affect IREE too much.
  # "-sALLOW_MEMORY_GROWTH=1"
  # TODO(scotttodd): tune this (figure out where memory is going and trim)
  # "-sINITIAL_MEMORY=33554432"
  # ------------------------------------------------------------------------- #
)
