# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT EMSCRIPTEN)
  return()
endif()

set(_NAME "iree_experimental_web_sample_webgpu")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
    device_webgpu.c
)
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "web-sample-webgpu")

target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

# Note: we have to be very careful about dependencies here.
#
# The general purpose libraries link in multiple executable loaders and HAL
# drivers/devices, which include code not compatible with Emscripten.
target_link_libraries(${_NAME}
  # C sources
  iree_base_loop_emscripten
  iree_runtime_runtime
  iree_experimental_webgpu_webgpu
  iree_experimental_webgpu_platform_emscripten_emscripten
  # JS sources
  "--js-library ${IREE_ROOT_DIR}/runtime/src/iree/base/internal/wait_handle_emscripten.js"
  "--js-library ${IREE_ROOT_DIR}/runtime/src/iree/base/loop_emscripten.js"
)

target_link_options(${_NAME} PRIVATE
  # https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-ccall-cwrap
  "-sEXPORTED_FUNCTIONS=['_setup_sample', '_cleanup_sample', '_load_program', '_inspect_program', '_unload_program', '_call_function', '_malloc', '_free']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall', 'cwrap', 'UTF8ToString', 'dynCall', 'addFunction']"
  #
  "-sUSE_WEBGPU=1"
  #
  "-sASSERTIONS=1"
  #
  # Programs loaded dynamically can require additional memory, so allow growth.
  "-sALLOW_MEMORY_GROWTH"
  #
  # Allow table growth to create new function pointers from JS that C can call.
  # This is used for the callback passed in to 'call_function'.
  "-sALLOW_TABLE_GROWTH"
  #
  # For https://emscripten.org/docs/debugging/Sanitizers.html#address-sanitizer
  # "-fsanitize=address"
  # "-sALLOW_MEMORY_GROWTH"
  #
  # https://developer.chrome.com/blog/wasm-debugging-2020/
  "-g"
  "-gseparate-dwarf"
)
