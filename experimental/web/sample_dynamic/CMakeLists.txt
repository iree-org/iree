# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT EMSCRIPTEN)
  return()
endif()

#-------------------------------------------------------------------------------
# Sync
#-------------------------------------------------------------------------------

set(_NAME "iree_experimental_web_sample_dynamic_sync")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
    device_sync.c
)
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "web-sample-dynamic-sync")

target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

# Note: we have to be very careful about dependencies here.
#
# The general purpose libraries link in multiple executable loaders and HAL
# drivers/devices, which include code not compatible with Emscripten.
target_link_libraries(${_NAME}
  iree_runtime_runtime
  iree_hal_local_executable_plugin_manager
  iree_hal_local_loaders_system_library_loader
  iree_hal_local_loaders_vmvx_module_loader
  iree_hal_drivers_local_sync_sync_driver
)

target_link_options(${_NAME} PRIVATE
  # https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-ccall-cwrap
  "-sEXPORTED_FUNCTIONS=['_setup_sample', '_cleanup_sample', '_load_program', '_inspect_program', '_unload_program', '_call_function', '_malloc']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
  #
  "-sASSERTIONS=1"
  #
  # Programs loaded dynamically can require additional memory, so allow growth.
  "-sALLOW_MEMORY_GROWTH"
  #
  # https://developer.chrome.com/blog/wasm-debugging-2020/
  "-g"
  "-gseparate-dwarf"
  #
  # Dynamic linking: https://emscripten.org/docs/compiling/Dynamic-Linking.html
  "-sMAIN_MODULE=2"
  # "-sALLOW_TABLE_GROWTH"
)
