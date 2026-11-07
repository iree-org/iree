# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
  iree_wasm_cc_binary(
    NAME
      wasi_exit_code_fixture
    MAIN
      "../wasm_test_main.mjs"
    SRCS
      "wasi_exit_code_fixture.c"
    TESTONLY
  )
  find_program(_NODE_EXECUTABLE node REQUIRED)
  add_test(
    NAME build_tools_wasm_test_wasi_exit_code_test
    COMMAND "${_NODE_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/wasi_exit_code_test.mjs"
      "${CMAKE_CURRENT_BINARY_DIR}/wasi_exit_code_fixture.mjs"
  )
  iree_configure_test(build_tools_wasm_test_wasi_exit_code_test)
  return()
endif()

# Smoke test for the freestanding wasm32 toolchain.
add_executable(build_tools_wasm_test_smoke_test smoke_test.c)

# End-to-end test for the wasm binary bundler.
#
# Compiles a minimal C program to wasm32, bundles it with the iree_syscall
# JS companion, and runs it via Node.js.
#
#   ctest -R bundler_test
iree_wasm_cc_test(
  NAME
    bundler_test
  MAIN
    "bundler_test_main.mjs"
  SRCS
    "bundler_test.c"
  DEPS
    iree::build_tools::wasm::libc::syscall_imports
)
