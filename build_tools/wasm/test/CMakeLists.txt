# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
