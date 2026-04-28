// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// End-to-end test for the wasm binary bundler.
//
// This C program is compiled to wasm32 (freestanding) and uses the iree_syscall
// imports (write) declared via __attribute__((import_module)). The bundler
// collects the JS companion (syscall_imports.js) from the dependency graph and
// produces a single .mjs file that provides the imports at runtime.
//
// The test verifies the full pipeline:
//   1. C -> .wasm (via wasm32 toolchain)
//   2. JS companion collection (via collect_wasm_js aspect)
//   3. DCE (via wasm import section parsing)
//   4. Bundling (via wasm_binary_bundler.py)
//   5. Runtime execution (via Node.js)

#include <stdint.h>

// Declare the iree_syscall.write import directly. This matches the declaration
// in syscall_imports.h but avoids depending on the full libc header (which is
// internal to the libc package).
__attribute__((import_module("iree_syscall"), import_name("write"))) int32_t
iree_syscall_write(int32_t fd, const void* buffer, int32_t length);

__attribute__((export_name("run_test"))) int run_test(void) {
  const char message[] = "PASS: bundler end-to-end test\n";
  iree_syscall_write(1, message, sizeof(message) - 1);
  return 0;
}
