// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Program termination and assertion failure for wasm32.
//
// abort() and exit() call JS imports for diagnostic reporting, then execute
// __builtin_trap() to ensure the wasm module actually stops. The trap
// instruction (unreachable) is a single byte and causes a RuntimeError that
// the JS host can catch.
//
// iree_wasm_assert_fail() is called by the assert() macro from assert.h.

#include <stdlib.h>

#include "syscall_imports.h"

_Noreturn void abort(void) {
  iree_wasm_exit(134);  // 128 + SIGABRT(6) — conventional abort exit code.
  __builtin_trap();
}

_Noreturn void exit(int status) {
  iree_wasm_exit(status);
  __builtin_trap();
}

_Noreturn void iree_wasm_assert_fail(const char* expression, const char* file,
                                     int line, const char* function) {
  iree_wasm_assert_fail_import(expression, file, line, function);
  __builtin_trap();
}
