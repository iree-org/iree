// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm import declarations for the freestanding libc.
//
// These functions are provided by the JS host and appear as wasm imports in the
// module's import section. The JS test harness, browser host, or Node.js loader
// must supply implementations for all of them.
//
// The import_module groups related imports:
//   "iree_syscall" — basic OS-level operations (write, clock, exit)

#ifndef IREE_WASM_LIBC_SYSCALL_IMPORTS_H_
#define IREE_WASM_LIBC_SYSCALL_IMPORTS_H_

#include <stddef.h>
#include <stdint.h>

#define IREE_WASM_IMPORT(module, name) \
  __attribute__((import_module(module), import_name(name)))

// Writes |length| bytes from |buffer| (a wasm linear memory pointer) to the
// output identified by |fd|. fd 1 = stdout, fd 2 = stderr. Returns the number
// of bytes written, or -1 on error.
IREE_WASM_IMPORT("iree_syscall", "write")
int32_t iree_wasm_write(int32_t fd, const void* buffer, int32_t length);

// Returns the current monotonic time in nanoseconds. Backed by
// performance.now() in the JS host (millisecond precision, microsecond
// resolution when cross-origin isolated).
IREE_WASM_IMPORT("iree_syscall", "clock_now_ns")
int64_t iree_wasm_clock_now_ns(void);

// Terminates the wasm module with the given exit code. The JS host should
// record the status and stop executing wasm code. This function should not
// return, but we don't mark it _Noreturn because the import mechanism can't
// enforce that — the caller follows up with __builtin_trap().
IREE_WASM_IMPORT("iree_syscall", "exit")
void iree_wasm_exit(int32_t status);

// Reports an assertion failure to the JS host for diagnostic logging.
// |expression|, |file|, and |function| are pointers into wasm linear memory
// (null-terminated C strings). The JS host should log these and the line
// number, then the caller traps.
IREE_WASM_IMPORT("iree_syscall", "assert_fail")
void iree_wasm_assert_fail_import(const char* expression, const char* file,
                                  int32_t line, const char* function);

#endif  // IREE_WASM_LIBC_SYSCALL_IMPORTS_H_
