// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <dlfcn.h> for wasm32 (stubs — no dynamic loading on wasm).

#ifndef IREE_WASM_LIBC_DLFCN_H_
#define IREE_WASM_LIBC_DLFCN_H_

#define RTLD_LAZY 1
#define RTLD_NOW 2
#define RTLD_LOCAL 0
#define RTLD_GLOBAL 0x100

void* dlopen(const char* filename, int flags);
int dlclose(void* handle);
void* dlsym(void* handle, const char* symbol);
char* dlerror(void);
int dladdr(const void* addr, void* info);

// Dl_info structure (for dladdr).
typedef struct {
  const char* dli_fname;
  void* dli_fbase;
  const char* dli_sname;
  void* dli_saddr;
} Dl_info;

#endif  // IREE_WASM_LIBC_DLFCN_H_
