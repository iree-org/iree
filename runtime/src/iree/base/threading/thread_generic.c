// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/thread.h"

#if defined(IREE_PLATFORM_GENERIC)

// Generic bare-metal platforms (such as riscv*-unknown-elf) don't provide a
// native threading API. Provide minimal stub implementations so that code can
// be linked, but return UNAVAILABLE from operations that would require
// operating-system threads.

struct iree_thread_t {};

struct iree_thread_override_t {};

IREE_API_EXPORT iree_status_t
iree_thread_create(iree_thread_entry_t entry, void* entry_arg,
                   iree_thread_create_params_t params,
                   iree_allocator_t allocator, iree_thread_t** out_thread) {
  (void)entry;
  (void)entry_arg;
  (void)params;
  (void)allocator;
  (void)out_thread;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "threads are not supported on this platform");
}

IREE_API_EXPORT void iree_thread_retain(iree_thread_t* thread) { (void)thread; }

IREE_API_EXPORT void iree_thread_release(iree_thread_t* thread) {
  (void)thread;
}

IREE_API_EXPORT uintptr_t iree_thread_id(iree_thread_t* thread) {
  (void)thread;
  return 0;
}

IREE_API_EXPORT iree_thread_override_t*
iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  (void)thread;
  (void)priority_class;
  return NULL;
}

IREE_API_EXPORT void iree_thread_override_end(
    iree_thread_override_t* override_token) {
  (void)override_token;
}

IREE_API_EXPORT void iree_thread_request_affinity(
    iree_thread_t* thread, iree_thread_affinity_t affinity) {
  (void)thread;
  (void)affinity;
}

IREE_API_EXPORT void iree_thread_resume(iree_thread_t* thread) { (void)thread; }

IREE_API_EXPORT void iree_thread_join(iree_thread_t* thread) { (void)thread; }

IREE_API_EXPORT void iree_thread_yield(void) {}

#endif  // IREE_PLATFORM_GENERIC
