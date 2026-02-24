// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_thread.h"

iree_status_t iree_async_proactor_thread_create(
    iree_async_proactor_t* proactor,
    iree_async_proactor_thread_options_t options, iree_allocator_t allocator,
    iree_async_proactor_thread_t** out_thread) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)proactor;
  (void)options;
  (void)allocator;
  *out_thread = NULL;
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "proactor_thread not yet implemented");
}

void iree_async_proactor_thread_retain(iree_async_proactor_thread_t* thread) {
  (void)thread;
}

void iree_async_proactor_thread_release(iree_async_proactor_thread_t* thread) {
  (void)thread;
}

void iree_async_proactor_thread_request_stop(
    iree_async_proactor_thread_t* thread) {
  (void)thread;
}

iree_status_t iree_async_proactor_thread_join(
    iree_async_proactor_thread_t* thread, iree_duration_t timeout) {
  (void)thread;
  (void)timeout;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "proactor_thread not yet implemented");
}

iree_status_t iree_async_proactor_thread_consume_status(
    iree_async_proactor_thread_t* thread) {
  (void)thread;
  return iree_ok_status();
}
