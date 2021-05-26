// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: must be first to ensure that we can define settings for all includes.
#include "iree/base/internal/wait_handle_impl.h"

#if IREE_WAIT_API == IREE_WAIT_API_KQUEUE

#include "iree/base/internal/wait_handle_posix.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_wait_set_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): iree_wait_set_s using a kqueue.
// Could just cast the kqueue() fd to iree_wait_set_s* to avoid allocs.
// https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/kqueue.2.html
struct iree_wait_set_t {
  int reserved;
};

iree_status_t iree_wait_set_allocate(iree_host_size_t capacity,
                                     iree_allocator_t allocator,
                                     iree_wait_set_t** out_set) {
  // TODO(benvanik): kqueue support
}

void iree_wait_set_free(iree_wait_set_t* set) {
  // TODO(benvanik): close()
}

iree_status_t iree_wait_set_insert(iree_wait_set_t* set,
                                   iree_wait_handle_t handle) {
  // TODO(benvanik): kqueue support
}

void iree_wait_set_erase(iree_wait_set_t* set, iree_wait_handle_t handle) {
  // TODO(benvanik): kqueue support
}

void iree_wait_set_clear(iree_wait_set_t* set) {
  // TODO(benvanik): kqueue support
}

iree_status_t iree_wait_all(iree_wait_set_t* set, iree_time_t deadline_ns) {
  // TODO(benvanik): kqueue support
}

iree_status_t iree_wait_any(iree_wait_set_t* set, iree_time_t deadline_ns,
                            iree_wait_handle_t* out_wake_handle) {
  // TODO(benvanik): kqueue support
}

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  // TODO(benvanik): kqueue support
}

#endif  // IREE_WAIT_API == IREE_WAIT_API_KQUEUE
