// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off: must be included before all other headers.
#include "iree/base/internal/wait_handle_impl.h"
// clang-format on

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/target_platform.h"

#if IREE_WAIT_API == IREE_WAIT_API_NULL

//===----------------------------------------------------------------------===//
// iree_wait_primitive_* raw calls
//===----------------------------------------------------------------------===//

void iree_wait_handle_close(iree_wait_handle_t* handle) {
  iree_wait_handle_deinitialize(handle);
}

//===----------------------------------------------------------------------===//
// iree_wait_set_t
//===----------------------------------------------------------------------===//

struct iree_wait_set_t {
  int reserved;
};

iree_status_t iree_wait_set_allocate(iree_host_size_t capacity,
                                     iree_allocator_t allocator,
                                     iree_wait_set_t** out_set) {
  *out_set = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "wait primitives not available on this platform");
}

void iree_wait_set_free(iree_wait_set_t* set) {}

bool iree_wait_set_is_empty(const iree_wait_set_t* set) { return true; }

iree_status_t iree_wait_set_insert(iree_wait_set_t* set,
                                   iree_wait_handle_t handle) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "wait primitives not available on this platform");
}

void iree_wait_set_erase(iree_wait_set_t* set, iree_wait_handle_t handle) {}

void iree_wait_set_clear(iree_wait_set_t* set) {}

iree_status_t iree_wait_all(iree_wait_set_t* set, iree_time_t deadline_ns) {
  return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                          "wait primitives not available on this platform");
}

iree_status_t iree_wait_any(iree_wait_set_t* set, iree_time_t deadline_ns,
                            iree_wait_handle_t* out_wake_handle) {
  return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                          "wait primitives not available on this platform");
}

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                          "wait primitives not available on this platform");
}

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//

iree_status_t iree_event_initialize(bool initial_state,
                                    iree_event_t* out_event) {
  memset(out_event, 0, sizeof(*out_event));
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "events not available on this platform");
}

void iree_event_deinitialize(iree_event_t* event) {}

void iree_event_set(iree_event_t* event) {}

void iree_event_reset(iree_event_t* event) {}

#endif  // IREE_WAIT_API == IREE_WAIT_API_NULL
