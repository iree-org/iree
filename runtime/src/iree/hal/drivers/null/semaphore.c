// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/null/semaphore.h"

#include "iree/hal/utils/semaphore_base.h"

//===----------------------------------------------------------------------===//
// iree_hal_null_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_null_semaphore_t {
  iree_hal_semaphore_t base;
  iree_allocator_t host_allocator;
} iree_hal_null_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_null_semaphore_vtable;

static iree_hal_null_semaphore_t* iree_hal_null_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_semaphore_vtable);
  return (iree_hal_null_semaphore_t*)base_value;
}

iree_status_t iree_hal_null_semaphore_create(
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_null_semaphore_t* semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*semaphore),
                                (void**)&semaphore));
  iree_hal_semaphore_initialize(&iree_hal_null_semaphore_vtable,
                                &semaphore->base);
  semaphore->host_allocator = host_allocator;

  // TODO(null): implement semaphores. Note that there is some basic support
  // provided for timepoints as part of iree/hal/utils/semaphore_base.h but the
  // actual synchronization aspects are handled by the implementation.
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "semaphore not implemented");

  if (iree_status_is_ok(status)) {
    *out_semaphore = &semaphore->base;
  } else {
    iree_hal_semaphore_release(&semaphore->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_null_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_null_semaphore_t* semaphore =
      iree_hal_null_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_null_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  *out_value = 0;
  iree_hal_null_semaphore_t* semaphore =
      iree_hal_null_semaphore_cast(base_semaphore);

  // TODO(null): return the current value of the semaphore by (depending on the
  // implementation) making a syscall to get it. It's expected that the value
  // may immediately change after being queried here.

  // TODO(null): if the value is IREE_HAL_SEMAPHORE_FAILURE_VALUE then return
  // the failure status cached from the fail call by cloning it (like `return
  // iree_status_clone(semaphore->failure_status)`).

  (void)semaphore;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "semaphore query not implemented");

  return status;
}

static iree_status_t iree_hal_null_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_null_semaphore_t* semaphore =
      iree_hal_null_semaphore_cast(base_semaphore);

  // TODO(null): validation is optional but encouraged if cheap: semaphores
  // must always be signaled to a value that is greater than the previous value
  // (not less-than-or-equal).

  // TODO(null): signals when the semaphore have failed should also fail and
  // because failed semaphores have their value set to
  // IREE_HAL_SEMAPHORE_FAILURE_VALUE that should happen naturally during
  // validation. If not then an IREE_STATUS_DATA_LOSS or IREE_STATUS_ABORTED
  // depending on how fatal such an occurrence is in the implementation.
  // Data-loss usually indicates an abort()-worthy situation where graceful
  // handling is not possible while Aborted indicates that an individual work
  // stream may be invalid but unrelated work streams may still progress.

  (void)semaphore;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "semaphore signal not implemented");

  return status;
}

static void iree_hal_null_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                         iree_status_t status) {
  iree_hal_null_semaphore_t* semaphore =
      iree_hal_null_semaphore_cast(base_semaphore);
  const iree_status_code_t status_code = iree_status_code(status);

  // TODO(null): if the semaphore has already failed and has a status set then
  // `IREE_IGNORE_ERROR(status)` and return without modifying anything. Note
  // that it's possible for fail to be called concurrently from multiple
  // threads.

  // TODO(null): set the value to `IREE_HAL_SEMAPHORE_FAILURE_VALUE` as expected
  // by the API.

  // TODO(null): take ownership of the status (no need to clone, the caller is
  // giving it to us) and keep it until the semaphore is destroyed.

  (void)semaphore;
  (void)status_code;
}

static iree_status_t iree_hal_null_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_null_semaphore_t* semaphore =
      iree_hal_null_semaphore_cast(base_semaphore);

  // TODO(null): if a failure status is set return
  // `iree_status_from_code(IREE_STATUS_ABORTED)`. Avoid a full status as it may
  // capture a backtrace and allocate and callers are expected to follow up a
  // failed wait with a query to get the status.

  // TODO(null): prefer having a fast-path for if the semaphore is
  // known-signaled in user-mode. This can usually avoid syscalls/ioctls and
  // potential context switches in polling cases.

  // TODO(null): check for `iree_timeout_is_immediate(timeout)` and return
  // immediately if the condition is not satisfied before waiting with
  // `iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED)`. Prefer the raw code
  // status instead of a full status object as immediate timeouts are used when
  // polling and a full status may capture a backtrace and allocate.

  (void)semaphore;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "semaphore wait not implemented");

  return status;
}

static const iree_hal_semaphore_vtable_t iree_hal_null_semaphore_vtable = {
    .destroy = iree_hal_null_semaphore_destroy,
    .query = iree_hal_null_semaphore_query,
    .signal = iree_hal_null_semaphore_signal,
    .fail = iree_hal_null_semaphore_fail,
    .wait = iree_hal_null_semaphore_wait,
};
