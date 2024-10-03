// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/semaphore.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_internal_semaphore_t
//===----------------------------------------------------------------------===//

static const iree_hal_semaphore_vtable_t
    iree_hal_amdgpu_internal_semaphore_vtable;

static iree_hal_amdgpu_internal_semaphore_t*
iree_hal_amdgpu_internal_semaphore_cast(iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_internal_semaphore_vtable);
  return (iree_hal_amdgpu_internal_semaphore_t*)base_value;
}

iree_status_t iree_hal_amdgpu_internal_semaphore_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_hal_semaphore_flags_t flags,
    iree_hal_amdgpu_device_semaphore_t* device_semaphore,
    iree_hal_amdgpu_internal_semaphore_release_callback_t release_callback,
    iree_hal_amdgpu_internal_semaphore_t* out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_semaphore, 0, sizeof(*out_semaphore));

  // Create the HSA signal.
  // HSA has two signal types: default and interrupt. A default signal is like a
  // futex and is relatively light-weight but the host can only busy-wait on it.
  // An interrupt signal involves the OS but allows for platform-level waits.
  //
  // TODO(benvanik): add a semaphore flag for device-only? It's hard to know
  // that in all cases but in the compiler we could do it for our locally-scoped
  // ones. We aggressively pool semaphores and don't track if there's host
  // waiters so for today we just take the hit and always use interrupt signals.
  // If we wanted device-only we'd set the HSA_AMD_SIGNAL_AMD_GPU_ONLY flag.
  uint64_t signal_flags = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_signal_create(
              libhsa, /*initial_value=*/0ull, /*num_consumers=*/0,
              /*consumers=*/NULL, signal_flags, &out_semaphore->signal));

  iree_hal_resource_initialize(&iree_hal_amdgpu_internal_semaphore_vtable,
                               &out_semaphore->resource);
  // Pooling behavior: maintain a 0 ref count until acquired.
  iree_atomic_ref_count_dec(&out_semaphore->resource.ref_count);
  out_semaphore->libhsa = libhsa;
  out_semaphore->flags = flags;
  out_semaphore->device_semaphore = device_semaphore;
  out_semaphore->release_callback = release_callback;

  // NOTE: today we assume the semaphore device memory is host-accessible. In
  // the future we may make device-only semaphores and would need to do a
  // host-to-device transfer to update the device semaphore values.
  out_semaphore->device_semaphore->signal = out_semaphore->signal;

  iree_hal_amdgpu_device_mutex_initialize(&device_semaphore->mutex);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_internal_semaphore_deinitialize(
    iree_hal_amdgpu_internal_semaphore_t* semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_IGNORE_ERROR(
      iree_hsa_signal_destroy(semaphore->libhsa, semaphore->signal));

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_internal_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_device_mutex_lock(&semaphore->device_semaphore->mutex);
  IREE_ASSERT_EQ(semaphore->device_semaphore->wake_list_head, NULL);
  IREE_ASSERT_EQ(semaphore->device_semaphore->wake_list_tail, NULL);
  iree_hal_amdgpu_device_mutex_unlock(&semaphore->device_semaphore->mutex);

  if (semaphore->release_callback.fn) {
    semaphore->release_callback.fn(semaphore->release_callback.user_data,
                                   semaphore);
  }

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_internal_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is(semaphore,
                              &iree_hal_amdgpu_internal_semaphore_vtable);
}

void iree_hal_amdgpu_internal_semaphore_reset(
    iree_hal_amdgpu_internal_semaphore_t* semaphore, uint64_t initial_value) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reset HSA signal value (just an atomic store).
  iree_hsa_signal_silent_store_relaxed(
      semaphore->libhsa, semaphore->device_semaphore->signal, initial_value);

  iree_hal_amdgpu_device_mutex_lock(&semaphore->device_semaphore->mutex);

  semaphore->device_semaphore->last_value = initial_value;

  IREE_ASSERT_EQ(semaphore->device_semaphore->wake_list_head, NULL);
  IREE_ASSERT_EQ(semaphore->device_semaphore->wake_list_tail, NULL);

  iree_hal_amdgpu_device_mutex_unlock(&semaphore->device_semaphore->mutex);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_internal_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  *out_value = 0;
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);

  // TODO(benvanik): return the current value of the semaphore by (depending on
  // the implementation) making a syscall to get it. It's expected that the
  // value may immediately change after being queried here.

  // TODO(benvanik): if the value is IREE_HAL_SEMAPHORE_FAILURE_VALUE then
  // return the failure status cached from the fail call by cloning it (like
  // `return iree_status_clone(semaphore->failure_status)`).

  (void)semaphore;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "semaphore query not implemented");

  return status;
}

static iree_status_t iree_hal_amdgpu_internal_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);

  // TODO(benvanik): validation is optional but encouraged if cheap: semaphores
  // must always be signaled to a value that is greater than the previous value
  // (not less-than-or-equal).

  // TODO(benvanik): signals when the semaphore have failed should also fail and
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

static void iree_hal_amdgpu_internal_semaphore_fail(
    iree_hal_semaphore_t* base_semaphore, iree_status_t status) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);
  const iree_status_code_t status_code = iree_status_code(status);

  // TODO(benvanik): if the semaphore has already failed and has a status set
  // then `IREE_IGNORE_ERROR(status)` and return without modifying anything.
  // Note that it's possible for fail to be called concurrently from multiple
  // threads.

  // TODO(benvanik): set the value to `IREE_HAL_SEMAPHORE_FAILURE_VALUE` as
  // expected by the API.

  // TODO(benvanik): take ownership of the status (no need to clone, the caller
  // is giving it to us) and keep it until the semaphore is destroyed.

  (void)semaphore;
  (void)status_code;
}

static iree_status_t iree_hal_amdgpu_internal_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);

  // TODO(benvanik): if a failure status is set return
  // `iree_status_from_code(IREE_STATUS_ABORTED)`. Avoid a full status as it may
  // capture a backtrace and allocate and callers are expected to follow up a
  // failed wait with a query to get the status.

  // TODO(benvanik): prefer having a fast-path for if the semaphore is
  // known-signaled in user-mode. This can usually avoid syscalls/ioctls and
  // potential context switches in polling cases.

  // TODO(benvanik): check for `iree_timeout_is_immediate(timeout)` and return
  // immediately if the condition is not satisfied before waiting with
  // `iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED)`. Prefer the raw code
  // status instead of a full status object as immediate timeouts are used when
  // polling and a full status may capture a backtrace and allocate.

  (void)semaphore;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "semaphore wait not implemented");

  return status;
}

static const iree_hal_semaphore_vtable_t
    iree_hal_amdgpu_internal_semaphore_vtable = {
        .destroy = iree_hal_amdgpu_internal_semaphore_destroy,
        .query = iree_hal_amdgpu_internal_semaphore_query,
        .signal = iree_hal_amdgpu_internal_semaphore_signal,
        .fail = iree_hal_amdgpu_internal_semaphore_fail,
        .wait = iree_hal_amdgpu_internal_semaphore_wait,
};
