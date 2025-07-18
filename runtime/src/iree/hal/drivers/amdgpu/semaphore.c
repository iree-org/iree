// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/semaphore.h"

#include "iree/hal/drivers/amdgpu/device/semaphore.h"

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
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_semaphore_options_t options,
    iree_hal_semaphore_flags_t flags,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_t* device_semaphore,
    iree_hal_amdgpu_internal_semaphore_release_callback_t release_callback,
    iree_hal_amdgpu_internal_semaphore_t* out_semaphore) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(device_semaphore);
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
              IREE_LIBHSA(libhsa), /*initial_value=*/0ull,
              /*num_consumers=*/0,
              /*consumers=*/NULL, signal_flags, &out_semaphore->signal));

  iree_hal_resource_initialize(&iree_hal_amdgpu_internal_semaphore_vtable,
                               &out_semaphore->resource);
  // Pooling behavior: maintain a 0 ref count until acquired.
  iree_atomic_ref_count_dec(&out_semaphore->resource.ref_count);
  out_semaphore->libhsa = libhsa;
  out_semaphore->options = options;
  out_semaphore->flags = flags;
  out_semaphore->device_semaphore = device_semaphore;
  out_semaphore->release_callback = release_callback;

  // NOTE: today we assume the semaphore device memory is host-accessible. In
  // the future we may make device-only semaphores and would need to do a
  // host-to-device transfer to update the device semaphore values.
  memset(device_semaphore, 0, sizeof(*device_semaphore));
  device_semaphore->signal = (iree_amd_signal_t*)out_semaphore->signal.handle;
  device_semaphore->host_semaphore = (uint64_t)out_semaphore;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_internal_semaphore_deinitialize(
    iree_hal_amdgpu_internal_semaphore_t* semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdgpu_libhsa_t* libhsa = semaphore->libhsa;

  IREE_IGNORE_ERROR(
      iree_hsa_signal_destroy(IREE_LIBHSA(libhsa), semaphore->signal));

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_internal_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  // If the semaphore failed we need to free the status object, if any.
  // The signal will be reset to a new initial value if it is reused.
  const hsa_signal_value_t old_value = iree_hsa_signal_exchange_scacquire(
      IREE_LIBHSA(semaphore->libhsa), semaphore->signal, 0);
  iree_hal_semaphore_failure_free((uint64_t)old_value);

  // Use the provided release callback to free or recycle the semaphore.
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
  // Reset the HSA signal value to the user-provided initial value.
  // Note that this is just a store here as we've already cleared any status
  // that may have been embedded in the value prior to it being returned to the
  // pool. We do a silent store here as no one should be waiting on the signal
  // and they don't need to be notified.
  //
  // NOTE: ROCR implements the silent calls by just routing to the normal ones
  // so this isn't actually silent. Darn.
  // https://github.com/ROCm/ROCR-Runtime/issues/316
  iree_hsa_signal_silent_store_screlease(IREE_LIBHSA(semaphore->libhsa),
                                         semaphore->signal, initial_value);
}

static iree_status_t iree_hal_amdgpu_internal_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);
  *out_value = 0;

  // Fast path for the common case of the semaphore being in a valid state.
  hsa_signal_value_t current_value = iree_hsa_signal_load_scacquire(
      IREE_LIBHSA(semaphore->libhsa), semaphore->signal);
  if (IREE_LIKELY(current_value < IREE_HAL_SEMAPHORE_FAILURE_VALUE)) {
    *out_value = current_value;
    return iree_ok_status();
  }

  // If the semaphore failed then interpret the failure as an IREE status
  // object. The semaphore retains the status until it is deinitialized and we
  // return a clone per caller.
  *out_value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  return iree_hal_semaphore_failure_as_status(current_value);
}

static iree_status_t iree_hal_amdgpu_internal_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);

  // Check that we are incrementing the value. This also handles cases where the
  // signal has failed as then the current value will always be larger than
  // whatever value we are setting it to.
  hsa_signal_value_t current_value = iree_hsa_signal_load_relaxed(
      IREE_LIBHSA(semaphore->libhsa), semaphore->signal);
  while (current_value != new_value) {
    if (new_value < current_value) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "semaphore signal requested to an older value; "
          "semaphores must be monotonically increasing (previous=%" PRIu64
          ", new=%" PRIu64 ")",
          current_value, new_value);
    }
    // We update the signal to the new value (and notify host waiters) with a
    // CAS. Immediately upon store some host thread or device agent may
    // immediately wake and process whatever data is being signaled as
    // available. If someone else came in and updated the value before us the
    // CAS will fail and we'll try again (unless doing so would be invalid).
    const hsa_signal_value_t observed_value = iree_hsa_signal_cas_scacq_screl(
        IREE_LIBHSA(semaphore->libhsa), semaphore->signal, current_value,
        (hsa_signal_value_t)new_value);
    if (observed_value == current_value) {
      // Swap took place.
      break;
    }
    current_value = observed_value;  // try again
  }

  // TODO(benvanik): update device-side semaphore entry and wake any schedulers
  // registered with it.

  return iree_ok_status();
}

static void iree_hal_amdgpu_internal_semaphore_fail(
    iree_hal_semaphore_t* base_semaphore, iree_status_t status) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);

  // Encode the status in a signal value.
  hsa_signal_value_t new_value = iree_hal_status_as_semaphore_failure(status);

  // Check to see if the semaphore has failed before assigning failure.
  // We do this in a loop to retry in races between when we check and when we
  // update to the failed value.
  hsa_signal_value_t current_value = iree_hsa_signal_load_scacquire(
      IREE_LIBHSA(semaphore->libhsa), semaphore->signal);
  while (current_value != new_value) {
    if (current_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      // Already failed. Ignore the new error.
      IREE_IGNORE_ERROR(status);
      return;
    }
    // Try to swap. If this succeeds and current_value == new_value then we've
    // either transferred ownership of the status to the signal or it was
    // already set to the same exact failure by someone else. Since statuses are
    // either codes (which have a chance of collision) or uniquely allocated
    // pointers there's no real risk of leaking.
    const hsa_signal_value_t observed_value = iree_hsa_signal_cas_scacq_screl(
        IREE_LIBHSA(semaphore->libhsa), semaphore->signal, current_value,
        new_value);
    if (observed_value == current_value) {
      // Swap took place.
      break;
    }
    current_value = observed_value;  // try again
  }
}

static iree_status_t iree_hal_amdgpu_internal_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);
  iree_hal_semaphore_list_t semaphore_list = {
      .count = 1,
      .semaphores = &base_semaphore,
      .payload_values = &value,
  };
  return iree_hal_amdgpu_wait_semaphores(semaphore->libhsa, semaphore->options,
                                         IREE_HAL_WAIT_MODE_ALL, semaphore_list,
                                         timeout);
}

static const iree_hal_semaphore_vtable_t
    iree_hal_amdgpu_internal_semaphore_vtable = {
        .destroy = iree_hal_amdgpu_internal_semaphore_destroy,
        .query = iree_hal_amdgpu_internal_semaphore_query,
        .signal = iree_hal_amdgpu_internal_semaphore_signal,
        .fail = iree_hal_amdgpu_internal_semaphore_fail,
        .wait = iree_hal_amdgpu_internal_semaphore_wait,
};

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_external_semaphore_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): external imported semaphore wrapper.

//===----------------------------------------------------------------------===//
// Semaphore Operations
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_t** out_handle) {
  if (iree_hal_amdgpu_internal_semaphore_isa(base_semaphore)) {
    iree_hal_amdgpu_internal_semaphore_t* semaphore =
        (iree_hal_amdgpu_internal_semaphore_t*)base_semaphore;
    *out_handle = semaphore->device_semaphore;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "semaphore is not from the AMDGPU backend and has no "
                          "corresponding device handle");
}

iree_status_t iree_hal_amdgpu_semaphore_hsa_signal(
    iree_hal_semaphore_t* base_semaphore, hsa_signal_t* out_signal) {
  if (iree_hal_amdgpu_internal_semaphore_isa(base_semaphore)) {
    iree_hal_amdgpu_internal_semaphore_t* semaphore =
        (iree_hal_amdgpu_internal_semaphore_t*)base_semaphore;
    *out_signal = semaphore->signal;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "semaphore is not from the AMDGPU backend and has no "
                          "corresponding HSA signal");
}

iree_status_t iree_hal_amdgpu_poll_semaphore(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_current_value) {
  if (iree_hal_amdgpu_internal_semaphore_isa(base_semaphore)) {
    iree_hal_amdgpu_internal_semaphore_t* semaphore =
        (iree_hal_amdgpu_internal_semaphore_t*)base_semaphore;
    hsa_signal_value_t current_value = iree_hsa_signal_load_scacquire(
        IREE_LIBHSA(semaphore->libhsa), semaphore->signal);
    if (IREE_UNLIKELY(current_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE)) {
      // If the semaphore failed then interpret the failure as an IREE status
      // object and clone it for the caller.
      return iree_hal_semaphore_failure_as_status(current_value);
    }
    *out_current_value = (uint64_t)current_value;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "only AMDGPU semaphores are supported; a fallback path for mixed "
      "semaphores would be needed for polling");
}

iree_status_t iree_hal_amdgpu_poll_semaphores(
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list) {
  // Poll every semaphore and check the >= condition.
  // In wait-any mode the first satisfied condition will return OK.
  // In wait-all mode the first unsatisfied condition will return
  // DEADLINE_EXCEEDED.
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    uint64_t current_value = 0ull;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_poll_semaphore(
        semaphore_list.semaphores[i], &current_value));
    if (current_value >= semaphore_list.payload_values[i]) {
      // Satisfied.
      if (wait_mode == IREE_HAL_WAIT_MODE_ANY) {
        // Only one semaphore needs to be reached in wait-any mode.
        return iree_ok_status();
      }
    } else {
      // Unsatisfied.
      if (wait_mode == IREE_HAL_WAIT_MODE_ALL) {
        // All semaphores need ot be reached in wait-all mode.
        return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
    }
  }
  // In wait-any mode if none were satisfied then return DEADLINE_EXCEEDED.
  // In wait-all mode if none were unsatisfied then return OK.
  return wait_mode == IREE_HAL_WAIT_MODE_ANY
             ? iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED)
             : iree_ok_status();
}

// The ROCR implementation of wait multiple is really bad/slow. We should really
// get that rewritten if we want to continue using it. We could easily go to
// hsaKmtWaitOnMultipleEvents_Ext ourselves but the special wait flag handling
// in core::Signal is something we can't directly touch. I'm not sure we
// actually need that, though, given that we have no way of keeping that in sync
// with device-side waits.
iree_status_t iree_hal_amdgpu_wait_semaphores(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_semaphore_options_t options, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(libhsa);
  if (semaphore_list.count == 0) return iree_ok_status();  // no-op
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, iree_timeout_as_duration_ns(timeout));

  // Fast-path for immediate timeouts using this API to poll.
  if (iree_timeout_is_immediate(timeout)) {
    iree_status_t poll_status =
        iree_hal_amdgpu_poll_semaphores(wait_mode, semaphore_list);
    IREE_TRACE_ZONE_END(z0);
    return poll_status;
  }

  // TODO(benvanik): use options.wait_active_for_ns to spin locally before we
  // call into ROCR (which is significantly more expensive).
  const hsa_wait_state_t wait_state =
      options.wait_active_for_ns == IREE_DURATION_INFINITE
          ? HSA_WAIT_STATE_ACTIVE
          : HSA_WAIT_STATE_BLOCKED;
  const iree_duration_t timeout_duration_ns =
      iree_timeout_is_infinite(timeout) ? UINT64_MAX
                                        : iree_timeout_as_duration_ns(timeout);

  // Fast-path for single semaphore waits.
  // ROCR's multi-wait is inefficient and we really want to avoid it if
  // possible.
  if (semaphore_list.count == 1) {
    hsa_signal_t signal;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdgpu_semaphore_hsa_signal(semaphore_list.semaphores[0],
                                             &signal),
        "retrieving HSA signal from semaphore");
    hsa_signal_value_t expected_value = semaphore_list.payload_values[0];
    iree_status_t status = iree_ok_status();
    hsa_signal_value_t current_value = iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(libhsa), signal, HSA_SIGNAL_CONDITION_GTE, expected_value,
        timeout_duration_ns, wait_state);
    if (IREE_UNLIKELY(current_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE)) {
      // If the semaphore failed then interpret the failure as an IREE status
      // object and clone it for the caller.
      status = iree_hal_semaphore_failure_as_status(current_value);
    } else if (current_value < expected_value) {
      // Assume timeout. It may be a spurious wake and we should try again until
      // the timeout duration has been reached.
      // TODO(benvanik): retry while timeout remaining.
      status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Build array-of-structs for the individual wait operations.
  hsa_signal_t* signals =
      iree_alloca(semaphore_list.count * sizeof(hsa_signal_t));
  hsa_signal_condition_t* conds =
      iree_alloca(semaphore_list.count * sizeof(hsa_signal_condition_t));
  hsa_signal_value_t* values =
      iree_alloca(semaphore_list.count * sizeof(hsa_signal_value_t));
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdgpu_semaphore_hsa_signal(semaphore_list.semaphores[i],
                                             &signals[i]),
        "retrieving HSA signal from semaphore");
    conds[i] = HSA_SIGNAL_CONDITION_GTE;
    values[i] = semaphore_list.payload_values[i];
  }

  // NOTE: hsa_amd_signal_wait_all/hsa_amd_signal_wait_any has relaxed memory
  // semantics and to have the proper acquire behavior we need to load the
  // signal value ourselves.
  iree_status_t status = iree_ok_status();
  switch (wait_mode) {
    case IREE_HAL_WAIT_MODE_ALL: {
      const uint32_t wait_result = iree_hsa_amd_signal_wait_all(
          IREE_LIBHSA(libhsa), semaphore_list.count, signals, conds, values,
          timeout_duration_ns, wait_state, /*satisfying_values=*/NULL);
      if (wait_result == 0) {
        // If the wait succeeded then check for errors.
        // This also issues an acquire fence on every semaphore.
        status = iree_hal_amdgpu_poll_semaphores(wait_mode, semaphore_list);
      } else {
        status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
    } break;
    case IREE_HAL_WAIT_MODE_ANY: {
      hsa_signal_value_t satisfying_value = 0;
      const uint32_t satisfying_index = iree_hsa_amd_signal_wait_any(
          IREE_LIBHSA(libhsa), semaphore_list.count, signals, conds, values,
          timeout_duration_ns, wait_state, &satisfying_value);
      if (satisfying_index != UINT32_MAX) {
        // Issue an acquire fence on the satisfying semaphore. This will
        // propagate errors if the wait succeeded because the semaphore was
        // signaled to a failure value. We could reuse the satisfying_value
        // above but we'd still need the acquire fence.
        //
        // Note that more than one semaphore make have had its condition
        // satisfied and more than one may be in a failure state; this API
        // doesn't exhaustively check.
        uint64_t current_value = 0ull;
        status = iree_hal_amdgpu_poll_semaphore(
            semaphore_list.semaphores[satisfying_index], &current_value);
      } else {
        status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
    } break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown wait mode %d", (int)wait_mode);
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
