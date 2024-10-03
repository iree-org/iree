// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/semaphore.h"

#include "iree/hal/drivers/amdgpu/device/semaphore.h"
#include "iree/hal/drivers/amdgpu/system.h"

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
    iree_hal_amdgpu_system_t* system, iree_hal_semaphore_flags_t flags,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_t* device_semaphore,
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
              IREE_LIBHSA(&system->libhsa), /*initial_value=*/0ull,
              /*num_consumers=*/0,
              /*consumers=*/NULL, signal_flags, &out_semaphore->signal));

  iree_hal_resource_initialize(&iree_hal_amdgpu_internal_semaphore_vtable,
                               &out_semaphore->resource);
  // Pooling behavior: maintain a 0 ref count until acquired.
  iree_atomic_ref_count_dec(&out_semaphore->resource.ref_count);
  out_semaphore->system = system;
  out_semaphore->flags = flags;
  out_semaphore->device_semaphore = device_semaphore;
  out_semaphore->release_callback = release_callback;

  // NOTE: today we assume the semaphore device memory is host-accessible. In
  // the future we may make device-only semaphores and would need to do a
  // host-to-device transfer to update the device semaphore values.
  memset(device_semaphore, 0, sizeof(*device_semaphore));
  device_semaphore->host_semaphore = (uint64_t)out_semaphore;
  device_semaphore->signal = out_semaphore->signal;

  iree_hal_amdgpu_device_mutex_initialize(&device_semaphore->mutex);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_internal_semaphore_deinitialize(
    iree_hal_amdgpu_internal_semaphore_t* semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdgpu_libhsa_t* libhsa = &semaphore->system->libhsa;

  IREE_IGNORE_ERROR(
      iree_hsa_signal_destroy(IREE_LIBHSA(libhsa), semaphore->signal));

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_internal_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_amdgpu_internal_semaphore_t* semaphore =
      iree_hal_amdgpu_internal_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdgpu_libhsa_t* libhsa = &semaphore->system->libhsa;

#if !defined(NDEBUG)
  // Device should have cleaned up the wake list. If not we may need to do so
  // here (as we'd still be linked in it). That's hard.
  iree_hal_amdgpu_device_mutex_lock(&semaphore->device_semaphore->mutex);
  IREE_ASSERT_EQ(semaphore->device_semaphore->wake_list_head, NULL);
  IREE_ASSERT_EQ(semaphore->device_semaphore->wake_list_tail, NULL);
  iree_hal_amdgpu_device_mutex_unlock(&semaphore->device_semaphore->mutex);
#endif  // !NDEBUG

  // If the semaphore failed we need to free the status object, if any.
  // The signal will be reset to a new initial value if it is reused.
  const hsa_signal_value_t old_value = iree_hsa_signal_exchange_relaxed(
      IREE_LIBHSA(libhsa), semaphore->signal, 0);
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
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdgpu_libhsa_t* libhsa = &semaphore->system->libhsa;

  // Reset the HSA signal value to the user-provided initial value.
  // Note that this is just a store here as we've already cleared any status
  // that may have been embedded in the value prior to it being returned to the
  // pool. We do a silent store here as no one should be waiting on the signal
  // and they don't need to be notified.
  iree_hsa_signal_silent_store_relaxed(IREE_LIBHSA(libhsa), semaphore->signal,
                                       initial_value);

  // NOTE: this is doing a write into device memory. The lock makes this
  // expensive as we must read the mutex state. Ideally we'd use an atomic
  // instead such that we could be write-only.
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

  // Fast path for the common case of the semaphore being in a valid state.
  uint64_t value = 0ull;
  iree_status_t status =
      iree_hal_amdgpu_poll_semaphore(base_semaphore, out_value);
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    *out_value = value;
    return status;
  }

  // If the semaphore failed then interpret the failure as an IREE status
  // object. The semaphore retains the status until it is deinitialized and we
  // return a clone per caller.
  *out_value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  iree_status_t failure_status = iree_hal_semaphore_failure_as_status(value);
  return iree_status_clone(failure_status);
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

  // DO NOT SUBMIT
  // need to process wake list

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
  iree_hal_semaphore_list_t semaphore_list = {
      .count = 1,
      .semaphores = &base_semaphore,
      .payload_values = &value,
  };
  return iree_hal_amdgpu_wait_semaphores(
      semaphore->system, IREE_HAL_WAIT_MODE_ALL, semaphore_list, timeout);
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

iree_status_t iree_hal_amdgpu_external_semaphore_notify(
    iree_hal_amdgpu_external_semaphore_t* semaphore, uint64_t payload) {
  // DO NOT SUBMIT iree_hal_amdgpu_external_semaphore_notify
  // from the device
  // signal platform handle
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

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

iree_status_t iree_hal_amdgpu_poll_semaphore(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_current_value) {
  if (iree_hal_amdgpu_internal_semaphore_isa(base_semaphore)) {
    iree_hal_amdgpu_internal_semaphore_t* semaphore =
        (iree_hal_amdgpu_internal_semaphore_t*)base_semaphore;
    hsa_signal_value_t current_value = iree_hsa_signal_load_relaxed(
        IREE_LIBHSA(&semaphore->system->libhsa), semaphore->signal);
    if (current_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      return iree_status_from_code(IREE_STATUS_ABORTED);
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

iree_status_t iree_hal_amdgpu_wait_semaphores(
    iree_hal_amdgpu_system_t* system, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(system);
  if (semaphore_list.count == 0) return iree_ok_status();  // no-op
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, iree_timeout_as_duration_ns(timeout));

  // Fast-path for immediate timeouts using this API to poll.
  if (iree_timeout_is_immediate(timeout)) {
    iree_status_t poll_status =
        iree_hal_amdgpu_poll_semaphores(wait_mode, semaphore_list);
    if (iree_status_is_ok(poll_status)) {
      // Acquire fence if the wait was successful. If unsuccessful we don't
      // care as state is undefined. In the full HSA wait path this happens
      // inside the HSA implementation.
      iree_atomic_thread_fence(iree_memory_order_acquire);
    }
    IREE_TRACE_ZONE_END(z0);
    return poll_status;
  }

  // Convert the timeout to a relative tick count in the system timestamp
  // frequency.
  uint64_t timeout_duration_ticks = 0;
  if (iree_timeout_is_infinite(timeout)) {
    timeout_duration_ticks = UINT64_MAX;
  } else {
    const iree_duration_t timeout_duration_ns =
        iree_timeout_as_duration_ns(timeout);
    timeout_duration_ticks =
        timeout_duration_ns / system->info.timestamp_frequency;
  }

  const iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;

  (void)libhsa;
  (void)timeout_duration_ticks;
  // DO NOT SUBMIT iree_hal_amdgpu_wait_semaphores
  // system->options.wait_active_for_ns;
  // wait with ACTIVE up to this time
  // wait with BLOCKED for the rest (subtract from duration)

  iree_status_t status = iree_ok_status();
  switch (wait_mode) {
    case IREE_HAL_WAIT_MODE_ALL: {
      // TODO(benvanik): use hsa_amd_signal_wait_all when landed:
      // https://github.com/ROCm/ROCR-Runtime/issues/241
      // For now we have to wait each one at a time which requires at least O(n)
      // syscalls.
      for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
        // DO NOT SUBMIT unwrap to signal list helper
        if (!iree_hal_amdgpu_internal_semaphore_isa(
                semaphore_list.semaphores[i])) {
          status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                    "unhandled semaphore type for host waits");
          break;
        }
        hsa_signal_t signal = iree_hal_amdgpu_internal_semaphore_cast(
                                  semaphore_list.semaphores[i])
                                  ->signal;

        // NOTE: 0 handles are ignored.
        // TODO(benvanik): https://github.com/ROCm/ROCR-Runtime/issues/252
        // tracks making this a supported behavior in the HSA API.
        if (!signal.handle) continue;

        iree_hsa_signal_wait_scacquire(IREE_LIBHSA(libhsa), signal,
                                       HSA_SIGNAL_CONDITION_GTE,
                                       semaphore_list.payload_values[i],
                                       UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
      }
    } break;
    case IREE_HAL_WAIT_MODE_ANY: {
      // Unfortunately hsa_amd_signal_wait_any does not allow 0 signal handles
      // but AQL does: in an AQL packet processor the 0 signals are no-ops.
      //
      // TODO(benvanik): https://github.com/ROCm/ROCR-Runtime/issues/252 tracks
      // making this a supported behavior in the HSA API.
      // DO NOT SUBMIT allocate conds/values
      hsa_signal_t* signals =
          iree_alloca(semaphore_list.count * sizeof(hsa_signal_t));
      hsa_signal_condition_t* conds =
          iree_alloca(semaphore_list.count * sizeof(hsa_signal_condition_t));
      hsa_signal_value_t* values =
          iree_alloca(semaphore_list.count * sizeof(hsa_signal_value_t));
      for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
        // DO NOT SUBMIT unwrap to signal list helper
        if (!iree_hal_amdgpu_internal_semaphore_isa(
                semaphore_list.semaphores[i])) {
          status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                    "unhandled semaphore type for host waits");
          break;
        }
        hsa_signal_t signal = iree_hal_amdgpu_internal_semaphore_cast(
                                  semaphore_list.semaphores[i])
                                  ->signal;
        // NOTE: 0 handles are ignored.
        if (!signal.handle) break;
        signals[i] = signal;
        conds[i] = HSA_SIGNAL_CONDITION_GTE;
        values[i] = semaphore_list.payload_values[i];
      }

      // NOTE: this will wake if the signal ever passes through 0 - it's
      // possible for it to be non-zero upon return if something else modifies
      // it (but we should never be doing that).
      //
      // NOTE: hsa_amd_signal_wait_any has relaxed memory semantics and to have
      // the proper acquire behavior we need to load the signal value ourselves.
      hsa_signal_value_t satisfying_value = 0;
      const uint32_t satisfying_index = iree_hsa_amd_signal_wait_any(
          IREE_LIBHSA(libhsa), semaphore_list.count, signals, conds, values,
          UINT64_MAX, HSA_WAIT_STATE_BLOCKED, &satisfying_value);
      if (satisfying_index != UINT32_MAX) {
        iree_hsa_signal_load_scacquire(IREE_LIBHSA(libhsa),
                                       signals[satisfying_index]);
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
