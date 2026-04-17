// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_host_call.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"

typedef struct iree_hal_amdgpu_host_call_state_t {
  // Resource header so existing reclaim cleanup owns this cold payload.
  iree_hal_resource_t resource;

  // Host allocator used for this state and cloned semaphore-list storage.
  iree_allocator_t host_allocator;

  // Device reported to the host-call callback. Borrowed from the queue.
  iree_hal_device_t* device;

  // Queue affinity reported to the host-call callback.
  iree_hal_queue_affinity_t queue_affinity;

  // User callback and user data captured at queue_host_call submission.
  iree_hal_host_call_t call;

  // User arguments copied at queue_host_call submission.
  uint64_t args[4];

  // Host-call flags captured at queue_host_call submission.
  iree_hal_host_call_flags_t flags;

  // Cloned signal list retained until the reclaim action runs.
  iree_hal_semaphore_list_t signal_semaphore_list;
} iree_hal_amdgpu_host_call_state_t;

static void iree_hal_amdgpu_host_call_state_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amdgpu_host_call_state_t* state =
      (iree_hal_amdgpu_host_call_state_t*)resource;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!iree_hal_semaphore_list_is_empty(state->signal_semaphore_list)) {
    iree_hal_semaphore_list_free(state->signal_semaphore_list,
                                 state->host_allocator);
  }
  iree_allocator_free(state->host_allocator, state);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_resource_vtable_t iree_hal_amdgpu_host_call_state_vtable =
    {
        .destroy = iree_hal_amdgpu_host_call_state_destroy,
};

iree_status_t iree_hal_amdgpu_host_queue_validate_host_call(
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  const iree_hal_host_call_flags_t known_flags =
      IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING |
      IREE_HAL_HOST_CALL_FLAG_WAIT_ACTIVE | IREE_HAL_HOST_CALL_FLAG_RELAXED;
  if (IREE_UNLIKELY(!call.fn)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host_call callback must be non-null");
  }
  if (IREE_UNLIKELY(!args)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host_call args must be non-null");
  }
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~known_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported host_call flags: 0x%" PRIx64, flags);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_call_state_create(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags,
    iree_hal_amdgpu_host_call_state_t** out_state) {
  *out_state = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_host_queue_validate_host_call(call, args, flags));
  iree_hal_amdgpu_host_call_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(queue->host_allocator, sizeof(*state),
                                (void**)&state));
  memset(state, 0, sizeof(*state));
  iree_hal_resource_initialize(&iree_hal_amdgpu_host_call_state_vtable,
                               &state->resource);
  state->host_allocator = queue->host_allocator;
  state->device = queue->logical_device;
  state->queue_affinity = queue->queue_affinity;
  state->call = call;
  memcpy(state->args, args, sizeof(state->args));
  state->flags = flags;

  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, state->host_allocator,
      &state->signal_semaphore_list);
  if (iree_status_is_ok(status)) {
    *out_state = state;
  } else {
    iree_hal_resource_release(&state->resource);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_host_call_fail_with_borrowed_status(
    iree_hal_semaphore_list_t signal_semaphore_list, iree_status_t status) {
  if (signal_semaphore_list.count == 0) {
    return;
  }
  iree_hal_semaphore_list_fail(signal_semaphore_list,
                               iree_status_clone(status));
}

static void iree_hal_amdgpu_host_call_signal_or_fail(
    iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_status_t signal_status =
      iree_hal_semaphore_list_signal(signal_semaphore_list, /*frontier=*/NULL);
  if (!iree_status_is_ok(signal_status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list, signal_status);
  }
}

// Consumes a callback status whose result is intentionally unobservable by the
// host-call API contract. NON_BLOCKING callbacks are fire-and-forget after the
// queue has signaled, and DEFERRED callbacks transfer completion ownership to
// the callback's cloned signal list.
static void iree_hal_amdgpu_host_call_consume_unobservable_status(
    iree_status_t status) {
  (void)iree_status_consume_code(status);
}

static void iree_hal_amdgpu_host_call_execute(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status) {
  (void)entry;
  iree_hal_amdgpu_host_call_state_t* state =
      (iree_hal_amdgpu_host_call_state_t*)user_data;

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_call_fail_with_borrowed_status(
        state->signal_semaphore_list, status);
    return;
  }

  const bool is_nonblocking =
      iree_any_bit_set(state->flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  if (is_nonblocking) {
    iree_status_t signal_status = iree_hal_semaphore_list_signal(
        state->signal_semaphore_list, /*frontier=*/NULL);
    if (!iree_status_is_ok(signal_status)) {
      iree_hal_semaphore_list_fail(state->signal_semaphore_list, signal_status);
      return;
    }
  }

  iree_hal_host_call_context_t context = {
      .device = state->device,
      .queue_affinity = state->queue_affinity,
      .signal_semaphore_list = is_nonblocking ? iree_hal_semaphore_list_empty()
                                              : state->signal_semaphore_list,
  };
  iree_status_t call_status =
      state->call.fn(state->call.user_data, state->args, &context);

  if (is_nonblocking) {
    iree_hal_amdgpu_host_call_consume_unobservable_status(call_status);
  } else if (iree_status_is_deferred(call_status)) {
    iree_hal_amdgpu_host_call_consume_unobservable_status(call_status);
  } else if (iree_status_is_ok(call_status)) {
    iree_hal_amdgpu_host_call_signal_or_fail(state->signal_semaphore_list);
  } else {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, call_status);
  }
}

iree_status_t iree_hal_amdgpu_host_queue_submit_host_call(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags, bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  iree_hal_amdgpu_host_call_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_call_state_create(
      queue, signal_semaphore_list, call, args, flags, &state));

  // Host callbacks observe host memory by default. RELAXED opts out when the
  // callback is known not to consume device-produced host-visible data.
  iree_hal_amdgpu_wait_resolution_t host_call_resolution = *resolution;
  if (!iree_any_bit_set(flags, IREE_HAL_HOST_CALL_FLAG_RELAXED)) {
    host_call_resolution.inline_acquire_scope =
        iree_hal_amdgpu_host_queue_max_fence_scope(
            host_call_resolution.inline_acquire_scope,
            IREE_HSA_FENCE_SCOPE_SYSTEM);
    host_call_resolution.barrier_acquire_scope =
        iree_hal_amdgpu_host_queue_max_fence_scope(
            host_call_resolution.barrier_acquire_scope,
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  }

  iree_hal_resource_t* operation_resources[1] = {
      &state->resource,
  };
  iree_hal_amdgpu_host_queue_barrier_submission_t submission;
  iree_status_t status =
      iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
          queue, &host_call_resolution, iree_hal_semaphore_list_empty(),
          IREE_ARRAYSIZE(operation_resources), out_ready, &submission);
  if (iree_status_is_ok(status) && *out_ready) {
    const uint64_t submission_id =
        iree_hal_amdgpu_host_queue_finish_barrier_submission(
            queue, &host_call_resolution, iree_hal_semaphore_list_empty(),
            (iree_hal_amdgpu_reclaim_action_t){
                .fn = iree_hal_amdgpu_host_call_execute,
                .user_data = state,
            },
            operation_resources, IREE_ARRAYSIZE(operation_resources),
            iree_hal_amdgpu_host_queue_post_commit_callback_null(),
            /*resource_set=*/NULL,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &submission);
    iree_hal_amdgpu_host_queue_record_profile_queue_event(
        queue, &host_call_resolution, signal_semaphore_list,
        &(iree_hal_amdgpu_host_queue_profile_event_info_t){
            .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL,
            .submission_id = submission_id,
            .operation_count = 1,
        });
  }
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_resource_release(&state->resource);
  }
  return status;
}
