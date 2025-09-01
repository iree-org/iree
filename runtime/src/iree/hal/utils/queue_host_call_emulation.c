// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/queue_host_call_emulation.h"

#if IREE_THREADING_ENABLE

#include "iree/base/internal/threading.h"

//===----------------------------------------------------------------------===//
// Emulated Host Call
//===----------------------------------------------------------------------===//

// Issues the host call on the calling thread and signals the semaphore list.
// Returns errors only if signaling fails; user call errors are propagated to
// the semaphore list.
static iree_status_t iree_hal_emulated_host_call_issue(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Non-blocking mode signals the semaphore list first.
  const bool is_nonblocking =
      iree_any_bit_set(flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  if (is_nonblocking) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_list_signal(signal_semaphore_list));
  }

  // Call the user function.
  iree_hal_host_call_context_t context = {
      .device = device,
      .queue_affinity = queue_affinity,
      .signal_semaphore_list = is_nonblocking ? iree_hal_semaphore_list_empty()
                                              : signal_semaphore_list,
  };
  iree_status_t call_status = call.fn(call.user_data, args, &context);

  if (is_nonblocking || iree_status_is_deferred(call_status)) {
    // User callback will signal in the future (or they are fire-and-forget).
  } else if (iree_status_is_ok(call_status)) {
    // Signal callback completed synchronously.
    iree_hal_semaphore_list_signal(signal_semaphore_list);
  } else {
    // If the user function failed we propagate the error to the semaphore list
    // (blocking) or ignore it (non-blocking, where we lost our chance).
    if (!is_nonblocking) {
      iree_hal_semaphore_list_fail(signal_semaphore_list, call_status);
    } else {
      iree_status_ignore(call_status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Heap-allocated state to track a host call that is in-flight.
typedef struct iree_hal_emulated_host_call_state_t {
  // Device the call was scheduled on. Unowned.
  iree_hal_device_t* device;
  // Queue affinity as originally requested.
  // We don't know where we'd actually run so we pass through without
  // modification.
  iree_hal_queue_affinity_t queue_affinity;
  // The transient thread waiting for the wait semaphores and issuing the call.
  iree_thread_t* thread;
  // Target function to call.
  iree_hal_host_call_t call;
  // User arguments.
  uint64_t args[4];
  // Flags controlling call behavior.
  iree_hal_host_call_flags_t flags;
  // Wait semaphores, stored at the end of the state structure.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Signal semaphores, stored at the end of the state structure.
  iree_hal_semaphore_list_t signal_semaphore_list;
} iree_hal_emulated_host_call_state_t;

// Waits, calls, and signals a host call.
// Resources will be released and the state will be deallocated prior to
// returning.
static int iree_hal_emulated_host_call_main(void* entry_arg) {
  iree_hal_emulated_host_call_state_t* state =
      (iree_hal_emulated_host_call_state_t*)entry_arg;

  // Wait for all semaphores to be reached.
  iree_status_t status = iree_hal_semaphore_list_wait(
      state->wait_semaphore_list, iree_infinite_timeout(),
      IREE_HAL_WAIT_FLAG_DEFAULT);

  // Release wait semaphores early.
  iree_hal_semaphore_list_release(state->wait_semaphore_list);

  // If non-blocking then immediately signal the dependencies instead of letting
  // the call do it. If there's dependent work in the queue it should be able to
  // progress after this point regardless of how long the host call takes.
  const bool is_nonblocking =
      iree_any_bit_set(state->flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  if (is_nonblocking) {
    // NOTE: the signals can fail in which case we never perform the call.
    // That's ok as failure to signal is considered a device-loss/death
    // situation as there's no telling what has gone wrong.
    status = iree_hal_semaphore_list_signal(state->signal_semaphore_list);
  }

  // Issue the call.
  if (iree_status_is_ok(status)) {
    status = iree_hal_emulated_host_call_issue(
        state->device, state->queue_affinity,
        is_nonblocking ? iree_hal_semaphore_list_empty()
                       : state->signal_semaphore_list,
        state->call, state->args, state->flags);
  }

  // If anything (wait, call, or signal) failed we need to fail all dependent
  // semaphores to propagate the error.
  if (!iree_status_is_ok(status)) {
    // Transfers status ownership.
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
    status = iree_status_from_code(IREE_STATUS_INTERNAL);
  }
  // NOTE: status is invalid here as we've transferred ownership to the
  // semaphore list via iree_hal_semaphore_list_fail.

  // Release signal semaphores.
  iree_hal_semaphore_list_release(state->signal_semaphore_list);

  // Deallocate state (note that we must take the thread handle locally).
  iree_allocator_t host_allocator =
      iree_hal_device_host_allocator(state->device);
  iree_thread_t* thread = state->thread;
  iree_allocator_free(host_allocator, state);

  // Release the thread and return.
  iree_thread_release(thread);
  return 0;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_emulated_host_call(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // If there are no wait semaphores we can immediately issue the call from the
  // calling thread. We still honor the non-blocking flag by signaling early
  // when set.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list)) {
    iree_status_t status = iree_hal_emulated_host_call_issue(
        device, queue_affinity, signal_semaphore_list, call, args, flags);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Allocate state structure for tracking the host call and waiter thread.
  // We embed all parameters in the state structure to avoid extra allocations.
  iree_hal_emulated_host_call_state_t* state = NULL;
  const iree_host_size_t semaphore_list_size = iree_host_align(
      (wait_semaphore_list.count + signal_semaphore_list.count) *
          sizeof(iree_hal_semaphore_t*),
      iree_max_align_t);
  const iree_host_size_t payload_list_size = iree_host_align(
      (wait_semaphore_list.count + signal_semaphore_list.count) *
          sizeof(uint64_t),
      iree_max_align_t);
  const iree_host_size_t total_length =
      iree_host_align(sizeof(*state), iree_max_align_t) + semaphore_list_size +
      payload_list_size;
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(device);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_length, (void**)&state));

  state->device = device;
  state->queue_affinity = queue_affinity;
  state->call = call;
  memcpy(state->args, args, sizeof(state->args));
  state->flags = flags;

  uint8_t* state_ptr =
      (uint8_t*)state + iree_host_align(sizeof(*state), iree_max_align_t);
  iree_hal_semaphore_t** semaphore_list_ptr = (iree_hal_semaphore_t**)state_ptr;
  state_ptr += semaphore_list_size;
  uint64_t* payload_list_ptr = (uint64_t*)state_ptr;
  state_ptr += payload_list_size;

  state->wait_semaphore_list.count = wait_semaphore_list.count;
  state->wait_semaphore_list.semaphores = semaphore_list_ptr;
  state->wait_semaphore_list.payload_values = payload_list_ptr;
  memcpy(state->wait_semaphore_list.semaphores, wait_semaphore_list.semaphores,
         wait_semaphore_list.count * sizeof(*semaphore_list_ptr));
  memcpy(state->wait_semaphore_list.payload_values,
         wait_semaphore_list.payload_values,
         wait_semaphore_list.count * sizeof(*payload_list_ptr));
  iree_hal_semaphore_list_retain(state->wait_semaphore_list);

  state->signal_semaphore_list.count = signal_semaphore_list.count;
  state->signal_semaphore_list.semaphores =
      semaphore_list_ptr + wait_semaphore_list.count;
  state->signal_semaphore_list.payload_values =
      payload_list_ptr + wait_semaphore_list.count;
  memcpy(state->signal_semaphore_list.semaphores,
         signal_semaphore_list.semaphores,
         signal_semaphore_list.count * sizeof(*semaphore_list_ptr));
  memcpy(state->signal_semaphore_list.payload_values,
         signal_semaphore_list.payload_values,
         signal_semaphore_list.count * sizeof(*payload_list_ptr));
  iree_hal_semaphore_list_retain(state->signal_semaphore_list);

  // Launch the thread to perform the wait.
  const iree_thread_create_params_t thread_params = {
      .name = iree_make_cstring_view("iree-hal-host-call"),
      .stack_size = 0,  // default
      .create_suspended = false,
      .priority_class = IREE_THREAD_PRIORITY_CLASS_HIGH,
  };
  iree_status_t status =
      iree_thread_create(iree_hal_emulated_host_call_main, state, thread_params,
                         host_allocator, &state->thread);

  // NOTE: if thread creation fails we never enqueued the waits and thus can
  // treat the failure like a failure to enqueue. We need to clean up the state
  // but do not need to signal dependencies as failures.
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_release(state->wait_semaphore_list);
    iree_hal_semaphore_list_release(state->signal_semaphore_list);
    iree_allocator_free(host_allocator, state);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // IREE_THREADING_ENABLE
