// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_worker.h"

#include "iree/hal/drivers/amdgpu/device/host.h"
#include "iree/hal/drivers/amdgpu/queue.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/trace_buffer.h"

//===----------------------------------------------------------------------===//
// Platform Support
//===----------------------------------------------------------------------===//

// Returns the IREE thread affinity that specifies one or more threads
// associated with the |numa_node|.
static iree_status_t iree_hal_amdgpu_host_agent_thread_affinity(
    const iree_hal_amdgpu_libhsa_t* libhsa, uint32_t numa_node,
    iree_thread_affinity_t* out_thread_affinity) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_thread_affinity, 0, sizeof(*out_thread_affinity));

  // DO NOT SUBMIT numactl
  // hsa4 shows cpu_set construction
  // may need magic iree_thread api?
  // pthread_setaffinity_np called from worker?
  // seal bit from affinity::id, use as group_specified
  // change specified to id_specified, use both?
  iree_status_t status = iree_ok_status();

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// HSA_PACKET_TYPE_BARRIER_AND
//===----------------------------------------------------------------------===//

// Issues a `HSA_PACKET_TYPE_BARRIER_AND` on the host worker.
// Signals completion when all dependencies are resolved.
static iree_status_t iree_hal_amdgpu_host_worker_issue_barrier_and(
    iree_hal_amdgpu_host_worker_t* worker,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const hsa_barrier_and_packet_t* packet) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): propagate failures according to spec?

  // TODO(benvanik): use hsa_amd_signal_wait_all when landed:
  // https://github.com/ROCm/ROCR-Runtime/issues/241
  // For now we have to wait each one at a time which requires at least O(n)
  // syscalls.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(packet->dep_signal); ++i) {
    // NOTE: 0 handles are ignored.
    // TODO(benvanik): https://github.com/ROCm/ROCR-Runtime/issues/252 tracks
    // making this a supported behavior in the HSA API.
    if (!packet->dep_signal[i].handle) continue;

    // NOTE: this will wake if the signal ever passes through 0 - it's possible
    // for it to be non-zero upon return if something else modifies it (but we
    // should never be doing that).
    iree_hsa_signal_wait_scacquire(IREE_LIBHSA(libhsa), packet->dep_signal[i],
                                   HSA_SIGNAL_CONDITION_EQ, 0u, UINT64_MAX,
                                   HSA_WAIT_STATE_BLOCKED);
  }

  if (packet->completion_signal.handle != 0) {
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(libhsa),
                                       packet->completion_signal, 1);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// HSA_PACKET_TYPE_BARRIER_OR
//===----------------------------------------------------------------------===//

// Issues a `HSA_PACKET_TYPE_BARRIER_OR` on the host worker.
// Signals completion when any dependency is resolved.
static iree_status_t iree_hal_amdgpu_host_worker_issue_barrier_or(
    iree_hal_amdgpu_host_worker_t* worker,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const hsa_barrier_or_packet_t* packet) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): propagate failures according to spec?

  // Unfortunately hsa_amd_signal_wait_any does not allow 0 signal handles but
  // AQL does: in an AQL packet processor the 0 signals are no-ops.
  //
  // TODO(benvanik): https://github.com/ROCm/ROCR-Runtime/issues/252 tracks
  // making this a supported behavior in the HSA API.
  hsa_signal_t signals[IREE_ARRAYSIZE(packet->dep_signal)];
  uint32_t signal_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(packet->dep_signal); ++i) {
    // NOTE: 0 handles are ignored.
    if (!packet->dep_signal[i].handle) break;
    signals[signal_count++] = packet->dep_signal[i];
  }
  hsa_signal_condition_t conds[5] = {
      HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ,
      HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ,
  };
  hsa_signal_value_t values[5] = {
      0, 0, 0, 0, 0,
  };

  // NOTE: this will wake if the signal ever passes through 0 - it's possible
  // for it to be non-zero upon return if something else modifies it (but we
  // should never be doing that).
  //
  // NOTE: hsa_amd_signal_wait_any has relaxed memory semantics and to have the
  // proper acquire behavior we need to load the signal value ourselves.
  hsa_signal_value_t satisfying_value = 0;
  const uint32_t satisfying_index = iree_hsa_amd_signal_wait_any(
      IREE_LIBHSA(libhsa), signal_count, signals, conds, values, UINT64_MAX,
      HSA_WAIT_STATE_BLOCKED, &satisfying_value);
  if (satisfying_index != UINT32_MAX) {
    iree_hsa_signal_load_scacquire(IREE_LIBHSA(libhsa),
                                   signals[satisfying_index]);
  }

  if (packet->completion_signal.handle != 0) {
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(libhsa),
                                       packet->completion_signal, 1);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// HSA_AMD_PACKET_TYPE_BARRIER_VALUE
//===----------------------------------------------------------------------===//

// Returns true if `|current_value| |condition| |desired_value|` is true.
static bool iree_hsa_condition_is_met(hsa_signal_condition32_t condition,
                                      hsa_signal_value_t current_value,
                                      hsa_signal_value_t desired_value) {
  switch (condition) {
    default:
    case HSA_SIGNAL_CONDITION_EQ:
      return current_value == desired_value;
    case HSA_SIGNAL_CONDITION_NE:
      return current_value != desired_value;
    case HSA_SIGNAL_CONDITION_LT:
      return current_value < desired_value;
    case HSA_SIGNAL_CONDITION_GTE:
      return current_value >= desired_value;
  }
}

// Issues a `HSA_AMD_PACKET_TYPE_BARRIER_VALUE` on the host worker.
// Signals completion when the value condition is met.
static iree_status_t iree_hal_amdgpu_host_worker_issue_barrier_value(
    iree_hal_amdgpu_host_worker_t* worker,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const hsa_amd_barrier_value_packet_t* packet) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): propagate failures according to spec?

  // HSA signal wait doesn't take a mask so if the mask is used we have to
  // emulate it. Hopefully most cases are not using a mask.
  if (packet->mask == UINT64_MAX) {
    // NOTE: this will wake if the signal ever meets the condition - it's
    // possible for it to be unsatisfied upon return if something else modifies
    // it (but we should never be doing that).
    iree_hsa_signal_wait_scacquire(IREE_LIBHSA(libhsa), packet->signal,
                                   packet->cond, packet->value, UINT64_MAX,
                                   HSA_WAIT_STATE_BLOCKED);
  } else {
    // Emulate a wait that takes a mask. This will wake each time the value
    // changes until the condition is met.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "emulated mask");
    hsa_signal_value_t value =
        iree_hsa_signal_load_scacquire(IREE_LIBHSA(libhsa), packet->signal);
    while (!iree_hsa_condition_is_met(packet->cond, value & packet->mask,
                                      packet->value)) {
      value = iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(libhsa), packet->signal, HSA_SIGNAL_CONDITION_NE, value,
          UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    }
  }

  if (packet->completion_signal.handle != 0) {
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(libhsa),
                                       packet->completion_signal, 1);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// HSA_PACKET_TYPE_AGENT_DISPATCH
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_host_retire_entry(
    iree_hal_amdgpu_host_worker_t* worker, iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* scheduler_queue_entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Enqueue device-side request to retire the entry.
  // This originates from work queued by the device that completes on the host
  // and that the device may otherwise not observe without being notified.
  iree_hal_amdgpu_queue_request_retire(queue, scheduler_queue_entry);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_pool_grow(
    iree_hal_amdgpu_host_worker_t* worker,
    iree_hal_amdgpu_device_allocator_pool_t* pool, uint8_t min_alignment,
    uint64_t allocation_size, uint64_t allocation_offset,
    iree_hal_amdgpu_device_allocation_handle_t* handle) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT pool grow
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_pool_trim(
    iree_hal_amdgpu_host_worker_t* worker,
    iree_hal_amdgpu_device_allocator_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT pool trim
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_post_release(
    iree_hal_amdgpu_host_worker_t* worker, iree_hal_resource_t* resources[4]) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release each resource. Some entries may be NULL.
  for (iree_host_size_t i = 0; i < 4; ++i) {
    iree_hal_resource_release(resources[i]);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_post_retire(
    iree_hal_amdgpu_host_worker_t* worker, iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* entry,
    iree_hal_amdgpu_host_call_retire_entry_arg2_t arg2,
    uint64_t allocation_token, iree_hal_resource_set_t* resource_set) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): tracing rendezvouz with entry submission (if we have a way
  // to do that). Likely need to just bite the bullet and add proper async
  // support to tracy.

  iree_status_t status = iree_hal_amdgpu_queue_retire_entry(
      queue, entry, arg2.has_signals, arg2.allocation_pool, allocation_token,
      resource_set);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_post_error(
    iree_hal_amdgpu_host_worker_t* worker,
    iree_hal_amdgpu_device_queue_entry_header_t* source_entry, uint64_t code,
    uint64_t arg0, uint64_t arg1) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, code);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, arg0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, arg1);

  // Map the code to a status.
  //
  // TODO(benvanik): allow posting allocated statuses from the device. To do so
  // we could perform a cloning of the status from device memory to host memory.
  // The device-side would need to have a status ringbuffer or something in
  // order to encode the source location/message.
  iree_status_t error_status = iree_ok_status();
  switch (code) {
    default:
      error_status = iree_make_status(IREE_STATUS_INTERNAL,
                                      "device-side error %" PRIX64 " (%" PRIi64
                                      "), arg0=%" PRIX64 " (%" PRIi64
                                      "), arg1=%" PRIX64 " (%" PRIi64 ")",
                                      code, code, arg0, arg0, arg1, arg1);
      break;
  }

  // If the error source queue entry was provided we retire it inline by failing
  // all signal statuses with the error status.
  const iree_hal_amdgpu_device_semaphore_list_t* semaphore_list =
      source_entry ? source_entry->signal_list : NULL;
  if (semaphore_list != NULL) {
    for (uint8_t i = 0; i < semaphore_list->count; ++i) {
      iree_hal_amdgpu_device_semaphore_t* device_semaphore =
          semaphore_list->entries[i].semaphore;
      iree_hal_semaphore_t* host_semaphore =
          (iree_hal_semaphore_t*)device_semaphore->host_semaphore;
      // Last semaphore gets the error status ownership, earlier need to clone.
      const iree_status_t fail_status = i < semaphore_list->count
                                            ? iree_status_clone(error_status)
                                            : error_status;
      iree_hal_semaphore_fail(host_semaphore, fail_status);
    }
  } else {
    // Fallback to a global error on the host. This will terminate the device.
    // We only preserve the first failure so only do this if we are going from a
    // valid state to a failed one.
    iree_status_t old_status = iree_ok_status();
    if (!iree_atomic_compare_exchange_strong(
            &worker->failure_status, (intptr_t*)&old_status,
            (intptr_t)error_status, iree_memory_order_acq_rel,
            iree_memory_order_relaxed /* old_status is unused */)) {
      // Previous status was not OK; drop our new status.
      IREE_IGNORE_ERROR(error_status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_post_signal(
    iree_hal_amdgpu_host_worker_t* worker,
    iree_hal_amdgpu_external_semaphore_t* semaphore, uint64_t payload) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Notify the external semaphore of its new value. It may make platform calls
  // or do other bookkeeping.
  iree_status_t status =
      iree_hal_amdgpu_external_semaphore_notify(semaphore, payload);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_post_trace_flush(
    iree_hal_amdgpu_host_worker_t* worker,
    iree_hal_amdgpu_trace_buffer_t* trace_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ask the trace buffer to flush itself.
  iree_status_t status = iree_hal_amdgpu_trace_buffer_flush(trace_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Issues a `HSA_PACKET_TYPE_AGENT_DISPATCH` on the host worker.
// Signals completion if the requested operation completes synchronously and
// otherwise may signal asynchronously after this function returns.
//
// If the operation is asynchronous then the worker->outstanding_signal must be
// incremented and decremented when the operation completes along with the
// packet completion signal.
static iree_status_t iree_hal_amdgpu_host_worker_issue_agent_dispatch(
    iree_hal_amdgpu_host_worker_t* worker,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const hsa_agent_dispatch_packet_t* packet) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT debugging/tracing on agent dispatch
  // TODO: propagate failures according to spec?
  // fprintf(stderr,
  //         "agent dispatch %u: %p %" PRIx64 " (%" PRId64 ") %" PRIx64
  //         " (%" PRId64 ") %" PRIx64 " (%" PRId64 ") %" PRIx64 " (%" PRId64
  //         ")\n",
  //         packet->type, packet->return_address, packet->arg[0],
  //         packet->arg[0], packet->arg[1], packet->arg[1], packet->arg[2],
  //         packet->arg[2], packet->arg[3], packet->arg[3]);

  iree_status_t status = iree_ok_status();
  switch (packet->type) {
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_RETIRE_ENTRY:
      status = iree_hal_amdgpu_host_retire_entry(
          worker, (iree_hal_amdgpu_queue_t*)packet->arg[0],
          (iree_hal_amdgpu_device_queue_entry_header_t*)packet->arg[1]);
      break;
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_GROW:
      status = iree_hal_amdgpu_host_pool_grow(
          worker, (iree_hal_amdgpu_device_allocator_pool_t*)packet->arg[0],
          packet->arg[2] & 0xFFu, packet->arg[2] >> 8, packet->arg[3],
          (iree_hal_amdgpu_device_allocation_handle_t*)packet->return_address);
      break;
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_TRIM:
      status = iree_hal_amdgpu_host_pool_trim(
          worker, (iree_hal_amdgpu_device_allocator_pool_t*)packet->arg[0]);
      break;
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RELEASE:
      status = iree_hal_amdgpu_host_post_release(
          worker, (iree_hal_resource_t**)packet->arg);
      break;
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RETIRE:
      status = iree_hal_amdgpu_host_post_retire(
          worker, (iree_hal_amdgpu_queue_t*)packet->arg[0],
          (iree_hal_amdgpu_device_queue_entry_header_t*)packet->arg[1],
          (iree_hal_amdgpu_host_call_retire_entry_arg2_t){
              .bits = packet->arg[2],
          },
          packet->arg[3], (iree_hal_resource_set_t*)packet->return_address);
      break;
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_ERROR:
      status = iree_hal_amdgpu_host_post_error(
          worker, (iree_hal_amdgpu_device_queue_entry_header_t*)packet->arg[0],
          packet->arg[1], packet->arg[2], packet->arg[3]);
      break;
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_SIGNAL:
      status = iree_hal_amdgpu_host_post_signal(
          worker, (iree_hal_amdgpu_external_semaphore_t*)packet->arg[0],
          packet->arg[1]);
      break;
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_TRACE_FLUSH:
      status = iree_hal_amdgpu_host_post_trace_flush(
          worker, (iree_hal_amdgpu_trace_buffer_t*)packet->arg[0]);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown type %u",
                                packet->type);
      break;
  }

  if (packet->completion_signal.handle != 0) {
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(libhsa),
                                       packet->completion_signal, 1);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_worker_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_host_worker_barrier(
    iree_hal_amdgpu_host_worker_t* worker,
    const iree_hal_amdgpu_libhsa_t* libhsa) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait until all asynchronous operations complete (signal reaches <= 0).
  // Note that this may fail and return a value greater than that for various
  // reasons.
  hsa_signal_value_t value = iree_hsa_signal_wait_scacquire(
      IREE_LIBHSA(libhsa), worker->outstanding_signal, HSA_SIGNAL_CONDITION_LT,
      1ull, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, value);

  // TODO(benvanik): a way to get errors back? we could reserve the high bit and
  // make all values with it set interpret as a status. A high bit set would
  // make the hsa_signal_value_t negative and allow the wait to be satisfied.
  iree_status_t status =
      value == 0ull ? iree_ok_status()
                    : iree_make_status(
                          IREE_STATUS_ABORTED,
                          "asynchronous work failed and the queue is invalid");

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static int iree_hal_amdgpu_host_worker_main(void* entry_arg) {
  iree_hal_amdgpu_host_worker_t* worker =
      (iree_hal_amdgpu_host_worker_t*)entry_arg;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdgpu_libhsa_t* libhsa = &worker->system->libhsa;

  // Main loop.
  const uint64_t queue_mask = worker->queue->size - 1;
  uint64_t last_packet_id = 0;
  uint64_t read_index = 0;
  while (true) {
    // Since we are in MULTI mode we just check that the packet ID changed but
    // don't trust it as an indication of what we should process as it may be
    // set out of order from multiple producers.
    const uint64_t new_packet_id = (uint64_t)iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(libhsa), worker->doorbell, HSA_SIGNAL_CONDITION_NE,
        last_packet_id, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    last_packet_id = new_packet_id;
    if (new_packet_id == UINT64_MAX) {
      // Exit signal.
      break;
    }

    // Drain all packets. Note that this may block and we may get new packets
    // enqueued while processing.
    while (read_index != iree_hsa_queue_load_write_index_scacquire(
                             IREE_LIBHSA(libhsa), worker->queue)) {
      IREE_TRACE_ZONE_BEGIN_NAMED(z_packet,
                                  "iree_hal_amdgpu_host_worker_process_packet");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_packet, read_index);

      // Reference packet in queue memory. Note that we want to get it out of
      // there ASAP to free up the space in the queue.
      hsa_agent_dispatch_packet_t* packet_ptr =
          (hsa_agent_dispatch_packet_t*)worker->queue->base_address +
          (read_index & queue_mask);

      // Spin until the packet is populated.
      // In AQL it's valid to bump the write index prior to the packet header
      // being updated and the queue must stall until it's no longer INVALID.
      //
      // Note that because this is expected to be cycles away we don't yield and
      // risk an OS context switch. If we hit cases where the write index and
      // packet header stores are split in time we'll want to do something
      // smarter like a backoff.
      uint32_t packet_type = HSA_PACKET_TYPE_INVALID;
      do {
        const uint32_t packet_header = iree_atomic_load(
            (iree_atomic_uint32_t*)packet_ptr, iree_memory_order_acquire);
        packet_type = (packet_header >> HSA_PACKET_HEADER_TYPE) &
                      ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
      } while (packet_type == HSA_PACKET_TYPE_INVALID);

      // Copy the packet locally and swap the packet back to INVALID so that
      // producers can overwrite it immediately. By bumping the read index
      // producers will be able to reserve it if they are waiting for capacity
      // to be available.
      uint8_t packet_data[64];
      memcpy(packet_data, packet_ptr, sizeof(packet_data));
      iree_atomic_store((iree_atomic_uint32_t*)packet_ptr,
                        (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE),
                        iree_memory_order_relaxed);
      iree_hsa_queue_store_read_index_screlease(IREE_LIBHSA(libhsa),
                                                worker->queue, ++read_index);

      // If the packet has a barrier bit set then we need to block until all
      // prior queue operations have completed. Most of our operations are
      // synchronous but it's possible to have async operations outstanding and
      // we need to wait for them.
      const uint16_t packet_header = *(const uint16_t*)packet_data;
      if (packet_header & (1u << HSA_PACKET_HEADER_BARRIER)) {
        iree_status_t barrier_status =
            iree_hal_amdgpu_host_worker_barrier(worker, libhsa);
        if (!iree_status_is_ok(barrier_status)) {
          // DO NOT SUBMIT barrier_status propagation
          IREE_TRACE_ZONE_APPEND_TEXT(
              z_packet,
              iree_status_code_string(iree_status_code(barrier_status)));
          IREE_IGNORE_ERROR(barrier_status);
        }
      }

      // Switch on packet type and issue.
      iree_status_t status = iree_ok_status();
      switch (packet_type) {
        case HSA_PACKET_TYPE_BARRIER_AND:
          status = iree_hal_amdgpu_host_worker_issue_barrier_and(
              worker, libhsa, (const hsa_barrier_and_packet_t*)packet_data);
          break;
        case HSA_PACKET_TYPE_BARRIER_OR:
          status = iree_hal_amdgpu_host_worker_issue_barrier_or(
              worker, libhsa, (const hsa_barrier_or_packet_t*)packet_data);
          break;
        case HSA_AMD_PACKET_TYPE_BARRIER_VALUE:
          status = iree_hal_amdgpu_host_worker_issue_barrier_value(
              worker, libhsa,
              (const hsa_amd_barrier_value_packet_t*)packet_data);
          break;
        case HSA_PACKET_TYPE_AGENT_DISPATCH:
          status = iree_hal_amdgpu_host_worker_issue_agent_dispatch(
              worker, libhsa, (const hsa_agent_dispatch_packet_t*)packet_data);
          break;
        default:
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "invalid packet type %u", packet_type);
          break;
      }
      if (!iree_status_is_ok(status)) {
        // DO NOT SUBMIT dispatch error propagation
        IREE_TRACE_ZONE_APPEND_TEXT(
            z_packet, iree_status_code_string(iree_status_code(status)));
        IREE_IGNORE_ERROR(status);
      }

      IREE_TRACE_ZONE_END(z_packet);
    }
  }

  // Wait for any outstanding asynchronous operations to complete.
  // This ensures that we don't free memory that may be in use by them.
  // Note that only this worker is allowed to wait on the signal so we have to
  // do it here.
  IREE_IGNORE_ERROR(iree_hal_amdgpu_host_worker_barrier(worker, libhsa));

  IREE_TRACE_ZONE_END(z0);
  return 0;
}

iree_status_t iree_hal_amdgpu_host_worker_initialize(
    iree_hal_amdgpu_system_t* system, iree_host_size_t host_ordinal,
    iree_host_size_t device_ordinal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_worker_t* out_worker) {
  IREE_ASSERT_ARGUMENT(out_worker);
  IREE_TRACE_ZONE_BEGIN(z0);

  hsa_agent_t host_agent = system->topology.cpu_agents[host_ordinal];

  memset(out_worker, 0, sizeof(*out_worker));
  out_worker->system = system;
  out_worker->host_agent = host_agent;
  out_worker->host_ordinal = host_ordinal;
  out_worker->failure_status = IREE_ATOMIC_VAR_INIT(0);

  // NUMA node.
  uint32_t host_agent_node = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_agent_get_info(IREE_LIBHSA(&system->libhsa), host_agent,
                                  HSA_AGENT_INFO_NODE, &host_agent_node));

  // Pin the thread to the NUMA node specified.
  // We don't care which core but do want it to be one of those associated with
  // the devices this worker is servicing.
  iree_thread_affinity_t thread_affinity = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_host_agent_thread_affinity(
              &system->libhsa, host_agent_node, &thread_affinity));

  // Create a semaphore for tracking outstanding asynchronous operations. It's
  // marked as only being "consumed" by the host agent (waited on). Other agents
  // and threads can signal it.
  iree_status_t status = iree_hsa_amd_signal_create(
      IREE_LIBHSA(&system->libhsa), 0ull, 1, &host_agent,
      /*attributes=*/0, &out_worker->outstanding_signal);

  // Create the doorbell for the soft queue. It's marked as only being
  // "consumed" by the host agent (waited on). Other agents can signal it.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(&system->libhsa), 0ull, 1, &host_agent,
        /*attributes=*/0, &out_worker->doorbell);
  }

  // Create and allocate the soft queue.
  // We cannot change the queue size after it is created and pick something
  // large enough to be able to reasonably satisfy all requests. Must be a power
  // of two. We allocate the queue in the host pool closest to the devices that
  // will be producing for it and where this worker will be consuming it.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_soft_queue_create(
        IREE_LIBHSA(&system->libhsa),
        system->host_memory_pools[host_ordinal].fine_region,
        IREE_HAL_AMDGPU_HOST_WORKER_QUEUE_CAPACITY, HSA_QUEUE_TYPE_MULTI,
        HSA_QUEUE_FEATURE_AGENT_DISPATCH, out_worker->doorbell,
        &out_worker->queue);
  }

  // Create the worker thread for handling device library requests.
  // The worker may start immediately and use the queue/doorbell.
  if (iree_status_is_ok(status)) {
    char thread_name[32];
    snprintf(thread_name, IREE_ARRAYSIZE(thread_name),
             "iree-amdgpu-host-%" PRIhsz "-%" PRIhsz, host_ordinal,
             device_ordinal);
    const iree_thread_create_params_t thread_params = {
        .name = iree_make_cstring_view(thread_name),
        .stack_size = 0,  // default
        .create_suspended = false,
        .priority_class = IREE_THREAD_PRIORITY_CLASS_HIGH,
        .initial_affinity = thread_affinity,
    };
    status =
        iree_thread_create(iree_hal_amdgpu_host_worker_main, out_worker,
                           thread_params, host_allocator, &out_worker->thread);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_worker_deinitialize(out_worker);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_worker_deinitialize(
    iree_hal_amdgpu_host_worker_t* worker) {
  IREE_ASSERT_ARGUMENT(worker);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdgpu_libhsa_t* libhsa = &worker->system->libhsa;

  // Mark the queue as inactive. This is likely a no-op for our soft queue from
  // the API perspective but can help tooling see our intent.
  if (worker->queue) {
    IREE_IGNORE_ERROR(
        iree_hsa_queue_inactivate(IREE_LIBHSA(libhsa), worker->queue));
  }

  // Signal the doorbell to the termination value. This should wake the worker
  // and have it exit.
  if (worker->doorbell.handle) {
    iree_hsa_signal_store_screlease(IREE_LIBHSA(libhsa), worker->doorbell,
                                    UINT64_MAX);
  }

  // Join thread after it has shut down.
  if (worker->thread) {
    iree_thread_release(worker->thread);
    worker->thread = NULL;
  }

  // Tear down HSA resources.
  if (worker->queue) {
    IREE_IGNORE_ERROR(
        iree_hsa_queue_destroy(IREE_LIBHSA(libhsa), worker->queue));
  }
  if (worker->doorbell.handle) {
    IREE_IGNORE_ERROR(
        iree_hsa_signal_destroy(IREE_LIBHSA(libhsa), worker->doorbell));
  }
  if (worker->outstanding_signal.handle) {
    IREE_IGNORE_ERROR(iree_hsa_signal_destroy(IREE_LIBHSA(libhsa),
                                              worker->outstanding_signal));
  }

  // If a failure status was set we need to dispose of it.
  iree_status_t failure_status = (iree_status_t)iree_atomic_exchange(
      &worker->failure_status, 0, iree_memory_order_acquire);
  iree_status_free(failure_status);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_host_worker_check_status(
    iree_hal_amdgpu_host_worker_t* worker) {
  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &worker->failure_status, iree_memory_order_acquire);
  return iree_status_clone(failure_status);
}

void iree_hal_amdgpu_host_worker_notify_completion(
    iree_hal_amdgpu_host_async_token_t token, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(token.worker);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(status)));

  // TODO(benvanik): we could track queue depth here or use the token to track
  // the async operation across threads.

  // If the async operation failed we need to set the sticky failure status.
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // Try to set our local status - we only preserve the first failure so only
    // do this if we are going from a valid semaphore to a failed one.
    iree_status_t old_status = iree_ok_status();
    if (!iree_atomic_compare_exchange_strong(
            &token.worker->failure_status, (intptr_t*)&old_status,
            (intptr_t)status, iree_memory_order_acq_rel,
            iree_memory_order_relaxed /* old_status is unused */)) {
      // Previous status was not OK; drop our new status.
      IREE_IGNORE_ERROR(status);
    }
  }

  iree_hal_amdgpu_host_worker_t* worker = token.worker;
  iree_hsa_signal_subtract_screlease(IREE_LIBHSA(&worker->system->libhsa),
                                     worker->outstanding_signal, 1);

  IREE_TRACE_ZONE_END(z0);
}
