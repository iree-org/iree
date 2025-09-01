// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_service.h"

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/host_client.h"

// A signal payload that is unlikely to ever be hit during real execution.
// This is used to indicate a default state that we want to detect changes on.
#define IREE_HAL_AMDGPU_INVALID_SIGNAL_VALUE ((hsa_signal_value_t) - 2)

static void iree_hal_amdgpu_host_service_fail(
    iree_hal_amdgpu_host_service_t* service, iree_status_t status);

//===----------------------------------------------------------------------===//
// HSA_PACKET_TYPE_BARRIER_AND
//===----------------------------------------------------------------------===//

// Issues an `HSA_PACKET_TYPE_BARRIER_AND` on the host worker.
// Signals completion when all dependencies are resolved.
static iree_status_t iree_hal_amdgpu_host_service_issue_barrier_and(
    iree_hal_amdgpu_host_service_t* service,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const hsa_barrier_and_packet_t* packet) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: this will wake if all signals ever passes through 0 - it's possible
  // for it to be non-zero upon return if something else modifies it (but we
  // should never be doing that).
  //
  // NOTE: hsa_amd_signal_wait_any has relaxed memory semantics and to have the
  // proper acquire behavior we need to load the signal value ourselves.
  static_assert(IREE_ARRAYSIZE(packet->dep_signal) == 5, "expecting 5 signals");
  hsa_signal_condition_t conds[5] = {
      HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ,
      HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ,
  };
  hsa_signal_value_t values[5] = {
      0, 0, 0, 0, 0,
  };
  const uint32_t satisfying_index = iree_hsa_amd_signal_wait_all(
      IREE_LIBHSA(libhsa), IREE_ARRAYSIZE(packet->dep_signal),
      (hsa_signal_t*)packet->dep_signal, conds, values, UINT64_MAX,
      HSA_WAIT_STATE_BLOCKED,
      /*satisfying_values=*/NULL);
  iree_status_t status = iree_ok_status();
  if (IREE_LIKELY(satisfying_index != UINT32_MAX)) {
    // NOTE: if all signals are null then wait_all will return index 0 as having
    // satisfied the wait even if it's invalid. We assume here there's at least
    // one valid signal (as otherwise why would the wait have been requested?)
    // and try to find the first, using the satisfying_index as a starting point
    // ideally pointing at a valid signal and we break from the loop
    // immediately.
    for (uint32_t i = satisfying_index; i < IREE_ARRAYSIZE(packet->dep_signal);
         ++i) {
      if (packet->dep_signal[i].handle) {
        iree_hsa_signal_load_scacquire(IREE_LIBHSA(libhsa),
                                       packet->dep_signal[i]);
        break;
      }
    }
  } else {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "hsa_amd_signal_wait_all failed");
  }

  // TODO(benvanik): figure out the expected behavior of completion signals on
  // failed barriers (it may be in the HSA spec). For now we don't signal and
  // rely on a global device loss to alert the user.
  if (iree_status_is_ok(status) && packet->completion_signal.handle != 0) {
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(libhsa),
                                       packet->completion_signal, 1);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// HSA_PACKET_TYPE_BARRIER_OR
//===----------------------------------------------------------------------===//

// Issues an `HSA_PACKET_TYPE_BARRIER_OR` on the host worker.
// Signals completion when any dependency is resolved.
static iree_status_t iree_hal_amdgpu_host_service_issue_barrier_or(
    iree_hal_amdgpu_host_service_t* service,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const hsa_barrier_or_packet_t* packet) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: this will wake if the signal ever passes through 0 - it's possible
  // for it to be non-zero upon return if something else modifies it (but we
  // should never be doing that).
  //
  // NOTE: hsa_amd_signal_wait_any has relaxed memory semantics and to have the
  // proper acquire behavior we need to load the signal value ourselves.
  static_assert(IREE_ARRAYSIZE(packet->dep_signal) == 5, "expecting 5 signals");
  hsa_signal_condition_t conds[5] = {
      HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ,
      HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ,
  };
  hsa_signal_value_t values[5] = {
      0, 0, 0, 0, 0,
  };
  hsa_signal_value_t satisfying_value = 0;
  const uint32_t satisfying_index = iree_hsa_amd_signal_wait_any(
      IREE_LIBHSA(libhsa), IREE_ARRAYSIZE(packet->dep_signal),
      (hsa_signal_t*)packet->dep_signal, conds, values, UINT64_MAX,
      HSA_WAIT_STATE_BLOCKED, &satisfying_value);
  iree_status_t status = iree_ok_status();
  if (IREE_LIKELY(satisfying_index != UINT32_MAX)) {
    iree_hsa_signal_load_scacquire(IREE_LIBHSA(libhsa),
                                   packet->dep_signal[satisfying_index]);
  } else {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "hsa_amd_signal_wait_any failed");
  }

  // TODO(benvanik): figure out the expected behavior of completion signals on
  // failed barriers (it may be in the HSA spec). For now we don't signal and
  // rely on a global device loss to alert the user.
  if (iree_status_is_ok(status) && packet->completion_signal.handle != 0) {
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(libhsa),
                                       packet->completion_signal, 1);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
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

// Issues an `HSA_AMD_PACKET_TYPE_BARRIER_VALUE` on the host worker.
// Signals completion when the value condition is met.
static iree_status_t iree_hal_amdgpu_host_service_issue_barrier_value(
    iree_hal_amdgpu_host_service_t* service,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const hsa_amd_barrier_value_packet_t* packet) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): propagate failures according to spec?

  // HSA signal wait doesn't take a mask so if the mask is used we have to
  // emulate it. Hopefully most cases are not using a mask.
  if (IREE_LIKELY(packet->mask == UINT64_MAX)) {
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

// IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_SIGNAL
static iree_status_t iree_hal_amdgpu_host_post_signal(
    iree_hal_amdgpu_host_service_t* service, iree_hal_semaphore_t* semaphore,
    uint64_t payload) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Notify the (likely) external semaphore of its new value. It may make
  // platform calls or do other bookkeeping.
  iree_status_t status = iree_hal_semaphore_signal(semaphore, payload);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RELEASE
static iree_status_t iree_hal_amdgpu_host_post_release(
    iree_hal_amdgpu_host_service_t* service, iree_host_size_t resource_count,
    iree_hal_resource_t* resources[]) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release each resource. Some entries may be NULL.
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    iree_hal_resource_release(resources[i]);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Issues an `HSA_PACKET_TYPE_AGENT_DISPATCH` on the host service.
// Signals completion if the requested operation completes synchronously and
// otherwise may signal asynchronously after this function returns.
//
// If the operation is asynchronous then the service->outstanding_signal must be
// incremented and decremented when the operation completes along with the
// packet completion signal.
static iree_status_t iree_hal_amdgpu_host_service_issue_agent_dispatch(
    iree_hal_amdgpu_host_service_t* service,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const hsa_agent_dispatch_packet_t* packet) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  switch (packet->type) {
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_SIGNAL:
      status = iree_hal_amdgpu_host_post_signal(
          service, (iree_hal_semaphore_t*)packet->arg[0], packet->arg[1]);
      break;
    case IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RELEASE:
      status = iree_hal_amdgpu_host_post_release(
          service, IREE_ARRAYSIZE(packet->arg),
          (iree_hal_resource_t**)packet->arg);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown type %u",
                                packet->type);
      break;
  }

  if (iree_status_is_ok(status) && packet->completion_signal.handle != 0) {
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(libhsa),
                                       packet->completion_signal, 1);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_service_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_host_service_barrier(
    iree_hal_amdgpu_host_service_t* service,
    const iree_hal_amdgpu_libhsa_t* libhsa) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait until all asynchronous operations complete (signal reaches <= 0).
  // Note that this may fail and return a value greater than that for various
  // reasons.
  hsa_signal_value_t value = iree_hsa_signal_wait_scacquire(
      IREE_LIBHSA(libhsa), service->outstanding_signal, HSA_SIGNAL_CONDITION_LT,
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

static int iree_hal_amdgpu_host_service_main(void* entry_arg) {
  iree_hal_amdgpu_host_service_t* service =
      (iree_hal_amdgpu_host_service_t*)entry_arg;
  const iree_hal_amdgpu_libhsa_t* libhsa = service->libhsa;

  // Main loop.
  const uint64_t queue_mask = service->queue->size - 1;
  uint64_t last_packet_id = IREE_HAL_AMDGPU_INVALID_SIGNAL_VALUE;
  uint64_t read_index = 0;
  while (true) {
    // Since we are in MULTI mode we just check that the packet ID changed but
    // don't trust it as an indication of what we should process as it may be
    // set out of order from multiple producers.
    const uint64_t new_packet_id = (uint64_t)iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(libhsa), service->doorbell, HSA_SIGNAL_CONDITION_NE,
        last_packet_id, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    last_packet_id = new_packet_id;
    if (new_packet_id == UINT64_MAX) {
      // Exit signal.
      break;
    }

    // Drain all packets. Note that this may block and we may get new packets
    // enqueued while processing.
    while (read_index != iree_hsa_queue_load_write_index_scacquire(
                             IREE_LIBHSA(libhsa), service->queue)) {
      IREE_TRACE_ZONE_BEGIN_NAMED(
          z_packet, "iree_hal_amdgpu_host_service_process_packet");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_packet, read_index);

      // Reference packet in queue memory. Note that we want to get it out of
      // there ASAP to free up the space in the queue.
      // NOTE: we cast to an agent packet here but don't yet know the type.
      // We only use the struct to parse the header bits common to all packets.
      hsa_agent_dispatch_packet_t* packet_ptr =
          (hsa_agent_dispatch_packet_t*)service->queue->base_address +
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
                                                service->queue, ++read_index);

      // If the packet has a barrier bit set then we need to block until all
      // prior queue operations have completed. Most of our operations are
      // synchronous but it's possible to have async operations outstanding and
      // we need to wait for them.
      const uint16_t packet_header = *(const uint16_t*)packet_data;
      if (packet_header & (1u << HSA_PACKET_HEADER_BARRIER)) {
        iree_status_t barrier_status =
            iree_hal_amdgpu_host_service_barrier(service, libhsa);
        if (!iree_status_is_ok(barrier_status)) {
          IREE_TRACE_ZONE_APPEND_TEXT(
              z_packet,
              iree_status_code_string(iree_status_code(barrier_status)));
          iree_hal_amdgpu_host_service_fail(service, barrier_status);
          break;
        }
      }

      // Switch on packet type and issue.
      iree_status_t status = iree_ok_status();
      switch (packet_type) {
        case HSA_PACKET_TYPE_BARRIER_AND:
          status = iree_hal_amdgpu_host_service_issue_barrier_and(
              service, libhsa, (const hsa_barrier_and_packet_t*)packet_data);
          break;
        case HSA_PACKET_TYPE_BARRIER_OR:
          status = iree_hal_amdgpu_host_service_issue_barrier_or(
              service, libhsa, (const hsa_barrier_or_packet_t*)packet_data);
          break;
        case HSA_PACKET_TYPE_VENDOR_SPECIFIC: {
          const hsa_amd_vendor_packet_header_t vendor_header =
              *(const hsa_amd_vendor_packet_header_t*)packet_data;
          switch (vendor_header.AmdFormat) {
            case HSA_AMD_PACKET_TYPE_BARRIER_VALUE:
              status = iree_hal_amdgpu_host_service_issue_barrier_value(
                  service, libhsa,
                  (const hsa_amd_barrier_value_packet_t*)packet_data);
              break;
            default:
              status = iree_make_status(IREE_STATUS_INTERNAL,
                                        "invalid vendor packet type %u",
                                        vendor_header.AmdFormat);
              break;
          }
          break;
        }
        case HSA_PACKET_TYPE_AGENT_DISPATCH:
          status = iree_hal_amdgpu_host_service_issue_agent_dispatch(
              service, libhsa, (const hsa_agent_dispatch_packet_t*)packet_data);
          break;
        default:
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "invalid packet type %u", packet_type);
          break;
      }
      if (!iree_status_is_ok(status)) {
        IREE_TRACE_ZONE_APPEND_TEXT(
            z_packet, iree_status_code_string(iree_status_code(status)));
        iree_hal_amdgpu_host_service_fail(service, status);
      }

      IREE_TRACE_ZONE_END(z_packet);
    }
  }

  // Wait for any outstanding asynchronous operations to complete.
  // This ensures that we don't free memory that may be in use by them.
  // Note that only this worker is allowed to wait on the signal so we have to
  // do it here.
  IREE_IGNORE_ERROR(iree_hal_amdgpu_host_service_barrier(service, libhsa));

  return 0;
}

iree_status_t iree_hal_amdgpu_host_service_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_host_size_t host_ordinal,
    hsa_agent_t host_agent, hsa_region_t host_fine_region,
    iree_host_size_t device_ordinal,
    iree_hal_amdgpu_error_callback_t error_callback,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_service_t* out_service) {
  IREE_ASSERT_ARGUMENT(out_service);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_service, 0, sizeof(*out_service));
  out_service->libhsa = libhsa;
  out_service->error_callback = error_callback;
  out_service->failure_code = IREE_ATOMIC_VAR_INIT(0);

  // NUMA node.
  uint32_t host_agent_node = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), host_agent,
                                  HSA_AGENT_INFO_NODE, &host_agent_node));

  // Pin the thread to the NUMA node specified.
  // We don't care which core but do want it to be one of those associated with
  // the devices this worker is servicing.
  iree_thread_affinity_t thread_affinity = {0};
  iree_thread_affinity_set_group_any(host_agent_node, &thread_affinity);

  // Create a semaphore for tracking outstanding asynchronous operations. It's
  // marked as only being "consumed" by the host agent (waited on). Other agents
  // and threads can signal it.
  iree_status_t status = iree_hsa_amd_signal_create(
      IREE_LIBHSA(libhsa), 0ull, 1, &host_agent,
      /*attributes=*/0, &out_service->outstanding_signal);

  // Create the doorbell for the soft queue. It's marked as only being
  // "consumed" by the host agent (waited on). Other agents can signal it.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(libhsa), IREE_HAL_AMDGPU_INVALID_SIGNAL_VALUE, 1,
        &host_agent,
        /*attributes=*/0, &out_service->doorbell);
  }

  // Create and allocate the soft queue.
  // We cannot change the queue size after it is created and pick something
  // large enough to be able to reasonably satisfy all requests. Must be a power
  // of two. We allocate the queue in the host pool closest to the devices that
  // will be producing for it and where this service will be consuming it.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_soft_queue_create(
        IREE_LIBHSA(libhsa), host_fine_region,
        IREE_HAL_AMDGPU_HOST_SERVICE_QUEUE_CAPACITY, HSA_QUEUE_TYPE_MULTI,
        HSA_QUEUE_FEATURE_AGENT_DISPATCH, out_service->doorbell,
        &out_service->queue);
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
        iree_thread_create(iree_hal_amdgpu_host_service_main, out_service,
                           thread_params, host_allocator, &out_service->thread);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_service_deinitialize(out_service);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_service_deinitialize(
    iree_hal_amdgpu_host_service_t* service) {
  IREE_ASSERT_ARGUMENT(service);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdgpu_libhsa_t* libhsa = service->libhsa;

  // Mark the queue as inactive. This is likely a no-op for our soft queue from
  // the API perspective but can help tooling see our intent.
  if (service->queue) {
    IREE_IGNORE_ERROR(
        iree_hsa_queue_inactivate(IREE_LIBHSA(libhsa), service->queue));
  }

  // Signal the doorbell to the termination value. This should wake the worker
  // and have it exit.
  if (service->doorbell.handle) {
    iree_hsa_signal_store_screlease(IREE_LIBHSA(libhsa), service->doorbell,
                                    UINT64_MAX);
  }

  // Join thread after it has shut down.
  if (service->thread) {
    iree_thread_join(service->thread);
    iree_thread_release(service->thread);
    service->thread = NULL;
  }

  // Tear down HSA resources.
  if (service->queue) {
    IREE_IGNORE_ERROR(
        iree_hsa_queue_destroy(IREE_LIBHSA(libhsa), service->queue));
  }
  if (service->doorbell.handle) {
    IREE_IGNORE_ERROR(
        iree_hsa_signal_destroy(IREE_LIBHSA(libhsa), service->doorbell));
  }
  if (service->outstanding_signal.handle) {
    IREE_IGNORE_ERROR(iree_hsa_signal_destroy(IREE_LIBHSA(libhsa),
                                              service->outstanding_signal));
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_host_service_fail(
    iree_hal_amdgpu_host_service_t* service, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(service);
  IREE_ASSERT(!iree_status_is_ok(status));
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(status)));

  // Try to set our local status - we only preserve the first failure so only
  // do this if we are going from a valid status to a failed one.
  uint64_t old_status_code = 0;
  uint64_t new_status_code = (uint64_t)iree_status_code(status);
  const bool first_failure = iree_atomic_compare_exchange_strong(
      &service->failure_code, &old_status_code, new_status_code,
      iree_memory_order_acq_rel,
      iree_memory_order_relaxed /* old_status is unused */);
  if (first_failure && service->error_callback.fn) {
    // Notify user-provided function; ownership of the status is transferred to
    // the callee.
    service->error_callback.fn(service->error_callback.user_data, status);
  } else {
    // No callback or callback already issued prior, drop the error.
    IREE_IGNORE_ERROR(status);
  }

  // Force the worker to exit (soon).
  if (service->doorbell.handle) {
    iree_hsa_signal_store_screlease(IREE_LIBHSA(service->libhsa),
                                    service->doorbell, UINT64_MAX);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amdgpu_host_service_notify_completion(
    iree_hal_amdgpu_host_async_token_t token, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(token.service);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): we could track queue depth here or use the token to track
  // the async operation across threads.

  // If the async operation failed we need to set the sticky failure status.
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // Enter the failure state; ownership of the status transferred.
    iree_hal_amdgpu_host_service_fail(token.service, status);
  }

  iree_hal_amdgpu_host_service_t* service = token.service;
  iree_hsa_signal_subtract_screlease(IREE_LIBHSA(service->libhsa),
                                     service->outstanding_signal, 1);

  IREE_TRACE_ZONE_END(z0);
}
