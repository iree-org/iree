// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_worker.h"

#include "iree/hal/drivers/amdgpu/device/host.h"

//===----------------------------------------------------------------------===//
// Platform Support
//===----------------------------------------------------------------------===//

// Returns the IREE thread affinity that specifies one or more threads
// associated with the NUMA node of |agent|.
static iree_status_t iree_hal_amdgpu_host_agent_thread_affinity(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    iree_thread_affinity_t* out_thread_affinity) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_thread_affinity, 0, sizeof(*out_thread_affinity));

  // DO NOT SUBMIT
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
// iree_hal_amdgpu_host_worker_t
//===----------------------------------------------------------------------===//

static int iree_hal_amdgpu_host_worker_main(void* entry_arg) {
  iree_hal_amdgpu_host_worker_t* worker =
      (iree_hal_amdgpu_host_worker_t*)entry_arg;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_libhsa_t* libhsa = &worker->system->libhsa;

  // DO NOT SUBMIT
  (void)worker;

  uint64_t queue_mask = worker->queue->size - 1;
  uint64_t last_packet_id = 0;
  uint64_t read_index = 0;
  while (true) {
    uint64_t new_packet_id = (uint64_t)iree_hsa_signal_wait_scacquire(
        libhsa, worker->doorbell, HSA_SIGNAL_CONDITION_NE, last_packet_id,
        UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    last_packet_id = new_packet_id;
    if (new_packet_id == UINT64_MAX) break;

    while (read_index !=
           iree_hsa_queue_load_write_index_scacquire(libhsa, worker->queue)) {
      hsa_agent_dispatch_packet_t* packet_ptr =
          (hsa_agent_dispatch_packet_t*)worker->queue->base_address +
          (read_index & queue_mask);

      // wait until packet populated
      uint32_t packet_type = HSA_PACKET_TYPE_INVALID;
      do {
        // NOTE: we assume this is waiting for at most a few cycles and spin.
        uint32_t packet_header =
            iree_atomic_load_int32((volatile iree_atomic_uint32_t*)packet_ptr,
                                   iree_memory_order_acquire);
        packet_type = (packet_header >> HSA_PACKET_HEADER_TYPE) &
                      ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
      } while (packet_type == HSA_PACKET_TYPE_INVALID);

      // copy packet locally
      union {
        uint8_t data[64];
        hsa_barrier_and_packet_t barrier_and;
        hsa_barrier_or_packet_t barrier_or;
        hsa_amd_barrier_value_packet_t barrier_value;
        hsa_agent_dispatch_packet_t agent_dispatch;
      } packet;
      memcpy(packet.data, packet_ptr, sizeof(packet.data));

      // swap packet back to invalid so that it can be reused immediately
      iree_atomic_store_int32(
          (volatile iree_atomic_uint32_t*)packet_ptr,
          (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE),
          iree_memory_order_relaxed);
      iree_hsa_queue_store_read_index_screlease(libhsa, worker->queue,
                                                ++read_index);

      // DO NOT SUBMIT
      switch (packet_type) {
        case HSA_PACKET_TYPE_AGENT_DISPATCH: {
        } break;
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return 0;
}

iree_status_t iree_hal_amdgpu_host_worker_initialize(
    iree_hal_amdgpu_system_t* system, hsa_agent_t agent,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_worker_t* out_worker) {
  IREE_ASSERT_ARGUMENT(out_worker);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_worker, 0, sizeof(*out_worker));
  out_worker->system = system;
  out_worker->agent = agent;

  // Pin the thread to the NUMA node specified.
  // We don't care which core but do want it to be one of those associated with
  // the devices this worker is servicing.
  iree_thread_affinity_t thread_affinity;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_host_agent_thread_affinity(&system->libhsa, agent,
                                                     &thread_affinity));

  // DO NOT SUBMIT
  iree_status_t status = iree_ok_status();

  // out_worker->queue;
  // out_worker->doorbell;

  // Create the worker thread for handling device library requests.
  // The worker may start immediately and use the queue/doorbell.
  if (iree_status_is_ok(status)) {
    uint32_t agent_node = 0;
    iree_hsa_agent_get_info(&system->libhsa, agent, HSA_AGENT_INFO_NODE,
                            &agent_node);
    char thread_name[32];
    snprintf(thread_name, IREE_ARRAYSIZE(thread_name), "iree-amdgpu-host-%d",
             agent_node);
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

  // DO NOT SUBMIT signal to UINT64_MAX?

  // Join thread after it has shut down.
  iree_thread_release(worker->thread);
  worker->thread = NULL;

  // DO NOT SUBMIT
  // worker->doorbell;
  // worker->queue;

  IREE_TRACE_ZONE_END(z0);
}
