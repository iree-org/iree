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

  // DO NOT SUBMIT
  (void)worker;

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
    const iree_thread_create_params_t thread_params = {
        // Developer-visible name for the thread displayed in tooling.
        // May be omitted for the system-default name (usually thread ID).
        .name = "XX",
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

  // Join thread after it has shut down.
  iree_thread_release(worker->thread);
  worker->thread = NULL;

  // DO NOT SUBMIT
  // worker->doorbell;
  // worker->queue;

  IREE_TRACE_ZONE_END(z0);
}
