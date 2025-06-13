// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_SERVICE_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_SERVICE_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/threading.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_service_t
//===----------------------------------------------------------------------===//

// Capacity in entries of the host service queue.
// This should not need to be too large as each queue is only able to have ~64
// outstanding operations but each operation may require several host calls.
// Effectively this is the maximum queue count sharing a single service * the
// maximum concurrency on that queue * the maximum pipeline depth of any
// pipeline that may use host services.
#define IREE_HAL_AMDGPU_HOST_SERVICE_QUEUE_CAPACITY (1 * 1024)

// A host service managing requests from one or more device queues.
// Multiple physical devices or queues on a single physical device can share the
// same host service. Multiple service workers can be used to reduce latency on
// high core-count systems or locate the worker closer to the devices it manages
// in NUMA systems.
//
// Thread-safe.
typedef struct iree_hal_amdgpu_host_service_t {
  // System this worker is used on.
  iree_hal_amdgpu_system_t* system;

  // Host CPU agent the worker is pinned to.
  // This is likely the nearest agent to the devices managed by the worker.
  hsa_agent_t host_agent;
  // Ordinal of the CPU agent within the topology.
  iree_host_size_t host_ordinal;

  // OS handle to the worker thread.
  iree_thread_t* thread;

  // A semaphore signal used to indicate the number of outstanding asynchronous
  // operations. 0 in the idle state, incremented for each new asynchronous
  // operation, and decremented when an asynchronous operation completes.
  // Used to implement the packet barrier bit.
  hsa_signal_t outstanding_signal;

  // If the service has received a fatal error from the device it will be stored
  // here.
  iree_atomic_uint64_t failure_status;

  // HSA soft queue for incoming requests from devices.
  hsa_queue_t* queue;
  // HSA doorbell indicating the queue has been updated.
  hsa_signal_t doorbell;
} iree_hal_amdgpu_host_service_t;

// Initializes the service state and launches the worker thread.
//
// TODO(benvanik): change device_ordinal to some other disambiguator if we
// decide to share host workers across devices.
iree_status_t iree_hal_amdgpu_host_service_initialize(
    iree_hal_amdgpu_system_t* system, iree_host_size_t host_ordinal,
    iree_host_size_t device_ordinal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_service_t* out_service);

// Deinitializes the service and terminates the worker thread.
void iree_hal_amdgpu_host_service_deinitialize(
    iree_hal_amdgpu_host_service_t* service);

// Returns a failure status if it is set on the |service|.
// The caller must propagate the status and assume the worker is dead if it
// returns a failure. Safe to call from any thread. The failure status is sticky
// and will be cloned and returned on all queries for the lifetime of the
// service.
iree_status_t iree_hal_amdgpu_host_service_query_status(
    iree_hal_amdgpu_host_service_t* service);

// An asynchronous host operation token.
typedef struct iree_hal_amdgpu_host_async_token_t {
  iree_hal_amdgpu_host_service_t* service;
} iree_hal_amdgpu_host_async_token_t;

// Notifies the service that |async_token| originated from that an asynchronous
// operation has completed. May be called from any thread. Must only be called
// once per asynchronous operation. The worker may be immediately deallocated
// after exiting and this should almost always be a tail call. If the operation
// failed a |status| can be provided and will be consumed by the call.
void iree_hal_amdgpu_host_service_notify_completion(
    iree_hal_amdgpu_host_async_token_t async_token, iree_status_t status);

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_SERVICE_H_
