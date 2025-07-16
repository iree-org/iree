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
#include "iree/hal/drivers/amdgpu/util/error_callback.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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
  // HSA library handle. Unowned.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Optional callback issued when the failure status is first set.
  iree_hal_amdgpu_error_callback_t error_callback;

  // If the service has received a fatal error from the device it will be stored
  // here as a status code to prevent duplicate error callbacks.
  iree_atomic_uint64_t failure_code;

  // OS handle to the worker thread.
  iree_thread_t* thread;

  // A semaphore signal used to indicate the number of outstanding asynchronous
  // operations. 0 in the idle state, incremented for each new asynchronous
  // operation, and decremented when an asynchronous operation completes.
  // Used to implement the packet barrier bit.
  hsa_signal_t outstanding_signal;

  // HSA soft queue for incoming requests from devices.
  hsa_queue_t* queue;
  // HSA doorbell indicating the queue has been updated.
  hsa_signal_t doorbell;
} iree_hal_amdgpu_host_service_t;

// Initializes the service state and launches the worker thread.
// |libhsa| must remain valid for the lifetime of the service.
//
// The |host_ordinal| and |device_ordinal| are used for naming the service
// worker thread. The worker thread will be pinned to the CPU |host_agent|
// affinity and have its underlying HSA queue allocated from |host_fine_region|.
//
// An optional |error_callback| can be provided to receive notification of the
// service entering the failure state. The callback may be issued from driver
// threads and must not re-enter the host service API or make any stateful HSA
// calls.
//
// TODO(benvanik): change device_ordinal to some other disambiguator if we
// decide to share host workers across devices. Today it is only used for
// thread/trace naming.
iree_status_t iree_hal_amdgpu_host_service_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_host_size_t host_ordinal,
    hsa_agent_t host_agent, hsa_region_t host_fine_region,
    iree_host_size_t device_ordinal,
    iree_hal_amdgpu_error_callback_t error_callback,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_service_t* out_service);

// Deinitializes the service and terminates the worker thread.
void iree_hal_amdgpu_host_service_deinitialize(
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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_SERVICE_H_
