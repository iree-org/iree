// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_HOST_CLIENT_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_HOST_CLIENT_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/queue.h"
#include "iree/hal/drivers/amdgpu/device/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_host_client_t
//===----------------------------------------------------------------------===//

typedef uint16_t iree_hal_amdgpu_device_host_call_t;
enum iree_hal_amdgpu_device_host_call_e {
  // Host will notify any registered listeners of the semaphore signal.
  // The semaphore provided is a host handle to a generic HAL semaphore and may
  // be of any device in the system - not just AMDGPU semaphores.
  //
  // Signature:
  //   arg0: iree_hal_semaphore_t* semaphore
  //   arg1: uint64_t payload
  //   arg2: unused
  //   arg3: unused
  //   return_address: unused
  //   completion_signal: unused
  IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_SIGNAL = 0u,

  // Host will call iree_hal_resource_release on each non-NULL resource pointer.
  // This is effectively a transfer operation indicating that the device will no
  // longer be using the resources.
  //
  // It's strongly recommended that iree_hal_resource_set_t is used where
  // appropriate so that the number of packets required to release a set of
  // resources can be kept small. The 4 available here is just enough for the
  // common case of submissions like execute that are a wait semaphore, the
  // command buffer, the binding table resource set, and the signal semaphore.
  //
  // TODO(benvanik): evaluate a version that takes a ringbuffer of uint64_t
  // pointers and make this a drain request instead. Then we can enqueue as many
  // as we want and kick the host to drain as it is able.
  //
  // Signature:
  //   arg0: iree_hal_resource_t* resource0
  //   arg1: iree_hal_resource_t* resource1
  //   arg2: iree_hal_resource_t* resource2
  //   arg3: iree_hal_resource_t* resource3
  //   return_address: unused
  //   completion_signal: optional, signaled when the release has completed
  IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RELEASE,
};

// Represents the host runtime thread that is managing host interrupts.
// One or more schedulers may share a single host queue. Any host calls that
// need to identify the scheduler or scheduler-related resources must pass those
// as arguments.
typedef struct iree_hal_amdgpu_device_host_client_t {
  // Host soft-queue processing device requests. May be servicing requests from
  // multiple device agents. Cached inline in device memory but references a
  // ringbuffer in host memory.
  iree_amd_cached_queue_t service_queue;
  // Optional trace buffer used when tracing infrastructure is available.
  iree_hal_amdgpu_device_trace_buffer_t* trace_buffer;
} iree_hal_amdgpu_device_host_client_t;

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Enqueues a unidirection host agent packet ("post").
// Since this is device->host only operation this only uses an acquire scope
// from the agent and releases to the entire system so the host agent can
// observe changes. The completion signal is optional and may be
// `iree_hsa_signal_null()`.
//
// NOTE: the barrier bit is set but the host processing is (today) synchronous
// with respect to other packets and generally only executes in FIFO order with
// respect to what each packet may affect anyway. We could tweak this in the
// future e.g. posts to flush a ringbuffer don't need to block and can be
// eagerly processed. Maybe. For non-post operations we'd rely on queue barrier
// packets.
//
// NOTE: we currently use the agent dispatch packet fields as intended (mostly)
// so that tooling that intercepts them can work. We don't have to, though, and
// could even have custom vendor packets instead to get the most bytes out of
// the channel.
void iree_hal_amdgpu_device_host_client_post(
    const iree_hal_amdgpu_device_host_client_t* IREE_AMDGPU_RESTRICT client,
    uint16_t type, uint64_t return_address, uint64_t arg0, uint64_t arg1,
    uint64_t arg2, uint64_t arg3, iree_hsa_signal_t completion_signal);

// Posts a semaphore signal notification to the host.
// This is only needed for external semaphores that are managed by the host.
void iree_hal_amdgpu_device_host_client_post_signal(
    const iree_hal_amdgpu_device_host_client_t* IREE_AMDGPU_RESTRICT client,
    uint64_t semaphore, uint64_t payload);

// Posts a multi-resource release request to the host.
// The host will call iree_hal_resource_release on each non-NULL resource
// pointer provided. The optional |completion_signal| will be signaled when the
// release has completed.
void iree_hal_amdgpu_device_host_client_post_release(
    const iree_hal_amdgpu_device_host_client_t* IREE_AMDGPU_RESTRICT client,
    uint64_t resource0, uint64_t resource1, uint64_t resource2,
    uint64_t resource3, iree_hsa_signal_t completion_signal);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_HOST_CLIENT_H_
