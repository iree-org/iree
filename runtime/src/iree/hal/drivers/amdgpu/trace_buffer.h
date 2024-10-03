// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_TRACE_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_TRACE_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amdgpu/util/device_library.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

typedef struct iree_hal_amdgpu_device_trace_buffer_kernargs_t
    iree_hal_amdgpu_device_trace_buffer_kernargs_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_trace_buffer_t
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

typedef struct iree_hal_amdgpu_device_trace_buffer_t
    iree_hal_amdgpu_device_trace_buffer_t;

// Tracing ringbuffer and adapter layer.
// Trace buffers are per-scheduler and collect events from both the scheduler
// control queue (modeled as a CPU thread) and execution queue (modeled as a
// GPU).
//
// Thread-safe; any host thread may flush a trace buffer at any time to acquire
// the latest trace events and the device-side scheduler will request flushes
// when the buffer is full.
typedef struct iree_hal_amdgpu_trace_buffer_t {
  // Unowned libhsa handle. Must be retained by the owner.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // /dev/kfd file handle if needed by the platform.
  // This is a workaround for not having access to the clock counters via HSA.
  // When supported there we can drop this and just use that new API.
  int kfd_fd;

  // GPU agent this trace buffer is local to.
  hsa_agent_t device_agent;
  // HSA_AMD_AGENT_INFO_DRIVER_UID of the device agent.
  uint32_t device_driver_uid;

  // Base pointers of the device library code in host and device memory.
  // Used for translating rodata from the loaded ELF.
  iree_hal_amdgpu_code_range_t device_library_code_range;

  // Ringbuffer in device memory.
  iree_hal_amdgpu_vmem_ringbuffer_t ringbuffer;

  // Device-side trace buffer that is managed by the tracing infrastructure.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_trace_buffer_t* device_buffer;

  // Last read offset in the ringbuffer.
  // When flushing the ringbuffer will be consumed from this point up to the
  // write_commit_offset.
  iree_atomic_uint64_t last_read_offset;

  // Unowned kernel information used for populating control packets.
  // Must remain live for the lifetime of the trace buffer.
  const iree_hal_amdgpu_device_kernels_t* kernels;

  // Tracing context used for sinking events to a tracing backend.
  iree_tracing_context_t* tracing_context;
} iree_hal_amdgpu_trace_buffer_t;

#else

typedef struct iree_hal_amdgpu_trace_buffer_t {
  int reserved;
} iree_hal_amdgpu_trace_buffer_t;

#endif  //  IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

// Initializes a host and device-side trace buffer.
// The trace buffer will be allocated in |memory_pool| and be accessible by
// only the specified |device_agent| owning the pool producing trace events and
// |host_agent| that flushes them.
//
// The |kernargs| used for launching device-side trace buffer work must be
// allocated by the caller and will be populated with the pointer to the
// device-side trace buffer allocation.
//
// Asynchronous device initialization will be issued on |control_queue| and
// |initialization_signal| will be decremented once it has completed.
//
// NOTE: if initialization fails callers must call
// iree_hal_amdgpu_trace_buffer_deinitialize after |initialization_signal| is
// reached.
iree_status_t iree_hal_amdgpu_trace_buffer_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, int kfd_fd,
    iree_string_view_t executor_name, hsa_agent_t host_agent,
    hsa_agent_t device_agent, hsa_queue_t* control_queue,
    hsa_amd_memory_pool_t memory_pool, iree_device_size_t ringbuffer_capacity,
    const iree_hal_amdgpu_device_library_t* device_library,
    const iree_hal_amdgpu_device_kernels_t* kernels,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_trace_buffer_kernargs_t*
        kernargs,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_trace_buffer_t* out_trace_buffer);

// Deinitializes a trace buffer and deallocates the device-side ringbuffer
// resources. Any remaining trace events will be flushed but callers should
// prefer calling flush explicitly prior.
void iree_hal_amdgpu_trace_buffer_deinitialize(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer);

// Flushes all trace events currently committed in the ringbuffer.
iree_status_t iree_hal_amdgpu_trace_buffer_flush(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer);

#endif  // IREE_HAL_DRIVERS_AMDGPU_TRACE_BUFFER_H_
