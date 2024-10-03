// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/queue.h"

#include "iree/base/internal/debugging.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/buffer_pool.h"
#include "iree/hal/drivers/amdgpu/command_buffer.h"
#include "iree/hal/drivers/amdgpu/device/scheduler.h"
#include "iree/hal/drivers/amdgpu/device/semaphore.h"
#include "iree/hal/drivers/amdgpu/host_worker.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/block_pool.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_options_t
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_queue_options_initialize(
    iree_hal_amdgpu_queue_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->flags = IREE_HAL_AMDGPU_QUEUE_FLAG_NONE;
  out_options->mode = IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULING_MODE_DEFAULT;
  out_options->control_queue_capacity =
      IREE_HAL_AMDGPU_DEFAULT_CONTROL_QUEUE_CAPACITY;
  out_options->execution_queue_count =
      IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_COUNT;
  out_options->execution_queue_capacity =
      IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_CAPACITY;
  out_options->kernarg_ringbuffer_capacity =
      IREE_HAL_AMDGPU_DEFAULT_KERNARG_RINGBUFFER_CAPACITY;
  out_options->trace_buffer_capacity =
      IREE_HAL_AMDGPU_DEFAULT_TRACE_BUFFER_CAPACITY;
}

// Verifies that the given |queue_capacity| is between the agent min/max
// requirements and a power-of-two.
static iree_status_t iree_hal_amdgpu_verify_hsa_queue_size(
    iree_string_view_t queue_name, iree_host_size_t queue_size,
    uint32_t queue_min_size, uint32_t queue_max_size) {
  // Queues must meet the min/max size requirements.
  if (queue_size < queue_min_size || queue_size > queue_max_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s queue capacity on this agent must be between "
        "HSA_AGENT_INFO_QUEUE_MIN_SIZE=%u and HSA_AGENT_INFO_QUEUE_MAX_SIZE=%u "
        "(provided %" PRIhsz ")",
        (int)queue_name.size, queue_name.data, queue_min_size, queue_max_size,
        queue_size);
  }

  // All queues must be a power-of-two due to ringbuffer masking.
  if (!iree_host_size_is_power_of_two(queue_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s queue capacity must be a power of two (provided %" PRIhsz ")",
        (int)queue_name.size, queue_name.data, queue_size);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_options_verify(
    const iree_hal_amdgpu_queue_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(libhsa);

  // Currently only one execution queue is supported.
  if (options->execution_queue_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "multiple hardware execution queues per HAL queue "
                            "are not yet implemented");
  }

  // Query agent min/max queue size.
  uint32_t queue_min_size = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), agent,
                                               HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                                               &queue_min_size));
  uint32_t queue_max_size = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), agent,
                                               HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                                               &queue_max_size));

  // Verify HSA queues.
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_verify_hsa_queue_size(
      IREE_SV("control"), options->control_queue_capacity, queue_min_size,
      queue_max_size));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_verify_hsa_queue_size(
      IREE_SV("execution"), options->execution_queue_capacity, queue_min_size,
      queue_max_size));

  // Verify kernarg ringbuffer capacity (our ringbuffer so no HSA min/max
  // required).
  if (!iree_device_size_is_power_of_two(options->kernarg_ringbuffer_capacity)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernarg ringbuffer capacity must be a power of two (provided %" PRIdsz
        ")",
        options->kernarg_ringbuffer_capacity);
  }

  // Verify trace buffer capacity (our ringbuffer so no HSA min/max required).
  if (options->trace_buffer_capacity &&
      !iree_device_size_is_power_of_two(options->trace_buffer_capacity)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "trace buffer capacity must be a power of two (provided %" PRIdsz ")",
        options->trace_buffer_capacity);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// HAL API Utilities
//===----------------------------------------------------------------------===//

// Resolves a HAL buffer to a device-side buffer reference.
// Verifies (roughly) that it's usable but not that it's accessible to any
// particular agent.
static iree_status_t iree_hal_amdgpu_resolve_buffer_ref(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_amdgpu_device_buffer_ref_t* out_ref) {
  iree_hal_amdgpu_device_buffer_type_t type = 0;
  uint64_t bits = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer(buffer, &type, &bits));
  out_ref->type = type;
  out_ref->offset = offset;
  out_ref->length = length;
  out_ref->value.bits = bits;
  return iree_ok_status();
}

// Resolves a HAL buffer binding to a device-side buffer reference.
// Verifies (roughly) that it's usable but not that it's accessible to any
// particular agent.
static iree_status_t iree_hal_amdgpu_resolve_binding(
    iree_hal_buffer_binding_t binding,
    iree_hal_amdgpu_device_buffer_ref_t* out_device_ref) {
  iree_hal_amdgpu_device_buffer_type_t type = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer(
      binding.buffer, &type, &out_device_ref->value.bits));
  out_device_ref->type = type;
  out_device_ref->offset = binding.offset;
  out_device_ref->length =
      binding.length != IREE_HAL_WHOLE_BUFFER
          ? binding.length
          : iree_hal_buffer_byte_length(binding.buffer) - binding.offset;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Inlined HSA Functions
//===----------------------------------------------------------------------===//
// These functions are on our critical path and we don't want to take an extern
// function pointer call that prevents all compiler optimization on each. Only
// if we statically linked HSA and ran LTO would this possibly be optimizable as
// a call. For all modern AMD GPU targets these are just simple atomic
// operations and the cache misses involved in walking the multiple pointers out
// to the final vtabled implementations cost many times more than the operations
// themselves.

// Inlined copy of hsa_queue_load_read_index_scacquire.
static uint64_t iree_hsa_queue_load_read_index_scacquire_inline(
    hsa_queue_t* IREE_AMDGPU_RESTRICT queue) {
  iree_amd_queue_t* amd_queue = (iree_amd_queue_t*)queue;
  return iree_atomic_load((iree_atomic_uint64_t*)&amd_queue->read_dispatch_id,
                          iree_memory_order_acquire);
}

// Inlined copy of hsa_queue_add_write_index_screlease.
static uint64_t iree_hsa_queue_add_write_index_screlease_inline(
    hsa_queue_t* IREE_AMDGPU_RESTRICT queue, uint64_t value) {
  iree_amd_queue_t* amd_queue = (iree_amd_queue_t*)queue;
  return iree_atomic_fetch_add(
      (iree_atomic_uint64_t*)&amd_queue->write_dispatch_id, value,
      iree_memory_order_release);
}

// Inlined copy of hsa_signal_store_relaxed for doorbell signals.
static void iree_hsa_signal_store_relaxed_doorbell_inline(hsa_signal_t signal,
                                                          uint64_t value) {
  iree_amd_signal_t* amd_signal = (iree_amd_signal_t*)signal.handle;
  iree_atomic_store((iree_atomic_uint64_t*)amd_signal->hardware_doorbell_ptr,
                    value, iree_memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// HSA Utilities
//===----------------------------------------------------------------------===//

// Launches a kernel on the given |queue|.
// |kernarg_address| must point to a kernarg region.
// A full barrier will be set with system/system fences.
// |completion_signal| will be incremented when enqueuing the operation and
// decremented when the operation completes.
void iree_hal_amdgpu_kernel_dispatch(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_queue_t* queue,
    IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t*
        kernel_args,
    const uint32_t grid_size[3], IREE_AMDGPU_DEVICE_PTR void* kernarg_address,
    hsa_signal_t completion_signal) {
  // Construct the packet in host memory where we don't care about write
  // order/caches.
  hsa_kernel_dispatch_packet_t packet = {
      .header = HSA_PACKET_TYPE_INVALID,
      .setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS,
      .workgroup_size_x = kernel_args->workgroup_size[0],
      .workgroup_size_y = kernel_args->workgroup_size[1],
      .workgroup_size_z = kernel_args->workgroup_size[2],
      .reserved0 = 0,
      .grid_size_x = grid_size[0],
      .grid_size_y = grid_size[1],
      .grid_size_z = grid_size[2],
      .private_segment_size = kernel_args->private_segment_size,
      .group_segment_size = kernel_args->group_segment_size,
      .kernel_object = kernel_args->kernel_object,
      .kernarg_address = kernarg_address,
      .completion_signal = completion_signal,
  };
  const uint16_t packet_header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  const uint32_t packet_header_setup = packet_header | (packet.setup << 16);

  // Increment signal prior to issuing.
  if (completion_signal.handle) {
    iree_hsa_signal_add_relaxed(IREE_LIBHSA(libhsa), completion_signal, 1u);
  }

  // Acquire a queue slot (spin if full).
  const uint64_t packet_id =
      iree_hsa_queue_add_write_index_screlease_inline(queue, 1);
  while ((packet_id - iree_hsa_queue_load_read_index_scacquire_inline(queue)) >=
         queue->size) {
    iree_thread_yield();
  }
  hsa_kernel_dispatch_packet_t* packet_ptr =
      (hsa_kernel_dispatch_packet_t*)queue->base_address +
      (packet_id & (queue->size - 1));

  // Copy the packet from host memory into kernarg memory.
  iree_memcpy_stream_dst(packet_ptr, &packet, sizeof(packet));

  // Mark the packet as ready (INVALID->KERNEL_DISPATCH).
  // The command processor may immediately begin executing it.
  iree_atomic_store((iree_atomic_uint32_t*)packet_ptr, packet_header_setup,
                    iree_memory_order_release);

  // Notify the command processor the packet is available.
  iree_hsa_signal_store_relaxed_doorbell_inline(queue->doorbell_signal,
                                                packet_id);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_t
//===----------------------------------------------------------------------===//

// Handles HSA queue error callbacks.
static void iree_hal_amdgpu_queue_hsa_callback(hsa_status_t status,
                                               hsa_queue_t* queue,
                                               void* user_data) {
  iree_hal_amdgpu_queue_t* parent_queue = (iree_hal_amdgpu_queue_t*)user_data;
  // DO NOT SUBMIT HSA queue callback error handler
  iree_debug_break();
  if (queue == parent_queue->control_queue) {
    // The primary control queue.
  } else {
    // An execution queue.
  }
}

// Creates the control queue used to perform scheduling and queue management.
// This runs at a high priority and (hopefully) infrequently.
static iree_status_t iree_hal_amdgpu_create_control_queue(
    iree_hal_amdgpu_system_t* system, iree_hal_amdgpu_queue_t* parent_queue,
    hsa_agent_t device_agent, hsa_amd_memory_pool_t memory_pool,
    iree_host_size_t queue_capacity, hsa_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_ARGUMENT(out_queue);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_queue = NULL;

  const iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;

  // Create the queue in MULTI-mode as we may have control dispatches enqueued
  // from any agent (host or device).
  //
  // NOTE: for control queues we could likely set the private/group segment
  // sizes to something small.
  hsa_queue_t* queue = NULL;
  iree_status_t status = iree_hsa_queue_create(
      IREE_LIBHSA(libhsa), device_agent, queue_capacity, HSA_QUEUE_TYPE_MULTI,
      iree_hal_amdgpu_queue_hsa_callback,
      /*callback_data=*/parent_queue,
      /*private_segment_size=*/UINT32_MAX,
      /*group_segment_size=*/UINT32_MAX, &queue);

  // Set queue priority hint to above all other default queues.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_queue_set_priority(IREE_LIBHSA(libhsa), queue,
                                             HSA_AMD_QUEUE_PRIORITY_HIGH);
  }

  if (iree_status_is_ok(status)) {
    *out_queue = queue;
  } else {
    if (queue) {
      IREE_IGNORE_ERROR(iree_hsa_queue_destroy(IREE_LIBHSA(libhsa), queue));
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_create_execution_queue(
    iree_hal_amdgpu_system_t* system, iree_hal_amdgpu_queue_t* parent_queue,
    hsa_agent_t device_agent, hsa_amd_memory_pool_t memory_pool,
    iree_host_size_t queue_capacity, hsa_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_ARGUMENT(out_queue);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_queue = NULL;

  const iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;

  // Create the queue in MULTI-mode because though we are only enqueuing work
  // from the local agent we may do so concurrently from several workgroups at a
  // time during command buffer issue and want the packet processor behavior
  // associated with MUTLI.
  hsa_queue_t* queue = NULL;
  iree_status_t status = iree_hsa_queue_create(
      IREE_LIBHSA(libhsa), device_agent, queue_capacity, HSA_QUEUE_TYPE_MULTI,
      iree_hal_amdgpu_queue_hsa_callback,
      /*callback_data=*/parent_queue,
      /*private_segment_size=*/UINT32_MAX,
      /*group_segment_size=*/UINT32_MAX, &queue);

  // Enable profiling so that we can get timestamps from signals.
  if (iree_status_is_ok(status)) {
    const bool profiler_enabled = iree_all_bits_set(
        parent_queue->flags, IREE_HAL_AMDGPU_QUEUE_FLAG_TRACE_EXECUTION);
    status = iree_hsa_amd_profiling_set_profiler_enabled(
        IREE_LIBHSA(libhsa), queue, profiler_enabled ? 1 : 0);
  }

  if (iree_status_is_ok(status)) {
    *out_queue = queue;
  } else {
    if (queue) {
      IREE_IGNORE_ERROR(iree_hsa_queue_destroy(IREE_LIBHSA(libhsa), queue));
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_allocate_scheduler(
    const iree_hal_amdgpu_queue_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t host_agent,
    hsa_agent_t device_agent, hsa_amd_memory_pool_t memory_pool,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_scheduler_t**
        out_scheduler,
    iree_hal_amdgpu_queue_scheduler_ptrs_t* out_scheduler_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_scheduler = NULL;
  memset(out_scheduler_ptrs, 0, sizeof(*out_scheduler_ptrs));

  const iree_host_size_t kernargs_offset =
      iree_host_align(sizeof(iree_hal_amdgpu_device_queue_scheduler_t), 64);
  const iree_host_size_t total_size =
      kernargs_offset +
      sizeof(iree_hal_amdgpu_device_queue_scheduler_kernargs_t) +
      options->execution_queue_count * sizeof(iree_hsa_queue_t*);
  uint8_t* slab = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_memory_pool_allocate(
              IREE_LIBHSA(libhsa), memory_pool, total_size,
              HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&slab));

  hsa_agent_t access_agents[2] = {
      host_agent,
      device_agent,
  };
  iree_status_t status = iree_hsa_amd_agents_allow_access(
      IREE_LIBHSA(libhsa), IREE_ARRAYSIZE(access_agents), access_agents, NULL,
      slab);

  if (iree_status_is_ok(status)) {
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_scheduler_t* scheduler =
        (iree_hal_amdgpu_device_queue_scheduler_t*)slab;
    *out_scheduler = scheduler;
    out_scheduler_ptrs->mailbox = &scheduler->mailbox;
    out_scheduler_ptrs->tick_action_set = &scheduler->tick_action_set;
    out_scheduler_ptrs->active_set = &scheduler->active_set;
    out_scheduler_ptrs->control_kernargs =
        (iree_hal_amdgpu_device_queue_scheduler_kernargs_t*)(slab +
                                                             kernargs_offset);
  } else {
    IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), slab));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_free_scheduler(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_device_queue_scheduler_t* scheduler) {
  if (!scheduler) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_IGNORE_ERROR(
      iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), scheduler));

  IREE_TRACE_ZONE_END(z0);
}

// Asynchronously dispatches the device-side scheduler initializer kernel:
// `iree_hal_amdgpu_device_queue_scheduler_initialize`.
static void iree_hal_amdgpu_initialize_scheduler(
    iree_hal_amdgpu_queue_t* queue,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_allocator_t* device_allocator,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_scheduler_kernargs_t*
        kernargs,
    hsa_signal_t completion_signal) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Distributed by groups of signals across all signals.
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
      &queue->kernels.iree_hal_amdgpu_device_queue_scheduler_initialize;
  const uint32_t grid_size[3] = {1, 1, 1};

  // Populate parameters prior to issuing.
  // TODO(benvanik): use an HSA API for copying this? Seems to work as-is.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_scheduler_params_t*
      params = &kernargs->storage.params;
  kernargs->scheduler = queue->scheduler;
  kernargs->params = params;

  params->mode = queue->mode;
  params->queue = (uint64_t)queue;
  params->host.queue = (iree_hsa_queue_t*)queue->host_worker->queue;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  params->host.trace_buffer = queue->trace_buffer.device_buffer;
#else
  params->host.trace_buffer = NULL;
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  params->allocator = device_allocator;
  params->control_kernargs = kernargs;

  params->kernarg_ringbuffer.base_ptr = queue->kernarg_ringbuffer.ring_base_ptr;
  params->kernarg_ringbuffer.capacity = queue->kernarg_ringbuffer.capacity;

  params->control_queue = (iree_hsa_queue_t*)queue->control_queue;
  params->execution_queue_count = queue->execution_queue_count;
  for (iree_host_size_t i = 0; i < queue->execution_queue_count; ++i) {
    params->execution_queues[i] = (iree_hsa_queue_t*)queue->execution_queues[i];
  }

  // Populate kernel information and object pointers for the agent this queue is
  // running on. Each agent will have its own kernel object pointers.
  memcpy(&params->kernels, &queue->kernels, sizeof(params->kernels));

  // TODO(benvanik): initialize the pool. It may not even be required.
  params->signal_pool.ptr = NULL;
  params->signal_pool.count = 0;
  params->signal_pool.values = NULL;

  // Dispatch the initialization kernel asynchronously.
  iree_hal_amdgpu_kernel_dispatch(&queue->system->libhsa, queue->control_queue,
                                  kernel_args, grid_size, kernargs,
                                  completion_signal);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_queue_initialize(
    iree_hal_amdgpu_queue_options_t options, iree_hal_amdgpu_system_t* system,
    hsa_agent_t device_agent, iree_host_size_t device_ordinal,
    iree_hal_amdgpu_device_allocator_t* device_allocator,
    iree_hal_amdgpu_host_worker_t* host_worker,
    iree_arena_block_pool_t* host_block_pool,
    iree_hal_amdgpu_block_allocators_t* block_allocators,
    iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_ARGUMENT(host_worker);
  IREE_ASSERT_ARGUMENT(host_block_pool);
  IREE_ASSERT_ARGUMENT(block_allocators);
  IREE_ASSERT_ARGUMENT(buffer_pool);
  IREE_ASSERT_ARGUMENT(out_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;

  memset(out_queue, 0, sizeof(*out_queue));

  out_queue->flags = options.flags;
  out_queue->mode = options.mode;
  out_queue->system = system;
  out_queue->device_agent = device_agent;
  out_queue->device_ordinal = device_ordinal;
  out_queue->host_worker = host_worker;
  out_queue->host_block_pool = host_block_pool;
  out_queue->block_allocators = block_allocators;
  out_queue->buffer_pool = buffer_pool;
  out_queue->execution_queue_count = options.execution_queue_count;

  // Verify configuration.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_queue_options_verify(&options, libhsa, device_agent));

  // Find a fine-grained memory pool on the agent used for our device-side
  // allocations.
  hsa_amd_memory_pool_t fine_memory_pool = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_find_fine_global_memory_pool(libhsa, device_agent,
                                                       &fine_memory_pool));

  // Create the HSA queues used for control and execution.
  // We have different queues as prioritization is per-queue and we want to
  // ensure the control queue has a higher priority. We could share a queue at
  // the risk of higher latency.
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_create_control_queue(
        system, out_queue, device_agent, fine_memory_pool,
        options.control_queue_capacity, &out_queue->control_queue);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < out_queue->execution_queue_count; ++i) {
      status = iree_hal_amdgpu_create_execution_queue(
          system, out_queue, device_agent, fine_memory_pool,
          options.execution_queue_capacity, &out_queue->execution_queues[i]);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Start idle (0).
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(IREE_LIBHSA(libhsa), 0ull, 0, NULL, 0,
                                        &out_queue->idle_signal);
  }

  // Populate kernel information and object pointers for the agent this queue is
  // running on. Each agent will have its own kernel object pointers.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_library_populate_agent_kernels(
        &out_queue->system->device_library, device_agent, &out_queue->kernels);
  }

  // Allocate the kernarg ringbuffer storage for exclusive access on the device.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_vmem_ringbuffer_initialize_with_topology(
        libhsa, device_agent, fine_memory_pool,
        options.kernarg_ringbuffer_capacity, &system->topology,
        IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE, &out_queue->kernarg_ringbuffer);
  }

  // Allocate device memory for the device-side scheduler.
  // We don't need to touch it from the host in the steady-state as we always go
  // through the soft queue.
  //
  // Various control kernarg storage is allocated as part of the scheduler
  // allocation and returned here so we can populate it. It's immutable once
  // populated and remains allocated on device for the lifetime of the
  // scheduler.
  iree_hal_amdgpu_device_queue_scheduler_kernargs_t* scheduler_kernargs = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_allocate_scheduler(
        &options, libhsa, host_worker->host_agent, device_agent,
        fine_memory_pool, &out_queue->scheduler, &out_queue->scheduler_ptrs);
    scheduler_kernargs = out_queue->scheduler_ptrs.control_kernargs;
  }

  // Initialize the per-queue tracing buffer, if tracing is enabled (and
  // otherwise this will no-op).
  if (iree_status_is_ok(status)) {
    iree_string_view_t executor_name = iree_string_view_empty();
    IREE_TRACE({
      static iree_atomic_int32_t queue_id = IREE_ATOMIC_VAR_INIT(0);
      char queue_name[32];
      int queue_name_length = snprintf(
          queue_name, sizeof(executor_name), "iree-amdgpu-q%d",
          iree_atomic_fetch_add(&queue_id, 1, iree_memory_order_seq_cst));
      IREE_LEAK_CHECK_DISABLE_PUSH();
      executor_name.data = malloc(queue_name_length + 1);
      executor_name.size = queue_name_length;
      memcpy((void*)executor_name.data, queue_name, queue_name_length + 1);
      IREE_LEAK_CHECK_DISABLE_POP();
    });
    status = iree_hal_amdgpu_trace_buffer_initialize(
        libhsa, system->kfd_fd, executor_name, host_worker->host_agent,
        device_agent, out_queue->control_queue, fine_memory_pool,
        options.trace_buffer_capacity, &system->device_library,
        &out_queue->kernels, &scheduler_kernargs->storage.trace_buffer,
        initialization_signal, host_allocator, &out_queue->trace_buffer);
  }

  // Asynchronously initialize the device scheduler.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_initialize_scheduler(
        out_queue, device_allocator, scheduler_kernargs, initialization_signal);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_queue_deinitialize(iree_hal_amdgpu_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdgpu_libhsa_t* libhsa = &queue->system->libhsa;

  // Wait for the queue to be idle.
  // TODO(benvanik): timeout and abort if it never decrements.
  // For now we just... hope.
  if (queue->idle_signal.handle) {
    iree_hsa_signal_wait_scacquire(IREE_LIBHSA(libhsa), queue->idle_signal,
                                   HSA_SIGNAL_CONDITION_LT, 1u, UINT64_MAX,
                                   HSA_WAIT_STATE_BLOCKED);
  }

  iree_hal_amdgpu_trace_buffer_deinitialize(&queue->trace_buffer);

  iree_hal_amdgpu_free_scheduler(libhsa, queue->scheduler);

  iree_hal_amdgpu_vmem_ringbuffer_deinitialize(libhsa,
                                               &queue->kernarg_ringbuffer);

  if (queue->idle_signal.handle) {
    IREE_IGNORE_ERROR(
        iree_hsa_signal_destroy(IREE_LIBHSA(libhsa), queue->idle_signal));
  }

  for (iree_host_size_t i = 0; i < queue->execution_queue_count; ++i) {
    hsa_queue_t* execution_queue = queue->execution_queues[i];
    if (execution_queue) {
      IREE_IGNORE_ERROR(
          iree_hsa_queue_destroy(IREE_LIBHSA(libhsa), execution_queue));
    }
  }
  if (queue->control_queue) {
    IREE_IGNORE_ERROR(
        iree_hsa_queue_destroy(IREE_LIBHSA(libhsa), queue->control_queue));
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amdgpu_queue_trim(iree_hal_amdgpu_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  // No-op today but if we allocated things per-queue (related to
  // tracing/profiling, most-likely) we'd trim them here.
}

// Requests that the scheduler run one tick if one is not already pending.
// The |action_set| will be OR'ed into the scheduler pending actions set.
static void iree_hal_amdgpu_queue_request_tick(
    iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_tick_action_set_t action_set) {
  IREE_ASSERT_ARGUMENT(queue);

#if 0
  // OR in the tick action request so that the scheduler knows to check the
  // mailbox. The returned value prior to the atomic OR lets us know if a tick
  // was pending - if one already was then we don't need to schedule a kernel
  // launch as our new work will be picked up as the tick proceeds. This only
  // works because the first thing the tick does is check and clear this value.
  //
  // Unfortunately the stall incurred reading the device memory in order to
  // evaluate the condition and exit early can be as large as 2us and that's
  // longer than it takes to actually enqueue the tick.
  //
  // DO NOT SUBMIT re-evaluate with the control queue in device memory - the
  // tick enqueue may only be fast because the queue is in host-local memory and
  // once we switch it could be several times slower.
  if (IREE_UNLIKELY(iree_amdgpu_scoped_atomic_fetch_or(
                        queue->scheduler_ptrs.tick_action_set, action_set,
                        iree_amdgpu_memory_order_release,
                        iree_amdgpu_memory_scope_system) !=
                    IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_NONE)) {
    return;  // tick already pending
  }
#else
  iree_amdgpu_scoped_atomic_fetch_or(
      queue->scheduler_ptrs.tick_action_set, action_set,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_system);
#endif

  // We could use the signal to ensure cleaner shutdowns by incrementing it on
  // each tick submission. That'd be extra work on top of the tick action so
  // today unless we need it we avoid it.
  hsa_signal_t completion_signal = {0};
  const uint32_t grid_size[3] = {1, 1, 1};
  iree_hal_amdgpu_kernel_dispatch(
      &queue->system->libhsa, queue->control_queue,
      &queue->kernels.iree_hal_amdgpu_device_queue_scheduler_tick, grid_size,
      queue->scheduler_ptrs.control_kernargs, completion_signal);
}

// Reserves a new queue entry of the given |type|.
// At least |base_size| bytes will be allocated for the queue entry in addition
// to any internal allocations like the semaphore lists which will be tacked
// onto the end.
//
// If |out_resource_set| is provided a resource set will be acquired from the
// block pool and returned to the caller. The resource set will be freed when
// the queue entry retires.
static iree_status_t iree_hal_amdgpu_queue_reserve_entry(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_device_queue_entry_type_t type, iree_host_size_t base_size,
    iree_host_size_t max_kernarg_capacity,
    IREE_AMDGPU_DEVICE_PTR void** out_entry,
    iree_hal_resource_set_t** out_resource_set) {
  *out_entry = NULL;
  if (out_resource_set) *out_resource_set = NULL;

  // There's a lot going on here. We need at least two allocations: the queue
  // entry in device-visible memory and the resource set tracking lifetime of
  // all referenced resources in host memory. The hope is that we have a 100%
  // hit rate in the pools in the steady state and this boils down to mostly the
  // pointer math to ensure we stick to only two allocations.

  // Allocate queue entry on the device.
  const iree_host_size_t wait_list_size =
      iree_hal_amdgpu_device_semaphore_list_size(wait_semaphore_list.count);
  const iree_host_size_t signal_list_size =
      iree_hal_amdgpu_device_semaphore_list_size(signal_semaphore_list.count);
  const iree_host_size_t total_size =
      base_size + wait_list_size + signal_list_size;
  iree_hal_amdgpu_block_allocator_t* block_allocator =
      total_size >= queue->block_allocators->small.page_size
          ? &queue->block_allocators->large
          : &queue->block_allocators->small;
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_entry_header_t* entry =
      NULL;
  iree_hal_amdgpu_block_token_t entry_token = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_block_allocator_allocate(
      block_allocator, total_size, (void**)&entry, &entry_token));
  entry->type = type;
  entry->flags = IREE_HAL_AMDGPU_DEVICE_DEVICE_QUEUE_ENTRY_FLAG_NONE;
  entry->allocation_token = entry_token.bits;
  entry->allocation_pool =
      block_allocator == &queue->block_allocators->large ? 1 : 0;

  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_list_t* wait_list =
      (iree_hal_amdgpu_device_semaphore_list_t*)((uint8_t*)entry + base_size);
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_list_t* signal_list =
      (iree_hal_amdgpu_device_semaphore_list_t*)((uint8_t*)entry + base_size +
                                                 wait_list_size);

  // Used to reserve kernarg space on device. Not allocated until issued.
  entry->max_kernarg_capacity = max_kernarg_capacity;

  entry->epoch = 0;             // managed by device scheduler on accept
  entry->active_bit_index = 0;  // managed by device scheduler on issue
  entry->kernarg_offset = 0;    // managed by device scheduler on issue
  entry->list_next = NULL;      // managed by device scheduler

  // Allocate a resource set used to track all of the resources associated with
  // the entry.
  iree_hal_resource_set_t* resource_set = NULL;
  iree_status_t status =
      iree_hal_resource_set_allocate(queue->host_block_pool, &resource_set);
  if (iree_status_is_ok(status) && wait_semaphore_list.count > 0) {
    status =
        iree_hal_resource_set_insert(resource_set, wait_semaphore_list.count,
                                     &wait_semaphore_list.semaphores[0]);
  }
  if (iree_status_is_ok(status) && signal_semaphore_list.count > 0) {
    status =
        iree_hal_resource_set_insert(resource_set, signal_semaphore_list.count,
                                     &signal_semaphore_list.semaphores[0]);
  }

  // Translate the semaphores referenced in the wait/signal lists to device-side
  // semaphore handles. This may fail if any semaphore is incompatible with the
  // device.
  entry->wait_list = wait_list;
  wait_list->count = (uint16_t)wait_semaphore_list.count;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < wait_semaphore_list.count; ++i) {
    status = iree_hal_amdgpu_semaphore_handle(wait_semaphore_list.semaphores[i],
                                              &wait_list->entries[i].semaphore);
    wait_list->entries[i].payload = wait_semaphore_list.payload_values[i];
  }
  entry->signal_list = signal_list;
  signal_list->count = (uint16_t)signal_semaphore_list.count;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < signal_semaphore_list.count; ++i) {
    status =
        iree_hal_amdgpu_semaphore_handle(signal_semaphore_list.semaphores[i],
                                         &signal_list->entries[i].semaphore);
    signal_list->entries[i].payload = signal_semaphore_list.payload_values[i];
  }

  if (iree_status_is_ok(status)) {
    entry->resource_set = (uint64_t)resource_set;
    *out_entry = entry;
    if (out_resource_set) *out_resource_set = resource_set;
  } else {
    if (resource_set) {
      iree_hal_resource_set_free(resource_set);
    }
    iree_hal_amdgpu_block_allocator_free(block_allocator, entry, entry_token);
  }
  return status;
}

// NOTE: this accesses device memory to perform the free and will be slow.
// Consider this useful for error handling cleanup only.
static void iree_hal_amdgpu_queue_free_entry(
    iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* entry) {
  if (!entry) return;

  // Free all resources retained by the entry (including the semaphore list).
  iree_hal_resource_set_free((iree_hal_resource_set_t*)entry->resource_set);

  // Free the allocation holding the queue entry back to the block pool.
  iree_hal_amdgpu_block_token_t entry_token = {entry->allocation_token};
  iree_hal_amdgpu_block_allocator_free(entry->allocation_pool
                                           ? &queue->block_allocators->large
                                           : &queue->block_allocators->small,
                                       entry, entry_token);
}

static void iree_hal_amdgpu_queue_commit_entry(
    iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Acquire a mailbox slot (spin if full).
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_mailbox_t* mailbox =
      queue->scheduler_ptrs.mailbox;
  const uint64_t entry_index = iree_atomic_fetch_add(&mailbox->write_index, 1,
                                                     iree_memory_order_release);
  while ((entry_index -
          iree_atomic_load(&mailbox->read_index, iree_memory_order_acquire)) >=
         IREE_ARRAYSIZE(mailbox->entries)) {
    iree_thread_yield();
  }
  const uint64_t entry_mask = IREE_ARRAYSIZE(mailbox->entries) - 1;
  IREE_AMDGPU_DEVICE_PTR iree_atomic_uint64_t* entry_ptr =
      (iree_atomic_uint64_t*)&mailbox->entries[entry_index & entry_mask];

  // Spin until the slot is available for use - the scheduler should be draining
  // it ASAP and changing it to INVALID when it no longer needs it.
  uint64_t invalid_entry = IREE_HAL_AMDGPU_DEVICE_MAILBOX_ENTRY_INVALID;
  while (!iree_atomic_compare_exchange_strong(
      entry_ptr, &invalid_entry, (uint64_t)entry, iree_memory_order_acq_rel,
      iree_memory_order_relaxed)) {
    iree_thread_yield();
  }

  // Kick off a scheduler run if one is not already pending.
  iree_hal_amdgpu_queue_request_tick(
      queue, IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_INCOMING);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amdgpu_queue_request_retire(
    iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* entry) {
  // Mark the queue entry as retired. The device-side scheduler may immediately
  // reclaim it if it is already running a tick.
  //
  // NOTE: this touches device memory and may be very slow.
  iree_amdgpu_scoped_atomic_fetch_or(
      &queue->scheduler_ptrs.active_set->retire_bits,
      1ul << entry->active_bit_index, iree_amdgpu_memory_order_release,
      iree_amdgpu_memory_scope_system);

  // Request a tick if one is not already pending. Ticks will always scan for
  // retired entries.
  iree_hal_amdgpu_queue_request_tick(
      queue, IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_RETIRE);
}

iree_status_t iree_hal_amdgpu_queue_retire_entry(
    iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* entry, bool has_signals,
    uint32_t allocation_pool, uint64_t allocation_token_bits,
    iree_hal_resource_set_t* resource_set) {
  // Signal any semaphores that were not able to be signaled by the device.
  // NOTE: this is going to read device memory and should be avoided unless
  // has_signals indicates we have signals to process.
  if (has_signals) {
    const iree_host_size_t signal_count = entry->signal_list->count;
    for (iree_host_size_t i = 0; i < signal_count; ++i) {
      // TODO(benvanik): external/callback semaphores.
      IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_t* semaphore =
          entry->signal_list->entries[i].semaphore;
      const uint64_t payload = entry->signal_list->entries[i].payload;
      IREE_RETURN_IF_ERROR(
          iree_hal_semaphore_signal(
              (iree_hal_semaphore_t*)semaphore->host_semaphore, payload),
          "signaling host-only semaphore");
    }
  }

  // Free all resources retained by the entry (including the semaphore list).
  iree_hal_resource_set_free(resource_set);

  // Free the allocation holding the queue entry back to the block pool.
  iree_hal_amdgpu_block_token_t entry_token = {allocation_token_bits};
  iree_hal_amdgpu_block_allocator_free(allocation_pool
                                           ? &queue->block_allocators->large
                                           : &queue->block_allocators->small,
                                       entry, entry_token);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_queue_alloca(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(queue);

  // TODO(benvanik): use params.queue_affinity to restrict access? By default
  // the device-side allocation pool is accessible to all devices in the system
  // but this can be inefficient.

  // TODO(benvanik): pool IDs.
  // DO NOT SUBMIT pool_id mapping
  iree_hal_amdgpu_device_allocation_pool_id_t pool_id = {
      .device_pool = NULL,
      .host_pool = 0,
  };

  // Allocate placeholder HAL buffer handle. This has no backing storage beyond
  // the allocation handle in the device memory pool.
  iree_hal_buffer_t* buffer = NULL;
  iree_hal_amdgpu_device_allocation_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_buffer_pool_acquire(
      queue->buffer_pool, params, allocation_size, &buffer, &handle));

  // NOTE: if entry reserve/commit fails we need to clean up the buffer.
  iree_hal_amdgpu_device_queue_alloca_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  iree_status_t status = iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_ALLOCA, sizeof(*entry),
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ALLOCA_MAX_KERNARG_CAPACITY, (void**)&entry,
      &resource_set);
  if (iree_status_is_ok(status)) {
    entry->pool_id = pool_id;
    entry->min_alignment = params.min_alignment;
    entry->allocation_size = allocation_size;
    entry->handle = handle;

    // Insert newly allocated buffer into resource set to keep it live for the
    // lifetime of the queue operation. Users usually don't allocate and then
    // immediately drop the last reference to something but they will in error
    // handling cases.
    status = iree_hal_resource_set_insert(resource_set, 1, &buffer);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
    *out_buffer = buffer;
  } else {
    iree_hal_amdgpu_queue_free_entry(queue, &entry->header);
    if (buffer) iree_hal_buffer_release(buffer);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_queue_dealloca(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);

  // Must be a transient buffer. This will fail for other buffer types.
  iree_hal_amdgpu_device_allocation_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_resolve_transient_buffer(buffer, &handle));

  iree_hal_amdgpu_device_queue_dealloca_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_DEALLOCA, sizeof(*entry),
      IREE_HAL_AMDGPU_DEVICE_QUEUE_DEALLOCA_MAX_KERNARG_CAPACITY,
      (void**)&entry, &resource_set));
  entry->handle = handle;

  // Insert the deallocated buffer handle into resource set to keep it live for
  // the lifetime of the queue operation. We have to keep it live as until the
  // deallocation in the queue timeline the handle may be in use.
  iree_status_t status = iree_hal_resource_set_insert(resource_set, 1, &buffer);

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_queue_fill(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern, uint8_t pattern_length,
    iree_hal_fill_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);

  iree_hal_amdgpu_device_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           target_buffer, target_offset, length, &target_ref),
                       "resolving `target_ref`");

  iree_hal_amdgpu_device_queue_fill_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_FILL, sizeof(*entry),
      IREE_HAL_AMDGPU_DEVICE_QUEUE_FILL_MAX_KERNARG_CAPACITY, (void**)&entry,
      &resource_set));
  entry->target_ref = target_ref;
  entry->pattern = pattern;
  entry->pattern_length = pattern_length;

  // Insert the target buffer into resource set to keep it live for the lifetime
  // of the queue operation.
  iree_status_t status =
      iree_hal_resource_set_insert(resource_set, 1, &target_buffer);

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_queue_update(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);

  iree_hal_amdgpu_device_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           target_buffer, target_offset, length, &target_ref),
                       "resolving `target_ref`");

  // NOTE: we allocate extra storage in the queue entry for the update contents.
  // This limits the size of the update data to the large block pool size minus
  // the queue entry overhead. If we wanted to fix the max update size as the
  // block size we'd have to allocate it separately.
  iree_hal_amdgpu_device_queue_update_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_UPDATE, sizeof(*entry) + length,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_UPDATE_MAX_KERNARG_CAPACITY, (void**)&entry,
      &resource_set));
  entry->source_ptr = (uint8_t*)entry + sizeof(*entry);
  entry->target_ref = target_ref;

  // Insert the target buffer into resource set to keep it live for the lifetime
  // of the queue operation. The source buffer is copied into the queue entry.
  iree_status_t status =
      iree_hal_resource_set_insert(resource_set, 1, &target_buffer);

  // Copy source contents into the queue entry.
  if (iree_status_is_ok(status)) {
    iree_memcpy_stream_dst((uint8_t*)entry + sizeof(*entry),
                           (const uint8_t*)source_buffer + source_offset,
                           length);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_queue_copy(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);

  iree_hal_amdgpu_device_buffer_ref_t source_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           source_buffer, source_offset, length, &source_ref),
                       "resolving `source_ref`");
  iree_hal_amdgpu_device_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           target_buffer, target_offset, length, &target_ref),
                       "resolving `target_ref`");

  iree_hal_amdgpu_device_queue_copy_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_COPY, sizeof(*entry),
      IREE_HAL_AMDGPU_DEVICE_QUEUE_COPY_MAX_KERNARG_CAPACITY, (void**)&entry,
      &resource_set));
  entry->source_ref = source_ref;
  entry->target_ref = target_ref;

  // Insert buffers into the resource set to keep them live for the lifetime of
  // the queue operation.
  const void* resources[] = {
      source_buffer,
      target_buffer,
  };
  iree_status_t status = iree_hal_resource_set_insert(
      resource_set, IREE_ARRAYSIZE(resources), resources);

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_queue_free_entry(queue, &entry->header);
  }
  return status;

  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_read(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  // TODO(benvanik): support device-side reads by relaying through the host
  // worker. We should be able to enqueue the operation and have the device ask
  // the host to read buffer ranges at the appropriate time. Today the
  // emulation is blocking.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "device-side reads not yet implemented");
}

iree_status_t iree_hal_amdgpu_queue_write(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  // TODO(benvanik): support device-side writes by relaying through the host
  // worker. We should be able to enqueue the operation and have the device ask
  // the host to write buffer ranges at the appropriate time. Today the
  // emulation is blocking.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "device-side writes not yet implemented");
}

static iree_status_t iree_hal_amdgpu_queue_barrier(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_hal_amdgpu_device_queue_barrier_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_BARRIER, sizeof(*entry),
      IREE_HAL_AMDGPU_DEVICE_QUEUE_BARRIER_MAX_KERNARG_CAPACITY, (void**)&entry,
      /*resource_set=*/NULL));
  iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_execute(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);

  // Fast-path barriers.
  if (command_buffer == NULL) {
    return iree_hal_amdgpu_queue_barrier(queue, wait_semaphore_list,
                                         signal_semaphore_list);
  }

  // Query the device-side resource requirements and per-device copy of the
  // command buffer information. All other information is handled device-side
  // during the issue of the execute operation.
  iree_hal_amdgpu_device_command_buffer_t* device_command_buffer = NULL;
  iree_host_size_t command_buffer_max_kernarg_capacity = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_buffer_query_execution_state(
      command_buffer, queue->device_ordinal, &device_command_buffer,
      &command_buffer_max_kernarg_capacity));

  // Kernarg requirements are for the device-side control dispatches as well as
  // execution of any block in the command buffer.
  const iree_host_size_t max_kernarg_capacity =
      IREE_HAL_AMDGPU_DEVICE_QUEUE_EXECUTE_MAX_KERNARG_CAPACITY(
          command_buffer_max_kernarg_capacity);

  // Reserve an entry in the queue for populating.
  // The device-side scheduler will not begin processing it until after it has
  // been committed below.
  iree_hal_amdgpu_device_queue_execute_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_EXECUTE,
      sizeof(*entry) +
          binding_table.count * sizeof(iree_hal_amdgpu_device_buffer_ref_t),
      max_kernarg_capacity, (void**)&entry, &resource_set));

  // NOTE: we only need to populate the flags and command buffer/binding table.
  // Other fields are setup when the operation is issued on device.

  // DO NOT SUBMIT queue entry flags

  iree_hal_amdgpu_device_execution_flags_t flags =
      IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_NONE;

  // TODO(benvanik): if there are few commands then set ISSUE_SERIALLY
  // (IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_ISSUE_SERIALLY) to reduce latency.
  // Serial issuing should really be per-block and we may want to turn this into
  // a block-level option.

  flags |= IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_SERIALIZE;
  flags |= IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_UNCACHED;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  flags |= IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL;
  flags |= IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_DISPATCH;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  entry->flags = flags;

  // Execution will begin at the entry block (block[0]).
  entry->command_buffer = device_command_buffer;

  // Resolve all provided binding table entries to their device handles or
  // pointers. Note that this may fail if any binding is invalid and we need to
  // clean up the allocated queue entry (we do this so that we can resolve
  // in-place and not need an extra allocation).
  //
  // TODO(benvanik): store in the entry instead of the state and allow the
  // device to resolve allocation_handles.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    status = iree_hal_amdgpu_resolve_binding(binding_table.bindings[i],
                                             &entry->state.bindings[i]);
    if (!iree_status_is_ok(status)) break;
  }

  // Insert all bindings into the queue entry resource set.
  if (iree_status_is_ok(status) && binding_table.count > 0) {
    status = iree_hal_resource_set_insert_strided(
        resource_set, binding_table.count, &binding_table.bindings[0].buffer,
        offsetof(iree_hal_buffer_binding_t, buffer),
        sizeof(iree_hal_buffer_binding_t));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_queue_flush(iree_hal_amdgpu_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  // No-op as we don't do any host-side buffering (today).
  return iree_ok_status();
}
