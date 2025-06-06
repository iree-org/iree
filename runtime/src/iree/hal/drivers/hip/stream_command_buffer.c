
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/stream_command_buffer.h"

#include "iree/hal/drivers/hip/hip_buffer.h"
#include "iree/hal/drivers/hip/native_executable.h"
#include "iree/hal/drivers/hip/rccl_channel.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/resource_set.h"
#include "iree/hal/utils/stream_tracing.h"

typedef struct iree_hal_hip_stream_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
  const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols;

  // Per-stream HIP tracing context.
  iree_hal_stream_tracing_context_t* tracing_context;
  iree_hal_stream_tracing_context_event_list_t tracing_event_list;

  hipStream_t hip_stream;

  // A resource set to maintain references to all resources used within the
  // command buffer. Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // Used for when we need HIP to be able to reference memory as it performs
  // asynchronous operations.
  iree_arena_allocator_t arena;

  // Iteratively constructed batch of collective operations.
  iree_hal_collective_batch_t collective_batch;
} iree_hal_hip_stream_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_hip_stream_command_buffer_vtable;

static iree_hal_hip_stream_command_buffer_t*
iree_hal_hip_stream_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_stream_command_buffer_vtable);
  return (iree_hal_hip_stream_command_buffer_t*)base_value;
}

iree_status_t iree_hal_hip_stream_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_hal_stream_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    hipStream_t stream, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(hip_symbols);
  IREE_ASSERT_ARGUMENT(nccl_symbols);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_stream_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_hip_stream_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->hip_symbols = hip_symbols;
  command_buffer->nccl_symbols = nccl_symbols;
  command_buffer->tracing_context = tracing_context;
  command_buffer->tracing_event_list.head = NULL;
  command_buffer->tracing_event_list.tail = NULL;
  command_buffer->hip_stream = stream;
  iree_arena_initialize(block_pool, &command_buffer->arena);

  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);

  if (iree_status_is_ok(status)) {
    iree_hal_collective_batch_initialize(&command_buffer->arena,
                                         command_buffer->resource_set,
                                         &command_buffer->collective_batch);
  }

  *out_command_buffer = &command_buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hip_stream_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_stream_tracing_free(command_buffer->tracing_context,
                               &command_buffer->tracing_event_list);

  iree_hal_collective_batch_deinitialize(&command_buffer->collective_batch);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_hip_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_hip_stream_command_buffer_vtable);
}

void iree_hal_hip_stream_notify_submitted_commands(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  if (!command_buffer->tracing_context) {
    return;
  }

  iree_hal_stream_tracing_notify_submitted(command_buffer->tracing_context,
                                           &command_buffer->tracing_event_list);
}

// Flushes any pending batched collective operations.
// Must be called before any other non-collective nodes are added to the graph
// or a barrier is encountered.
static iree_status_t iree_hal_hip_stream_command_buffer_flush_collectives(
    iree_hal_hip_stream_command_buffer_t* command_buffer) {
  // NOTE: we could move this out into callers by way of an always-inline shim -
  // that would make this a single compare against the command buffer state we
  // are likely to access immediately after anyway and keep overheads minimal.
  if (IREE_LIKELY(iree_hal_collective_batch_is_empty(
          &command_buffer->collective_batch))) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_hip_nccl_submit_batch(
      command_buffer->nccl_symbols, command_buffer->tracing_context,
      &command_buffer->tracing_event_list, &command_buffer->collective_batch,
      command_buffer->hip_stream);
  iree_hal_collective_batch_clear(&command_buffer->collective_batch);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_stream_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HAL_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE,
      /*file_name=*/NULL, 0, /*line=*/0, "iree_hal_hip_stream_command_buffer",
      strlen("iree_hal_hip_stream_command_buffer"),
      /*name=*/NULL, 0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  // Reset the arena as there should be nothing using it now that we've
  // dispatched all our operations inline.
  // NOTE: the resource set may contain resources we need to drop as we don't
  //       need to keep them live any longer than it takes to schedule the
  //       operations. In a real command buffer we would be this stream command
  //       buffer is strictly used to perform inline execution/replay of
  //       deferred command buffers that are retaining the resources already.
  iree_arena_reset(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(command_buffer->arena.block_pool,
                                         &command_buffer->resource_set));

  IREE_HAL_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HAL_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE,
      location ? location->file.data : NULL, location ? location->file.size : 0,
      location ? location->line : 0,
      /*func_name=*/NULL, 0, label.data, label.size);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HAL_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);

  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_IF_ERROR(
      iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  // Nothing to do for barriers between memory operations or dispatches--HIP
  // stream semantics guarantees execution and memory visibility in program
  // order.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hip_stream_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hip_stream_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hip_stream_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  // We could mark the memory as invalidated so that if managed HIP does not
  // try to copy it back to the host.
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  hipDeviceptr_t target_device_buffer = iree_hal_hip_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  hipDeviceptr_t dst = (uint8_t*)target_device_buffer + target_offset;
  size_t num_elements = target_ref.length / pattern_length;
  IREE_HAL_STREAM_TRACE_ZONE_BEGIN(command_buffer->tracing_context,
                                   &command_buffer->tracing_event_list,
                                   IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);

  switch (pattern_length) {
    case 4: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipMemsetD32Async(dst, *(const uint32_t*)(pattern), num_elements,
                            command_buffer->hip_stream),
          "hipMemsetD32Async");
      break;
    }
    case 2: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipMemsetD16Async(dst, *(const uint16_t*)(pattern), num_elements,
                            command_buffer->hip_stream),
          "hipMemsetD16Async");
      break;
    }
    case 1: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipMemsetD8Async(dst, *(const uint8_t*)(pattern), num_elements,
                           command_buffer->hip_stream),
          "hipMemsetD8Async");
      break;
    }
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported fill pattern length");
  }
  IREE_HAL_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  // Allocate scratch space in the arena for the data and copy it in.
  // The update buffer API requires that the command buffer capture the host
  // memory at the time the method is called in case the caller wants to reuse
  // the memory. Because HIP memcpys are async if we didn't copy it's possible
  // for the reused memory to change before the stream reaches the copy
  // operation and get the wrong data.
  const uint8_t* src = (const uint8_t*)source_buffer + source_offset;
  if (command_buffer->arena.block_pool) {
    uint8_t* storage = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&command_buffer->arena, target_ref.length,
                                (void**)&storage));
    memcpy(storage, src, target_ref.length);
    src = storage;
  }

  // Issue the copy using the scratch memory as the source.
  hipDeviceptr_t target_device_buffer = iree_hal_hip_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  hipDeviceptr_t dst = (uint8_t*)target_device_buffer +
                       iree_hal_buffer_byte_offset(target_ref.buffer) +
                       target_ref.offset;
  IREE_HAL_STREAM_TRACE_ZONE_BEGIN(command_buffer->tracing_context,
                                   &command_buffer->tracing_event_list,
                                   IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);
  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hip_symbols,
      hipMemcpyHtoDAsync(dst, (void*)src, target_ref.length,
                         command_buffer->hip_stream),
      "hipMemcpyHtoDAsync");
  IREE_HAL_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));
  IREE_HAL_STREAM_TRACE_ZONE_BEGIN(command_buffer->tracing_context,
                                   &command_buffer->tracing_event_list,
                                   IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);

  hipDeviceptr_t target_device_buffer = iree_hal_hip_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  hipDeviceptr_t source_device_buffer = iree_hal_hip_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_ref.buffer));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  hipDeviceptr_t dst = (uint8_t*)target_device_buffer + target_offset;
  hipDeviceptr_t src = (uint8_t*)source_device_buffer + source_offset;

  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hip_symbols,
      hipMemcpyAsync(dst, src, target_ref.length, hipMemcpyDeviceToDevice,
                     command_buffer->hip_stream),
      "hipMemcpyAsync");

  IREE_HAL_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_binding_t send_binding = {
      .buffer = send_ref.buffer,
      .offset = send_ref.offset,
      .length = send_ref.length,
  };
  iree_hal_buffer_binding_t recv_binding = {
      .buffer = recv_ref.buffer,
      .offset = recv_ref.offset,
      .length = recv_ref.length,
  };
  iree_status_t status = iree_hal_collective_batch_append(
      &command_buffer->collective_batch, channel, op, param, send_binding,
      recv_binding, element_count);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_stream_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    const uint32_t workgroup_count[3], iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // If any of the workgroup counts are zero, we can skip execution
  // of the kernel. This prevents a 'hipErrorInvalidConfiguration' error when
  // launching the kernel.
  if (workgroup_count[0] == 0 || workgroup_count[1] == 0 ||
      workgroup_count[2] == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  const iree_hal_hip_kernel_params_t* kernel_params = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_native_executable_lookup_kernel_params(
              executable, entry_point, command_buffer->base.queue_affinity,
              &kernel_params));

  IREE_HAL_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      IREE_HAL_STREAM_TRACING_VERBOSITY_FINE,
      kernel_params->debug_info.source_filename.data,
      kernel_params->debug_info.source_filename.size,
      kernel_params->debug_info.source_line,
      kernel_params->debug_info.function_name.data,
      kernel_params->debug_info.function_name.size,
      /*name=*/NULL, 0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  // We append push constants to the end of descriptors to form a linear chain
  // of kernel arguments.
  iree_host_size_t kernel_params_count =
      kernel_params->binding_count + kernel_params->constant_count;
  iree_host_size_t kernel_params_length = kernel_params_count * sizeof(void*);

  // TODO: use packed parameters instead of the indirection mechanism - this
  // would avoid additional driver overhead to reflect and repack them all.
  //
  // Each kernel_params[i] is itself a pointer to the corresponding
  // element at the *second* inline allocation at the end of the current
  // segment.
  iree_host_size_t total_size = kernel_params_length * 2;
  uint8_t* storage_base = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, total_size,
                              (void**)&storage_base));
  void** params_ptr = (void**)storage_base;
  hipDeviceptr_t* payload_ptr =
      (hipDeviceptr_t*)((uint8_t*)params_ptr + kernel_params_length);
  for (iree_host_size_t i = 0; i < kernel_params_count; i++) {
    params_ptr[i] = &payload_ptr[i];
  }
  for (iree_host_size_t i = 0; i < bindings.count; i++) {
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    hipDeviceptr_t device_ptr = NULL;
    if (binding->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &binding->buffer));
      hipDeviceptr_t device_buffer = iree_hal_hip_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(binding->buffer));
      iree_device_size_t offset = iree_hal_buffer_byte_offset(binding->buffer);
      device_ptr = (uint8_t*)device_buffer + offset + binding->offset;
    }
    payload_ptr[i] = device_ptr;
  }

  // As commented in the above, what each kernel parameter points to is a
  // hipDeviceptr_t, which as the size of a pointer on the target machine. we
  // are just storing a 32-bit value for the push constant here instead. So we
  // must process one element each type, for 64-bit machines.
  for (iree_host_size_t i = 0; i < kernel_params->constant_count; i++) {
    *((uint32_t*)params_ptr[kernel_params->binding_count + i]) =
        ((const uint32_t*)constants.data)[i];
  }

  iree_status_t status = IREE_HIP_CALL_TO_STATUS(
      command_buffer->hip_symbols,
      hipModuleLaunchKernel(
          kernel_params->function, workgroup_count[0], workgroup_count[1],
          workgroup_count[2], kernel_params->block_dims[0],
          kernel_params->block_dims[1], kernel_params->block_dims[2],
          kernel_params->block_shared_memory_size, command_buffer->hip_stream,
          params_ptr, NULL),
      "hipModuleLaunchKernel");

  IREE_HAL_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_stream_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_ref_t workgroups_ref, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect dispatch not yet implemented");
}

iree_hal_stream_tracing_context_event_list_t
iree_hal_hip_stream_command_buffer_tracing_events(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  return command_buffer->tracing_event_list;
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_hip_stream_command_buffer_vtable = {
        .destroy = iree_hal_hip_stream_command_buffer_destroy,
        .begin = iree_hal_hip_stream_command_buffer_begin,
        .end = iree_hal_hip_stream_command_buffer_end,
        .begin_debug_group =
            iree_hal_hip_stream_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_hip_stream_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_hip_stream_command_buffer_execution_barrier,
        .signal_event = iree_hal_hip_stream_command_buffer_signal_event,
        .reset_event = iree_hal_hip_stream_command_buffer_reset_event,
        .wait_events = iree_hal_hip_stream_command_buffer_wait_events,
        .advise_buffer = iree_hal_hip_stream_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_hip_stream_command_buffer_fill_buffer,
        .update_buffer = iree_hal_hip_stream_command_buffer_update_buffer,
        .copy_buffer = iree_hal_hip_stream_command_buffer_copy_buffer,
        .collective = iree_hal_hip_stream_command_buffer_collective,
        .dispatch = iree_hal_hip_stream_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_hip_stream_command_buffer_dispatch_indirect,
};
