// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/stream_command_buffer.h"

#include "iree/base/tracing.h"
#include "iree/hal/drivers/cuda/cuda_buffer.h"
#include "iree/hal/drivers/cuda/cuda_event.h"
#include "iree/hal/drivers/cuda/native_executable.h"
#include "iree/hal/drivers/cuda/nccl_channel.h"
#include "iree/hal/drivers/cuda/pipeline_layout.h"
#include "iree/hal/drivers/cuda/status_util.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/resource_set.h"

#define IREE_HAL_CUDA_MAX_BINDING_COUNT 64
// Kernel arguments contains binding and push constants.
#define IREE_HAL_CUDA_MAX_KERNEL_ARG 128

typedef struct {
  iree_hal_command_buffer_t base;
  iree_hal_cuda_context_wrapper_t* context;
  iree_hal_cuda_tracing_context_t* tracing_context;
  CUstream stream;

  // Maintains a reference to all resources used within the command buffer.
  // Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // Used for when we need CUDA to be able to reference memory as it performs
  // asynchronous operations.
  iree_arena_allocator_t arena;

  // Iteratively constructed batch of collective operations.
  iree_hal_collective_batch_t collective_batch;

  int32_t push_constant[IREE_HAL_CUDA_MAX_PUSH_CONSTANT_COUNT];

  // Keep track of the current set of kernel arguments.
  void* current_descriptor[IREE_HAL_CUDA_MAX_KERNEL_ARG];
  CUdeviceptr* device_ptrs[IREE_HAL_CUDA_MAX_KERNEL_ARG];
} iree_hal_cuda_stream_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_stream_command_buffer_vtable;

static iree_hal_cuda_stream_command_buffer_t*
iree_hal_cuda_stream_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_stream_command_buffer_vtable);
  return (iree_hal_cuda_stream_command_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda_stream_command_buffer_create(
    iree_hal_device_t* device, iree_hal_cuda_context_wrapper_t* context,
    iree_hal_cuda_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, CUstream stream,
    iree_arena_block_pool_t* block_pool,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_stream_command_buffer_t* command_buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(context->host_allocator, sizeof(*command_buffer),
                            (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device, mode, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
        binding_capacity, &iree_hal_cuda_stream_command_buffer_vtable,
        &command_buffer->base);
    command_buffer->context = context;
    command_buffer->tracing_context = tracing_context;
    command_buffer->stream = stream;
    iree_arena_initialize(block_pool, &command_buffer->arena);
    for (size_t i = 0; i < IREE_HAL_CUDA_MAX_KERNEL_ARG; i++) {
      command_buffer->current_descriptor[i] = &command_buffer->device_ptrs[i];
    }

    status = iree_hal_resource_set_allocate(block_pool,
                                            &command_buffer->resource_set);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_collective_batch_initialize(&command_buffer->arena,
                                         command_buffer->resource_set,
                                         &command_buffer->collective_batch);
  }

  *out_command_buffer = &command_buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_stream_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_collective_batch_deinitialize(&command_buffer->collective_batch);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(command_buffer->context->host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_cuda_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_command_buffer_dyn_cast(
      command_buffer, &iree_hal_cuda_stream_command_buffer_vtable);
}

static void* iree_hal_cuda_stream_command_buffer_dyn_cast(
    iree_hal_command_buffer_t* command_buffer, const void* vtable) {
  if (vtable == &iree_hal_cuda_stream_command_buffer_vtable) {
    IREE_HAL_ASSERT_TYPE(command_buffer, vtable);
    return command_buffer;
  }
  return NULL;
}

// Flushes any pending batched collective operations.
// Must be called before any other non-collective nodes are added to the graph
// or a barrier is encountered.
static iree_status_t iree_hal_cuda_stream_command_buffer_flush_collectives(
    iree_hal_cuda_stream_command_buffer_t* command_buffer) {
  // NOTE: we could move this out into callers by way of an always-inline shim -
  // that would make this a single compare against the command buffer state we
  // are likely to access immediately after anyway and keep overheads minimal.
  if (IREE_LIKELY(iree_hal_collective_batch_is_empty(
          &command_buffer->collective_batch))) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_cuda_nccl_submit_batch(
      command_buffer->context, command_buffer->tracing_context,
      &command_buffer->collective_batch, command_buffer->stream);
  iree_hal_collective_batch_reset(&command_buffer->collective_batch);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_stream_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  iree_arena_reset(&command_buffer->arena);

  IREE_CUDA_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->stream,
      /*file_name=*/NULL, 0,
      /*line=*/0, /*func_name=*/NULL, 0, "iree_hal_cuda_stream_command_buffer",
      strlen("iree_hal_cuda_stream_command_buffer"));

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  IREE_CUDA_TRACE_ZONE_END(command_buffer->tracing_context,
                           command_buffer->stream);

  return iree_ok_status();
}

static void iree_hal_cuda_stream_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_CUDA_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->stream,
      location ? location->file.data : NULL, location ? location->file.size : 0,
      location ? location->line : 0, /*func_name=*/NULL, 0, label.data,
      label.size);

  // TODO: pass along to CUPTI if available.
}

static void iree_hal_cuda_stream_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  // TODO: pass along to CUPTI if available.

  IREE_CUDA_TRACE_ZONE_END(command_buffer->tracing_context,
                           command_buffer->stream);
}

static iree_status_t iree_hal_cuda_stream_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));
  // TODO(jinchen62): implement CUDA barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));
  // TODO(jinchen62): implement CUDA barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));
  // TODO(jinchen62): implement CUDA barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));
  // TODO(jinchen62): implement CUDA barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // We could mark the memory as invalidated so that if managed CUDA does not
  // try to copy it back to the host.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  CUdeviceptr dst = target_device_buffer + target_offset;
  size_t num_elements = length / pattern_length;
  switch (pattern_length) {
    case 4: {
      CUDA_RETURN_IF_ERROR(
          command_buffer->context->syms,
          cuMemsetD32Async(dst, *(const uint32_t*)(pattern), num_elements,
                           command_buffer->stream),
          "cuMemsetD32Async");
      break;
    }
    case 2: {
      CUDA_RETURN_IF_ERROR(
          command_buffer->context->syms,
          cuMemsetD16Async(dst, *(const uint16_t*)(pattern), num_elements,
                           command_buffer->stream),
          "cuMemsetD16Async");
      break;
    }
    case 1: {
      CUDA_RETURN_IF_ERROR(
          command_buffer->context->syms,
          cuMemsetD8Async(dst, *(const uint8_t*)(pattern), num_elements,
                          command_buffer->stream),
          "cuMemsetD8Async");
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported fill pattern length");
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  // Allocate scratch space in the arena for the data and copy it in.
  // The update buffer API requires that the command buffer capture the host
  // memory at the time the method is called in case the caller wants to reuse
  // the memory. Because CUDA memcpys are async if we didn't copy it's possible
  // for the reused memory to change before the stream reaches the copy
  // operation and get the wrong data.
  const uint8_t* src = (const uint8_t*)source_buffer + source_offset;
  if (command_buffer->arena.block_pool) {
    uint8_t* storage = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&command_buffer->arena, length, (void**)&storage));
    memcpy(storage, src, length);
    src = storage;
  }

  // Issue the copy using the scratch memory as the source.
  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  CUdeviceptr dst = target_device_buffer +
                    iree_hal_buffer_byte_offset(target_buffer) + target_offset;
  CUDA_RETURN_IF_ERROR(
      command_buffer->context->syms,
      cuMemcpyHtoDAsync_v2(dst, src, length, command_buffer->stream),
      "cuMemcpyHtoDAsync_v2");

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  CUdeviceptr source_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_buffer));
  source_offset += iree_hal_buffer_byte_offset(source_buffer);
  CUdeviceptr dst = target_device_buffer + target_offset;
  CUdeviceptr src = source_device_buffer + source_offset;
  CUDA_RETURN_IF_ERROR(command_buffer->context->syms,
                       cuMemcpyAsync(dst, src, length, command_buffer->stream),
                       "cuMemcpyAsync");

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  return iree_hal_collective_batch_append(&command_buffer->collective_batch,
                                          channel, op, param, send_binding,
                                          recv_binding, element_count);
}

static iree_status_t iree_hal_cuda_stream_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);

  iree_host_size_t constant_base_index = offset / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < values_length / sizeof(int32_t); i++) {
    command_buffer->push_constant[i + constant_base_index] =
        ((uint32_t*)values)[i];
  }

  return iree_ok_status();
}

// Tie together the binding index and its index in |bindings| array.
typedef struct {
  uint32_t index;
  uint32_t binding;
} iree_hal_cuda_binding_mapping_t;

// Helper to sort the binding based on their binding index.
static int compare_binding_index(const void* a, const void* b) {
  const iree_hal_cuda_binding_mapping_t buffer_a =
      *(const iree_hal_cuda_binding_mapping_t*)a;
  const iree_hal_cuda_binding_mapping_t buffer_b =
      *(const iree_hal_cuda_binding_mapping_t*)b;
  return buffer_a.binding < buffer_b.binding ? -1 : 1;
}

static iree_status_t iree_hal_cuda_stream_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);

  iree_host_size_t base_binding =
      iree_hal_cuda_base_binding_index(pipeline_layout, set);

  // Convention with the compiler side. We map bindings to kernel argument.
  // We compact the bindings to get a dense set of arguments and keep them order
  // based on the binding index.
  // Sort the binding based on the binding index and map the array index to the
  // argument index.
  iree_hal_cuda_binding_mapping_t binding_used[IREE_HAL_CUDA_MAX_BINDING_COUNT];
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    iree_hal_cuda_binding_mapping_t buffer = {i, bindings[i].binding};
    binding_used[i] = buffer;
  }
  // TODO: remove this sort - it's thankfully small (1-8 on average) but we
  // should be able to avoid it like we do on the CPU side with a bitmap.
  qsort(binding_used, binding_count, sizeof(iree_hal_cuda_binding_mapping_t),
        compare_binding_index);
  assert(binding_count < IREE_HAL_CUDA_MAX_BINDING_COUNT &&
         "binding count larger than the max expected.");

  for (iree_host_size_t i = 0; i < binding_count; i++) {
    iree_hal_descriptor_set_binding_t binding = bindings[binding_used[i].index];
    CUdeviceptr device_ptr =
        binding.buffer
            ? (iree_hal_cuda_buffer_device_pointer(
                   iree_hal_buffer_allocated_buffer(binding.buffer)) +
               iree_hal_buffer_byte_offset(binding.buffer) + binding.offset)
            : 0;
    *((CUdeviceptr*)command_buffer->current_descriptor[i + base_binding]) =
        device_ptr;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  iree_hal_cuda_kernel_params_t kernel_params;
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_native_executable_entry_point_kernel_params(
          executable, entry_point, &kernel_params));

  IREE_CUDA_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->stream,
      kernel_params.source_filename.data, kernel_params.source_filename.size,
      kernel_params.source_line, /*func_name=*/NULL, 0,
      kernel_params.function_name.data, kernel_params.function_name.size);

  // Patch the push constants in the kernel arguments.
  iree_host_size_t num_constants =
      iree_hal_cuda_pipeline_layout_num_constants(kernel_params.layout);
  iree_host_size_t constant_base_index =
      iree_hal_cuda_push_constant_index(kernel_params.layout);
  for (iree_host_size_t i = 0; i < num_constants; i++) {
    *((uint32_t*)command_buffer->current_descriptor[i + constant_base_index]) =
        command_buffer->push_constant[i];
  }

  CUDA_RETURN_IF_ERROR(
      command_buffer->context->syms,
      cuLaunchKernel(kernel_params.function, workgroup_x, workgroup_y,
                     workgroup_z, kernel_params.block_size[0],
                     kernel_params.block_size[1], kernel_params.block_size[2],
                     kernel_params.shared_memory_size, command_buffer->stream,
                     command_buffer->current_descriptor, NULL),
      "cuLaunchKernel");

  IREE_CUDA_TRACE_ZONE_END(command_buffer->tracing_context,
                           command_buffer->stream);

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation of dispatch indirect");
}

static iree_status_t iree_hal_cuda_stream_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  // TODO(#10144): support indirect command buffers with deferred command
  // buffers or graphs. We likely just want to switch to graphs.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect command buffers not yet implemented");
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_stream_command_buffer_vtable = {
        .destroy = iree_hal_cuda_stream_command_buffer_destroy,
        .dyn_cast = iree_hal_cuda_stream_command_buffer_dyn_cast,
        .begin = iree_hal_cuda_stream_command_buffer_begin,
        .end = iree_hal_cuda_stream_command_buffer_end,
        .begin_debug_group =
            iree_hal_cuda_stream_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_cuda_stream_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_cuda_stream_command_buffer_execution_barrier,
        .signal_event = iree_hal_cuda_stream_command_buffer_signal_event,
        .reset_event = iree_hal_cuda_stream_command_buffer_reset_event,
        .wait_events = iree_hal_cuda_stream_command_buffer_wait_events,
        .discard_buffer = iree_hal_cuda_stream_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_cuda_stream_command_buffer_fill_buffer,
        .update_buffer = iree_hal_cuda_stream_command_buffer_update_buffer,
        .copy_buffer = iree_hal_cuda_stream_command_buffer_copy_buffer,
        .collective = iree_hal_cuda_stream_command_buffer_collective,
        .push_constants = iree_hal_cuda_stream_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_cuda_stream_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_cuda_stream_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_cuda_stream_command_buffer_dispatch_indirect,
        .execute_commands =
            iree_hal_cuda_stream_command_buffer_execute_commands,
};
