// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/graph_command_buffer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "experimental/cuda2/cuda_buffer.h"
#include "experimental/cuda2/cuda_dynamic_symbols.h"
#include "experimental/cuda2/cuda_status_util.h"
#include "experimental/cuda2/native_executable.h"
#include "experimental/cuda2/nccl_channel.h"
#include "experimental/cuda2/pipeline_layout.h"
#include "iree/base/api.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/resource_set.h"

// The maximal number of descriptor bindings supported in the CUDA HAL driver.
#define IREE_HAL_CUDA_MAX_BINDING_COUNT 64
// The maximal number of kernel arguments supported in the CUDA HAL driver for
// descriptor bindings and push constants.
#define IREE_HAL_CUDA_MAX_KERNEL_ARG 128

// Command buffer implementation that directly records into CUDA graphs.
// The command buffer records the commands on the calling thread without
// additional threading indirection.
typedef struct iree_hal_cuda2_graph_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  const iree_hal_cuda2_dynamic_symbols_t* symbols;

  // A resource set to maintain references to all resources used within the
  // command buffer.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // This is used for when we need CUDA to be able to reference memory as it
  // performs asynchronous operations.
  iree_arena_allocator_t arena;

  CUcontext cu_context;
  // The CUDA graph under construction.
  CUgraph graph;
  CUgraphExec exec;

  // The last node added to the command buffer.
  // We need to track it as we are currently serializing all the nodes (each
  // node depends on the previous one).
  CUgraphNode last_node;

  // Iteratively constructed batch of collective operations.
  iree_hal_collective_batch_t collective_batch;

  int32_t push_constant[IREE_HAL_CUDA_MAX_PUSH_CONSTANT_COUNT];

  // The current set of kernel arguments.
  void* current_descriptor[];
} iree_hal_cuda2_graph_command_buffer_t;
// + Additional inline allocation for holding all kernel arguments.

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda2_graph_command_buffer_vtable;

static iree_hal_cuda2_graph_command_buffer_t*
iree_hal_cuda2_graph_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda2_graph_command_buffer_vtable);
  return (iree_hal_cuda2_graph_command_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda2_graph_command_buffer_create(
    iree_hal_device_t* device,
    const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols, CUcontext context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda2_graph_command_buffer_t* command_buffer = NULL;
  size_t total_size = sizeof(*command_buffer) +
                      IREE_HAL_CUDA_MAX_KERNEL_ARG * sizeof(void*) +
                      IREE_HAL_CUDA_MAX_KERNEL_ARG * sizeof(CUdeviceptr);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device, mode, command_categories, queue_affinity, binding_capacity,
      &iree_hal_cuda2_graph_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->symbols = cuda_symbols;
  iree_arena_initialize(block_pool, &command_buffer->arena);
  command_buffer->cu_context = context;
  command_buffer->graph = NULL;
  command_buffer->exec = NULL;
  command_buffer->last_node = NULL;

  CUdeviceptr* device_ptrs = (CUdeviceptr*)(command_buffer->current_descriptor +
                                            IREE_HAL_CUDA_MAX_KERNEL_ARG);
  for (size_t i = 0; i < IREE_HAL_CUDA_MAX_KERNEL_ARG; i++) {
    command_buffer->current_descriptor[i] = &device_ptrs[i];
  }

  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);

  if (iree_status_is_ok(status)) {
    iree_hal_collective_batch_initialize(&command_buffer->arena,
                                         command_buffer->resource_set,
                                         &command_buffer->collective_batch);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda2_graph_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Drop any pending collective batches before we tear things down.
  iree_hal_collective_batch_clear(&command_buffer->collective_batch);

  if (command_buffer->graph != NULL) {
    IREE_CUDA_IGNORE_ERROR(command_buffer->symbols,
                           cuGraphDestroy(command_buffer->graph));
    command_buffer->graph = NULL;
  }
  if (command_buffer->exec != NULL) {
    IREE_CUDA_IGNORE_ERROR(command_buffer->symbols,
                           cuGraphExecDestroy(command_buffer->exec));
    command_buffer->exec = NULL;
  }
  command_buffer->last_node = NULL;

  iree_hal_collective_batch_deinitialize(&command_buffer->collective_batch);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_cuda2_graph_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_cuda2_graph_command_buffer_vtable);
}

CUgraphExec iree_hal_cuda2_graph_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  return command_buffer->exec;
}

// Flushes any pending batched collective operations.
// Must be called before any other non-collective nodes are added to the graph
// or a barrier is encountered.
static iree_status_t iree_hal_cuda2_graph_command_buffer_flush_collectives(
    iree_hal_cuda2_graph_command_buffer_t* command_buffer) {
  // NOTE: we could move this out into callers by way of an always-inline shim -
  // that would make this a single compare against the command buffer state we
  // are likely to access immediately after anyway and keep overheads minimal.
  if (IREE_LIKELY(iree_hal_collective_batch_is_empty(
          &command_buffer->collective_batch))) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(#9580): use CUDA graph capture so that the NCCL calls end up in the
  // graph:
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
  //
  // Something like:
  //  syms->cuStreamBeginCapture(nccl_stream);
  //  iree_hal_cuda2_nccl_submit_batch(command_buffer->context,
  //                                  &command_buffer->collective_batch,
  //                                  nccl_stream);
  //  syms->cuStreamEndCapture(nccl_stream, &child_graph);
  //  syms->cuGraphAddChildGraphNode(..., child_graph);
  //  syms->cuGraphDestroy(child_graph);  // probably, I think it gets cloned
  //
  // Note that we'll want to create a scratch stream that we use to perform the
  // capture - we could memoize that on the command buffer or on the device
  // (though that introduces potential threading issues). There may be a special
  // stream mode for these capture-only streams that is lighter weight than a
  // normal stream.
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "CUDA graph capture of collective operations not yet implemented");

  iree_hal_collective_batch_clear(&command_buffer->collective_batch);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);

  if (command_buffer->graph != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer cannot be re-recorded");
  }

  // Create a new empty graph to record into.
  IREE_CUDA_RETURN_IF_ERROR(command_buffer->symbols,
                            cuGraphCreate(&command_buffer->graph, /*flags=*/0),
                            "cuGraphCreate");

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);

  // Flush any pending collective batches.
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda2_graph_command_buffer_flush_collectives(command_buffer));

  // Reset state used during recording.
  command_buffer->last_node = NULL;

  // Compile the graph.
  CUgraphNode error_node = NULL;
  iree_status_t status = IREE_CURESULT_TO_STATUS(
      command_buffer->symbols,
      cuGraphInstantiate(&command_buffer->exec, command_buffer->graph,
                         &error_node,
                         /*logBuffer=*/NULL,
                         /*bufferSize=*/0));
  if (iree_status_is_ok(status)) {
    // No longer need the source graph used for construction.
    IREE_CUDA_IGNORE_ERROR(command_buffer->symbols,
                           cuGraphDestroy(command_buffer->graph));
    command_buffer->graph = NULL;
  }

  iree_hal_resource_set_freeze(command_buffer->resource_set);

  return iree_ok_status();
}

static void iree_hal_cuda2_graph_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // TODO(benvanik): tracy event stack.
}

static void iree_hal_cuda2_graph_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO(benvanik): tracy event stack.
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda2_graph_command_buffer_flush_collectives(command_buffer));

  // TODO: Implement barrier with Graph edges. Right now all the nodes are
  // serialized so this is a no-op.

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda2_graph_command_buffer_flush_collectives(command_buffer));

  // TODO: Implement barrier with Graph edges. Right now all the nodes are
  // serialized so this is a no-op.

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda2_graph_command_buffer_flush_collectives(command_buffer));

  // TODO: Implement barrier with Graph edges. Right now all the nodes are
  // serialized so this is a no-op.

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda2_graph_command_buffer_flush_collectives(command_buffer));

  // TODO: Implement barrier with Graph edges. Right now all the nodes are
  // serialized so this is a no-op.

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // We could mark the memory as invalidated so that if this is a managed buffer
  // CUDA does not try to copy it back to the host.
  return iree_ok_status();
}

// Splats a pattern value of 1/2/4 bytes out to a 4 byte value.
static uint32_t iree_hal_cuda2_splat_pattern(const void* pattern,
                                             size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *(const uint8_t*)(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *(const uint16_t*)(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *(const uint32_t*)(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_cuda2_graph_command_buffer_flush_collectives(command_buffer));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &target_buffer));

  CUdeviceptr target_device_buffer = iree_hal_cuda2_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  uint32_t pattern_4byte =
      iree_hal_cuda2_splat_pattern(pattern, pattern_length);
  CUDA_MEMSET_NODE_PARAMS params = {
      .dst = target_device_buffer + target_offset,
      .elementSize = pattern_length,
      .pitch = 0,                        // unused if height == 1
      .width = length / pattern_length,  // element count
      .height = 1,
      .value = pattern_4byte,
  };

  // Serialize all the nodes for now.
  CUgraphNode dep[] = {command_buffer->last_node};
  size_t numNode = command_buffer->last_node ? 1 : 0;
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddMemsetNode(&command_buffer->last_node, command_buffer->graph,
                           dep, numNode, &params, command_buffer->cu_context),
      "cuGraphAddMemsetNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_cuda2_graph_command_buffer_flush_collectives(command_buffer));

  // Allocate scratch space in the arena for the data and copy it in.
  // The update buffer API requires that the command buffer capture the host
  // memory at the time the method is called in case the caller wants to reuse
  // the memory. Because CUDA memcpys are async if we didn't copy it's possible
  // for the reused memory to change before the stream reaches the copy
  // operation and get the wrong data.
  uint8_t* storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_arena_allocate(&command_buffer->arena, length, (void**)&storage));
  memcpy(storage, (const uint8_t*)source_buffer + source_offset, length);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &target_buffer));

  CUdeviceptr target_device_buffer = iree_hal_cuda2_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  CUDA_MEMCPY3D params = {
      .srcMemoryType = CU_MEMORYTYPE_HOST,
      .srcHost = storage,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstDevice = target_device_buffer,
      .dstXInBytes = iree_hal_buffer_byte_offset(target_buffer) + target_offset,
      .WidthInBytes = length,
      .Height = 1,
      .Depth = 1,
  };

  // Serialize all the nodes for now.
  CUgraphNode dep[] = {command_buffer->last_node};
  size_t numNode = command_buffer->last_node ? 1 : 0;

  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddMemcpyNode(&command_buffer->last_node, command_buffer->graph,
                           dep, numNode, &params, command_buffer->cu_context),
      "cuGraphAddMemcpyNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_cuda2_graph_command_buffer_flush_collectives(command_buffer));

  const iree_hal_buffer_t* buffers[2] = {source_buffer, target_buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_resource_set_insert(command_buffer->resource_set, 2, buffers));

  CUdeviceptr target_device_buffer = iree_hal_cuda2_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  CUdeviceptr source_device_buffer = iree_hal_cuda2_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_buffer));
  source_offset += iree_hal_buffer_byte_offset(source_buffer);

  CUDA_MEMCPY3D params = {
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcDevice = source_device_buffer,
      .srcXInBytes = source_offset,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstDevice = target_device_buffer,
      .dstXInBytes = target_offset,
      .WidthInBytes = length,
      .Height = 1,
      .Depth = 1,
  };

  // Serialize all the nodes for now.
  CUgraphNode dep[] = {command_buffer->last_node};
  size_t numNode = command_buffer->last_node ? 1 : 0;

  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddMemcpyNode(&command_buffer->last_node, command_buffer->graph,
                           dep, numNode, &params, command_buffer->cu_context),
      "cuGraphAddMemcpyNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  return iree_hal_collective_batch_append(&command_buffer->collective_batch,
                                          channel, op, param, send_binding,
                                          recv_binding, element_count);
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  iree_host_size_t constant_base_index = offset / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < values_length / sizeof(int32_t); i++) {
    command_buffer->push_constant[i + constant_base_index] =
        ((uint32_t*)values)[i];
  }
  return iree_ok_status();
}

typedef struct {
  // The original index into the iree_hal_descriptor_set_binding_t array.
  uint32_t index;
  // The descriptor binding number.
  uint32_t binding;
} iree_hal_cuda2_binding_mapping_t;

// Compares two iree_hal_cuda2_binding_mapping_t according to the descriptor
// binding number.
static int compare_binding_index(const void* a, const void* b) {
  const iree_hal_cuda2_binding_mapping_t buffer_a =
      *(const iree_hal_cuda2_binding_mapping_t*)a;
  const iree_hal_cuda2_binding_mapping_t buffer_b =
      *(const iree_hal_cuda2_binding_mapping_t*)b;
  return buffer_a.binding < buffer_b.binding ? -1 : 1;
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  IREE_ASSERT_LT(binding_count, IREE_HAL_CUDA_MAX_BINDING_COUNT,
                 "binding count larger than the max expected");
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t base_binding =
      iree_hal_cuda2_pipeline_layout_base_binding_index(pipeline_layout, set);

  // Convention with the compiler side. We map descriptor bindings to kernel
  // argument. We compact the descriptor binding number ranges to get a dense
  // set of kernel arguments and keep them ordered based on the descriptor
  // binding index.
  iree_hal_cuda2_binding_mapping_t
      sorted_bindings[IREE_HAL_CUDA_MAX_BINDING_COUNT];
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    sorted_bindings[i].index = i;
    sorted_bindings[i].binding = bindings[i].binding;
  }
  // Sort the binding based on the binding index and map the (base offset +
  // array index) to the kernel argument index.
  // TODO: remove this sort - it's thankfully small (1-8 on average) but we
  // should be able to avoid it like we do on the CPU side with a bitmap.
  qsort(sorted_bindings, binding_count,
        sizeof(iree_hal_cuda2_binding_mapping_t), compare_binding_index);

  for (iree_host_size_t i = 0; i < binding_count; i++) {
    const iree_hal_descriptor_set_binding_t* binding =
        &bindings[sorted_bindings[i].index];
    CUdeviceptr device_ptr = 0;
    if (binding->buffer) {
      CUdeviceptr device_buffer = iree_hal_cuda2_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(binding->buffer));
      iree_device_size_t offset = iree_hal_buffer_byte_offset(binding->buffer);
      device_ptr = device_buffer + offset + binding->offset;
    };
    *((CUdeviceptr*)command_buffer->current_descriptor[base_binding + i]) =
        device_ptr;
    if (binding->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &binding->buffer));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_cuda2_graph_command_buffer_t* command_buffer =
      iree_hal_cuda2_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_cuda2_graph_command_buffer_flush_collectives(command_buffer));

  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  iree_hal_cuda2_kernel_params_t kernel_params;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda2_native_executable_entry_point_kernel_params(
              executable, entry_point, &kernel_params));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  // Patch the push constants in the kernel arguments.
  iree_host_size_t num_constants =
      iree_hal_cuda2_pipeline_layout_push_constant_count(kernel_params.layout);
  iree_host_size_t base_index =
      iree_hal_cuda2_pipeline_layout_push_constant_index(kernel_params.layout);
  for (iree_host_size_t i = 0; i < num_constants; i++) {
    *((uint32_t*)command_buffer->current_descriptor[base_index + i]) =
        command_buffer->push_constant[i];
  }

  CUDA_KERNEL_NODE_PARAMS params = {
      .func = kernel_params.function,
      .blockDimX = kernel_params.block_size[0],
      .blockDimY = kernel_params.block_size[1],
      .blockDimZ = kernel_params.block_size[2],
      .gridDimX = workgroup_x,
      .gridDimY = workgroup_y,
      .gridDimZ = workgroup_z,
      .kernelParams = command_buffer->current_descriptor,
      .sharedMemBytes = kernel_params.shared_memory_size,
  };

  // Serialize all the nodes for now.
  CUgraphNode dep[] = {command_buffer->last_node};
  size_t numNodes = command_buffer->last_node ? 1 : 0;

  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddKernelNode(&command_buffer->last_node, command_buffer->graph,
                           dep, numNodes, &params),
      "cuGraphAddKernelNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect dispatch not yet implemented");
}

static iree_status_t iree_hal_cuda2_graph_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  // TODO(#10144): support indirect command buffers by adding subgraph nodes and
  // tracking the binding table for future cuGraphExecKernelNodeSetParams usage.
  // Need to look into how to update the params of the subgraph nodes - is the
  // graph exec the outer one and if so will it allow node handles from the
  // subgraphs?
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect command buffers not yet implemented");
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda2_graph_command_buffer_vtable = {
        .destroy = iree_hal_cuda2_graph_command_buffer_destroy,
        .begin = iree_hal_cuda2_graph_command_buffer_begin,
        .end = iree_hal_cuda2_graph_command_buffer_end,
        .begin_debug_group =
            iree_hal_cuda2_graph_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_cuda2_graph_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_cuda2_graph_command_buffer_execution_barrier,
        .signal_event = iree_hal_cuda2_graph_command_buffer_signal_event,
        .reset_event = iree_hal_cuda2_graph_command_buffer_reset_event,
        .wait_events = iree_hal_cuda2_graph_command_buffer_wait_events,
        .discard_buffer = iree_hal_cuda2_graph_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_cuda2_graph_command_buffer_fill_buffer,
        .update_buffer = iree_hal_cuda2_graph_command_buffer_update_buffer,
        .copy_buffer = iree_hal_cuda2_graph_command_buffer_copy_buffer,
        .collective = iree_hal_cuda2_graph_command_buffer_collective,
        .push_constants = iree_hal_cuda2_graph_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_cuda2_graph_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_cuda2_graph_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_cuda2_graph_command_buffer_dispatch_indirect,
        .execute_commands =
            iree_hal_cuda2_graph_command_buffer_execute_commands,
};
