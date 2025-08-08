// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/graph.h"
#include "experimental/streaming/internal.h"
#include "iree/base/api.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_streaming_graph_exec_t (instantiation)
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_graph_block_type_e {
  // iree_hal_device_queue_barrier
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER = 0,
  // iree_hal_device_queue_fill
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL,
  // iree_hal_device_queue_copy
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY,
  // iree_hal_device_queue_host_call
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
  // iree_hal_device_queue_dispatch
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH,
  // iree_hal_device_queue_execute
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
} iree_hal_streaming_graph_block_type_t;

typedef void (*iree_hal_streaming_host_callback_t)(void* user_data);

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
typedef struct iree_hal_streaming_graph_barrier_block_attrs_t {
  iree_hal_execute_flags_t flags;
} iree_hal_streaming_graph_barrier_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL
typedef struct iree_hal_streaming_graph_fill_block_attrs_t {
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint64_t pattern;
  iree_host_size_t pattern_length;
  iree_hal_fill_flags_t flags;
} iree_hal_streaming_graph_fill_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY
typedef struct iree_hal_streaming_graph_copy_block_attrs_t {
  iree_hal_buffer_t* source_buffer;
  iree_device_size_t source_offset;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  iree_hal_copy_flags_t flags;
} iree_hal_streaming_graph_copy_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL
typedef struct iree_hal_streaming_graph_host_call_block_attrs_t {
  iree_hal_streaming_host_callback_t fn;
  void* user_data;
  iree_hal_host_call_flags_t flags;
} iree_hal_streaming_graph_host_call_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH
typedef struct iree_hal_streaming_graph_dispatch_block_attrs_t {
  iree_hal_executable_t* executable;
  iree_host_size_t entry_point;
  iree_hal_dispatch_config_t config;
  iree_const_byte_span_t constants;
  iree_hal_buffer_ref_list_t bindings;
  iree_hal_dispatch_flags_t flags;
} iree_hal_streaming_graph_dispatch_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE
typedef struct iree_hal_streaming_graph_execute_block_attrs_t {
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_execute_flags_t flags;
} iree_hal_streaming_graph_execute_block_attrs_t;

// Block-specific data stored at the end of the block allocation.
typedef union iree_hal_streaming_graph_block_attrs_t {
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
  iree_hal_streaming_graph_barrier_block_attrs_t barrier;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL
  iree_hal_streaming_graph_fill_block_attrs_t fill;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY
  iree_hal_streaming_graph_copy_block_attrs_t copy;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL
  iree_hal_streaming_graph_host_call_block_attrs_t host_call;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH
  iree_hal_streaming_graph_dispatch_block_attrs_t dispatch;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE
  iree_hal_streaming_graph_execute_block_attrs_t execute;
} iree_hal_streaming_graph_block_attrs_t;

// Represents an atomically executable block of work in a graph.
typedef struct iree_hal_streaming_graph_block_t {
  iree_hal_streaming_graph_block_type_t type;

  // First node in sorted array.
  uint32_t node_start_index;
  // Number of nodes in this block.
  uint32_t node_count;

  // Semaphore synchronization.
  uint16_t wait_semaphore_count;
  uint16_t signal_semaphore_count;

  // Variable-length data follows:
  // - uint16_t wait_semaphore_indices[wait_semaphore_count]
  // - uint32_t wait_payload_deltas[wait_semaphore_count]
  // - uint16_t signal_semaphore_indices[signal_semaphore_count]
  // - uint32_t signal_payload_deltas[signal_semaphore_count]
  // - iree_hal_streaming_graph_block_attrs_t attrs (based on type)
} iree_hal_streaming_graph_block_t;

// Pointers to all variable-length arrays in a block.
typedef struct iree_hal_streaming_graph_block_ptrs_t {
  uint16_t* wait_semaphore_indices;
  uint32_t* wait_payload_deltas;
  uint16_t* signal_semaphore_indices;
  uint32_t* signal_payload_deltas;
  iree_hal_streaming_graph_block_attrs_t* attrs;
} iree_hal_streaming_graph_block_ptrs_t;

typedef struct iree_hal_streaming_graph_exec_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  iree_hal_streaming_context_t* context;  // retained
  iree_hal_streaming_graph_t* graph;      // retained

  // Arena allocator used for block allocations and inlined data.
  iree_arena_allocator_t arena_allocator;

  // Immutable block list created during instantiate.
  iree_hal_streaming_graph_block_t** blocks;
  uint32_t block_count;

  // Semaphore pool for internal synchronization.
  uint32_t semaphore_count;
  iree_hal_semaphore_t** semaphores;
  uint64_t* semaphore_base_values;

  // Resource set for automatic cleanup.
  iree_hal_resource_set_t* resource_set;

  unsigned long long flags;

  // Mutex needed for launch/update.
  iree_slim_mutex_t mutex;
} iree_hal_streaming_graph_exec_t;

static void iree_hal_streaming_graph_exec_destroy(
    iree_hal_streaming_graph_exec_t* exec);

// Internal: Create an exec object (called by graph.c).
iree_status_t iree_hal_streaming_graph_exec_create(
    iree_hal_streaming_context_t* context, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_allocator_t host_allocator,
    iree_hal_streaming_graph_exec_t** out_exec) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_exec);
  *out_exec = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_exec_t* exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*exec), (void**)&exec));

  iree_atomic_ref_count_init(&exec->ref_count);
  exec->host_allocator = host_allocator;
  exec->context = context;
  iree_hal_streaming_context_retain(exec->context);
  exec->graph = graph;
  iree_hal_streaming_graph_retain(exec->graph);
  iree_arena_initialize(&context->device_entry->block_pool,
                        &exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->semaphore_base_values = NULL;
  exec->resource_set = NULL;
  exec->flags = flags;
  iree_slim_mutex_initialize(&exec->mutex);

  // Create resource set for automatic cleanup.
  iree_status_t status = iree_hal_resource_set_allocate(
      &context->device_entry->block_pool, &exec->resource_set);

  if (iree_status_is_ok(status)) {
    *out_exec = exec;
  } else {
    iree_hal_streaming_graph_exec_destroy(exec);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_graph_exec_destroy(
    iree_hal_streaming_graph_exec_t* exec) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release all resources via resource set.
  // This handles command buffers, semaphores, buffers, executables, etc.
  iree_hal_resource_set_free(exec->resource_set);

  // NOTE: the arena contains all blocks and semaphores so we don't need to free
  // those.
  iree_arena_deinitialize(&exec->arena_allocator);

  iree_hal_streaming_graph_release(exec->graph);
  iree_hal_streaming_context_release(exec->context);
  iree_slim_mutex_deinitialize(&exec->mutex);

  iree_allocator_t host_allocator = exec->host_allocator;
  iree_allocator_free(host_allocator, exec);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_graph_exec_retain(
    iree_hal_streaming_graph_exec_t* exec) {
  if (exec) {
    iree_atomic_ref_count_inc(&exec->ref_count);
  }
}

void iree_hal_streaming_graph_exec_release(
    iree_hal_streaming_graph_exec_t* exec) {
  if (exec && iree_atomic_ref_count_dec(&exec->ref_count) == 1) {
    iree_hal_streaming_graph_exec_destroy(exec);
  }
}

// Calculate the size needed for a block with variable-length arrays.
static iree_host_size_t iree_hal_streaming_graph_block_calculate_size(
    uint16_t wait_semaphore_count, uint16_t signal_semaphore_count) {
  iree_host_size_t size = sizeof(iree_hal_streaming_graph_block_t);
  size += wait_semaphore_count * sizeof(uint16_t);  // wait_semaphore_indices
  size += wait_semaphore_count * sizeof(uint32_t);  // wait_payload_deltas
  size +=
      signal_semaphore_count * sizeof(uint16_t);  // signal_semaphore_indices
  size += signal_semaphore_count * sizeof(uint32_t);  // signal_payload_deltas
  size += sizeof(iree_hal_streaming_graph_block_attrs_t);  // type-specific data
  return size;
}

// Get pointers to all variable-length arrays in a block.
static inline void iree_hal_streaming_graph_block_get_ptrs(
    iree_hal_streaming_graph_block_t* block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  uint8_t* ptr = (uint8_t*)block + sizeof(*block);

  out_ptrs->wait_semaphore_indices = (uint16_t*)ptr;
  ptr +=
      block->wait_semaphore_count * sizeof(*out_ptrs->wait_semaphore_indices);

  out_ptrs->wait_payload_deltas = (uint32_t*)ptr;
  ptr += block->wait_semaphore_count * sizeof(*out_ptrs->wait_payload_deltas);

  out_ptrs->signal_semaphore_indices = (uint16_t*)ptr;
  ptr += block->signal_semaphore_count *
         sizeof(*out_ptrs->signal_semaphore_indices);

  out_ptrs->signal_payload_deltas = (uint32_t*)ptr;
  ptr +=
      block->signal_semaphore_count * sizeof(*out_ptrs->signal_payload_deltas);

  out_ptrs->attrs = (iree_hal_streaming_graph_block_attrs_t*)ptr;
}

// Allocates a block with variable-length arrays.
static iree_status_t iree_hal_streaming_graph_block_allocate(
    iree_arena_allocator_t* arena_allocator,
    iree_hal_streaming_graph_block_type_t type, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  const iree_host_size_t total_size =
      iree_hal_streaming_graph_block_calculate_size(wait_semaphore_count,
                                                    signal_semaphore_count);
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena_allocator, total_size, (void**)&block));

  block->type = type;
  block->node_start_index = node_start_index;
  block->node_count = node_count;
  block->wait_semaphore_count = wait_semaphore_count;
  block->signal_semaphore_count = signal_semaphore_count;

  iree_hal_streaming_graph_block_get_ptrs(block, out_ptrs);
  *out_block = block;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_barrier_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_execute_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  // Set barrier attributes.
  iree_hal_streaming_graph_barrier_block_attrs_t* attrs =
      &out_ptrs->attrs->barrier;
  attrs->flags = flags;

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_fill_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags, iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL, node_start_index,
              node_count, wait_semaphore_count, signal_semaphore_count, &block,
              out_ptrs));

  // Set fill attributes.
  iree_hal_streaming_graph_fill_block_attrs_t* attrs = &out_ptrs->attrs->fill;
  attrs->target_buffer = target_buffer;
  attrs->target_offset = target_offset;
  attrs->length = length;
  attrs->flags = flags;

  // Copy pattern data if provided.
  if (pattern_length > 0) {
    IREE_ASSERT(pattern_length < sizeof(attrs->pattern));
    memcpy(&attrs->pattern, pattern, pattern_length);
    attrs->pattern_length = pattern_length;
  }

  // Add buffer to resource set.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(exec->resource_set, 1, &target_buffer));

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_copy_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_copy_flags_t flags, iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  iree_hal_streaming_graph_block_ptrs_t ptrs = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY, node_start_index,
              node_count, wait_semaphore_count, signal_semaphore_count, &block,
              &ptrs));

  // Set copy attributes.
  iree_hal_streaming_graph_copy_block_attrs_t* attrs = &out_ptrs->attrs->copy;
  attrs->source_buffer = source_buffer;
  attrs->source_offset = source_offset;
  attrs->target_buffer = target_buffer;
  attrs->target_offset = target_offset;
  attrs->length = length;
  attrs->flags = flags;

  // Add buffers to resource set.
  void* resources[2] = {source_buffer, target_buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(exec->resource_set,
                                       IREE_ARRAYSIZE(resources), resources));

  *out_block = block;
  *out_ptrs = ptrs;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_host_call_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, void (*fn)(void* user_data),
    void* user_data, iree_hal_host_call_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  // Set host call attributes.
  iree_hal_streaming_graph_host_call_block_attrs_t* attrs =
      &out_ptrs->attrs->host_call;
  attrs->fn = fn;
  attrs->user_data = user_data;
  attrs->flags = flags;

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_dispatch_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_executable_t* executable,
    iree_host_size_t entry_point, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  // Set dispatch attributes.
  iree_hal_streaming_graph_dispatch_block_attrs_t* attrs =
      &out_ptrs->attrs->dispatch;
  attrs->executable = executable;
  attrs->entry_point = entry_point;
  attrs->config = config;
  attrs->flags = flags;

  // Copy constants if provided.
  if (constants.data_length > 0) {
    void* constants_copy = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&exec->arena_allocator, constants.data_length,
                                &constants_copy));
    memcpy(constants_copy, constants.data, constants.data_length);
    attrs->constants =
        iree_make_const_byte_span(constants_copy, constants.data_length);
  } else {
    attrs->constants = iree_const_byte_span_empty();
  }

  // Copy bindings if provided.
  if (bindings.count > 0) {
    iree_hal_buffer_ref_t* bindings_copy = NULL;
    const iree_host_size_t bindings_size =
        bindings.count * sizeof(*bindings_copy);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&exec->arena_allocator, bindings_size,
                                (void**)&bindings_copy));
    memcpy(bindings_copy, bindings.values, bindings_size);
    attrs->bindings.count = bindings.count;
    attrs->bindings.values = bindings_copy;

    // Add buffers to resource set.
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
        exec->resource_set, bindings.count, bindings.values,
        offsetof(iree_hal_buffer_ref_t, buffer),
        sizeof(iree_hal_buffer_ref_t)));
  } else {
    attrs->bindings = iree_hal_buffer_ref_list_empty();
  }

  // Add executable to resource set.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(exec->resource_set, 1, &executable));

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_execute_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_execute_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  iree_hal_streaming_graph_execute_block_attrs_t* attrs =
      &out_ptrs->attrs->execute;
  attrs->flags = flags;

  // Create command buffer.
  // Note that we set UNRETAINED as we ensure the resources we have to track are
  // retained at the graph exec level and CUDA/HIP don't make any statements
  // about resource lifetime.
  //
  // TODO: limit queue affinity to the device being instantiated on, if scoped
  // to a queue. Currently we are assuming we are targeting a single
  // iree_hal_device_t but it should really be a pair of (iree_hal_device_t,
  // queue_affinity_mask).
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_create(
              exec->context->device, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
              IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
              /*binding_capacity=*/0, &attrs->command_buffer));

  // Add to resource set for cleanup.
  iree_status_t status = iree_hal_resource_set_insert(exec->resource_set, 1,
                                                      &attrs->command_buffer);

  // We don't technically retain a reference to it past here on the stack, just
  // in the resource set associated with the exec.
  iree_hal_command_buffer_release(attrs->command_buffer);

  if (iree_status_is_ok(status)) {
    *out_block = block;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_hal_streaming_node_index_set_t {
  uint32_t values[8];
  uint32_t count : 31;
  uint32_t invalid : 1;
} iree_hal_streaming_node_index_set_t;

// Resets the set to empty.
static inline void iree_hal_streaming_node_index_set_reset(
    iree_hal_streaming_node_index_set_t* set) {
  set->count = 0;
  set->invalid = 0;
}

// Returns true if the |set| is invalid or |value| is present.
static bool iree_hal_streaming_node_index_set_test_hazard(
    const iree_hal_streaming_node_index_set_t* set, uint32_t value) {
  if (set->invalid) return true;
  for (uint32_t i = 0; i < set->count; ++i) {
    if (set->values[i] == value) {
      return true;
    }
  }
  return false;
}

// Inserts |value| into the |set|.
// If the set has reached capacity it is set to invalid and all future tests
// will return a hazard.
static void iree_hal_streaming_node_index_set_insert(
    iree_hal_streaming_node_index_set_t* set, uint32_t value) {
  if (set->count >= IREE_ARRAYSIZE(set->values)) {
    set->invalid = 1;
    return;
  }
  set->values[set->count++] = value;
}

// Helper to record nodes from a partition into a command buffer.
static iree_status_t iree_hal_streaming_graph_record_partition(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_sort_node_t* sorted_nodes,
    uint32_t node_start_index, uint32_t node_count,
    const uint32_t* node_index_map, uint8_t stream_id,
    iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Begin recording command buffer.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin(command_buffer));

  // Scope the partition into a debug group.
  // TODO: propagate graph information (name, origin, etc).
  const iree_string_view_t label_name = iree_make_cstring_view("tbd_partition");
  const iree_hal_label_location_t* location = NULL;
  const iree_hal_label_color_t label_color = iree_hal_label_color_unspecified();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin_debug_group(command_buffer, label_name,
                                                    label_color, location));

  // Record nodes assigned to this stream.
  //
  // We only insert barriers when there's a dependency between a node already
  // recorded since the last barrier that the new node has a dependency on.
  // If the partitioning/sort did a good job the node should be in an order
  // where there are spans of ~1-4 operations that can run concurrently.
  // We use a small linear scan set to make the test for hazards faster: we have
  // the original unsorted node indices of dependencies but not the sorted ones
  // we'd need to index into the sorted_nodes list and this avoids needing to
  // do that mapping.
  iree_status_t status = iree_ok_status();
  uint32_t in_stream_count = 0;
  iree_hal_streaming_node_index_set_t barrier_index_set;
  iree_hal_streaming_node_index_set_reset(&barrier_index_set);
  for (uint32_t i = 0; iree_status_is_ok(status) && i < node_count; ++i) {
    iree_hal_streaming_graph_sort_node_t* sort_node =
        &sorted_nodes[node_start_index + i];
    // Ignore nodes from other streams.
    if (sort_node->stream_id != stream_id) continue;
    iree_hal_streaming_graph_node_t* node = sort_node->node;
    if (in_stream_count > 1) {
      // Insert a barrier between the previous node and this one, if needed.
      // Barriers are only required if there is a dependency edge between two
      // nodes. Note that this edge may span backwards a bit and to elide the
      // barrier we need to scan between the node that began the current barrier
      // block and this node in execution (sorted) order.
      //
      // TODO: SIMD-ify this - in the naive cases we have 1 dependency to check
      // in the set and that's best as a linear scan today. If we start to see
      // dependency sets that are larger (~4 or ~8) then SIMD would be better as
      // we can scan the whole dependency list against the set 4 or 8x faster.
      //
      // The data structure here isn't great - we really should bake out this
      // indirection during one of our earlier passes and set hazard bits
      // somewhere. The slowest part of the recording process is this loop (when
      // >1).
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        const uint32_t dependency_sort_index =
            node_index_map[node->dependencies[j]->node_index];
        const bool has_hazard = iree_hal_streaming_node_index_set_test_hazard(
            &barrier_index_set, dependency_sort_index);
        if (has_hazard) {
          // This node has a dependency on one or more nodes issued between the
          // last barrier and this. We have to insert an execution barrier to
          // ensure they complete.
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0, iree_hal_command_buffer_execution_barrier(
                      command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
                      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
                      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, NULL, 0, NULL));
          iree_hal_streaming_node_index_set_reset(&barrier_index_set);
        }
        iree_hal_streaming_node_index_set_insert(&barrier_index_set,
                                                 sort_node->sorted_index);
      }
    }
    ++in_stream_count;
    switch (node->type) {
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL: {
        const iree_hal_streaming_graph_kernel_node_attrs_t* attrs =
            &node->attrs.kernel;
        iree_hal_streaming_symbol_t* symbol = attrs->symbol;
        const iree_hal_dispatch_config_t config = {
            .workgroup_size =
                {
                    attrs->block_dim[0],
                    attrs->block_dim[1],
                    attrs->block_dim[2],
                },
            .workgroup_count =
                {
                    attrs->grid_dim[0],
                    attrs->grid_dim[1],
                    attrs->grid_dim[2],
                },
            .dynamic_workgroup_local_memory = attrs->shared_memory_bytes,
        };
        const iree_hal_dispatch_flags_t flags =
            attrs->bindings.count
                ? IREE_HAL_DISPATCH_FLAG_NONE
                : IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
        status = iree_hal_command_buffer_dispatch(
            command_buffer, symbol->module->executable, symbol->export_ordinal,
            config, attrs->constants, attrs->bindings, flags);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY: {
        const iree_hal_streaming_graph_memcpy_node_attrs_t* attrs =
            &node->attrs.memcpy;
        status = iree_hal_command_buffer_copy_buffer(
            command_buffer,
            iree_hal_streaming_convert_range_buffer_ref(attrs->src_ref,
                                                        attrs->size),
            iree_hal_streaming_convert_range_buffer_ref(attrs->dst_ref,
                                                        attrs->size),
            attrs->flags);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET: {
        const iree_hal_streaming_graph_memset_node_attrs_t* attrs =
            &node->attrs.memset;
        status = iree_hal_command_buffer_fill_buffer(
            command_buffer,
            iree_hal_streaming_convert_range_buffer_ref(
                attrs->dst_ref, attrs->pattern_size * attrs->count),
            &attrs->pattern, attrs->pattern_size, attrs->flags);
        break;
      }
      default: {
        // Non-recordable nodes shouldn't be here.
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "non-recordable node type %d in recordable partition",
            (int)node->type);
        break;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end_debug_group(command_buffer);
  }

  // End recording.
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_instantiate_locked(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Use the new scheduler to analyze and partition the graph.
  iree_hal_streaming_graph_schedule_t schedule;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_schedule_nodes(
              node_blocks, node_count, &exec->arena_allocator, &schedule));

  // Allocate block array.
  exec->block_count = schedule.block_count;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&exec->arena_allocator,
                              exec->block_count * sizeof(*exec->blocks),
                              (void**)&exec->blocks));

  // Calculate semaphore count needed.
  // We need semaphores at partition boundaries for synchronization.
  // Multi-stream partitions need join semaphores.
  //
  // DO NOT SUBMIT we don't want semaphores per block, just per max layer size
  // we can use timelines to advance between them?
  // this is bad
  uint32_t semaphore_count = 0;
  for (iree_host_size_t i = 0; i < schedule.partition_count - 1; i++) {
    if (schedule.partitions[i].stream_count > 1) {
      // Multi-stream partition needs one semaphore per stream for join.
      semaphore_count += schedule.partitions[i].stream_count;
    } else {
      // Single stream needs one semaphore.
      semaphore_count += 1;
    }
  }
  exec->semaphore_count = semaphore_count;

  if (exec->semaphore_count > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_arena_allocate(&exec->arena_allocator,
                            exec->semaphore_count * sizeof(*exec->semaphores),
                            (void**)&exec->semaphores));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(
                &exec->arena_allocator,
                exec->semaphore_count * sizeof(*exec->semaphore_base_values),
                (void**)&exec->semaphore_base_values));

    // Create internal semaphores.
    for (uint32_t i = 0; i < exec->semaphore_count; i++) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_semaphore_create(exec->context->device, 0ull,
                                        IREE_HAL_SEMAPHORE_FLAG_NONE,
                                        &exec->semaphores[i]));
      exec->semaphore_base_values[i] = 0;
    }

    // Add semaphores to the resource set for automatic cleanup.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert(
                exec->resource_set, exec->semaphore_count, exec->semaphores));
  }

  // Create blocks from partitions.
  uint32_t block_index = 0;
  uint32_t semaphore_index = 0;
  for (iree_host_size_t p = 0; p < schedule.partition_count; p++) {
    const iree_hal_streaming_graph_partition_t* partition =
        &schedule.partitions[p];

    // Determine wait semaphores for chaining FROM the previous partition.
    // The first partition waits on the original submission stream timeline
    // semaphores. Subsequent partitions wait on the previous partitions
    // semaphores, of which there may be several for a join operation.
    // Note that we don't allocate space for the initial semaphores as those are
    // part of the submission, not the exec object.
    uint16_t wait_semaphore_count = 0;
    if (p > 0) {
      iree_hal_streaming_graph_partition_t* prev_partition =
          &schedule.partitions[p - 1];
      wait_semaphore_count =
          prev_partition->stream_count > 1 ? prev_partition->stream_count : 1;
    }

    // Determine signal semaphores for chaining TO the next partition.
    // Each partition gets at least one signal semaphore while multi-stream
    // partitions get one per stream to allow the subsequent partition to join
    // them. The final partition signals the original submission stream timeline
    // semaphores. Note that we don't allocate space for the final semaphores as
    // those are part of the submission, not the exec object.
    uint16_t signal_semaphore_count = 0;
    if (p < schedule.partition_count - 1) {
      signal_semaphore_count =
          partition->stream_count > 1 ? partition->stream_count : 1;
    }

    if (partition->type == IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE) {
      // DO NOT SUBMIT single node optimization, needs refactoring of this
      // function.
      // If only one node is in the partition and it's recordable, we
      // may be able to route it to a dedicated partition type. if
      const uint8_t stream_count = partition->stream_count;
      const uint32_t partition_wait_semaphore_start =
          semaphore_index - wait_semaphore_count;
      const uint32_t partition_signal_semaphore_start = semaphore_index;
      for (uint8_t s = 0; s < stream_count; s++) {
        iree_hal_streaming_graph_block_t* block = NULL;

        // All streams in partition wait on same semaphores from previous.
        // Each stream in multi-stream partition signals its own semaphore.
        // Single stream signals all semaphores for the partition.
        // But if this is the last partition (signal_semaphore_count=0), don't
        // signal.
        uint16_t block_signal_count = 0;
        if (signal_semaphore_count > 0) {
          block_signal_count = (stream_count > 1) ? 1 : signal_semaphore_count;
        }
        iree_hal_streaming_graph_block_ptrs_t ptrs;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_create_execute_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, block_signal_count,
                    IREE_HAL_EXECUTE_FLAG_NONE, &block, &ptrs));

        // Record nodes for this stream into the command buffer.
        // For single stream (s=0), records all nodes.
        // For multi-stream, records nodes filtered by stream_id.
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_record_partition(
                    exec, schedule.sorted_nodes, partition->start_index,
                    partition->count, schedule.node_index_map, s,
                    ptrs.attrs->execute.command_buffer));

        // Set up semaphore indices.
        if (wait_semaphore_count > 0) {
          for (uint16_t w = 0; w < wait_semaphore_count; w++) {
            ptrs.wait_semaphore_indices[w] = partition_wait_semaphore_start + w;
            ptrs.wait_payload_deltas[w] = 1;
          }
        }
        if (block_signal_count > 0) {
          if (stream_count > 1) {
            // Multi-stream: each stream signals its own semaphore.
            ptrs.signal_semaphore_indices[0] =
                partition_signal_semaphore_start + s;
            ptrs.signal_payload_deltas[0] = 1;
          } else {
            // Single stream: signal all semaphores.
            for (uint16_t i = 0; i < block_signal_count; i++) {
              ptrs.signal_semaphore_indices[i] =
                  partition_signal_semaphore_start + i;
              ptrs.signal_payload_deltas[i] = 1;
            }
          }
        }

        exec->blocks[block_index++] = block;
      }

      // Advance semaphore index by the number of signal semaphores.
      semaphore_index += signal_semaphore_count;
    } else {
      // Set up semaphore indices.
      iree_hal_streaming_graph_block_t* block = NULL;
      iree_hal_streaming_graph_block_ptrs_t ptrs;
      if (partition->type ==
          IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL) {
        // Host call gets its own block.
        iree_hal_streaming_graph_node_t* node =
            schedule.sorted_nodes[partition->start_index].node;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_create_host_call_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, signal_semaphore_count,
                    node->attrs.host.fn, node->attrs.host.user_data,
                    IREE_HAL_HOST_CALL_FLAG_NONE, &block, &ptrs));
      } else {
        // Empty/barrier partition.
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_create_barrier_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, signal_semaphore_count,
                    IREE_HAL_EXECUTE_FLAG_NONE, &block, &ptrs));
      }
      if (wait_semaphore_count > 0) {
        for (uint16_t w = 0; w < wait_semaphore_count; w++) {
          ptrs.wait_semaphore_indices[w] =
              semaphore_index - wait_semaphore_count + w;
          ptrs.wait_payload_deltas[w] = 1;
        }
      }
      if (signal_semaphore_count > 0) {
        for (uint16_t i = 0; i < signal_semaphore_count; i++) {
          ptrs.signal_semaphore_indices[i] = semaphore_index + i;
          ptrs.signal_payload_deltas[i] = 1;
        }
      }
      // Advance semaphore index by all signal semaphores.
      semaphore_index += signal_semaphore_count;
      exec->blocks[block_index++] = block;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_host_callback(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_host_callback_t call_fn =
      (iree_hal_streaming_host_callback_t)args[0];
  void* call_user_data = (void*)args[1];
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, args[0]);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, args[1]);
  call_fn(call_user_data);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_exec_launch(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Flush stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  // Query current values of internal semaphores for delta encoding.
  // Strategy A: Reuse semaphores - query current values.
  for (uint32_t i = 0; i < exec->semaphore_count; i++) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_query(exec->semaphores[i],
                                     &exec->semaphore_base_values[i]));
  }

  // Mutex needed for launch per CUDA docs.
  iree_slim_mutex_lock(&exec->mutex);

  // Save the stream's initial state.
  uint64_t stream_wait_value = stream->completed_value;
  ++stream->pending_value;
  uint64_t stream_signal_value = stream->pending_value;

  // Submit all blocks. They will synchronize with each other via internal
  // semaphores. This allows concurrent execution of independent blocks.
  // Track new semaphore base values to update after all blocks are submitted.
  // We copy existing base values so we preserve any that aren't updated during
  // the loop below.
  uint64_t* new_base_values = NULL;
  if (exec->semaphore_count > 0) {
    const iree_host_size_t base_values_size =
        exec->semaphore_count * sizeof(uint64_t);
    new_base_values = (uint64_t*)iree_alloca(base_values_size);
    memcpy(new_base_values, exec->semaphore_base_values, base_values_size);
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t block_index = 0;
       iree_status_is_ok(status) && block_index < exec->block_count;
       block_index++) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[block_index];

    // Get pointers to block data.
    iree_hal_streaming_graph_block_ptrs_t ptrs;
    iree_hal_streaming_graph_block_get_ptrs(block, &ptrs);

    // Build wait and signal semaphore lists from block's stored indices.
    // Calculate total semaphores needed for this block.
    // +2 for potential stream timeline semaphores (first block waits, last
    // signals).
    const iree_host_size_t total_semaphores =
        block->wait_semaphore_count + block->signal_semaphore_count + 2;

    // Allocate arrays for this block.
    iree_hal_semaphore_t** semaphore_array =
        (iree_hal_semaphore_t**)iree_alloca(total_semaphores *
                                            sizeof(iree_hal_semaphore_t*));
    uint64_t* value_array =
        (uint64_t*)iree_alloca(total_semaphores * sizeof(uint64_t));

    // Subset for wait and signal.
    // Maximum wait count is block->wait_semaphore_count + 1 (for stream
    // timeline).
    iree_hal_semaphore_t** wait_sems = semaphore_array;
    uint64_t* wait_vals = value_array;
    iree_hal_semaphore_t** signal_sems =
        semaphore_array + (block->wait_semaphore_count + 1);
    uint64_t* signal_vals = value_array + (block->wait_semaphore_count + 1);

    // First block waits on stream timeline.
    iree_host_size_t wait_count = 0;
    if (block_index == 0 && stream_wait_value > 0) {
      wait_sems[wait_count] = stream->timeline_semaphore;
      wait_vals[wait_count] = stream_wait_value;
      wait_count++;
    }

    // Add internal wait semaphores based on block's indices.
    for (uint16_t i = 0; i < block->wait_semaphore_count; i++) {
      const uint16_t semaphore_index = ptrs.wait_semaphore_indices[i];
      const uint32_t delta = ptrs.wait_payload_deltas[i];
      wait_sems[wait_count] = exec->semaphores[semaphore_index];
      wait_vals[wait_count] =
          exec->semaphore_base_values[semaphore_index] + delta;
      wait_count++;
    }

    // Add internal signal semaphores based on block's indices.
    iree_host_size_t signal_count = 0;
    for (uint16_t i = 0; i < block->signal_semaphore_count; i++) {
      const uint16_t semaphore_index = ptrs.signal_semaphore_indices[i];
      const uint32_t delta = ptrs.signal_payload_deltas[i];
      signal_sems[signal_count] = exec->semaphores[semaphore_index];
      signal_vals[signal_count] =
          exec->semaphore_base_values[semaphore_index] + delta;
      // Track new base value for next launch (update after all blocks
      // submitted).
      new_base_values[semaphore_index] = signal_vals[signal_count];
      signal_count++;
    }

    // Last block signals stream timeline.
    if (block_index == exec->block_count - 1) {
      signal_sems[signal_count] = stream->timeline_semaphore;
      signal_vals[signal_count] = stream_signal_value;
      signal_count++;
    }

    iree_hal_semaphore_list_t wait_semaphores = {
        .count = wait_count,
        .semaphores = wait_sems,
        .payload_values = wait_vals,
    };
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = signal_count,
        .semaphores = signal_sems,
        .payload_values = signal_vals,
    };

    // Submit block based on type.
    switch (block->type) {
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER: {
        status = iree_hal_device_queue_barrier(
            stream->context->device, stream->queue_affinity, wait_semaphores,
            signal_semaphores, ptrs.attrs->barrier.flags);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL: {
        status = iree_hal_device_queue_fill(
            stream->context->device, stream->queue_affinity, wait_semaphores,
            signal_semaphores, ptrs.attrs->fill.target_buffer,
            ptrs.attrs->fill.target_offset, ptrs.attrs->fill.length,
            &ptrs.attrs->fill.pattern, ptrs.attrs->fill.pattern_length,
            ptrs.attrs->fill.flags);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY: {
        status = iree_hal_device_queue_copy(
            stream->context->device, stream->queue_affinity, wait_semaphores,
            signal_semaphores, ptrs.attrs->copy.source_buffer,
            ptrs.attrs->copy.source_offset, ptrs.attrs->copy.target_buffer,
            ptrs.attrs->copy.target_offset, ptrs.attrs->copy.length,
            ptrs.attrs->copy.flags);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH: {
        iree_hal_buffer_ref_list_t bindings_list = {
            .count = ptrs.attrs->dispatch.bindings.count,
            .values = ptrs.attrs->dispatch.bindings.values,
        };
        status = iree_hal_device_queue_dispatch(
            stream->context->device, stream->queue_affinity, wait_semaphores,
            signal_semaphores, ptrs.attrs->dispatch.executable,
            ptrs.attrs->dispatch.entry_point, ptrs.attrs->dispatch.config,
            ptrs.attrs->dispatch.constants, bindings_list,
            ptrs.attrs->dispatch.flags);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE: {
        status = iree_hal_device_queue_execute(
            stream->context->device, stream->queue_affinity, wait_semaphores,
            signal_semaphores, ptrs.attrs->execute.command_buffer,
            iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL: {
        uint64_t call_args[4] = {
            (uint64_t)ptrs.attrs->host_call.fn,
            (uint64_t)ptrs.attrs->host_call.user_data,
        };
        status = iree_hal_device_queue_host_call(
            stream->context->device, stream->queue_affinity, wait_semaphores,
            signal_semaphores,
            iree_hal_make_host_call(iree_hal_streaming_graph_host_callback,
                                    NULL),
            call_args, ptrs.attrs->host_call.flags);
        break;
      }
      default: {
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "unsupported block type %u", block->type);
        break;
      }
    }
  }

  // Update semaphore base values for next launch now that all blocks are
  // submitted.
  if (iree_status_is_ok(status) && exec->semaphore_count > 0) {
    memcpy(exec->semaphore_base_values, new_base_values,
           exec->semaphore_count * sizeof(uint64_t));
  }

  iree_slim_mutex_unlock(&exec->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_update(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_graph_t* graph) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Mutex needed for update per CUDA docs.
  iree_slim_mutex_lock(&exec->mutex);

  // TODO: Update the executable graph from the new template graph.
  // For now, just return unsupported.
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "graph update not yet implemented");

  iree_slim_mutex_unlock(&exec->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
