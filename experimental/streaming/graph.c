// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/graph.h"

#include "experimental/streaming/internal.h"

//===----------------------------------------------------------------------===//
// iree_hal_streaming_graph_t (template)
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_graph_destroy(iree_hal_streaming_graph_t* graph);

iree_status_t iree_hal_streaming_graph_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_graph_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_graph_t** out_graph) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_graph);
  *out_graph = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_t* graph = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*graph), (void**)&graph));

  iree_atomic_ref_count_init(&graph->ref_count);

  // Initialize the arena using the device's block pool.
  iree_hal_streaming_device_t* device = context->device_entry;
  iree_arena_initialize(&device->block_pool, &graph->arena);
  graph->arena_allocator = iree_arena_allocator(&graph->arena);

  graph->node_blocks = NULL;
  graph->current_node_block = NULL;
  graph->node_count = 0;
  graph->root_blocks = NULL;
  graph->current_root_block = NULL;
  graph->root_count = 0;
  graph->flags = flags;
  graph->context = context;
  iree_hal_streaming_context_retain(context);
  iree_slim_mutex_initialize(&graph->mutex);
  graph->host_allocator = host_allocator;

  *out_graph = graph;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_streaming_graph_destroy(
    iree_hal_streaming_graph_t* graph) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reset the arena - this frees all nodes and arrays at once.
  // The arena returns all blocks to the device's block pool for reuse.
  iree_arena_deinitialize(&graph->arena);

  // Release context.
  iree_hal_streaming_context_release(graph->context);

  // Deinitialize synchronization.
  iree_slim_mutex_deinitialize(&graph->mutex);

  // Free graph memory itself (not allocated from arena).
  const iree_allocator_t host_allocator = graph->host_allocator;
  iree_allocator_free(host_allocator, graph);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_graph_retain(iree_hal_streaming_graph_t* graph) {
  if (graph) {
    iree_atomic_ref_count_inc(&graph->ref_count);
  }
}

void iree_hal_streaming_graph_release(iree_hal_streaming_graph_t* graph) {
  if (graph && iree_atomic_ref_count_dec(&graph->ref_count) == 1) {
    iree_hal_streaming_graph_destroy(graph);
  }
}

// Helper to allocate a graph node with trailing dependencies and extra data.
static iree_status_t iree_hal_streaming_graph_allocate_node(
    iree_allocator_t allocator, iree_host_size_t dependency_count,
    iree_host_size_t extra_data_size,
    iree_hal_streaming_graph_node_t** out_node, uint8_t** out_extra_data) {
  IREE_ASSERT_ARGUMENT(out_node);
  *out_node = NULL;
  if (out_extra_data) *out_extra_data = NULL;

  // Calculate total size needed.
  const iree_host_size_t node_size = sizeof(iree_hal_streaming_graph_node_t);
  const iree_host_size_t deps_size =
      dependency_count * sizeof(iree_hal_streaming_graph_node_t*);
  iree_host_size_t total_size = node_size + deps_size;

  // Align for extra data if needed.
  if (extra_data_size > 0) {
    total_size = iree_host_align(total_size, iree_max_align_t);
    const iree_host_size_t extra_data_offset = total_size;
    total_size += extra_data_size;

    // Allocate the entire block.
    iree_hal_streaming_graph_node_t* node = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, total_size, (void**)&node));

    *out_node = node;
    if (out_extra_data) {
      *out_extra_data = (uint8_t*)node + extra_data_offset;
    }
  } else {
    // Allocate just the node and dependencies.
    iree_hal_streaming_graph_node_t* node = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, total_size, (void**)&node));
    *out_node = node;
  }

  return iree_ok_status();
}

// Helper to allocate a new block for node storage.
static iree_status_t iree_hal_streaming_allocate_node_block(
    iree_allocator_t allocator, iree_host_size_t capacity,
    iree_hal_streaming_node_block_t** out_block) {
  const iree_host_size_t block_size =
      sizeof(iree_hal_streaming_node_block_t) +
      capacity * sizeof(iree_hal_streaming_graph_node_t*);

  iree_hal_streaming_node_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, block_size, (void**)&block));

  block->next = NULL;
  block->capacity = capacity;
  block->count = 0;
  *out_block = block;
  return iree_ok_status();
}

// Helper to add a node to the graph.
static iree_status_t iree_hal_streaming_graph_add_node(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_graph_node_t* node) {
  // Assign unique index to the node that can be used to get the logical index
  // in the graph for use as dependency references.
  node->node_index = (uint32_t)graph->node_count;

  // Add to node blocks.
  if (!graph->current_node_block ||
      graph->current_node_block->count >= graph->current_node_block->capacity) {
    // Need a new block.
    const iree_host_size_t block_capacity =
        graph->node_count < 64 ? 16 : 64;  // Grow block size for larger graphs.

    iree_hal_streaming_node_block_t* new_block = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_allocate_node_block(
        graph->arena_allocator, block_capacity, &new_block));

    // Chain the new block.
    if (graph->current_node_block) {
      graph->current_node_block->next = new_block;
    } else {
      graph->node_blocks = new_block;
    }
    graph->current_node_block = new_block;
  }

  graph->current_node_block->nodes[graph->current_node_block->count++] = node;
  ++graph->node_count;

  // Add to root nodes if no dependencies.
  if (node->dependency_count == 0) {
    if (!graph->current_root_block || graph->current_root_block->count >=
                                          graph->current_root_block->capacity) {
      // Need a new root block.
      const iree_host_size_t block_capacity = 8;

      iree_hal_streaming_node_block_t* new_block = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_streaming_allocate_node_block(
          graph->arena_allocator, block_capacity, &new_block));

      // Chain the new block.
      if (graph->current_root_block) {
        graph->current_root_block->next = new_block;
      } else {
        graph->root_blocks = new_block;
      }
      graph->current_root_block = new_block;
    }

    graph->current_root_block->nodes[graph->current_root_block->count++] = node;
    graph->root_count++;
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_add_empty_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate node with dependencies in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // iree_hal_streaming_graph_empty_attrs_t* attrs = &node->attrs.empty;

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status)) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_kernel_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(params);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify the symbol is a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol is not a function (type=%d)", symbol->type);
  }

  // Allocate node with dependencies and params storage in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  const iree_host_size_t constants_size =
      iree_host_align(symbol->parameters.constant_bytes, iree_max_align_t);
  const iree_host_size_t bindings_size = iree_host_align(
      symbol->parameters.binding_count * sizeof(iree_hal_buffer_ref_t),
      iree_max_align_t);
  uint8_t* extra_data = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count,
              constants_size + bindings_size, &node, &extra_data));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // Copy kernel dispatch parameters.
  iree_hal_streaming_graph_kernel_node_attrs_t* attrs = &node->attrs.kernel;
  attrs->symbol = symbol;
  memcpy(attrs->grid_dim, params->grid_dim, sizeof(params->grid_dim));
  memcpy(attrs->block_dim, params->block_dim, sizeof(params->block_dim));
  attrs->shared_memory_bytes = params->shared_memory_bytes;

  // Unpack parameters.
  void* constants = extra_data;
  attrs->constants =
      iree_make_const_byte_span(constants, symbol->parameters.constant_bytes);
  attrs->bindings.count = symbol->parameters.binding_count;
  attrs->bindings.values =
      (iree_hal_buffer_ref_t*)(extra_data + constants_size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_unpack_parameters(
              graph->context, &symbol->parameters, params->buffer, constants,
              &attrs->bindings));

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status)) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_memcpy_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_host_size_t size,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(graph->context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(graph->context, src, &src_ref),
      "resolving `src` buffer ref %p", (void*)src);

  // Allocate node with dependencies in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // Copy memcpy data.
  iree_hal_streaming_graph_memcpy_node_attrs_t* attrs = &node->attrs.memcpy;
  attrs->dst_ref = dst_ref;
  attrs->src_ref = src_ref;
  attrs->size = size;

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status)) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_memset_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_deviceptr_t dst,
    uint32_t pattern, iree_device_size_t pattern_size, iree_device_size_t count,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(graph->context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);

  // Allocate node with dependencies in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // Copy memset data.
  iree_hal_streaming_graph_memset_node_attrs_t* attrs = &node->attrs.memset;
  attrs->dst_ref = dst_ref;
  attrs->pattern = pattern;
  attrs->pattern_size = pattern_size;
  attrs->count = count;

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status)) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_host_call_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void (*fn)(void*), void* user_data,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate node with dependencies in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // Copy host function data.
  iree_hal_streaming_graph_host_call_node_attrs_t* attrs = &node->attrs.host;
  attrs->fn = fn;
  attrs->user_data = user_data;

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status)) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_streaming_graph_exec_t (instantiation)
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_instantiate(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_hal_streaming_graph_exec_t** out_exec) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_exec);
  *out_exec = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create an uninitialized exec object.
  iree_hal_streaming_graph_exec_t* exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_exec_create(graph->context, graph, flags,
                                               graph->host_allocator, &exec));

  // Mutex needed for instantiate per CUDA docs.
  iree_slim_mutex_lock(&graph->mutex);

  // Instantiate the graph exec on the given context and with our nodes.
  // NOTE: this must happen under the graph lock (so nodes cannot change).
  iree_status_t status = iree_hal_streaming_graph_exec_instantiate_locked(
      exec, graph->node_blocks, graph->node_count);

  iree_slim_mutex_unlock(&graph->mutex);

  if (iree_status_is_ok(status)) {
    *out_exec = exec;
  } else {
    iree_hal_streaming_graph_exec_release(exec);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Stream capture internal functions
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_begin_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&stream->mutex);

  // Check if already capturing.
  if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is already capturing");
  }

  // Flush any pending operations before starting capture.
  if (stream->command_buffer) {
    iree_status_t status = iree_hal_streaming_stream_flush(stream);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Create a new graph for capture.
  iree_status_t status = iree_hal_streaming_graph_create(
      stream->context, /*flags=*/0, stream->host_allocator,
      &stream->capture_graph);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Set capture state.
  stream->capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE;
  stream->capture_mode = mode;
  stream->capture_id++;  // Increment capture ID.

  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_end_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_t** out_graph) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&stream->mutex);

  // Check capture status.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is not capturing");
  }

  iree_hal_streaming_graph_t* graph = stream->capture_graph;
  stream->capture_graph = NULL;

  // Clear capture state.
  stream->capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  stream->capture_id = 0;

  // Reset dependency count but keep the buffer for reuse.
  stream->capture_dependency_count = 0;
  // Note: keeping capture_dependencies and capture_dependency_capacity
  // unchanged for reuse in next capture session.

  iree_slim_mutex_unlock(&stream->mutex);

  if (out_graph) {
    *out_graph = graph;
  } else if (graph) {
    iree_hal_streaming_graph_release(graph);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_capture_status(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_status_t* out_status,
    unsigned long long* out_id) {
  IREE_ASSERT_ARGUMENT(stream);

  iree_slim_mutex_lock(&stream->mutex);

  if (out_status) {
    *out_status = stream->capture_status;
  }
  if (out_id) {
    *out_id = stream->capture_id;
  }

  iree_slim_mutex_unlock(&stream->mutex);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_is_capturing(
    iree_hal_streaming_stream_t* stream, bool* out_is_capturing) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_is_capturing);

  iree_slim_mutex_lock(&stream->mutex);
  *out_is_capturing =
      (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);
  iree_slim_mutex_unlock(&stream->mutex);

  return iree_ok_status();
}

// Helper to grow the capture dependencies array.
static iree_status_t iree_hal_streaming_grow_capture_dependencies(
    iree_hal_streaming_stream_t* stream, iree_host_size_t required_capacity) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, required_capacity);

  // Calculate new capacity (at least 2x required).
  const iree_host_size_t new_capacity = required_capacity * 2;

  // Use realloc to potentially extend in-place.
  iree_status_t status = iree_allocator_realloc(
      stream->host_allocator,
      new_capacity * sizeof(iree_hal_streaming_graph_node_t*),
      (void**)&stream->capture_dependencies);
  if (iree_status_is_ok(status)) {
    stream->capture_dependency_capacity = new_capacity;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_update_capture_dependencies(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_capture_dependencies_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, dependency_count);

  iree_slim_mutex_lock(&stream->mutex);

  // Check if capturing.
  if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is not actively capturing");
  }

  // Calculate total count based on mode.
  const iree_host_size_t total_count =
      (mode == IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD)
          ? stream->capture_dependency_count + dependency_count
          : dependency_count;

  // Grow dependency array if needed.
  if (total_count > stream->capture_dependency_capacity) {
    iree_status_t status =
        iree_hal_streaming_grow_capture_dependencies(stream, total_count);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Copy dependencies based on mode.
  if (dependency_count > 0) {
    void* dest =
        (mode == IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD)
            ? stream->capture_dependencies + stream->capture_dependency_count
            : stream->capture_dependencies;
    memcpy(dest, dependencies, dependency_count * sizeof(*dependencies));
  }

  stream->capture_dependency_count = total_count;

  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
