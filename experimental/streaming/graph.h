// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_GRAPH_H_
#define IREE_EXPERIMENTAL_STREAMING_GRAPH_H_

#include "experimental/streaming/internal.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Graph partitioning and instantiation
//===----------------------------------------------------------------------===//

// Chained block for growing arrays without reallocation.
typedef struct iree_hal_streaming_node_block_t {
  struct iree_hal_streaming_node_block_t* next;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_hal_streaming_graph_node_t* nodes[];
} iree_hal_streaming_node_block_t;

// Graph structure (template).
typedef struct iree_hal_streaming_graph_t {
  iree_atomic_ref_count_t ref_count;

  // Arena allocator for all graph allocations.
  iree_arena_allocator_t arena;
  iree_allocator_t arena_allocator;

  // Graph nodes stored in chained blocks.
  iree_hal_streaming_node_block_t* node_blocks;
  iree_hal_streaming_node_block_t* current_node_block;
  iree_host_size_t node_count;

  // Root nodes stored in chained blocks.
  iree_hal_streaming_node_block_t* root_blocks;
  iree_hal_streaming_node_block_t* current_root_block;
  iree_host_size_t root_count;

  uint32_t flags;
  iree_hal_streaming_context_t* context;

  // Mutex only needed for instantiate/clone operations per CUDA docs.
  iree_slim_mutex_t mutex;

  iree_allocator_t host_allocator;
} iree_hal_streaming_graph_t;

// Type of partition - determines how nodes are executed.
enum iree_hal_streaming_graph_partition_type_e {
  // Can go in command buffer (count 1 may also be optimizable into a queue op).
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE = 0,
  // Must be separate host call.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL,
  // Barrier node.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_EMPTY,
};
typedef uint8_t iree_hal_streaming_graph_partition_type_t;

// Describes a partition of nodes that can be executed together.
typedef struct iree_hal_streaming_graph_partition_t {
  // Index into sorted_nodes array.
  uint32_t start_index;
  uint32_t count;
  iree_hal_streaming_graph_partition_type_t type;
  // Number of independent workstreams (~1-4).
  uint8_t stream_count;
} iree_hal_streaming_graph_partition_t;

// Chained block for growing partition arrays without reallocation.
typedef struct iree_hal_streaming_graph_partition_block_t {
  struct iree_hal_streaming_graph_partition_block_t* next;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_hal_streaming_graph_partition_t partitions[];
} iree_hal_streaming_graph_partition_block_t;

iree_status_t iree_hal_streaming_graph_exec_create(
    iree_hal_streaming_context_t* context, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_allocator_t host_allocator,
    iree_hal_streaming_graph_exec_t** out_exec);

iree_status_t iree_hal_streaming_graph_exec_instantiate_locked(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count);

// Augmented node for sorting and partitioning.
typedef struct iree_hal_streaming_graph_sort_node_t {
  // Pointer to original node.
  iree_hal_streaming_graph_node_t* node;
  // Index in original linked list.
  uint32_t original_index;
  // Position in topological order.
  uint32_t sorted_index;
  // Maximum sorted index of all dependencies.
  uint32_t max_dependency_index;
  // Assigned partition ID.
  uint32_t partition_id;
  // For topological sort.
  uint16_t in_degree;
  // Cached from node->type.
  uint8_t type;
  // Workstream within partition (~4).
  uint8_t stream_id;
} iree_hal_streaming_graph_sort_node_t;
static_assert(sizeof(iree_hal_streaming_graph_sort_node_t) <= 32,
              "really want 2 per cache line");

// A produced schedule.
// References are into the arena used during scheduling and only valid as long
// as it is.
typedef struct iree_hal_streaming_graph_schedule_t {
  // Sorted nodes with some additional information from analysis.
  iree_hal_streaming_graph_sort_node_t* sorted_nodes;
  // A map of original unsorted graph node_index to sorted_nodes index.
  uint32_t* node_index_map;
  // Partition descriptors in execution order.
  iree_hal_streaming_graph_partition_t* partitions;
  // Total number of partitions.
  iree_host_size_t partition_count;
  // Total number of blocks across all partitions.
  iree_host_size_t block_count;
} iree_hal_streaming_graph_schedule_t;

// Unified scheduler that performs topological sorting, partitioning, and
// workstream detection in an efficient three-phase algorithm.
//
// Phase 1: Linearize nodes and detect if already sorted
// Phase 2: Topological sort if needed (with fast path)
// Phase 3: Partition into executable blocks with workstream detection
//
// Returns sorted nodes array and partition descriptors with workstream info
// allocated from the provided |arena|.
// |out_total_block_count| is the total number of graph blocks required
// calculated as the number of non-recordable partitions + the total number of
// streams in all recordable partitions.
iree_status_t iree_hal_streaming_graph_schedule_nodes(
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count,
    iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_schedule_t* out_schedule);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_GRAPH_H_
