// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/graph.h"
#include "experimental/streaming/internal.h"
#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// Tuning Parameters and Heuristics
//===----------------------------------------------------------------------===//

// Minimum number of nodes in a partition to consider workstream detection.
// Smaller partitions don't benefit from the overhead of multiple streams.
#define IREE_HAL_STREAMING_GRAPH_MIN_PARTITION_SIZE_FOR_STREAMS 16

// Maximum number of nodes in a single partition before forcing a split.
// Prevents command buffers from growing too large.
#define IREE_HAL_STREAMING_GRAPH_MAX_PARTITION_SIZE 2048

// Maximum number of concurrent workstreams to detect within a partition.
// Hardware typically has limited concurrent execution contexts.
#define IREE_HAL_STREAMING_GRAPH_MAX_WORKSTREAMS 8

// Minimum nodes per workstream to justify creating separate streams.
// Each stream has overhead, so ensure sufficient work.
#define IREE_HAL_STREAMING_GRAPH_MIN_NODES_PER_STREAM 32

// Maximum reachability mask bits for efficient workstream detection.
// Beyond this, fall back to single stream for simplicity.
#define IREE_HAL_STREAMING_GRAPH_MAX_REACHABILITY_BITS 64

// Threshold for "small" graphs that use simplified algorithms.
#define IREE_HAL_STREAMING_GRAPH_SMALL_GRAPH_THRESHOLD 32

// Maximum dependency fan-out before considering node as synchronization point.
#define IREE_HAL_STREAMING_GRAPH_MAX_FAN_OUT 8

//===----------------------------------------------------------------------===//
// DAG Scheduling Algorithm Overview
//===----------------------------------------------------------------------===//
//
// This file implements an efficient three-phase algorithm for transforming a
// directed acyclic graph (DAG) of heterogeneous nodes into an optimized
// execution schedule with automatic workstream detection.
//
// Algorithm phases:
// 1. PREPARE: Linearize nodes and detect if already topologically sorted
// 2. SORT: Perform topological sorting if needed (with fast path)
// 3. PARTITION: Group nodes into executable blocks with workstream detection
//
// Example 1: Linear Stream with Small Concurrency
// ================================================
// Input DAG:
//   K1 ---> K2 ---> K3 ---> H1 ---> K4 ---> K5
//            \                 /
//             M1 -------------+
//
// Node types: K=Kernel(recordable), M=Memcpy(recordable),
// H=Host(non-recordable)
//
// After scheduling:
// +-------------------------------------+
// | Partition 0: RECORDABLE (2 streams) |
// |  Stream 0: K1 -> K2 -> K3           |
// |  Stream 1: M1                       |
// +-------------------------------------+
// +-------------------------------------+
// | Partition 1: HOST (1 stream)        |
// |  Stream 0: H1                       |
// +-------------------------------------+
// +-------------------------------------+
// | Partition 2: RECORDABLE (1 stream)  |
// |  Stream 0: K4 -> K5                 |
// +-------------------------------------+
//
// Efficiency: O(N + E) where N=6 nodes, E=6 edges
// Memory: 6 * 24 bytes (sort nodes) + 3 * 16 bytes (partitions) = 192 bytes
//
// Example 2: Multiple Concurrent Streams
// =======================================
// Input DAG:
//   K1 ---> K2 ---> K5 ---> H1 ---> K9
//    |               \              ^
//    +--> M1 --> M2 --> K6 ---------+
//    |                              |
//    +--> K3 --> K4 --> K7 --> K8 --+
//
// After scheduling:
// +-------------------------------------+
// | Partition 0: RECORDABLE (3 streams) |
// |  Stream 0: K1 -> K2 -> K5 -> K6     |
// |  Stream 1: M1 -> M2                 |
// |  Stream 2: K3 -> K4 -> K7 -> K8     |
// +-------------------------------------+
// +-------------------------------------+
// | Partition 1: HOST (1 stream)        |
// |  Stream 0: H1                       |
// +-------------------------------------+
// +-------------------------------------+
// | Partition 2: RECORDABLE (1 stream)  |
// |  Stream 0: K9                       |
// +-------------------------------------+
//
// Efficiency: O(N + E) where N=10 nodes, E=11 edges
// Memory: 10 * 24 bytes (sort nodes) + 3 * 16 bytes (partitions) = 288 bytes
//
// Performance characteristics:
// - Time Complexity: O(N + E) for all phases
// - Space Complexity: O(N) with 24 bytes per node + O(P) partitions
// - Cache Behavior: Sequential access patterns, prefetch-friendly
// - Scalability: Handles 1-100,000 nodes efficiently (as much as possible given
//                the design constraints imposed by the CUDA Graphs API)
//
// Worst case scenarios:
// 1. All non-recordable nodes: P = N partitions, O(N) space
// 2. Fully connected DAG: E = N*(N-1)/2 edges, O(N²) edge processing
// 3. Deep linear chain: No parallelism opportunities
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Phase 1: Preparation - Linearize and Detect Sorting
//===----------------------------------------------------------------------===//

// Linearize nodes from chained blocks and detects if already sorted.
// Returns true if nodes are already in topological order.
//
// Algorithm:
// 1. Walk chained blocks copying to sort_nodes array
// 2. Build index mapping from node_index to sort_nodes position
// 3. For each node, check if all dependencies have lower indices
// 4. Cache node type to avoid indirection later
//
// Complexity: O(N * avg_deps) ~= O(N) for sparse graphs
static bool iree_hal_streaming_graph_prepare_nodes(
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count,
    iree_hal_streaming_graph_sort_node_t* sort_nodes,
    uint32_t* node_index_map) {
  // Linearize from chained blocks.
  uint32_t index = 0;
  for (iree_hal_streaming_node_block_t* block = node_blocks; block;
       block = block->next) {
    if (block->next) {
      IREE_PREFETCH_RO(block->next, 1);  // Prefetch next block.
    }

    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      // TODO(benvanik): see if the allocation is guaranteed calloc - if so, we
      // can initialize fewer fields.
      sort_nodes[index] = (iree_hal_streaming_graph_sort_node_t){
          .original_index = index,
          .sorted_index = index,  // Initially assume sorted.
          .max_dependency_index = 0,
          .partition_id = 0,
          .in_degree = 0,
          .type = (uint8_t)node->type,
          .stream_id = 0,
          .node = node,
      };
      // Map the node's original index to its position in sort_nodes.
      node_index_map[node->node_index] = index;
      ++index;
    }
  }

  // Now check if dependencies maintain topological order.
  // This requires a second pass since we need all nodes linearized first.
  // Note that we can early exit on the first unsorted node we find. We could
  // possibly store this bit back on the graph to avoid needing to do this walk
  // on subsequent instantiations.
  bool is_sorted = true;
  for (uint32_t i = 0; is_sorted && i < node_count; ++i) {
    iree_hal_streaming_graph_node_t* node = sort_nodes[i].node;
    if (node->dependency_count > 0) {
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        // Look up dependency's position in sort_nodes using the mapping.
        uint32_t dep_index = node_index_map[node->dependencies[j]->node_index];
        if (dep_index >= i) {
          is_sorted = false;
          break;
        }
      }
    }
  }

  return is_sorted;
}

//===----------------------------------------------------------------------===//
// Phase 2: Topological Sort with Optimization
//===----------------------------------------------------------------------===//

// Performs topological sort using modified Kahn's algorithm.
// Fast path for already-sorted graphs (common case).
//
// Algorithm when sorting needed:
// 1. Calculate in-degrees for all nodes
// 2. Queue nodes with zero in-degree
// 3. Process queue, updating in-degrees
// 4. Reorder array in-place using cycle detection
//
// Complexity: O(N + E) where E = total edges
static iree_status_t iree_hal_streaming_graph_topological_sort(
    iree_hal_streaming_graph_sort_node_t* nodes, uint32_t node_count,
    uint32_t* node_index_map, iree_arena_allocator_t* arena,
    bool is_already_sorted) {
  if (is_already_sorted) {
    // Fast path: just compute max dependencies.
    // This is needed for partition boundary detection.
    for (uint32_t i = 0; i < node_count; ++i) {
      uint32_t max_dep = 0;
      for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
        // Use the mapping for O(1) lookup.
        const uint32_t dep_index =
            node_index_map[nodes[i].node->dependencies[j]->node_index];
        if (dep_index < i && dep_index != UINT32_MAX) {
          max_dep = iree_max(max_dep, dep_index);
        }
      }
      nodes[i].max_dependency_index = max_dep;
    }
    return iree_ok_status();
  }

  // Full topological sort using Kahn's algorithm.
  uint32_t* queue = NULL;
  const iree_host_size_t queue_size = node_count * sizeof(*queue);
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, queue_size, (void**)&queue));

  // Step 1: Calculate in-degrees.
  for (uint32_t i = 0; i < node_count; ++i) {
    nodes[i].in_degree = (uint16_t)nodes[i].node->dependency_count;
  }

  // Step 2: Find zero in-degree nodes.
  uint32_t queue_head = 0;
  uint32_t queue_tail = 0;
  for (uint32_t i = 0; i < node_count; ++i) {
    if (nodes[i].in_degree == 0) {
      queue[queue_tail++] = i;
    }
  }

  // Step 3: Process queue.
  uint32_t sorted_count = 0;
  while (queue_head < queue_tail) {
    uint32_t current = queue[queue_head++];
    nodes[current].sorted_index = sorted_count++;

    // Update max dependency for this node.
    uint32_t max_dep = 0;
    for (uint32_t j = 0; j < nodes[current].node->dependency_count; ++j) {
      // Use the mapping for O(1) lookup.
      uint32_t dep_index =
          node_index_map[nodes[current].node->dependencies[j]->node_index];
      if (dep_index != UINT32_MAX) {
        uint32_t dep_sorted_index = nodes[dep_index].sorted_index;
        max_dep = iree_max(max_dep, dep_sorted_index);
      }
    }
    nodes[current].max_dependency_index = max_dep;

    // Decrement in-degrees of nodes that depend on current.
    // This requires finding reverse edges (who depends on current).
    for (uint32_t i = 0; i < node_count; ++i) {
      if (i == current) continue;
      iree_hal_streaming_graph_node_t* node = nodes[i].node;
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        if (node->dependencies[j] == nodes[current].node) {
          if (--nodes[i].in_degree == 0) {
            queue[queue_tail++] = i;
          }
          break;  // Each node appears at most once in dependency list.
        }
      }
    }
  }

  if (sorted_count != node_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "graph contains cycles (%u nodes processed of %u)",
                            sorted_count, node_count);
  }

  // Step 4: Reorder array in-place based on sorted_index.
  // Use cycle-following algorithm to minimize copies.
  // TODO: see if we can use a faster sort (or switch based on count).
  iree_hal_streaming_graph_sort_node_t temp;
  for (uint32_t i = 0; i < node_count; ++i) {
    while (nodes[i].sorted_index != i) {
      uint32_t target = nodes[i].sorted_index;
      temp = nodes[target];
      nodes[target] = nodes[i];
      nodes[i] = temp;
    }
  }

  // Update the mapping to reflect the new sorted order.
  for (uint32_t i = 0; i < node_count; ++i) {
    node_index_map[nodes[i].node->node_index] = i;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Phase 3: Partitioning with Workstream Detection
//===----------------------------------------------------------------------===//

// State for tracking independent workstreams during partitioning.
typedef struct iree_hal_streaming_graph_stream_state_t {
  // Nodes reachable in this stream.
  uint64_t reachability_mask;
  // Last node added to stream.
  uint32_t last_node_index;
} iree_hal_streaming_graph_stream_state_t;

typedef struct iree_uint32x2_t {
  uint32_t values[2];
} iree_uint32x2_t;

// Partitions sorted nodes into executable blocks and detects independent
// workstreams within recordable partitions.
//
// Algorithm:
// 1. Scan for recordable vs non-recordable boundaries
// 2. Within recordable sections, detect independent workstreams
// 3. Track reachability masks to determine stream assignment
// 4. Merge streams when convergence detected
//
// Tuning heuristics:
// - Partitions < MIN_PARTITION_SIZE_FOR_STREAMS (~32) use single stream
// - Partitions limited to MAX_PARTITION_SIZE (~1024) nodes
// - Maximum MAX_WORKSTREAMS (~4) concurrent streams per partition
// - New streams require MIN_NODES_PER_STREAM (~4) nodes
// - Reachability analysis limited to MAX_REACHABILITY_BITS (64) nodes
// - High fan-out (> MAX_FAN_OUT=~8) forces stream convergence
//
// Complexity: O(N) with up to MAX_WORKSTREAMS workstreams per partition
//
// Returns [partition_count, block_count].
static iree_uint32x2_t iree_hal_streaming_graph_partition_with_streams(
    iree_hal_streaming_graph_sort_node_t* nodes, uint32_t node_count,
    uint32_t* node_index_map,
    iree_hal_streaming_graph_partition_t* partitions) {
  uint32_t partition_count = 0;
  uint32_t block_count = 0;

  // State for tracking independent workstreams.
  iree_hal_streaming_graph_stream_state_t
      streams[IREE_HAL_STREAMING_GRAPH_MAX_WORKSTREAMS];
  uint8_t active_streams = 0;

  for (uint32_t i = 0; i < node_count;) {
    bool is_recordable =
        iree_hal_streaming_graph_node_is_recordable(nodes[i].type);
    if (!is_recordable) {
      // Non-recordable node gets its own partition.
      iree_hal_streaming_graph_partition_type_t partition_type;
      switch (nodes[i].type) {
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY:
          partition_type = IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_EMPTY;
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL:
          partition_type = IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL;
          break;
        default:
          partition_type = IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_EMPTY;
          break;
      }
      partitions[partition_count] = (iree_hal_streaming_graph_partition_t){
          .start_index = i,
          .count = 1,
          .type = partition_type,
          .stream_count = 1,
      };
      nodes[i].partition_id = partition_count;
      nodes[i].stream_id = 0;
      ++partition_count;
      ++block_count;
      ++i;
      active_streams = 0;
    } else {
      // Start recordable partition.
      uint32_t recordable_start = i;
      memset(streams, 0, sizeof(streams));
      active_streams = 0;

      // Extend partition with compatible nodes up to size limit.
      // A node can be added if all its dependencies are before the partition
      // start (i.e., max_dependency_index < recordable_start).
      uint32_t partition_size = 0;
      while (i < node_count &&
             partition_size < IREE_HAL_STREAMING_GRAPH_MAX_PARTITION_SIZE &&
             iree_hal_streaming_graph_node_is_recordable(nodes[i].type)) {
        // Check if dependencies are satisfied.
        bool deps_satisfied = true;
        for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
          // Use the mapping for O(1) lookup.
          uint32_t dep_index =
              node_index_map[nodes[i].node->dependencies[j]->node_index];
          if (dep_index >= recordable_start && dep_index < i) {
            // Dependency is within this partition - OK.
          } else if (dep_index >= i) {
            // Dependency is ahead - can't include this node.
            deps_satisfied = false;
            break;
          }
        }
        if (!deps_satisfied) {
          break;  // End this partition.
        }

        // Only perform workstream detection for sufficiently large partitions.
        const bool use_workstreams =
            (i - recordable_start) >=
            IREE_HAL_STREAMING_GRAPH_MIN_PARTITION_SIZE_FOR_STREAMS;

        // Determine which stream this node belongs to.
        uint8_t assigned_stream = 0;
        uint8_t connected_streams = 0;

        // Check dependencies within this partition.
        for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
          // Use the mapping for O(1) lookup.
          uint32_t dep_index =
              node_index_map[nodes[i].node->dependencies[j]->node_index];
          if (dep_index >= recordable_start && dep_index < i) {
            // Dependency is within this partition.
            uint8_t dep_stream = nodes[dep_index].stream_id;
            connected_streams |= (1 << dep_stream);
          }
        }

        if (use_workstreams && connected_streams == 0 &&
            active_streams < IREE_HAL_STREAMING_GRAPH_MAX_WORKSTREAMS) {
          // Check if enough nodes remain to justify a new stream.
          uint32_t remaining_in_partition =
              IREE_HAL_STREAMING_GRAPH_MAX_PARTITION_SIZE - partition_size;
          if (remaining_in_partition >=
              IREE_HAL_STREAMING_GRAPH_MIN_NODES_PER_STREAM) {
            // No dependencies - new independent stream!
            assigned_stream = active_streams++;
            streams[assigned_stream].last_node_index = i;
            if ((i - recordable_start) <
                IREE_HAL_STREAMING_GRAPH_MAX_REACHABILITY_BITS) {
              streams[assigned_stream].reachability_mask =
                  1ULL << (i - recordable_start);
            }
          } else {
            // Not enough nodes - use stream 0.
            assigned_stream = 0;
            if (active_streams == 0) active_streams = 1;
          }
        } else if (use_workstreams &&
                   iree_math_count_ones_u32(connected_streams) == 1) {
          // Depends on single stream.
          assigned_stream =
              iree_math_count_trailing_zeros_u32(connected_streams);
          streams[assigned_stream].last_node_index = i;
          if ((i - recordable_start) <
              IREE_HAL_STREAMING_GRAPH_MAX_REACHABILITY_BITS) {
            streams[assigned_stream].reachability_mask |=
                1ULL << (i - recordable_start);
          }
        } else {
          // Either: no workstreams, multiple dependencies, or merge point.
          const uint32_t dep_count =
              iree_math_count_ones_u32(connected_streams);
          // Check for high fan-out synchronization point.
          const bool is_sync_point = (nodes[i].node->dependency_count >
                                      IREE_HAL_STREAMING_GRAPH_MAX_FAN_OUT);
          if (!use_workstreams || dep_count > 1 || is_sync_point) {
            // Collapse to single stream.
            assigned_stream = 0;
            for (uint32_t k = recordable_start; k < i; ++k) {
              nodes[k].stream_id = 0;  // Reset all to stream 0.
            }
            active_streams = 1;
          } else {
            // Single dependency or first node - use stream 0.
            assigned_stream = 0;
            if (active_streams == 0) active_streams = 1;
          }
        }

        nodes[i].partition_id = partition_count;
        nodes[i].stream_id = assigned_stream;
        ++i;
        ++partition_size;
      }

      const uint32_t stream_count = active_streams > 0 ? active_streams : 1;
      partitions[partition_count] = (iree_hal_streaming_graph_partition_t){
          .start_index = recordable_start,
          .count = i - recordable_start,
          .type = IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE,
          .stream_count = stream_count,
      };
      ++partition_count;
      block_count += stream_count;
    }
  }

  return (iree_uint32x2_t){{partition_count, block_count}};
}

//===----------------------------------------------------------------------===//
// Main Scheduling Entry Point
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_schedule_nodes(
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count,
    iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_schedule_t* out_schedule) {
  IREE_ASSERT_ARGUMENT(out_schedule);

  if (node_count == 0) {
    memset(out_schedule, 0, sizeof(*out_schedule));
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, node_count);

  // Allocate all working memory from arena.
  iree_hal_streaming_graph_sort_node_t* sorted_nodes = NULL;
  const iree_host_size_t sorted_nodes_size = node_count * sizeof(*sorted_nodes);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, sorted_nodes_size, (void**)&sorted_nodes));

  // Allocate mapping from original node_index to sorted position.
  uint32_t* node_index_map = NULL;
  const iree_host_size_t map_size = node_count * sizeof(*node_index_map);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, map_size, (void**)&node_index_map));

  // Phase 1: Prepare - linearize and detect if sorted.
  const bool is_sorted = iree_hal_streaming_graph_prepare_nodes(
      node_blocks, node_count, sorted_nodes, node_index_map);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, is_sorted);

  // Phase 2: Sort - topological ordering.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_topological_sort(
              sorted_nodes, node_count, node_index_map, arena, is_sorted));

  // TODO: try to avoid allocating an O(node) partition capacity? We could use
  // linked blocks, though it does make walking the partitions slightly more
  // complicated.
  iree_hal_streaming_graph_partition_t* partitions = NULL;
  const iree_host_size_t partitions_size = node_count * sizeof(*partitions);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, partitions_size, (void**)&partitions));

  // Phase 3: Partition - group into executable blocks.
  const iree_uint32x2_t partition_block_counts =
      iree_hal_streaming_graph_partition_with_streams(
          sorted_nodes, node_count, node_index_map, partitions);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, partition_block_counts.values[0]);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, partition_block_counts.values[1]);

  out_schedule->sorted_nodes = sorted_nodes;
  out_schedule->node_index_map = node_index_map;
  out_schedule->partitions = partitions;
  out_schedule->partition_count = partition_block_counts.values[0];
  out_schedule->block_count = partition_block_counts.values[1];

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
