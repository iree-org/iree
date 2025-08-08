// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_TEST_GRAPH_UTIL_H_
#define IREE_EXPERIMENTAL_STREAMING_TEST_GRAPH_UTIL_H_

#include "experimental/streaming/internal.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Graph generation utilities for benchmarking
//===----------------------------------------------------------------------===//

// Node type distribution for mixed graphs.
typedef struct iree_hal_streaming_graph_node_distribution_t {
  // Percentage of each node type (should sum to 100).
  uint32_t dispatch_percent;
  uint32_t memcpy_percent;
  uint32_t memset_percent;
  uint32_t host_call_percent;
  uint32_t empty_percent;
} iree_hal_streaming_graph_node_distribution_t;

// Parameters for generating test graphs.
typedef struct iree_hal_streaming_graph_gen_params_t {
  // Total number of nodes to generate.
  iree_host_size_t node_count;

  // Node type distribution.
  iree_hal_streaming_graph_node_distribution_t distribution;

  // Concurrency level (0-100): percentage of nodes that can run concurrently.
  uint32_t concurrency_percent;

  // For kernel nodes: number of arguments to pack.
  uint32_t kernel_arg_count;

  // For memcpy/memset nodes: size of operation in bytes.
  iree_device_size_t memory_op_size;

  // Random seed for deterministic generation.
  uint32_t random_seed;
} iree_hal_streaming_graph_gen_params_t;

// Test kernel symbol for benchmarking.
typedef struct iree_hal_streaming_test_symbol_t {
  iree_hal_streaming_symbol_t base;
  // Simple test data.
  char name[64];
} iree_hal_streaming_test_symbol_t;

//===----------------------------------------------------------------------===//
// Linear sequence generation
//===----------------------------------------------------------------------===//

// Creates a linear chain of nodes where each depends on the previous.
// All nodes will be of the same type based on distribution (first non-zero).
iree_status_t iree_hal_streaming_graph_gen_linear_sequence(
    iree_hal_streaming_graph_t* graph, iree_host_size_t node_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t** out_first_node,
    iree_hal_streaming_graph_node_t** out_last_node);

//===----------------------------------------------------------------------===//
// Fan-out/fan-in patterns
//===----------------------------------------------------------------------===//

// Creates a fan-out pattern: 1 root node -> N child nodes.
iree_status_t iree_hal_streaming_graph_gen_fanout(
    iree_hal_streaming_graph_t* graph, iree_host_size_t fanout_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t** out_root_node,
    iree_hal_streaming_graph_node_t*** out_leaf_nodes);

// Creates a fan-in pattern: N source nodes -> 1 sink node.
iree_status_t iree_hal_streaming_graph_gen_fanin(
    iree_hal_streaming_graph_t* graph, iree_host_size_t fanin_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t*** out_source_nodes,
    iree_hal_streaming_graph_node_t** out_sink_node);

//===----------------------------------------------------------------------===//
// Diamond/fork-join patterns
//===----------------------------------------------------------------------===//

// Creates a diamond pattern: 1 root -> N parallel -> 1 sink.
iree_status_t iree_hal_streaming_graph_gen_diamond(
    iree_hal_streaming_graph_t* graph, iree_host_size_t parallel_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t** out_root_node,
    iree_hal_streaming_graph_node_t** out_sink_node);

//===----------------------------------------------------------------------===//
// Mixed/complex patterns
//===----------------------------------------------------------------------===//

// Creates a graph with mixed node types and dependencies based on params.
iree_status_t iree_hal_streaming_graph_gen_mixed(
    iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_gen_params_t* params,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t*** out_nodes,
    iree_host_size_t* out_node_count);

// Creates an interleaved pattern (e.g., 20 dispatches, 1 host call, repeat).
iree_status_t iree_hal_streaming_graph_gen_interleaved(
    iree_hal_streaming_graph_t* graph, iree_host_size_t recordable_batch_size,
    iree_host_size_t host_call_interval, iree_host_size_t total_nodes,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t** out_first_node,
    iree_hal_streaming_graph_node_t** out_last_node);

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Creates a test symbol for use in kernel nodes.
iree_status_t iree_hal_streaming_test_symbol_create(
    const char* name, iree_allocator_t allocator,
    iree_hal_streaming_test_symbol_t** out_symbol);

void iree_hal_streaming_test_symbol_destroy(
    iree_hal_streaming_test_symbol_t* symbol);

// Simple host call function for benchmarking.
void iree_hal_streaming_test_host_call(void* user_data);

// Helper to add a single test node of the specified type.
iree_status_t iree_hal_streaming_graph_add_test_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_test_symbol_t* test_symbol,
    const iree_hal_streaming_graph_gen_params_t* params,
    iree_hal_streaming_graph_node_t** out_node);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_TEST_GRAPH_UTIL_H_
