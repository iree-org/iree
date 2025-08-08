// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/test/graph_util.h"

#include <stdlib.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Test symbol management
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_test_symbol_create(
    const char* name, iree_allocator_t allocator,
    iree_hal_streaming_test_symbol_t** out_symbol) {
  iree_hal_streaming_test_symbol_t* symbol = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*symbol), (void**)&symbol));
  memset(symbol, 0, sizeof(*symbol));

  // Initialize base symbol fields.
  symbol->base.module = NULL;  // Not attached to a real module.
  strncpy(symbol->name, name, sizeof(symbol->name) - 1);
  symbol->base.name = iree_make_string_view(symbol->name, strlen(symbol->name));
  symbol->base.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol->base.entry_point = 0;
  symbol->base.max_threads_per_block = 256;
  symbol->base.shared_size_bytes = 0;
  symbol->base.num_regs = 32;
  symbol->base.max_dynamic_shared_size_bytes = 0;

  *out_symbol = symbol;
  return iree_ok_status();
}

void iree_hal_streaming_test_symbol_destroy(
    iree_hal_streaming_test_symbol_t* symbol) {
  if (!symbol) return;
  iree_allocator_free(iree_allocator_system(), symbol);
}

//===----------------------------------------------------------------------===//
// Host call function
//===----------------------------------------------------------------------===//

void iree_hal_streaming_test_host_call(void* user_data) {
  // Simple no-op for benchmarking.
  (void)user_data;
}

//===----------------------------------------------------------------------===//
// Helper to add test nodes
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_add_test_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_test_symbol_t* test_symbol,
    const iree_hal_streaming_graph_gen_params_t* params,
    iree_hal_streaming_graph_node_t** out_node) {
  switch (node_type) {
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY:
      return iree_hal_streaming_graph_add_empty_node(
          graph, dependencies, dependency_count, out_node);

    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL: {
      // Create test dispatch params.
      iree_hal_streaming_dispatch_params_t dispatch_params = {
          .grid_dim = {1, 1, 1},
          .block_dim = {256, 1, 1},
          .shared_memory_bytes = 0,
          .kernel_params = NULL,
          .extra = NULL,
          .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_NONE,
      };

      // Allocate kernel args if requested.
      void** kernel_args = NULL;
      if (params && params->kernel_arg_count > 0) {
        kernel_args = (void**)calloc(params->kernel_arg_count, sizeof(void*));
        // Fill with dummy pointers.
        for (uint32_t i = 0; i < params->kernel_arg_count; ++i) {
          kernel_args[i] = (void*)(uintptr_t)(0x1000 + i * 8);
        }
        dispatch_params.kernel_params = kernel_args;
      }

      iree_status_t status = iree_hal_streaming_graph_add_kernel_node(
          graph, dependencies, dependency_count,
          (iree_hal_streaming_symbol_t*)test_symbol, &dispatch_params,
          out_node);

      free(kernel_args);
      return status;
    }

    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY: {
      // Use dummy device pointers for benchmarking.
      iree_hal_streaming_deviceptr_t src = 0x2000;
      iree_hal_streaming_deviceptr_t dst = 0x3000;
      iree_device_size_t size = params ? params->memory_op_size : 1024;

      return iree_hal_streaming_graph_add_memcpy_node(
          graph, dependencies, dependency_count, dst, src, size, out_node);
    }

    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET: {
      iree_hal_streaming_deviceptr_t dst = 0x4000;
      uint32_t pattern = 0;
      iree_host_size_t pattern_size = 4;
      iree_device_size_t count = params ? params->memory_op_size / 4 : 256;

      return iree_hal_streaming_graph_add_memset_node(
          graph, dependencies, dependency_count, dst, pattern, pattern_size,
          count, out_node);
    }

    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL:
      return iree_hal_streaming_graph_add_host_call_node(
          graph, dependencies, dependency_count,
          iree_hal_streaming_test_host_call, NULL, out_node);

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported node type: %d", node_type);
  }
}

//===----------------------------------------------------------------------===//
// Linear sequence generation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_gen_linear_sequence(
    iree_hal_streaming_graph_t* graph, iree_host_size_t node_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t** out_first_node,
    iree_hal_streaming_graph_node_t** out_last_node) {
  if (node_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "node_count must be > 0");
  }

  iree_hal_streaming_graph_node_t* prev_node = NULL;
  iree_hal_streaming_graph_node_t* first_node = NULL;

  for (iree_host_size_t i = 0; i < node_count; ++i) {
    iree_hal_streaming_graph_node_t* node = NULL;
    iree_hal_streaming_graph_node_t* deps[] = {prev_node};
    iree_host_size_t dep_count = prev_node ? 1 : 0;

    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_test_node(
        graph, node_type, deps, dep_count, test_symbol, NULL, &node));

    if (i == 0) {
      first_node = node;
    }
    prev_node = node;
  }

  if (out_first_node) *out_first_node = first_node;
  if (out_last_node) *out_last_node = prev_node;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Fan-out pattern generation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_gen_fanout(
    iree_hal_streaming_graph_t* graph, iree_host_size_t fanout_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t** out_root_node,
    iree_hal_streaming_graph_node_t*** out_leaf_nodes) {
  // Create root node.
  iree_hal_streaming_graph_node_t* root = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_test_node(
      graph, node_type, NULL, 0, test_symbol, NULL, &root));

  // Allocate leaf nodes array if requested.
  iree_hal_streaming_graph_node_t** leaf_nodes = NULL;
  if (out_leaf_nodes) {
    leaf_nodes = (iree_hal_streaming_graph_node_t**)calloc(
        fanout_count, sizeof(iree_hal_streaming_graph_node_t*));
  }

  // Create fan-out nodes.
  for (iree_host_size_t i = 0; i < fanout_count; ++i) {
    iree_hal_streaming_graph_node_t* node = NULL;
    iree_hal_streaming_graph_node_t* deps[] = {root};

    iree_status_t status = iree_hal_streaming_graph_add_test_node(
        graph, node_type, deps, 1, test_symbol, NULL, &node);
    if (!iree_status_is_ok(status)) {
      free(leaf_nodes);
      return status;
    }

    if (leaf_nodes) {
      leaf_nodes[i] = node;
    }
  }

  if (out_root_node) *out_root_node = root;
  if (out_leaf_nodes) *out_leaf_nodes = leaf_nodes;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Fan-in pattern generation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_gen_fanin(
    iree_hal_streaming_graph_t* graph, iree_host_size_t fanin_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t*** out_source_nodes,
    iree_hal_streaming_graph_node_t** out_sink_node) {
  // Allocate source nodes array.
  iree_hal_streaming_graph_node_t** source_nodes =
      (iree_hal_streaming_graph_node_t**)calloc(
          fanin_count, sizeof(iree_hal_streaming_graph_node_t*));

  // Create source nodes.
  for (iree_host_size_t i = 0; i < fanin_count; ++i) {
    iree_status_t status = iree_hal_streaming_graph_add_test_node(
        graph, node_type, NULL, 0, test_symbol, NULL, &source_nodes[i]);
    if (!iree_status_is_ok(status)) {
      free(source_nodes);
      return status;
    }
  }

  // Create sink node with all sources as dependencies.
  iree_hal_streaming_graph_node_t* sink = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_test_node(
      graph, node_type, source_nodes, fanin_count, test_symbol, NULL, &sink);

  if (!iree_status_is_ok(status)) {
    free(source_nodes);
    return status;
  }

  if (out_source_nodes) {
    *out_source_nodes = source_nodes;
  } else {
    free(source_nodes);
  }
  if (out_sink_node) *out_sink_node = sink;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Diamond pattern generation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_gen_diamond(
    iree_hal_streaming_graph_t* graph, iree_host_size_t parallel_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t** out_root_node,
    iree_hal_streaming_graph_node_t** out_sink_node) {
  // Create root node.
  iree_hal_streaming_graph_node_t* root = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_test_node(
      graph, node_type, NULL, 0, test_symbol, NULL, &root));

  // Create parallel nodes.
  iree_hal_streaming_graph_node_t** parallel_nodes =
      (iree_hal_streaming_graph_node_t**)calloc(
          parallel_count, sizeof(iree_hal_streaming_graph_node_t*));

  for (iree_host_size_t i = 0; i < parallel_count; ++i) {
    iree_hal_streaming_graph_node_t* deps[] = {root};
    iree_status_t status = iree_hal_streaming_graph_add_test_node(
        graph, node_type, deps, 1, test_symbol, NULL, &parallel_nodes[i]);
    if (!iree_status_is_ok(status)) {
      free(parallel_nodes);
      return status;
    }
  }

  // Create sink node.
  iree_hal_streaming_graph_node_t* sink = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_test_node(
      graph, node_type, parallel_nodes, parallel_count, test_symbol, NULL,
      &sink);

  free(parallel_nodes);

  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (out_root_node) *out_root_node = root;
  if (out_sink_node) *out_sink_node = sink;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Mixed pattern generation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_gen_mixed(
    iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_gen_params_t* params,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t*** out_nodes,
    iree_host_size_t* out_node_count) {
  // Initialize random number generator.
  srand(params->random_seed);

  // Allocate nodes array.
  iree_hal_streaming_graph_node_t** nodes =
      (iree_hal_streaming_graph_node_t**)calloc(
          params->node_count, sizeof(iree_hal_streaming_graph_node_t*));

  // Calculate node type thresholds.
  uint32_t dispatch_threshold = params->distribution.dispatch_percent;
  uint32_t memcpy_threshold =
      dispatch_threshold + params->distribution.memcpy_percent;
  uint32_t memset_threshold =
      memcpy_threshold + params->distribution.memset_percent;
  uint32_t host_threshold =
      memset_threshold + params->distribution.host_call_percent;

  for (iree_host_size_t i = 0; i < params->node_count; ++i) {
    // Determine node type based on distribution.
    uint32_t type_roll = rand() % 100;
    iree_hal_streaming_graph_node_type_t node_type;

    if (type_roll < dispatch_threshold) {
      node_type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
    } else if (type_roll < memcpy_threshold) {
      node_type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY;
    } else if (type_roll < memset_threshold) {
      node_type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET;
    } else if (type_roll < host_threshold) {
      node_type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL;
    } else {
      node_type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY;
    }

    // Determine dependencies based on concurrency level.
    iree_hal_streaming_graph_node_t** deps = NULL;
    iree_host_size_t dep_count = 0;

    if (i > 0) {
      uint32_t concurrency_roll = rand() % 100;
      if (concurrency_roll >= params->concurrency_percent) {
        // Create dependency on previous node(s).
        dep_count = 1;
        deps = &nodes[i - 1];
      }
    }

    iree_status_t status = iree_hal_streaming_graph_add_test_node(
        graph, node_type, deps, dep_count, test_symbol, params, &nodes[i]);
    if (!iree_status_is_ok(status)) {
      free(nodes);
      return status;
    }
  }

  if (out_nodes) {
    *out_nodes = nodes;
  } else {
    free(nodes);
  }
  if (out_node_count) *out_node_count = params->node_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Interleaved pattern generation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_gen_interleaved(
    iree_hal_streaming_graph_t* graph, iree_host_size_t recordable_batch_size,
    iree_host_size_t host_call_interval, iree_host_size_t total_nodes,
    iree_hal_streaming_test_symbol_t* test_symbol,
    iree_hal_streaming_graph_node_t** out_first_node,
    iree_hal_streaming_graph_node_t** out_last_node) {
  iree_hal_streaming_graph_node_t* prev_node = NULL;
  iree_hal_streaming_graph_node_t* first_node = NULL;

  for (iree_host_size_t i = 0; i < total_nodes; ++i) {
    // Determine node type based on position.
    iree_hal_streaming_graph_node_type_t node_type;
    if ((i + 1) % host_call_interval == 0) {
      node_type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL;
    } else {
      node_type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
    }

    iree_hal_streaming_graph_node_t* node = NULL;
    iree_hal_streaming_graph_node_t* deps[] = {prev_node};
    iree_host_size_t dep_count = prev_node ? 1 : 0;

    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_test_node(
        graph, node_type, deps, dep_count, test_symbol, NULL, &node));

    if (i == 0) {
      first_node = node;
    }
    prev_node = node;
  }

  if (out_first_node) *out_first_node = first_node;
  if (out_last_node) *out_last_node = prev_node;
  return iree_ok_status();
}
