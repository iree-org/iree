// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/test/graph_util.h"

#include <stdlib.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Test module management
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_test_module_load(
    iree_hal_streaming_context_t* context, iree_allocator_t allocator,
    iree_hal_streaming_module_t** out_module) {
  // Hardcoded path to the compiled test kernels.
  const char* module_path =
      "/home/ben/src/iree/experimental/streaming/test/kernels/compiled/"
      "test_kernels.cpu.so";

  // Load the module from file.
  iree_status_t status = iree_hal_streaming_module_create_from_file(
      context, IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA,
      iree_make_cstring_view(module_path), allocator, out_module);

  if (iree_status_is_ok(status)) {
    // Verify the module has the expected kernels.
    status = iree_hal_streaming_test_module_verify(*out_module);
    if (!iree_status_is_ok(status)) {
      iree_hal_streaming_module_release(*out_module);
      *out_module = NULL;
    }
  }

  return status;
}

iree_status_t iree_hal_streaming_test_module_verify(
    iree_hal_streaming_module_t* module) {
  // Expected kernel names in order matching the enum.
  const char* expected_names[] = {
      "nop_kernel",         // IREE_HAL_STREAMING_TEST_KERNEL_NOP
      "simple_add_raw",     // IREE_HAL_STREAMING_TEST_KERNEL_SIMPLE_ADD_RAW
      "simple_add_hal",     // IREE_HAL_STREAMING_TEST_KERNEL_SIMPLE_ADD_HAL
      "many_params_raw",    // IREE_HAL_STREAMING_TEST_KERNEL_MANY_PARAMS_RAW
      "many_params_hal",    // IREE_HAL_STREAMING_TEST_KERNEL_MANY_PARAMS_HAL
      "memory_copy",        // IREE_HAL_STREAMING_TEST_KERNEL_MEMORY_COPY
      "compute_intensive",  // IREE_HAL_STREAMING_TEST_KERNEL_COMPUTE_INTENSIVE
  };
  const iree_host_size_t expected_count =
      sizeof(expected_names) / sizeof(expected_names[0]);

  if (module->symbol_count != expected_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "test module has %zu symbols, expected %zu",
                            module->symbol_count, expected_count);
  }

  for (iree_host_size_t i = 0; i < expected_count; ++i) {
    if (!iree_string_view_equal(module->symbols[i].name,
                                iree_make_cstring_view(expected_names[i]))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test module symbol %zu is '%.*s', expected '%s'",
                              i, (int)module->symbols[i].name.size,
                              module->symbols[i].name.data, expected_names[i]);
    }
  }

  return iree_ok_status();
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
    iree_host_size_t dependency_count, iree_hal_streaming_module_t* test_module,
    iree_hal_streaming_test_kernel_type_t kernel_type,
    const iree_hal_streaming_graph_generate_params_t* params,
    iree_hal_streaming_graph_node_t** out_node) {
  switch (node_type) {
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY:
      return iree_hal_streaming_graph_add_empty_node(
          graph, dependencies, dependency_count, out_node);

    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL: {
      // Get the appropriate kernel symbol from the module.
      if (!test_module || kernel_type >= test_module->symbol_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid test module or kernel type");
      }
      iree_hal_streaming_symbol_t* symbol = &test_module->symbols[kernel_type];

      // Create dispatch params based on kernel type.
      // For raw kernels, pack params into buffer; for HAL kernels, use separate
      // arrays.
      iree_hal_streaming_dispatch_params_t dispatch_params = {
          .grid_dim = {1, 1, 1},
          .block_dim = {256, 1, 1},
          .shared_memory_bytes = 0,
          .buffer = NULL,
          .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_NONE,
      };

      // Allocate parameter buffer on stack if needed.
      // Size depends on kernel type - use parameter info from symbol.
      if (symbol->parameters.buffer_size > 0) {
        dispatch_params.buffer = iree_alloca(symbol->parameters.buffer_size);
        uint8_t* buffer = (uint8_t*)dispatch_params.buffer;
        
        // Fill with appropriate test data based on kernel type.
        switch (kernel_type) {
          case IREE_HAL_STREAMING_TEST_KERNEL_SIMPLE_ADD_RAW: {
            // 3 pointers + size n.
            IREE_ASSERT(params && params->buffer_a && params->buffer_b && params->buffer_c);
            uint64_t* ptrs = (uint64_t*)buffer;
            ptrs[0] = params->buffer_a->device_ptr;  // Input A pointer.
            ptrs[1] = params->buffer_b->device_ptr;  // Input B pointer.
            ptrs[2] = params->buffer_c->device_ptr;  // Output C pointer.
            uint32_t* size = (uint32_t*)(buffer + 24);
            *size = 256;  // Array size.
            break;
          }
          case IREE_HAL_STREAMING_TEST_KERNEL_SIMPLE_ADD_HAL: {
            // Parameter buffer contains 3 bindings + 1 constant interleaved.
            // Total size: 28 bytes (3 * 8 + 1 * 4).
            IREE_ASSERT(params && params->buffer_a && params->buffer_b && params->buffer_c);
            uint64_t* bindings = (uint64_t*)buffer;
            bindings[0] = params->buffer_a->device_ptr;  // Binding 0 - input A.
            bindings[1] = params->buffer_b->device_ptr;  // Binding 1 - input B.
            bindings[2] = params->buffer_c->device_ptr;  // Binding 2 - output C.
            // Add the constant n at offset 24.
            uint32_t* constant = (uint32_t*)(buffer + 24);
            *constant = 256;  // Array size n.
            break;
          }
          case IREE_HAL_STREAMING_TEST_KERNEL_MANY_PARAMS_RAW: {
            // 2 pointers + 16 scalars.
            IREE_ASSERT(params && params->buffer_a && params->buffer_b);
            uint64_t* ptrs = (uint64_t*)buffer;
            ptrs[0] = params->buffer_a->device_ptr;  // in_ptr.
            ptrs[1] = params->buffer_b->device_ptr;  // out_ptr.
            uint32_t* scalars = (uint32_t*)(buffer + 16);
            for (int i = 0; i < 16; i++) {
              scalars[i] = 0x100 + i;  // Test scalar values.
            }
            break;
          }
          case IREE_HAL_STREAMING_TEST_KERNEL_MANY_PARAMS_HAL: {
            // 10 bindings - these are device pointers that get looked up.
            IREE_ASSERT(params && params->buffer_a && params->buffer_b && params->buffer_c);
            uint64_t* bindings = (uint64_t*)buffer;
            // Use the three provided buffers, repeat as needed.
            for (uint32_t i = 0; i < 10; i++) {
              if (i < 5) {
                bindings[i] = params->buffer_a->device_ptr;  // First 5 use buffer_a.
              } else if (i < 8) {
                bindings[i] = params->buffer_b->device_ptr;  // Next 3 use buffer_b.
              } else {
                bindings[i] = params->buffer_c->device_ptr;  // Last 2 use buffer_c.
              }
            }
            // Note: Constants for HAL kernels are added by the caller.
            break;
          }
          default:
            // Fill with default test data.
            for (uint32_t i = 0; i < symbol->parameters.buffer_size; i += 4) {
              *(uint32_t*)(buffer + i) = 0x1000 + i;
            }
            break;
        }
      }

      iree_status_t status = iree_hal_streaming_graph_add_kernel_node(
          graph, dependencies, dependency_count, symbol, &dispatch_params,
          out_node);

      return status;
    }

    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY: {
      IREE_ASSERT(params && params->buffer_a && params->buffer_b);
      iree_hal_streaming_deviceptr_t src = params->buffer_a->device_ptr;
      iree_hal_streaming_deviceptr_t dst = params->buffer_b->device_ptr;
      iree_device_size_t size = params->memory_op_size;

      return iree_hal_streaming_graph_add_memcpy_node(
          graph, dependencies, dependency_count, dst, src, size, out_node);
    }

    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET: {
      IREE_ASSERT(params && params->buffer_c);
      iree_hal_streaming_deviceptr_t dst = params->buffer_c->device_ptr;
      uint32_t pattern = 0;
      iree_host_size_t pattern_size = 4;
      iree_device_size_t count = params->memory_op_size / 4;

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

iree_status_t iree_hal_streaming_graph_generate_linear_sequence(
    iree_hal_streaming_graph_t* graph, iree_host_size_t node_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_module_t* test_module,
    iree_hal_streaming_test_kernel_type_t kernel_type,
    const iree_hal_streaming_graph_generate_params_t* params,
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
        graph, node_type, deps, dep_count, test_module, kernel_type, params,
        &node));

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

iree_status_t iree_hal_streaming_graph_generate_fanout(
    iree_hal_streaming_graph_t* graph, iree_host_size_t fanout_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_module_t* test_module,
    iree_hal_streaming_test_kernel_type_t kernel_type,
    const iree_hal_streaming_graph_generate_params_t* params,
    iree_hal_streaming_graph_node_t** out_root_node,
    iree_hal_streaming_graph_node_t*** out_leaf_nodes) {
  // Create root node.
  iree_hal_streaming_graph_node_t* root = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_test_node(
      graph, node_type, NULL, 0, test_module, kernel_type, params, &root));

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
        graph, node_type, deps, 1, test_module, kernel_type, params, &node);
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

iree_status_t iree_hal_streaming_graph_generate_fanin(
    iree_hal_streaming_graph_t* graph, iree_host_size_t fanin_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_module_t* test_module,
    iree_hal_streaming_test_kernel_type_t kernel_type,
    const iree_hal_streaming_graph_generate_params_t* params,
    iree_hal_streaming_graph_node_t*** out_source_nodes,
    iree_hal_streaming_graph_node_t** out_sink_node) {
  // Allocate source nodes array.
  iree_hal_streaming_graph_node_t** source_nodes =
      (iree_hal_streaming_graph_node_t**)calloc(
          fanin_count, sizeof(iree_hal_streaming_graph_node_t*));

  // Create source nodes.
  for (iree_host_size_t i = 0; i < fanin_count; ++i) {
    iree_status_t status = iree_hal_streaming_graph_add_test_node(
        graph, node_type, NULL, 0, test_module, kernel_type, params,
        &source_nodes[i]);
    if (!iree_status_is_ok(status)) {
      free(source_nodes);
      return status;
    }
  }

  // Create sink node with all sources as dependencies.
  iree_hal_streaming_graph_node_t* sink = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_test_node(
      graph, node_type, source_nodes, fanin_count, test_module, kernel_type,
      params, &sink);

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

iree_status_t iree_hal_streaming_graph_generate_diamond(
    iree_hal_streaming_graph_t* graph, iree_host_size_t parallel_count,
    iree_hal_streaming_graph_node_type_t node_type,
    iree_hal_streaming_module_t* test_module,
    iree_hal_streaming_test_kernel_type_t kernel_type,
    const iree_hal_streaming_graph_generate_params_t* params,
    iree_hal_streaming_graph_node_t** out_root_node,
    iree_hal_streaming_graph_node_t** out_sink_node) {
  // Create root node.
  iree_hal_streaming_graph_node_t* root = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_test_node(
      graph, node_type, NULL, 0, test_module, kernel_type, params, &root));

  // Create parallel nodes.
  iree_hal_streaming_graph_node_t** parallel_nodes =
      (iree_hal_streaming_graph_node_t**)calloc(
          parallel_count, sizeof(iree_hal_streaming_graph_node_t*));

  for (iree_host_size_t i = 0; i < parallel_count; ++i) {
    iree_hal_streaming_graph_node_t* deps[] = {root};
    iree_status_t status = iree_hal_streaming_graph_add_test_node(
        graph, node_type, deps, 1, test_module, kernel_type, params,
        &parallel_nodes[i]);
    if (!iree_status_is_ok(status)) {
      free(parallel_nodes);
      return status;
    }
  }

  // Create sink node.
  iree_hal_streaming_graph_node_t* sink = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_test_node(
      graph, node_type, parallel_nodes, parallel_count, test_module,
      kernel_type, params, &sink);

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

iree_status_t iree_hal_streaming_graph_generate_mixed(
    iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_generate_params_t* params,
    iree_hal_streaming_module_t* test_module,
    iree_hal_streaming_test_kernel_type_t kernel_type,
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
        graph, node_type, deps, dep_count, test_module, kernel_type, params,
        &nodes[i]);
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

iree_status_t iree_hal_streaming_graph_generate_interleaved(
    iree_hal_streaming_graph_t* graph, iree_host_size_t recordable_batch_size,
    iree_host_size_t host_call_interval, iree_host_size_t total_nodes,
    iree_hal_streaming_module_t* test_module,
    iree_hal_streaming_test_kernel_type_t kernel_type,
    const iree_hal_streaming_graph_generate_params_t* params,
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
        graph, node_type, deps, dep_count, test_module, kernel_type, params,
        &node));

    if (i == 0) {
      first_node = node;
    }
    prev_node = node;
  }

  if (out_first_node) *out_first_node = first_node;
  if (out_last_node) *out_last_node = prev_node;
  return iree_ok_status();
}
