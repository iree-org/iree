// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/internal.h"
#include "experimental/streaming/test/graph_util.h"
#include "iree/base/api.h"
#include "iree/testing/benchmark.h"

//===----------------------------------------------------------------------===//
// Graph recording benchmarks
//===----------------------------------------------------------------------===//

namespace {

static iree_status_t InitializeStreamingContext(
    iree_hal_streaming_context_t** out_context) {
  IREE_RETURN_IF_ERROR(iree_hal_streaming_init_global(
      IREE_HAL_STREAMING_INIT_FLAG_NONE, iree_allocator_system()));
  iree_host_size_t device_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_device_count(&device_count));
  if (device_count == 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "no devices available");
  }
  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(0);
  IREE_RETURN_IF_ERROR(iree_hal_streaming_device_get_or_create_primary_context(
      device, out_context));
  return iree_ok_status();
}

static void CleanupStreamingContext() { iree_hal_streaming_cleanup_global(); }

// Benchmark graph creation overhead.
IREE_BENCHMARK_FN(BM_GraphCreate) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_optimization_barrier(graph);
    iree_hal_streaming_graph_release(graph);
  }

  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_GraphCreate);

// Benchmark adding empty nodes.
IREE_BENCHMARK_FN(BM_GraphAddEmptyNode) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_node_t* prev_node = nullptr;
    for (iree_host_size_t i = 0; i < node_count; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      iree_hal_streaming_graph_node_t* deps[] = {prev_node};
      iree_host_size_t dep_count = prev_node ? 1 : 0;

      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_empty_node(
          graph, deps, dep_count, &node));
      prev_node = node;
    }

    iree_optimization_barrier(prev_node);
    iree_hal_streaming_graph_release(graph);
  }
  iree_benchmark_set_items_processed(benchmark_state, node_count);

  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddEmptyNode, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddEmptyNode, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddEmptyNode, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddEmptyNode, 1000);

// Benchmark adding kernel nodes with HAL binding lookup.
IREE_BENCHMARK_FN(BM_GraphAddKernelNode) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  // Load the test kernel module.
  iree_hal_streaming_module_t* test_module = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_module_load(
      context, iree_allocator_system(), &test_module));

  // Use simple_add_hal kernel to test HAL binding lookup overhead.
  iree_hal_streaming_test_kernel_type_t kernel_type =
      IREE_HAL_STREAMING_TEST_KERNEL_SIMPLE_ADD_HAL;
  iree_hal_streaming_symbol_t* symbol = &test_module->symbols[kernel_type];

  // Allocate device buffers for the kernel.
  const iree_device_size_t buffer_size = 256 * sizeof(float);
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_a));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_b));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_c));

  // Create parameter buffer for simple_add_hal.
  // The buffer contains 3 binding pointers + 1 constant interleaved.
  // Total: 3 * sizeof(uint64_t) + 1 * sizeof(uint32_t) = 28 bytes.
  uint8_t param_buffer[28];
  uint64_t* bindings = (uint64_t*)param_buffer;
  bindings[0] = buffer_a->device_ptr;  // Binding 0 at offset 0.
  bindings[1] = buffer_b->device_ptr;  // Binding 1 at offset 8.
  bindings[2] = buffer_c->device_ptr;  // Binding 2 at offset 16.
  uint32_t* constant = (uint32_t*)(param_buffer + 24);
  *constant = 256;  // Constant n at offset 24.

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_node_t* prev_node = nullptr;
    for (iree_host_size_t i = 0; i < node_count; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      iree_hal_streaming_graph_node_t* deps[] = {prev_node};
      iree_host_size_t dep_count = prev_node ? 1 : 0;

      iree_hal_streaming_dispatch_params_t params = {};
      params.grid_dim[0] = params.grid_dim[1] = params.grid_dim[2] = 1;
      params.block_dim[0] = 256;
      params.block_dim[1] = params.block_dim[2] = 1;
      params.buffer = param_buffer;

      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_kernel_node(
          graph, deps, dep_count, symbol, &params, &node));
      prev_node = node;
    }

    iree_optimization_barrier(prev_node);
    iree_hal_streaming_graph_release(graph);
  }
  iree_benchmark_set_items_processed(benchmark_state, node_count);

  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_a));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_b));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_c));
  iree_hal_streaming_module_release(test_module);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNode, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNode, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNode, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNode, 1000);

// Benchmark adding kernel nodes with arguments.
IREE_BENCHMARK_FN(BM_GraphAddKernelNodeWithArgs) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  // Load the test kernel module.
  iree_hal_streaming_module_t* test_module = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_module_load(
      context, iree_allocator_system(), &test_module));

  // Use many_params_raw kernel for testing with many parameters.
  iree_hal_streaming_test_kernel_type_t kernel_type =
      IREE_HAL_STREAMING_TEST_KERNEL_MANY_PARAMS_RAW;
  iree_hal_streaming_symbol_t* symbol = &test_module->symbols[kernel_type];

  // Allocate real device buffers for kernel pointers.
  const iree_device_size_t buffer_size = 256 * sizeof(float);
  iree_hal_streaming_buffer_t* in_buffer = nullptr;
  iree_hal_streaming_buffer_t* out_buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &in_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &out_buffer));

  // Allocate parameter buffer for many_params_raw (80 bytes for 20 params).
  uint8_t param_buffer[80];  // many_params_raw needs 80 bytes.
  // Fill with test data:
  // - 2 pointers (16 bytes)
  // - 16 scalar parameters (64 bytes)
  uint64_t* ptrs = (uint64_t*)param_buffer;
  ptrs[0] = in_buffer->device_ptr;   // in_ptr - real device pointer
  ptrs[1] = out_buffer->device_ptr;  // out_ptr - real device pointer
  uint32_t* scalars = (uint32_t*)(param_buffer + 16);
  for (iree_host_size_t i = 0; i < 16; ++i) {
    scalars[i] = 0x100 + i;  // Test scalar values.
  }

  while (iree_benchmark_keep_running(benchmark_state, 100)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    // Add 100 kernel nodes with the specified arg count.
    iree_hal_streaming_graph_node_t* prev_node = nullptr;
    for (iree_host_size_t i = 0; i < 100; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      iree_hal_streaming_graph_node_t* deps[] = {prev_node};
      iree_host_size_t dep_count = prev_node ? 1 : 0;

      iree_hal_streaming_dispatch_params_t params = {};
      params.grid_dim[0] = params.grid_dim[1] = params.grid_dim[2] = 1;
      params.block_dim[0] = 256;
      params.block_dim[1] = params.block_dim[2] = 1;
      params.buffer = param_buffer;

      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_kernel_node(
          graph, deps, dep_count, symbol, &params, &node));
      prev_node = node;
    }

    iree_optimization_barrier(prev_node);
    iree_hal_streaming_graph_release(graph);
  }
  iree_benchmark_set_items_processed(benchmark_state, 100);

  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(in_buffer));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(out_buffer));
  iree_hal_streaming_module_release(test_module);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeWithArgs, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeWithArgs, 4);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeWithArgs, 8);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeWithArgs, 16);

// Benchmark adding kernel nodes with many HAL bindings.
IREE_BENCHMARK_FN(BM_GraphAddKernelNodeManyHAL) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  // Load the test kernel module.
  iree_hal_streaming_module_t* test_module = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_module_load(
      context, iree_allocator_system(), &test_module));

  // Use many_params_hal kernel for testing many HAL bindings.
  iree_hal_streaming_test_kernel_type_t kernel_type =
      IREE_HAL_STREAMING_TEST_KERNEL_MANY_PARAMS_HAL;
  iree_hal_streaming_symbol_t* symbol = &test_module->symbols[kernel_type];

  // Allocate 10 device buffers for the kernel bindings.
  const iree_device_size_t buffer_size = 256 * sizeof(float);
  iree_hal_streaming_buffer_t* buffers[10];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
        context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
        &buffers[i]));
  }

  // Create parameter buffer for many_params_hal.
  // 10 bindings (80 bytes) + 10 constants (40 bytes) = 120 bytes total.
  uint8_t param_buffer[120];
  uint64_t* bindings = (uint64_t*)param_buffer;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    bindings[i] = buffers[i]->device_ptr;
  }
  // Add the 10 constants after the bindings.
  uint32_t* constants = (uint32_t*)(param_buffer + 80);
  constants[0] = 256;         // n
  constants[1] = 0x3f800000;  // weight0 (1.0f as uint32)
  constants[2] = 0x3f000000;  // weight1 (0.5f as uint32)
  constants[3] = 0x3e800000;  // weight2 (0.25f as uint32)
  constants[4] = 0x3e000000;  // weight3 (0.125f as uint32)
  constants[5] = 0x3d800000;  // weight4 (0.0625f as uint32)
  constants[6] = 0;           // offset
  constants[7] = 1;           // stride
  constants[8] = 0;           // flags
  constants[9] = 1;           // mode

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_node_t* prev_node = nullptr;
    for (iree_host_size_t i = 0; i < node_count; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      iree_hal_streaming_graph_node_t* deps[] = {prev_node};
      iree_host_size_t dep_count = prev_node ? 1 : 0;

      iree_hal_streaming_dispatch_params_t params = {};
      params.grid_dim[0] = params.grid_dim[1] = params.grid_dim[2] = 1;
      params.block_dim[0] = 256;
      params.block_dim[1] = params.block_dim[2] = 1;
      params.buffer = param_buffer;

      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_kernel_node(
          graph, deps, dep_count, symbol, &params, &node));
      prev_node = node;
    }

    iree_optimization_barrier(prev_node);
    iree_hal_streaming_graph_release(graph);
  }
  iree_benchmark_set_items_processed(benchmark_state, node_count);

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    iree_hal_streaming_memory_free_device(
        context, iree_hal_streaming_buffer_device_pointer(buffers[i]));
  }
  iree_hal_streaming_module_release(test_module);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeManyHAL, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeManyHAL, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeManyHAL, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeManyHAL, 1000);

// Benchmark adding memcpy nodes.
IREE_BENCHMARK_FN(BM_GraphAddMemcpyNode) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  // Allocate real device buffers for memcpy operations.
  const iree_device_size_t buffer_size = 1024;
  iree_hal_streaming_buffer_t* src_buffer = nullptr;
  iree_hal_streaming_buffer_t* dst_buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &src_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &dst_buffer));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_node_t* prev_node = nullptr;
    for (iree_host_size_t i = 0; i < node_count; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      iree_hal_streaming_graph_node_t* deps[] = {prev_node};
      iree_host_size_t dep_count = prev_node ? 1 : 0;
      iree_hal_streaming_deviceptr_t src = src_buffer->device_ptr;
      iree_hal_streaming_deviceptr_t dst = dst_buffer->device_ptr;
      iree_device_size_t size = 1024;
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_memcpy_node(
          graph, deps, dep_count, dst, src, size, &node));
      prev_node = node;
    }

    iree_optimization_barrier(prev_node);
    iree_hal_streaming_graph_release(graph);
  }
  iree_benchmark_set_items_processed(benchmark_state, node_count);

  // Clean up allocated buffers.
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(src_buffer));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(dst_buffer));
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddMemcpyNode, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddMemcpyNode, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddMemcpyNode, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddMemcpyNode, 1000);

// Benchmark adding host call nodes.
IREE_BENCHMARK_FN(BM_GraphAddHostCallNode) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_node_t* prev_node = nullptr;
    for (iree_host_size_t i = 0; i < node_count; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      iree_hal_streaming_graph_node_t* deps[] = {prev_node};
      iree_host_size_t dep_count = prev_node ? 1 : 0;
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_host_call_node(
          graph, deps, dep_count, iree_hal_streaming_test_host_call, nullptr,
          &node));
      prev_node = node;
    }

    iree_optimization_barrier(prev_node);
    iree_hal_streaming_graph_release(graph);
  }
  iree_benchmark_set_items_processed(benchmark_state, node_count);

  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddHostCallNode, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddHostCallNode, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddHostCallNode, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddHostCallNode, 1000);

// Benchmark adding nodes with complex dependencies.
IREE_BENCHMARK_FN(BM_GraphAddNodeWithDependencies) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t dep_count = iree_benchmark_get_range(benchmark_state, 0);
  iree_hal_streaming_graph_node_t** dep_nodes =
      (iree_hal_streaming_graph_node_t**)iree_alloca(
          dep_count * sizeof(iree_hal_streaming_graph_node_t*));
  while (iree_benchmark_keep_running(benchmark_state, 100)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    // Create dependency nodes.
    for (iree_host_size_t i = 0; i < dep_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_empty_node(
          graph, nullptr, 0, &dep_nodes[i]));
    }

    // Add 100 nodes each with dep_count dependencies.
    for (iree_host_size_t i = 0; i < 100; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_empty_node(
          graph, dep_nodes, dep_count, &node));
    }

    iree_hal_streaming_graph_release(graph);
  }
  iree_benchmark_set_items_processed(benchmark_state, dep_count + 100);

  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddNodeWithDependencies, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddNodeWithDependencies, 4);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddNodeWithDependencies, 8);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddNodeWithDependencies, 16);

// Benchmark building a complete mixed graph.
IREE_BENCHMARK_FN(BM_GraphBuildMixedRaw) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  // Load the test kernel module.
  iree_hal_streaming_module_t* test_module = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_module_load(
      context, iree_allocator_system(), &test_module));

  // Allocate device buffers for memcpy/memset operations in mixed graph.
  const iree_device_size_t buffer_size = 4096;
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_a));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_b));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_c));

  // Use simple_add_hal kernel to test HAL binding lookup in mixed graphs.
  iree_hal_streaming_test_kernel_type_t kernel_type =
      IREE_HAL_STREAMING_TEST_KERNEL_SIMPLE_ADD_RAW;
  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_generate_params_t params = {};
    params.node_count = node_count;
    params.distribution.dispatch_percent = 60;
    params.distribution.memcpy_percent = 20;
    params.distribution.memset_percent = 10;
    params.distribution.host_call_percent = 10;
    params.concurrency_percent = 50;
    params.kernel_arg_count = 4;
    params.memory_op_size = 1024;
    params.random_seed = 42;
    params.buffer_a = buffer_a;
    params.buffer_b = buffer_b;
    params.buffer_c = buffer_c;

    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_generate_mixed(
        graph, &params, test_module, kernel_type, nullptr, nullptr));

    iree_hal_streaming_graph_release(graph);
  }
  iree_benchmark_set_items_processed(benchmark_state, node_count);

  // Clean up allocated buffers.
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_a));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_b));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_c));
  iree_hal_streaming_module_release(test_module);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphBuildMixedRaw, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphBuildMixedRaw, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphBuildMixedRaw, 10000);

// Benchmark building a complete mixed graph.
IREE_BENCHMARK_FN(BM_GraphBuildMixedHAL) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  // Load the test kernel module.
  iree_hal_streaming_module_t* test_module = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_module_load(
      context, iree_allocator_system(), &test_module));

  // Allocate device buffers for memcpy/memset operations in mixed graph.
  const iree_device_size_t buffer_size = 4096;
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_a));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_b));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_c));

  // Use simple_add_hal kernel to test HAL binding lookup in mixed graphs.
  iree_hal_streaming_test_kernel_type_t kernel_type =
      IREE_HAL_STREAMING_TEST_KERNEL_SIMPLE_ADD_HAL;
  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_generate_params_t params = {};
    params.node_count = node_count;
    params.distribution.dispatch_percent = 60;
    params.distribution.memcpy_percent = 20;
    params.distribution.memset_percent = 10;
    params.distribution.host_call_percent = 10;
    params.concurrency_percent = 50;
    params.kernel_arg_count = 4;
    params.memory_op_size = 1024;
    params.random_seed = 42;
    params.buffer_a = buffer_a;
    params.buffer_b = buffer_b;
    params.buffer_c = buffer_c;

    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_generate_mixed(
        graph, &params, test_module, kernel_type, nullptr, nullptr));

    iree_hal_streaming_graph_release(graph);
  }
  iree_benchmark_set_items_processed(benchmark_state, node_count);

  // Clean up allocated buffers.
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_a));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_b));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_c));
  iree_hal_streaming_module_release(test_module);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphBuildMixedHAL, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphBuildMixedHAL, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphBuildMixedHAL, 10000);

}  // namespace
