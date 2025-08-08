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
    for (int32_t i = 0; i < node_count; ++i) {
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

  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddEmptyNode, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddEmptyNode, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddEmptyNode, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddEmptyNode, 1000);

// Benchmark adding kernel nodes.
IREE_BENCHMARK_FN(BM_GraphAddKernelNode) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  iree_hal_streaming_test_symbol_t* symbol = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_symbol_create(
      "test_kernel", iree_allocator_system(), &symbol));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_node_t* prev_node = nullptr;
    for (int32_t i = 0; i < node_count; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      iree_hal_streaming_graph_node_t* deps[] = {prev_node};
      iree_host_size_t dep_count = prev_node ? 1 : 0;

      iree_hal_streaming_dispatch_params_t params = {};
      params.grid_dim[0] = params.grid_dim[1] = params.grid_dim[2] = 1;
      params.block_dim[0] = 256;
      params.block_dim[1] = params.block_dim[2] = 1;

      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_kernel_node(
          graph, deps, dep_count, (iree_hal_streaming_symbol_t*)symbol, &params,
          &node));
      prev_node = node;
    }

    iree_optimization_barrier(prev_node);
    iree_hal_streaming_graph_release(graph);
  }

  iree_hal_streaming_test_symbol_destroy(symbol);
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

  iree_hal_streaming_test_symbol_t* symbol = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_symbol_create(
      "test_kernel", iree_allocator_system(), &symbol));

  // Allocate kernel args.
  int64_t arg_count = iree_benchmark_get_range(benchmark_state, 0);
  void** kernel_args = (void**)calloc(arg_count, sizeof(void*));
  for (int32_t i = 0; i < arg_count; ++i) {
    kernel_args[i] = (void*)(uintptr_t)(0x1000 + i * 8);
  }

  while (iree_benchmark_keep_running(benchmark_state, 100)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    // Add 100 kernel nodes with the specified arg count.
    iree_hal_streaming_graph_node_t* prev_node = nullptr;
    for (int32_t i = 0; i < 100; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      iree_hal_streaming_graph_node_t* deps[] = {prev_node};
      iree_host_size_t dep_count = prev_node ? 1 : 0;

      iree_hal_streaming_dispatch_params_t params = {};
      params.grid_dim[0] = params.grid_dim[1] = params.grid_dim[2] = 1;
      params.block_dim[0] = 256;
      params.block_dim[1] = params.block_dim[2] = 1;
      params.kernel_params = kernel_args;

      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_kernel_node(
          graph, deps, dep_count, (iree_hal_streaming_symbol_t*)symbol, &params,
          &node));
      prev_node = node;
    }

    iree_optimization_barrier(prev_node);
    iree_hal_streaming_graph_release(graph);
  }

  free(kernel_args);
  iree_hal_streaming_test_symbol_destroy(symbol);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeWithArgs, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeWithArgs, 4);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeWithArgs, 8);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddKernelNodeWithArgs, 16);

// Benchmark adding memcpy nodes.
IREE_BENCHMARK_FN(BM_GraphAddMemcpyNode) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_node_t* prev_node = nullptr;
    for (int32_t i = 0; i < node_count; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      iree_hal_streaming_graph_node_t* deps[] = {prev_node};
      iree_host_size_t dep_count = prev_node ? 1 : 0;

      iree_hal_streaming_deviceptr_t src = 0x2000 + i * 0x1000;
      iree_hal_streaming_deviceptr_t dst = 0x3000 + i * 0x1000;
      iree_device_size_t size = 1024;

      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_memcpy_node(
          graph, deps, dep_count, dst, src, size, &node));
      prev_node = node;
    }

    iree_optimization_barrier(prev_node);
    iree_hal_streaming_graph_release(graph);
  }

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
    for (int32_t i = 0; i < node_count; ++i) {
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
  while (iree_benchmark_keep_running(benchmark_state, 100)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    // Create dependency nodes.
    iree_hal_streaming_graph_node_t** dep_nodes =
        (iree_hal_streaming_graph_node_t**)calloc(
            dep_count, sizeof(iree_hal_streaming_graph_node_t*));

    for (int32_t i = 0; i < dep_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_empty_node(
          graph, nullptr, 0, &dep_nodes[i]));
    }

    // Add 100 nodes each with dep_count dependencies.
    for (int32_t i = 0; i < 100; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_empty_node(
          graph, dep_nodes, dep_count, &node));
    }

    free(dep_nodes);
    iree_hal_streaming_graph_release(graph);
  }

  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddNodeWithDependencies, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddNodeWithDependencies, 4);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddNodeWithDependencies, 8);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphAddNodeWithDependencies, 16);

// Benchmark building a complete mixed graph.
IREE_BENCHMARK_FN(BM_GraphBuildMixed) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  iree_hal_streaming_test_symbol_t* symbol = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_symbol_create(
      "test_kernel", iree_allocator_system(), &symbol));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);
  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
        context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));

    iree_hal_streaming_graph_gen_params_t params = {};
    params.node_count = node_count;
    params.distribution.dispatch_percent = 60;
    params.distribution.memcpy_percent = 20;
    params.distribution.memset_percent = 10;
    params.distribution.host_call_percent = 10;
    params.concurrency_percent = 50;
    params.kernel_arg_count = 4;
    params.memory_op_size = 1024;
    params.random_seed = 42;

    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_gen_mixed(
        graph, &params, symbol, nullptr, nullptr));

    iree_hal_streaming_graph_release(graph);
  }

  iree_hal_streaming_test_symbol_destroy(symbol);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphBuildMixed, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphBuildMixed, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphBuildMixed, 10000);

}  // namespace
