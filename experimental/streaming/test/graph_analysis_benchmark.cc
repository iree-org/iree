// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/graph.h"
#include "experimental/streaming/internal.h"
#include "experimental/streaming/test/graph_util.h"
#include "iree/base/api.h"
#include "iree/testing/benchmark.h"

//===----------------------------------------------------------------------===//
// Graph scheduling benchmarks
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

// Creates a graph with the specified pattern and node count.
static iree_status_t create_test_graph(iree_hal_streaming_context_t* context,
                                       const char* pattern,
                                       iree_host_size_t node_count,
                                       iree_hal_streaming_graph_t** out_graph) {
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
      context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      out_graph));

  iree_hal_streaming_test_symbol_t* symbol = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_symbol_create(
      "test_kernel", iree_allocator_system(), &symbol));

  iree_status_t status = iree_ok_status();

  if (strcmp(pattern, "linear") == 0) {
    status = iree_hal_streaming_graph_gen_linear_sequence(
        *out_graph, node_count, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL,
        symbol, nullptr, nullptr);
  } else if (strcmp(pattern, "fanout") == 0) {
    status = iree_hal_streaming_graph_gen_fanout(
        *out_graph, node_count, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL,
        symbol, nullptr, nullptr);
  } else if (strcmp(pattern, "fanin") == 0) {
    status = iree_hal_streaming_graph_gen_fanin(
        *out_graph, node_count, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL,
        symbol, nullptr, nullptr);
  } else if (strcmp(pattern, "diamond") == 0) {
    status = iree_hal_streaming_graph_gen_diamond(
        *out_graph, node_count / 2, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL,
        symbol, nullptr, nullptr);
  } else if (strcmp(pattern, "mixed") == 0) {
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

    status = iree_hal_streaming_graph_gen_mixed(*out_graph, &params, symbol,
                                                nullptr, nullptr);
  } else if (strcmp(pattern, "interleaved") == 0) {
    status = iree_hal_streaming_graph_gen_interleaved(
        *out_graph, 20, 21, node_count, symbol, nullptr, nullptr);
  }

  iree_hal_streaming_test_symbol_destroy(symbol);
  return status;
}

// Benchmark scheduling for linear chain of nodes.
IREE_BENCHMARK_FN(BM_GraphSchedule_Linear) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(
      create_test_graph(context, "linear", node_count, &graph));

  // Create arena for scheduling.
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                   &block_pool);

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool, &arena);

    // Get the node blocks from the graph (this is implementation-specific).
    // For benchmarking, we need to extract the internal node blocks.
    iree_hal_streaming_node_block_t* node_blocks = nullptr;
    iree_host_size_t actual_node_count = node_count;

    // Schedule the nodes.
    iree_hal_streaming_graph_schedule_t schedule = {};
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_schedule_nodes(
        node_blocks, actual_node_count, &arena, &schedule));

    iree_optimization_barrier(&schedule);
    iree_arena_deinitialize(&arena);
  }

  iree_hal_streaming_graph_release(graph);
  iree_arena_block_pool_deinitialize(&block_pool);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Linear, 1);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Linear, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Linear, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Linear, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Linear, 10000);

// Benchmark scheduling for fanout pattern.
IREE_BENCHMARK_FN(BM_GraphSchedule_Fanout) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t fanout_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(
      create_test_graph(context, "fanout", fanout_count, &graph));

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                   &block_pool);

  while (iree_benchmark_keep_running(benchmark_state, fanout_count + 1)) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool, &arena);

    iree_hal_streaming_node_block_t* node_blocks = nullptr;
    iree_host_size_t node_count = fanout_count + 1;  // root + leaves

    iree_hal_streaming_graph_schedule_t schedule = {};
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_schedule_nodes(
        node_blocks, node_count, &arena, &schedule));

    iree_optimization_barrier(&schedule);
    iree_arena_deinitialize(&arena);
  }

  iree_hal_streaming_graph_release(graph);
  iree_arena_block_pool_deinitialize(&block_pool);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Fanout, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Fanout, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Fanout, 1000);

// Benchmark scheduling for diamond pattern.
IREE_BENCHMARK_FN(BM_GraphSchedule_Diamond) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t parallel_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(
      create_test_graph(context, "diamond", parallel_count * 2, &graph));

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                   &block_pool);

  while (iree_benchmark_keep_running(benchmark_state, parallel_count + 2)) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool, &arena);

    iree_hal_streaming_node_block_t* node_blocks = nullptr;
    iree_host_size_t node_count = parallel_count + 2;  // root + parallel + sink

    iree_hal_streaming_graph_schedule_t schedule = {};
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_schedule_nodes(
        node_blocks, node_count, &arena, &schedule));

    iree_optimization_barrier(&schedule);
    iree_arena_deinitialize(&arena);
  }

  iree_hal_streaming_graph_release(graph);
  iree_arena_block_pool_deinitialize(&block_pool);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Diamond, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Diamond, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Diamond, 1000);

// Benchmark scheduling for mixed node types.
IREE_BENCHMARK_FN(BM_GraphSchedule_Mixed) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(create_test_graph(context, "mixed", node_count, &graph));

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(256 * 1024, iree_allocator_system(),
                                   &block_pool);

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool, &arena);

    iree_hal_streaming_node_block_t* node_blocks = nullptr;
    iree_host_size_t actual_node_count = node_count;

    iree_hal_streaming_graph_schedule_t schedule = {};
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_schedule_nodes(
        node_blocks, actual_node_count, &arena, &schedule));

    iree_optimization_barrier(&schedule);
    iree_arena_deinitialize(&arena);
  }

  iree_hal_streaming_graph_release(graph);
  iree_arena_block_pool_deinitialize(&block_pool);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Mixed, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Mixed, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Mixed, 10000);

// Benchmark scheduling for interleaved pattern.
IREE_BENCHMARK_FN(BM_GraphSchedule_Interleaved) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(
      create_test_graph(context, "interleaved", node_count, &graph));

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(256 * 1024, iree_allocator_system(),
                                   &block_pool);

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool, &arena);

    iree_hal_streaming_node_block_t* node_blocks = nullptr;
    iree_host_size_t actual_node_count = node_count;

    iree_hal_streaming_graph_schedule_t schedule = {};
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_schedule_nodes(
        node_blocks, actual_node_count, &arena, &schedule));

    iree_optimization_barrier(&schedule);
    iree_arena_deinitialize(&arena);
  }

  iree_hal_streaming_graph_release(graph);
  iree_arena_block_pool_deinitialize(&block_pool);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Interleaved, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Interleaved, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphSchedule_Interleaved, 10000);

}  // namespace
