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
// Graph instantiation benchmarks
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

// Helper to create and populate a test graph.
static iree_status_t create_populated_graph(
    iree_hal_streaming_context_t* context, const char* pattern,
    iree_host_size_t node_count, iree_hal_streaming_graph_t** out_graph) {
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

// Benchmark instantiating a linear sequence graph.
IREE_BENCHMARK_FN(BM_GraphInstantiate_Linear) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  // Pre-create the graph outside timing loop.
  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(
      create_populated_graph(context, "linear", node_count, &graph));

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;

    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

    iree_optimization_barrier(exec);

    if (exec) {
      iree_hal_streaming_graph_exec_release(exec);
    }
  }

  iree_hal_streaming_graph_release(graph);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Linear, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Linear, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Linear, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Linear, 10000);

// Benchmark instantiating a diamond pattern graph.
IREE_BENCHMARK_FN(BM_GraphInstantiate_Diamond) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t parallel_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(
      create_populated_graph(context, "diamond", parallel_count * 2, &graph));

  while (iree_benchmark_keep_running(benchmark_state, parallel_count + 2)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;

    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

    iree_optimization_barrier(exec);

    if (exec) {
      iree_hal_streaming_graph_exec_release(exec);
    }
  }

  iree_hal_streaming_graph_release(graph);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Diamond, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Diamond, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Diamond, 1000);

// Benchmark instantiating a mixed node type graph.
IREE_BENCHMARK_FN(BM_GraphInstantiate_Mixed) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(
      create_populated_graph(context, "mixed", node_count, &graph));

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;

    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

    iree_optimization_barrier(exec);

    if (exec) {
      iree_hal_streaming_graph_exec_release(exec);
    }
  }

  iree_hal_streaming_graph_release(graph);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Mixed, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Mixed, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Mixed, 10000);

// Benchmark instantiating an interleaved pattern graph.
IREE_BENCHMARK_FN(BM_GraphInstantiate_Interleaved) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(
      create_populated_graph(context, "interleaved", node_count, &graph));

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;

    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

    iree_optimization_barrier(exec);

    if (exec) {
      iree_hal_streaming_graph_exec_release(exec);
    }
  }

  iree_hal_streaming_graph_release(graph);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Interleaved, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Interleaved, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_Interleaved, 10000);

// Benchmark instantiating with different flags.
IREE_BENCHMARK_FN(BM_GraphInstantiate_WithFlags) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t flag_bits = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(create_populated_graph(context, "mixed", 1000, &graph));

  iree_hal_streaming_graph_instantiate_flags_t flags =
      (iree_hal_streaming_graph_instantiate_flags_t)flag_bits;

  while (iree_benchmark_keep_running(benchmark_state, 1000)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;

    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_graph_instantiate(graph, flags, &exec));

    iree_optimization_barrier(exec);

    if (exec) {
      iree_hal_streaming_graph_exec_release(exec);
    }
  }

  iree_hal_streaming_graph_release(graph);
  CleanupStreamingContext();
  return iree_ok_status();
}
// Test different flag combinations.
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_WithFlags, 0);  // NONE
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_WithFlags,
                             1);  // AUTO_FREE_ON_LAUNCH
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_WithFlags, 2);  // UPLOAD
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_WithFlags,
                             4);  // DEVICE_LAUNCH
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphInstantiate_WithFlags,
                             8);  // USE_NODE_PRIORITY

// Benchmark repeated instantiation of the same graph.
IREE_BENCHMARK_FN(BM_GraphInstantiate_Repeated) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_RETURN_IF_ERROR(create_populated_graph(context, "mixed", 100, &graph));

  while (iree_benchmark_keep_running(benchmark_state, 10)) {
    // Instantiate the same graph 10 times.
    iree_hal_streaming_graph_exec_t* execs[10] = {};

    for (int i = 0; i < 10; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
          graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &execs[i]));
    }

    for (int i = 0; i < 10; ++i) {
      if (execs[i]) iree_hal_streaming_graph_exec_release(execs[i]);
    }
  }

  iree_hal_streaming_graph_release(graph);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_GraphInstantiate_Repeated);

}  // namespace
