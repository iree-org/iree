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
// Graph execution launch benchmarks
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

// Helper to create, populate, and instantiate a test graph.
static iree_status_t create_instantiated_graph(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream,
    const char* pattern, iree_host_size_t node_count,
    iree_hal_streaming_graph_t** out_graph,
    iree_hal_streaming_graph_exec_t** out_exec) {
  // Create and populate the graph.
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

  if (!iree_status_is_ok(status)) {
    return status;
  }

  // Instantiate the graph.
  return iree_hal_streaming_graph_instantiate(
      *out_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, out_exec);
}

// Benchmark launching a linear sequence graph.
IREE_BENCHMARK_FN(BM_GraphExecLaunch_Linear) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  // Create a stream for execution.
  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0, iree_allocator_system(),
      &stream));

  // Pre-create and instantiate the graph outside timing loop.
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_RETURN_IF_ERROR(create_instantiated_graph(context, stream, "linear",
                                                 node_count, &graph, &exec));

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    // Time only the launch, not the execution.
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_launch(exec, stream));

    // Barrier to prevent optimization.
    iree_optimization_barrier(exec);

    // Wait for completion outside of timing.
    iree_benchmark_pause_timing(benchmark_state);
    if (stream) {
      iree_hal_streaming_stream_synchronize(stream);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_hal_streaming_graph_exec_release(exec);
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Linear, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Linear, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Linear, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Linear, 10000);

// Benchmark launching a diamond pattern graph.
IREE_BENCHMARK_FN(BM_GraphExecLaunch_Diamond) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t parallel_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0, iree_allocator_system(),
      &stream));

  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_RETURN_IF_ERROR(create_instantiated_graph(
      context, stream, "diamond", parallel_count * 2, &graph, &exec));

  while (iree_benchmark_keep_running(benchmark_state, parallel_count + 2)) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_launch(exec, stream));

    iree_optimization_barrier(exec);

    iree_benchmark_pause_timing(benchmark_state);
    if (stream) {
      iree_hal_streaming_stream_synchronize(stream);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_hal_streaming_graph_exec_release(exec);
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Diamond, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Diamond, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Diamond, 1000);

// Benchmark launching a mixed node type graph.
IREE_BENCHMARK_FN(BM_GraphExecLaunch_Mixed) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0, iree_allocator_system(),
      &stream));

  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_RETURN_IF_ERROR(create_instantiated_graph(context, stream, "mixed",
                                                 node_count, &graph, &exec));

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_launch(exec, stream));

    iree_optimization_barrier(exec);

    iree_benchmark_pause_timing(benchmark_state);
    if (stream) {
      iree_hal_streaming_stream_synchronize(stream);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_hal_streaming_graph_exec_release(exec);
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Mixed, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Mixed, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Mixed, 10000);

// Benchmark launching an interleaved pattern graph.
IREE_BENCHMARK_FN(BM_GraphExecLaunch_Interleaved) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0, iree_allocator_system(),
      &stream));

  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_RETURN_IF_ERROR(create_instantiated_graph(context, stream, "interleaved",
                                                 node_count, &graph, &exec));

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_launch(exec, stream));

    iree_optimization_barrier(exec);

    iree_benchmark_pause_timing(benchmark_state);
    if (stream) {
      iree_hal_streaming_stream_synchronize(stream);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_hal_streaming_graph_exec_release(exec);
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Interleaved, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Interleaved, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Interleaved, 10000);

// Benchmark repeated launches of the same graph exec.
IREE_BENCHMARK_FN(BM_GraphExecLaunch_Repeated) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t launch_count = iree_benchmark_get_range(benchmark_state, 0);

  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0, iree_allocator_system(),
      &stream));

  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_RETURN_IF_ERROR(
      create_instantiated_graph(context, stream, "mixed", 100, &graph, &exec));

  while (iree_benchmark_keep_running(benchmark_state, launch_count)) {
    for (int64_t i = 0; i < launch_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_launch(exec, stream));
    }

    // Wait for all launches to complete outside timing.
    iree_benchmark_pause_timing(benchmark_state);
    if (stream) {
      iree_hal_streaming_stream_synchronize(stream);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_hal_streaming_graph_exec_release(exec);
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Repeated, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Repeated, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Repeated, 1000);

// Benchmark launch overhead for tiny graphs.
IREE_BENCHMARK_FN(BM_GraphExecLaunch_Tiny) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0, iree_allocator_system(),
      &stream));

  // Create a single-node graph.
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_RETURN_IF_ERROR(
      create_instantiated_graph(context, stream, "linear", 1, &graph, &exec));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_launch(exec, stream));

    iree_optimization_barrier(exec);

    // For tiny graphs, synchronization overhead matters.
    iree_benchmark_pause_timing(benchmark_state);
    if (stream) {
      iree_hal_streaming_stream_synchronize(stream);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_hal_streaming_graph_exec_release(exec);
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_GraphExecLaunch_Tiny);

// Benchmark launch overhead for large graphs.
IREE_BENCHMARK_FN(BM_GraphExecLaunch_Large) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0, iree_allocator_system(),
      &stream));

  // Create a 50,000 node graph.
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_RETURN_IF_ERROR(create_instantiated_graph(context, stream, "mixed",
                                                 50000, &graph, &exec));

  while (iree_benchmark_keep_running(benchmark_state, 50000)) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_exec_launch(exec, stream));

    iree_optimization_barrier(exec);

    iree_benchmark_pause_timing(benchmark_state);
    if (stream) {
      iree_hal_streaming_stream_synchronize(stream);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_hal_streaming_graph_exec_release(exec);
  iree_hal_streaming_graph_release(graph);
  iree_hal_streaming_stream_release(stream);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_GraphExecLaunch_Large);

}  // namespace
