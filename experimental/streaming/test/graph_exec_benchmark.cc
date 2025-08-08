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

// State structure to manage all benchmark resources.
struct ExecBenchmarkState {
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  iree_hal_streaming_module_t* module = nullptr;
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  iree_hal_streaming_context_t* context = nullptr;
};

// Helper to create, populate, and instantiate a test graph.
static iree_status_t CreateExecBenchmarkState(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream,
    const char* pattern, iree_host_size_t node_count,
    ExecBenchmarkState** out_state) {
  ExecBenchmarkState* state = new ExecBenchmarkState();
  state->context = context;
  *out_state = state;

  // Create and populate the graph.
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
      context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &state->graph));

  // Allocate device buffers needed for SIMPLE_ADD_RAW and memcpy/memset.
  const iree_device_size_t buffer_size = 256 * sizeof(float);
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &state->buffer_a));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &state->buffer_b));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_device(
      context, buffer_size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &state->buffer_c));

  // Load the test kernel module.
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_module_load(
      context, iree_allocator_system(), &state->module));

  // Use simple_add_raw kernel for benchmarking.
  iree_hal_streaming_test_kernel_type_t kernel_type =
      IREE_HAL_STREAMING_TEST_KERNEL_SIMPLE_ADD_RAW;

  // Set up params with buffers.
  iree_hal_streaming_graph_generate_params_t gen_params = {};
  gen_params.buffer_a = state->buffer_a;
  gen_params.buffer_b = state->buffer_b;
  gen_params.buffer_c = state->buffer_c;
  gen_params.memory_op_size = 1024;

  iree_status_t status = iree_ok_status();

  if (strcmp(pattern, "linear") == 0) {
    status = iree_hal_streaming_graph_generate_linear_sequence(
        state->graph, node_count, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL,
        state->module, kernel_type, &gen_params, nullptr, nullptr);
  } else if (strcmp(pattern, "diamond") == 0) {
    status = iree_hal_streaming_graph_generate_diamond(
        state->graph, node_count / 2, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL,
        state->module, kernel_type, &gen_params, nullptr, nullptr);
  } else if (strcmp(pattern, "mixed") == 0) {
    // Mixed pattern needs additional params.
    gen_params.node_count = node_count;
    gen_params.distribution.dispatch_percent = 60;
    gen_params.distribution.memcpy_percent = 20;
    gen_params.distribution.memset_percent = 10;
    gen_params.distribution.host_call_percent = 10;
    gen_params.concurrency_percent = 50;
    gen_params.kernel_arg_count = 4;
    gen_params.random_seed = 42;

    status = iree_hal_streaming_graph_generate_mixed(state->graph, &gen_params,
                                                     state->module, kernel_type,
                                                     nullptr, nullptr);
  } else if (strcmp(pattern, "interleaved") == 0) {
    status = iree_hal_streaming_graph_generate_interleaved(
        state->graph, 20, 21, node_count, state->module, kernel_type,
        &gen_params, nullptr, nullptr);
  }

  // NOTE: We keep the module alive as long as the graph exists.
  // The module will be released in DestroyExecBenchmarkState().

  if (!iree_status_is_ok(status)) {
    return status;
  }

  // Instantiate the graph.
  return iree_hal_streaming_graph_instantiate(
      state->graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE,
      &state->exec);
}

// Destroys the benchmark state and releases all resources.
static void DestroyExecBenchmarkState(ExecBenchmarkState* state) {
  if (!state) return;

  // Release in reverse order of creation.
  iree_hal_streaming_graph_exec_release(state->exec);
  iree_hal_streaming_graph_release(state->graph);
  iree_hal_streaming_module_release(state->module);

  // Clean up buffers after graph is released.
  iree_hal_streaming_memory_free_device(state->context,
                                        state->buffer_a->device_ptr);
  iree_hal_streaming_memory_free_device(state->context,
                                        state->buffer_b->device_ptr);
  iree_hal_streaming_memory_free_device(state->context,
                                        state->buffer_c->device_ptr);

  delete state;
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
  ExecBenchmarkState* state = nullptr;
  IREE_RETURN_IF_ERROR(
      CreateExecBenchmarkState(context, stream, "linear", node_count, &state));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    // Time only the launch, not the execution.
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_graph_exec_launch(state->exec, stream));

    // Barrier to prevent optimization.
    iree_optimization_barrier(state->exec);

    // Wait for completion outside of timing.
    iree_benchmark_pause_timing(benchmark_state);
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    iree_benchmark_resume_timing(benchmark_state);
    IREE_RETURN_IF_ERROR(sync_status);
  }

  DestroyExecBenchmarkState(state);
  iree_hal_streaming_stream_release(stream);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Linear, 10);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Linear, 100);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Linear, 1000);
IREE_BENCHMARK_REGISTER_ARGS(BM_GraphExecLaunch_Linear, 4096);
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

  ExecBenchmarkState* state = nullptr;
  IREE_RETURN_IF_ERROR(CreateExecBenchmarkState(context, stream, "diamond",
                                                parallel_count * 2, &state));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_graph_exec_launch(state->exec, stream));
    iree_optimization_barrier(state->exec);
    iree_benchmark_pause_timing(benchmark_state);
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    iree_benchmark_resume_timing(benchmark_state);
    IREE_RETURN_IF_ERROR(sync_status);
  }

  DestroyExecBenchmarkState(state);
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

  ExecBenchmarkState* state = nullptr;
  IREE_RETURN_IF_ERROR(
      CreateExecBenchmarkState(context, stream, "mixed", node_count, &state));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_graph_exec_launch(state->exec, stream));
    iree_optimization_barrier(state->exec);
    iree_benchmark_pause_timing(benchmark_state);
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    iree_benchmark_resume_timing(benchmark_state);
    IREE_RETURN_IF_ERROR(sync_status);
  }

  DestroyExecBenchmarkState(state);
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

  ExecBenchmarkState* state = nullptr;
  IREE_RETURN_IF_ERROR(CreateExecBenchmarkState(context, stream, "interleaved",
                                                node_count, &state));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_graph_exec_launch(state->exec, stream));
    iree_optimization_barrier(state->exec);
    iree_benchmark_pause_timing(benchmark_state);
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    iree_benchmark_resume_timing(benchmark_state);
    IREE_RETURN_IF_ERROR(sync_status);
  }

  DestroyExecBenchmarkState(state);
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

  ExecBenchmarkState* state = nullptr;
  IREE_RETURN_IF_ERROR(
      CreateExecBenchmarkState(context, stream, "mixed", 100, &state));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    for (int64_t i = 0; i < launch_count; ++i) {
      IREE_RETURN_IF_ERROR(
          iree_hal_streaming_graph_exec_launch(state->exec, stream));
    }
    iree_benchmark_pause_timing(benchmark_state);
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    iree_benchmark_resume_timing(benchmark_state);
    IREE_RETURN_IF_ERROR(sync_status);
  }

  DestroyExecBenchmarkState(state);
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
  ExecBenchmarkState* state = nullptr;
  IREE_RETURN_IF_ERROR(
      CreateExecBenchmarkState(context, stream, "linear", 1, &state));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_graph_exec_launch(state->exec, stream));
    iree_optimization_barrier(state->exec);
    iree_benchmark_pause_timing(benchmark_state);
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    iree_benchmark_resume_timing(benchmark_state);
    IREE_RETURN_IF_ERROR(sync_status);
  }

  DestroyExecBenchmarkState(state);
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
  ExecBenchmarkState* state = nullptr;
  IREE_RETURN_IF_ERROR(
      CreateExecBenchmarkState(context, stream, "mixed", 50000, &state));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_graph_exec_launch(state->exec, stream));
    iree_optimization_barrier(state->exec);
    iree_benchmark_pause_timing(benchmark_state);
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    iree_benchmark_resume_timing(benchmark_state);
    IREE_RETURN_IF_ERROR(sync_status);
  }

  DestroyExecBenchmarkState(state);
  iree_hal_streaming_stream_release(stream);
  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_GraphExecLaunch_Large);

}  // namespace
