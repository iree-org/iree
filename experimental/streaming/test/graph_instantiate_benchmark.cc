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
    iree_host_size_t node_count, iree_hal_streaming_graph_t** out_graph,
    iree_hal_streaming_buffer_t** out_buffer_a,
    iree_hal_streaming_buffer_t** out_buffer_b,
    iree_hal_streaming_buffer_t** out_buffer_c) {
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_create(
      context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      out_graph));

  // Allocate device buffers needed for SIMPLE_ADD_RAW and memcpy/memset.
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

  // Load the test kernel module.
  iree_hal_streaming_module_t* test_module = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_test_module_load(
      context, iree_allocator_system(), &test_module));

  // Use simple_add_raw kernel for benchmarking.
  iree_hal_streaming_test_kernel_type_t kernel_type =
      IREE_HAL_STREAMING_TEST_KERNEL_SIMPLE_ADD_RAW;

  // Set up params with buffers.
  iree_hal_streaming_graph_generate_params_t gen_params = {};
  gen_params.buffer_a = buffer_a;
  gen_params.buffer_b = buffer_b;
  gen_params.buffer_c = buffer_c;
  gen_params.memory_op_size = 1024;

  iree_status_t status = iree_ok_status();
  if (strcmp(pattern, "linear") == 0) {
    status = iree_hal_streaming_graph_generate_linear_sequence(
        *out_graph, node_count, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL,
        test_module, kernel_type, &gen_params, nullptr, nullptr);
  } else if (strcmp(pattern, "diamond") == 0) {
    status = iree_hal_streaming_graph_generate_diamond(
        *out_graph, node_count / 2, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL,
        test_module, kernel_type, &gen_params, nullptr, nullptr);
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

    status = iree_hal_streaming_graph_generate_mixed(
        *out_graph, &gen_params, test_module, kernel_type, nullptr, nullptr);
  } else if (strcmp(pattern, "interleaved") == 0) {
    status = iree_hal_streaming_graph_generate_interleaved(
        *out_graph, 20, 21, node_count, test_module, kernel_type, &gen_params,
        nullptr, nullptr);
  }

  iree_hal_streaming_module_release(test_module);

  // Return buffers to caller for cleanup after graph is released.
  if (out_buffer_a) *out_buffer_a = buffer_a;
  if (out_buffer_b) *out_buffer_b = buffer_b;
  if (out_buffer_c) *out_buffer_c = buffer_c;

  return status;
}

// Benchmark instantiating a linear sequence graph.
IREE_BENCHMARK_FN(BM_GraphInstantiate_Linear) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(InitializeStreamingContext(&context));

  int64_t node_count = iree_benchmark_get_range(benchmark_state, 0);

  // Pre-create the graph outside timing loop.
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  IREE_RETURN_IF_ERROR(create_populated_graph(
      context, "linear", node_count, &graph, &buffer_a, &buffer_b, &buffer_c));

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));
    iree_optimization_barrier(exec);
    iree_hal_streaming_graph_exec_release(exec);
  }

  iree_hal_streaming_graph_release(graph);

  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_a));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_b));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_c));

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
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  IREE_RETURN_IF_ERROR(create_populated_graph(context, "diamond",
                                              parallel_count * 2, &graph,
                                              &buffer_a, &buffer_b, &buffer_c));

  while (iree_benchmark_keep_running(benchmark_state, parallel_count + 2)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));
    iree_optimization_barrier(exec);
    iree_hal_streaming_graph_exec_release(exec);
  }

  iree_hal_streaming_graph_release(graph);

  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_a));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_b));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_c));

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
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  IREE_RETURN_IF_ERROR(create_populated_graph(
      context, "mixed", node_count, &graph, &buffer_a, &buffer_b, &buffer_c));

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));
    iree_optimization_barrier(exec);
    iree_hal_streaming_graph_exec_release(exec);
  }

  iree_hal_streaming_graph_release(graph);

  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_a));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_b));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_c));

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
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  IREE_RETURN_IF_ERROR(create_populated_graph(context, "interleaved",
                                              node_count, &graph, &buffer_a,
                                              &buffer_b, &buffer_c));

  while (iree_benchmark_keep_running(benchmark_state, node_count)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));
    iree_optimization_barrier(exec);
    iree_hal_streaming_graph_exec_release(exec);
  }

  iree_hal_streaming_graph_release(graph);

  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_a));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_b));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_c));

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
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  IREE_RETURN_IF_ERROR(create_populated_graph(context, "mixed", 1000, &graph,
                                              &buffer_a, &buffer_b, &buffer_c));

  iree_hal_streaming_graph_instantiate_flags_t flags =
      (iree_hal_streaming_graph_instantiate_flags_t)flag_bits;

  while (iree_benchmark_keep_running(benchmark_state, 1000)) {
    iree_hal_streaming_graph_exec_t* exec = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_graph_instantiate(graph, flags, &exec));
    iree_optimization_barrier(exec);
    iree_hal_streaming_graph_exec_release(exec);
  }

  iree_hal_streaming_graph_release(graph);

  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_a));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_b));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_c));

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
  iree_hal_streaming_buffer_t* buffer_a = nullptr;
  iree_hal_streaming_buffer_t* buffer_b = nullptr;
  iree_hal_streaming_buffer_t* buffer_c = nullptr;
  IREE_RETURN_IF_ERROR(create_populated_graph(context, "mixed", 100, &graph,
                                              &buffer_a, &buffer_b, &buffer_c));

  while (iree_benchmark_keep_running(benchmark_state, 10)) {
    // Instantiate the same graph 10 times.
    iree_hal_streaming_graph_exec_t* execs[10] = {};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(execs); ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_instantiate(
          graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &execs[i]));
    }
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(execs); ++i) {
      iree_hal_streaming_graph_exec_release(execs[i]);
    }
  }

  iree_hal_streaming_graph_release(graph);

  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_a));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_b));
  iree_hal_streaming_memory_free_device(
      context, iree_hal_streaming_buffer_device_pointer(buffer_c));

  CleanupStreamingContext();
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_GraphInstantiate_Repeated);

}  // namespace
