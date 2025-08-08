// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/streaming/binding/hip/api.h"

#define HIP_CHECK(call)                                                 \
  do {                                                                  \
    hipError_t err = (call);                                            \
    if (err != hipSuccess) {                                            \
      fprintf(stderr, "HIP error at %s:%d: %s returned %d\n", __FILE__, \
              __LINE__, #call, err);                                    \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

int main(int argc, char** argv) {
  // if (argc < 2) {
  //   fprintf(stderr, "Usage: %s <module_file>\n", argv[0]);
  //   fprintf(stderr, "Example: %s kernels/compiled/vector_add.hsaco\n",
  //   argv[0]); return 1;
  // }

  const char* module_path =
      "/home/ben/src/iree/experimental/streaming/kernels/compiled/"
      "vector_add.cpu.so";  // argv[1];
  const int n = 1024;       // number of elements
  const size_t size = n * sizeof(float);

  printf("HIP Graph API Demo\n");
  printf("==================\n");
  printf("Module path: %s\n", module_path);
  printf("Vector size: %d elements\n", n);

  // Get device count (doesn't require initialization).
  printf("\n1. Checking for HIP devices...\n");
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  printf("   Found %d device(s)\n", device_count);

  if (device_count == 0) {
    fprintf(stderr, "Error: No HIP devices found\n");
    return 1;
  }

  // Get device properties (still doesn't require initialization).
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("   Using device 0: %s\n", props.name);

  // Set device 0 as current (implicitly initializes HIP runtime and creates
  // context). This is the canonical HIP runtime API initialization pattern.
  printf("\n2. Setting device 0 (initializes HIP and creates context)...\n");
  HIP_CHECK(hipSetDevice(0));

  // Load module from file.
  printf("\n3. Loading module from %s...\n", module_path);
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, module_path));

  // Get kernel function.
  printf("\n4. Getting 'vector_add' function...\n");
  hipFunction_t vector_add_func;
  HIP_CHECK(hipModuleGetFunction(&vector_add_func, module, "vector_add"));

  // Allocate host memory.
  printf("\n5. Allocating host memory...\n");
  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);
  float* h_C2 = (float*)malloc(size);  // For second launch verification
  if (!h_A || !h_B || !h_C || !h_C2) {
    fprintf(stderr, "Error: Failed to allocate host memory\n");
    return 1;
  }

  // Initialize input vectors with predictable pattern.
  printf("   Initializing input vectors...\n");
  for (int i = 0; i < n; i++) {
    h_A[i] = i * 2.0f;  // A[i] = i * 2
    h_B[i] = i * 3.0f;  // B[i] = i * 3
  }

  // Allocate device memory.
  printf("\n6. Allocating device memory...\n");
  hipDeviceptr_t d_A, d_B, d_C;
  HIP_CHECK(hipMalloc(&d_A, size));
  HIP_CHECK(hipMalloc(&d_B, size));
  HIP_CHECK(hipMalloc(&d_C, size));

  // Copy input data to device.
  printf("\n7. Copying input data to device...\n");
  HIP_CHECK(hipMemcpyHtoD(d_A, h_A, size));
  HIP_CHECK(hipMemcpyHtoD(d_B, h_B, size));

  // Create a stream for graph execution.
  printf("\n8. Creating stream...\n");
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Create a graph.
  printf("\n9. Creating graph...\n");
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare kernel node parameters.
  printf("\n10. Adding kernel node to graph...\n");

  // Calculate grid and block dimensions.
  const int threads_per_block = 256;
  const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
  printf("    Grid: %d blocks\n", blocks_per_grid);
  printf("    Block: %d threads\n", threads_per_block);

  // Prepare kernel parameters.
  void* kernel_params[] = {d_A, d_B, d_C, (void*)(uintptr_t)n};

  // Create kernel node params structure.
  hipKernelNodeParams kernel_node_params = {0};
  kernel_node_params.func = vector_add_func;
  kernel_node_params.gridDim.x = blocks_per_grid;
  kernel_node_params.gridDim.y = 1;
  kernel_node_params.gridDim.z = 1;
  kernel_node_params.blockDim.x = threads_per_block;
  kernel_node_params.blockDim.y = 1;
  kernel_node_params.blockDim.z = 1;
  kernel_node_params.sharedMemBytes = 0;
  kernel_node_params.kernelParams = kernel_params;
  kernel_node_params.extra = NULL;

  // Add kernel node to graph (no dependencies).
  hipGraphNode_t kernel_node;
  HIP_CHECK(
      hipGraphAddKernelNode(&kernel_node, graph, NULL, 0, &kernel_node_params));

  // Instantiate the graph.
  printf("\n11. Instantiating graph...\n");
  hipGraphExec_t graph_exec;
  hipGraphNode_t error_node;
  char log_buffer[1024];

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, &error_node, log_buffer,
                                sizeof(log_buffer)));

  if (strlen(log_buffer) > 0) {
    printf("    Graph instantiation log: %s\n", log_buffer);
  }

  // Launch the graph on the stream (first time).
  printf("\n12. Launching graph on stream (first execution)...\n");
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));

  // Synchronize the stream.
  printf("    Synchronizing stream...\n");
  HIP_CHECK(hipStreamSynchronize(stream));

  // Copy result back to host.
  printf("    Copying result back to host...\n");
  HIP_CHECK(hipMemcpyDtoH(h_C, d_C, size));

  // Verify first execution results.
  printf("    Verifying first execution results...\n");
  int errors1 = 0;
  const float epsilon = 1e-5f;

  for (int i = 0; i < n; i++) {
    float expected = h_A[i] + h_B[i];  // should be i * 5.0f
    float actual = h_C[i];
    float diff = fabsf(expected - actual);

    if (diff > epsilon) {
      if (errors1 < 5) {
        printf("    Error at index %d: expected %.2f, got %.2f\n", i, expected,
               actual);
      }
      errors1++;
    }
  }

  if (errors1 == 0) {
    printf(
        "     SUCCESS: First execution - all %d elements computed "
        "correctly!\n",
        n);
  } else {
    printf(
        "     FAILURE: First execution - %d errors found out of %d elements\n",
        errors1, n);
  }

  // Modify input data for second execution.
  printf("\n13. Modifying input data for second execution...\n");
  for (int i = 0; i < n; i++) {
    h_A[i] = i * 4.0f;  // A[i] = i * 4
    h_B[i] = i * 1.0f;  // B[i] = i * 1
  }
  HIP_CHECK(hipMemcpyHtoD(d_A, h_A, size));
  HIP_CHECK(hipMemcpyHtoD(d_B, h_B, size));

  // Launch the graph on the stream again (second time).
  printf("\n14. Launching graph on stream (second execution)...\n");
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));

  // Synchronize the stream.
  printf("    Synchronizing stream...\n");
  HIP_CHECK(hipStreamSynchronize(stream));

  // Copy result back to host.
  printf("    Copying result back to host...\n");
  HIP_CHECK(hipMemcpyDtoH(h_C2, d_C, size));

  // Verify second execution results.
  printf("    Verifying second execution results...\n");
  int errors2 = 0;

  for (int i = 0; i < n; i++) {
    float expected =
        h_A[i] + h_B[i];  // should be i * 5.0f (different calculation)
    float actual = h_C2[i];
    float diff = fabsf(expected - actual);

    if (diff > epsilon) {
      if (errors2 < 5) {
        printf("    Error at index %d: expected %.2f, got %.2f\n", i, expected,
               actual);
      }
      errors2++;
    }
  }

  if (errors2 == 0) {
    printf(
        "     SUCCESS: Second execution - all %d elements computed "
        "correctly!\n",
        n);
  } else {
    printf(
        "     FAILURE: Second execution - %d errors found out of %d "
        "elements\n",
        errors2, n);
  }

  // Print sample results from both executions.
  if (errors1 == 0 && errors2 == 0) {
    printf("\n15. Sample results:\n");
    printf("    First execution (A[i] = i*2, B[i] = i*3):\n");
    for (int i = 0; i < 3; i++) {
      printf("      C[%d] = %.2f\n", i, h_C[i]);
    }
    printf("    Second execution (A[i] = i*4, B[i] = i*1):\n");
    for (int i = 0; i < 3; i++) {
      printf("      C[%d] = %.2f\n", i, h_C2[i]);
    }
  }

  // Cleanup.
  printf("\n16. Cleaning up...\n");

  // Destroy graph execution.
  HIP_CHECK(hipGraphExecDestroy(graph_exec));

  // Destroy graph.
  HIP_CHECK(hipGraphDestroy(graph));

  // Destroy stream.
  HIP_CHECK(hipStreamDestroy(stream));

  // Free device memory.
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  // Free host memory.
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C2);

  // Unload module.
  HIP_CHECK(hipModuleUnload(module));

  // Reset device (destroys primary context).
  HIP_CHECK(hipDeviceReset());

  printf("\nDone!\n");

  // Extension: full global deinitialize.
  hipHALDeinit();

  return (errors1 == 0 && errors2 == 0) ? 0 : 1;
}
