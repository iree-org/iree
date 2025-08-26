// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/streaming/binding/cuda/api.h"

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s returned %s (%d)\n", __FILE__, \
              __LINE__, #call, cudaGetErrorString(err), err);                 \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define CU_CHECK(call)                                                \
  do {                                                                \
    CUresult err = (call);                                            \
    if (err != CUDA_SUCCESS) {                                        \
      fprintf(stderr, "CUDA driver error at %s:%d: %s returned %d\n", \
              __FILE__, __LINE__, #call, err);                        \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <module_file>\n", argv[0]);
    fprintf(stderr, "Example: %s kernels/compiled/vector_add.hsaco\n", argv[0]);
    return 1;
  }

  const char* module_path = argv[1];
  const int n = 1024;  // number of elements
  const size_t size = n * sizeof(float);

  printf("CUDA Runtime API Vector Addition Demo\n");
  printf("=====================================\n");
  printf("Module path: %s\n", module_path);
  printf("Vector size: %d elements\n", n);

  // Initialize CUDA driver (required for module loading).
  // This is commonly done in frameworks that mix Runtime and Driver APIs.
  printf("\n1. Initializing CUDA driver for module loading...\n");
  CU_CHECK(cuInit(0));

  // Get device count and properties using Runtime API.
  printf("\n2. Querying devices (Runtime API)...\n");
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  printf("   Found %d device(s)\n", device_count);

  if (device_count == 0) {
    fprintf(stderr, "Error: No CUDA devices found\n");
    return 1;
  }

  // Get device properties using Runtime API.
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("   Using device 0: %s\n", prop.name);
  printf("   Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("   Total global memory: %.2f GB\n",
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

  // Set the device using Runtime API.
  printf("\n3. Setting device 0 (Runtime API)...\n");
  CUDA_CHECK(cudaSetDevice(0));

  // Load module using Driver API (no Runtime API equivalent).
  printf("\n4. Loading module from %s (Driver API)...\n", module_path);
  CUmodule module;
  CU_CHECK(cuModuleLoad(&module, module_path));

  // Get kernel function using Driver API.
  printf("\n5. Getting 'vector_add' function (Driver API)...\n");
  CUfunction vector_add_func;
  CU_CHECK(cuModuleGetFunction(&vector_add_func, module, "vector_add"));

  // Allocate host memory using standard malloc.
  printf("\n6. Allocating host memory...\n");
  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);
  if (!h_A || !h_B || !h_C) {
    fprintf(stderr, "Error: Failed to allocate host memory\n");
    return 1;
  }

  // Initialize input vectors with predictable pattern.
  printf("   Initializing input vectors...\n");
  for (int i = 0; i < n; i++) {
    h_A[i] = i * 2.0f;  // A[i] = i * 2
    h_B[i] = i * 3.0f;  // B[i] = i * 3
  }

  // Allocate device memory using Runtime API.
  printf("\n7. Allocating device memory (Runtime API)...\n");
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void**)&d_A, size));
  CUDA_CHECK(cudaMalloc((void**)&d_B, size));
  CUDA_CHECK(cudaMalloc((void**)&d_C, size));

  // Copy input data to device using Runtime API.
  printf("\n8. Copying input data to device (Runtime API)...\n");
  CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // Launch kernel using Runtime API's cudaLaunchKernel.
  printf("\n9. Launching kernel (Runtime API)...\n");

  // Calculate grid and block dimensions.
  const int threads_per_block = 256;
  const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
  printf("   Grid: %d blocks\n", blocks_per_grid);
  printf("   Block: %d threads\n", threads_per_block);

  // Set up kernel parameters.
  // The vector_add kernel takes: (const float* a, const float* b, float* c, int
  // n).
  void* kernel_params[] = {&d_A, &d_B, &d_C, (void*)&n};

  // Launch kernel using cudaLaunchKernel.
  // Note: cudaLaunchKernel expects a function pointer from a linked kernel,
  // but we can cast the CUfunction to use it (this is what PyTorch does).
  dim3 grid_dim = {blocks_per_grid, 1, 1};
  dim3 block_dim = {threads_per_block, 1, 1};

  CUDA_CHECK(cudaLaunchKernel(
      (const void*)vector_add_func,  // kernel function (cast from CUfunction)
      grid_dim,                      // grid dimensions
      block_dim,                     // block dimensions
      kernel_params,                 // kernel arguments
      0,                             // shared memory size
      NULL                           // stream (NULL = default)
      ));

  // Check for kernel launch errors.
  CUDA_CHECK(cudaGetLastError());

  // Synchronize device using Runtime API.
  printf("\n10. Synchronizing device (Runtime API)...\n");
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host using Runtime API.
  printf("\n11. Copying result back to host (Runtime API)...\n");
  CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  // Verify results.
  printf("\n12. Verifying results...\n");
  int errors = 0;
  const float epsilon = 1e-5f;

  for (int i = 0; i < n; i++) {
    float expected = h_A[i] + h_B[i];  // should be i * 5.0f
    float actual = h_C[i];
    float diff = fabsf(expected - actual);

    if (diff > epsilon) {
      // Print only the first 10 errors.
      if (errors < 10) {
        printf("   Error at index %d: expected %.2f, got %.2f\n", i, expected,
               actual);
      }
      errors++;
    }
  }

  if (errors == 0) {
    printf("   ✓ SUCCESS: All %d elements computed correctly!\n", n);

    // Print a few sample results.
    printf("\n   Sample results:\n");
    for (int i = 0; i < 5; i++) {
      printf("   C[%d] = A[%d] + B[%d] = %.2f + %.2f = %.2f\n", i, i, i, h_A[i],
             h_B[i], h_C[i]);
    }
    printf("   ...\n");
    for (int i = n - 3; i < n; i++) {
      printf("   C[%d] = A[%d] + B[%d] = %.2f + %.2f = %.2f\n", i, i, i, h_A[i],
             h_B[i], h_C[i]);
    }
  } else {
    printf("   ✗ FAILURE: %d errors found out of %d elements\n", errors, n);
  }

  // Cleanup.
  printf("\n13. Cleaning up...\n");

  // Free device memory using Runtime API.
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  // Free host memory.
  free(h_A);
  free(h_B);
  free(h_C);

  // Unload module using Driver API.
  printf("   Unloading module (Driver API)...\n");
  CU_CHECK(cuModuleUnload(module));

  // Reset device using Runtime API.
  printf("   Resetting device (Runtime API)...\n");
  CUDA_CHECK(cudaDeviceReset());

  printf("\nDone!\n");

  return (errors == 0) ? 0 : 1;
}
