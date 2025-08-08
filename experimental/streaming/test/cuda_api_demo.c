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

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    CUresult err = (call);                                               \
    if (err != CUDA_SUCCESS) {                                           \
      fprintf(stderr, "CUDA error at %s:%d: %s returned %d\n", __FILE__, \
              __LINE__, #call, err);                                     \
      exit(1);                                                           \
    }                                                                    \
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

  printf("CUDA API Vector Addition Demo\n");
  printf("=========================\n");
  printf("Module path: %s\n", module_path);
  printf("Vector size: %d elements\n", n);

  // Initialize CUDA.
  printf("\n1. Initializing CUDA...\n");
  CUDA_CHECK(cuInit(0));

  // Get device count.
  int device_count = 0;
  CUDA_CHECK(cuDeviceGetCount(&device_count));
  printf("   Found %d device(s)\n", device_count);

  if (device_count == 0) {
    fprintf(stderr, "Error: No CUDA devices found\n");
    return 1;
  }

  // Get device handle.
  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));

  // Get device name.
  char device_name[256];
  CUDA_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));
  printf("   Using device 0: %s\n", device_name);

  // Create context.
  printf("\n2. Creating context...\n");
  CUcontext context;
  CUDA_CHECK(cuCtxCreate(&context, 0, device));

  // Load module from file.
  printf("\n3. Loading module from %s...\n", module_path);
  CUmodule module;
  CUDA_CHECK(cuModuleLoad(&module, module_path));

  // Get kernel function.
  printf("\n4. Getting 'vector_add' function...\n");
  CUfunction vector_add_func;
  CUDA_CHECK(cuModuleGetFunction(&vector_add_func, module, "vector_add"));

  // Allocate host memory.
  printf("\n5. Allocating host memory...\n");
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

  // Allocate device memory.
  printf("\n6. Allocating device memory...\n");
  CUdeviceptr d_A, d_B, d_C;
  CUDA_CHECK(cuMemAlloc(&d_A, size));
  CUDA_CHECK(cuMemAlloc(&d_B, size));
  CUDA_CHECK(cuMemAlloc(&d_C, size));

  // Copy input data to device.
  printf("\n7. Copying input data to device...\n");
  CUDA_CHECK(cuMemcpyHtoD(d_A, h_A, size));
  CUDA_CHECK(cuMemcpyHtoD(d_B, h_B, size));

  // Prepare kernel launch parameters.
  printf("\n8. Launching kernel...\n");

  // The vector_add kernel takes: (const float* a, const float* b, float* c,
  // uint32_t n).
  void* kernel_params[] = {(void*)d_A, (void*)d_B, (void*)d_C,
                           (void*)(uintptr_t)n};

  // Calculate grid and block dimensions.
  const int threads_per_block = 256;
  const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
  printf("   Grid: %d blocks\n", blocks_per_grid);
  printf("   Block: %d threads\n", threads_per_block);

  // Launch kernel.
  CUDA_CHECK(cuLaunchKernel(
      vector_add_func, blocks_per_grid, 1, 1,  // grid dimensions (x, y, z)
      threads_per_block, 1, 1,                 // block dimensions (x, y, z)
      0,                                       // shared memory size
      NULL,                                    // stream (NULL = default)
      kernel_params,                           // kernel parameters
      NULL                                     // extra (unused)
      ));

  // Synchronize.
  printf("\n9. Synchronizing...\n");
  CUDA_CHECK(cuCtxSynchronize());

  // Copy result back to host.
  printf("\n10. Copying result back to host...\n");
  CUDA_CHECK(cuMemcpyDtoH(h_C, d_C, size));

  // Verify results.
  printf("\n11. Verifying results...\n");
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
  printf("\n12. Cleaning up...\n");

  // Free device memory.
  CUDA_CHECK(cuMemFree(d_A));
  CUDA_CHECK(cuMemFree(d_B));
  CUDA_CHECK(cuMemFree(d_C));

  // Free host memory.
  free(h_A);
  free(h_B);
  free(h_C);

  // Unload module.
  CUDA_CHECK(cuModuleUnload(module));

  // Destroy context.
  CUDA_CHECK(cuCtxDestroy(context));

  printf("\nDone!\n");

  // Extension: full global deinitialize.
  cuHALDeinit();

  return (errors == 0) ? 0 : 1;
}
