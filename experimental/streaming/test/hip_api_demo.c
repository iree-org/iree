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
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <module_file>\n", argv[0]);
    fprintf(stderr, "Example: %s kernels/compiled/vector_add.hsaco\n", argv[0]);
    return 1;
  }

  const char* module_path = argv[1];
  const int n = 1024;  // number of elements
  const size_t size = n * sizeof(float);

  printf("HIP API Vector Addition Demo\n");
  printf("========================\n");
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
  hipDeviceptr_t d_A, d_B, d_C;
  HIP_CHECK(hipMalloc(&d_A, size));
  HIP_CHECK(hipMalloc(&d_B, size));
  HIP_CHECK(hipMalloc(&d_C, size));

  // Copy input data to device.
  printf("\n7. Copying input data to device...\n");
  HIP_CHECK(hipMemcpyHtoD(d_A, h_A, size));
  HIP_CHECK(hipMemcpyHtoD(d_B, h_B, size));

  // Prepare kernel launch parameters.
  printf("\n8. Launching kernel...\n");

  // The vector_add kernel takes: (const float* a, const float* b, float* c,
  // uint32_t n).
  void* kernel_params[] = {d_A, d_B, d_C, (void*)(uintptr_t)n};

  // Launch configuration.
  int threads_per_block = 256;
  int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
  printf("   Grid: %d blocks, Block: %d threads\n", blocks_per_grid,
         threads_per_block);

  // Launch kernel.
  HIP_CHECK(hipModuleLaunchKernel(vector_add_func, blocks_per_grid, 1,
                                  1,                        // grid dimensions
                                  threads_per_block, 1, 1,  // block dimensions
                                  0,                        // shared memory
                                  0,              // stream (0 = default)
                                  kernel_params,  // kernel parameters
                                  NULL            // extra options
                                  ));

  // Wait for kernel to complete.
  printf("\n9. Synchronizing device...\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Copy result back to host.
  printf("\n10. Copying result back to host...\n");
  HIP_CHECK(hipMemcpyDtoH(h_C, d_C, size));

  // Verify result.
  printf("\n11. Verifying result...\n");
  int errors = 0;
  for (int i = 0; i < n; i++) {
    float expected = h_A[i] + h_B[i];
    if (fabs(h_C[i] - expected) > 1e-5) {
      if (errors < 5) {  // Only print first 5 errors
        printf("   Error at index %d: expected %.2f, got %.2f\n", i, expected,
               h_C[i]);
      }
      errors++;
    }
  }

  if (errors == 0) {
    printf("   ✓ SUCCESS: All %d elements computed correctly!\n", n);
    // Print last few results as sanity check
    printf("   Last results:\n");
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
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  // Free host memory.
  free(h_A);
  free(h_B);
  free(h_C);

  // Unload module.
  HIP_CHECK(hipModuleUnload(module));

  // Reset device (destroys primary context).
  // This is optional but good practice.
  HIP_CHECK(hipDeviceReset());

  printf("\nDone!\n");

  // Extension: full global deinitialize.
  hipHALDeinit();

  return (errors == 0) ? 0 : 1;
}
