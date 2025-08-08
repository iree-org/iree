// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Matrix multiplication kernel for testing CUDA HAL
// Compilable for AMDGPU, SPIR-V, and CPU targets

#ifdef __AMDGCN__
// AMDGPU specific
#define KERNEL_ATTR __attribute__((amdgpu_kernel))
#define GET_GLOBAL_ID_X()             \
  (__builtin_amdgcn_workitem_id_x() + \
   __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x())
#define GET_GLOBAL_ID_Y()             \
  (__builtin_amdgcn_workitem_id_y() + \
   __builtin_amdgcn_workgroup_id_y() * __builtin_amdgcn_workgroup_size_y())
#define GET_LOCAL_ID_X() __builtin_amdgcn_workitem_id_x()
#define GET_LOCAL_ID_Y() __builtin_amdgcn_workitem_id_y()
#define BARRIER() __builtin_amdgcn_s_barrier()
#elif defined(__SPIRV__)
// SPIR-V specific
#define KERNEL_ATTR __attribute__((kernel))
extern unsigned get_global_id(unsigned dim);
extern unsigned get_local_id(unsigned dim);
extern void barrier(unsigned flags);
#define GET_GLOBAL_ID_X() get_global_id(0)
#define GET_GLOBAL_ID_Y() get_global_id(1)
#define GET_LOCAL_ID_X() get_local_id(0)
#define GET_LOCAL_ID_Y() get_local_id(1)
#define BARRIER() barrier(1)  // CLK_LOCAL_MEM_FENCE
#else
// CPU fallback
#define KERNEL_ATTR
#define GET_GLOBAL_ID_X() 0
#define GET_GLOBAL_ID_Y() 0
#define GET_LOCAL_ID_X() 0
#define GET_LOCAL_ID_Y() 0
#define BARRIER()
#endif

// Simple matrix multiplication: C = A * B
// A is M x K, B is K x N, C is M x N
KERNEL_ATTR
void matrix_multiply(const float* a, const float* b, float* c, unsigned int m,
                     unsigned int n, unsigned int k) {
#if defined(__AMDGCN__) || defined(__SPIRV__)
  unsigned int row = GET_GLOBAL_ID_Y();
  unsigned int col = GET_GLOBAL_ID_X();

  if (row < m && col < n) {
    float sum = 0.0f;
    for (unsigned int i = 0; i < k; i++) {
      sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
  }
#else
  // CPU version - triple nested loop
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (unsigned int l = 0; l < k; l++) {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = sum;
    }
  }
#endif
}

// Tiled matrix multiplication with shared memory (for GPUs)
// Uses 16x16 tiles
#define TILE_SIZE 16

#if defined(__AMDGCN__)
// AMD GPU shared memory
// #define SHARED __attribute__((address_space(3)))
#define SHARED
#elif defined(__SPIRV__)
// SPIR-V/OpenCL local memory
#define SHARED __local
#else
// CPU doesn't have shared memory
#define SHARED
#endif

KERNEL_ATTR
void matrix_multiply_tiled(const float* a, const float* b, float* c,
                           unsigned int m, unsigned int n, unsigned int k) {
#if defined(__AMDGCN__) || defined(__SPIRV__)
  // Shared memory tiles
  SHARED float tile_a[TILE_SIZE][TILE_SIZE];
  SHARED float tile_b[TILE_SIZE][TILE_SIZE];

  unsigned int row = GET_GLOBAL_ID_Y();
  unsigned int col = GET_GLOBAL_ID_X();
  unsigned int local_row = GET_LOCAL_ID_Y();
  unsigned int local_col = GET_LOCAL_ID_X();

  float sum = 0.0f;

  // Loop over tiles
  for (unsigned int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    // Load tile from A
    if (row < m && (tile * TILE_SIZE + local_col) < k) {
      tile_a[local_row][local_col] = a[row * k + tile * TILE_SIZE + local_col];
    } else {
      tile_a[local_row][local_col] = 0.0f;
    }

    // Load tile from B
    if ((tile * TILE_SIZE + local_row) < k && col < n) {
      tile_b[local_row][local_col] =
          b[(tile * TILE_SIZE + local_row) * n + col];
    } else {
      tile_b[local_row][local_col] = 0.0f;
    }

    // Synchronize to ensure tiles are loaded
    BARRIER();

    // Compute partial dot product
    for (unsigned int i = 0; i < TILE_SIZE; i++) {
      sum += tile_a[local_row][i] * tile_b[i][local_col];
    }

    // Synchronize before loading next tiles
    BARRIER();
  }

  // Write result
  if (row < m && col < n) {
    c[row * n + col] = sum;
  }
#else
  // CPU version - use simple multiplication
  matrix_multiply(a, b, c, m, n, k);
#endif
}

// Matrix-vector multiplication: y = A * x
// A is M x N, x is N x 1, y is M x 1
KERNEL_ATTR
void matrix_vector_multiply(const float* a, const float* x, float* y,
                            unsigned int m, unsigned int n) {
#if defined(__AMDGCN__) || defined(__SPIRV__)
  unsigned int row = GET_GLOBAL_ID_X();

  if (row < m) {
    float sum = 0.0f;
    for (unsigned int i = 0; i < n; i++) {
      sum += a[row * n + i] * x[i];
    }
    y[row] = sum;
  }
#else
  // CPU version
  for (unsigned int i = 0; i < m; i++) {
    float sum = 0.0f;
    for (unsigned int j = 0; j < n; j++) {
      sum += a[i * n + j] * x[j];
    }
    y[i] = sum;
  }
#endif
}
