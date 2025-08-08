// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CPU implementation of matrix multiplication using IREE executable library
// interface

#include <string.h>

#include "iree/hal/local/executable_library.h"

//===----------------------------------------------------------------------===//
// Kernel implementations
//===----------------------------------------------------------------------===//

// Simple matrix multiplication: C = A * B
// A is M x K, B is K x N, C is M x N
static int matrix_multiply_impl(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Get pointers to buffers
  const float* a = (const float*)dispatch_state->binding_ptrs[0];
  const float* b = (const float*)dispatch_state->binding_ptrs[1];
  float* c = (float*)dispatch_state->binding_ptrs[2];

  // Get dimensions from push constants
  // Layout: [m, n, k]
  uint32_t m = dispatch_state->constants[0];
  uint32_t n = dispatch_state->constants[1];
  uint32_t k = dispatch_state->constants[2];

  // Get workgroup info for 2D dispatch
  uint32_t workgroup_id_y = workgroup_state->workgroup_id_y;
  uint32_t workgroup_id_x = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count_y = dispatch_state->workgroup_count_y;
  uint32_t workgroup_count_x = dispatch_state->workgroup_count_x;

  // Calculate the tile this workgroup processes
  uint32_t rows_per_workgroup = (m + workgroup_count_y - 1) / workgroup_count_y;
  uint32_t cols_per_workgroup = (n + workgroup_count_x - 1) / workgroup_count_x;

  uint32_t row_start = workgroup_id_y * rows_per_workgroup;
  uint32_t row_end = row_start + rows_per_workgroup;
  if (row_end > m) row_end = m;

  uint32_t col_start = workgroup_id_x * cols_per_workgroup;
  uint32_t col_end = col_start + cols_per_workgroup;
  if (col_end > n) col_end = n;

  // Perform matrix multiplication for this tile
  for (uint32_t i = row_start; i < row_end; i++) {
    for (uint32_t j = col_start; j < col_end; j++) {
      float sum = 0.0f;
      for (uint32_t l = 0; l < k; l++) {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = sum;
    }
  }

  return 0;
}

// Tiled matrix multiplication with local memory optimization
// Uses blocking to improve cache performance
static int matrix_multiply_tiled_impl(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Get pointers to buffers
  const float* a = (const float*)dispatch_state->binding_ptrs[0];
  const float* b = (const float*)dispatch_state->binding_ptrs[1];
  float* c = (float*)dispatch_state->binding_ptrs[2];

  // Get dimensions from push constants
  uint32_t m = dispatch_state->constants[0];
  uint32_t n = dispatch_state->constants[1];
  uint32_t k = dispatch_state->constants[2];

// Define tile size (must match GPU version)
#define TILE_SIZE 16

  // Get workgroup info
  uint32_t workgroup_id_y = workgroup_state->workgroup_id_y;
  uint32_t workgroup_id_x = workgroup_state->workgroup_id_x;

  // Calculate the output tile position
  uint32_t tile_row = workgroup_id_y * TILE_SIZE;
  uint32_t tile_col = workgroup_id_x * TILE_SIZE;

  // Use local memory if available for tile caching
  float* local_mem = (float*)workgroup_state->local_memory;
  float tile_a[TILE_SIZE][TILE_SIZE];
  float tile_b[TILE_SIZE][TILE_SIZE];
  float tile_c[TILE_SIZE][TILE_SIZE] = {{0}};

  // If we have local memory, use it for better cache behavior
  float (*work_tile_a)[TILE_SIZE] =
      local_mem ? (float (*)[TILE_SIZE])local_mem : tile_a;
  float (*work_tile_b)[TILE_SIZE] =
      local_mem ? (float (*)[TILE_SIZE])(local_mem + TILE_SIZE * TILE_SIZE)
                : tile_b;
  float (*work_tile_c)[TILE_SIZE] =
      local_mem ? (float (*)[TILE_SIZE])(local_mem + 2 * TILE_SIZE * TILE_SIZE)
                : tile_c;

  // Initialize output tile
  for (int i = 0; i < TILE_SIZE; i++) {
    for (int j = 0; j < TILE_SIZE; j++) {
      work_tile_c[i][j] = 0.0f;
    }
  }

  // Loop over tiles in K dimension
  for (uint32_t tile_k = 0; tile_k < k; tile_k += TILE_SIZE) {
    // Load tile from A
    for (int i = 0; i < TILE_SIZE; i++) {
      for (int j = 0; j < TILE_SIZE; j++) {
        uint32_t row = tile_row + i;
        uint32_t col = tile_k + j;
        if (row < m && col < k) {
          work_tile_a[i][j] = a[row * k + col];
        } else {
          work_tile_a[i][j] = 0.0f;
        }
      }
    }

    // Load tile from B
    for (int i = 0; i < TILE_SIZE; i++) {
      for (int j = 0; j < TILE_SIZE; j++) {
        uint32_t row = tile_k + i;
        uint32_t col = tile_col + j;
        if (row < k && col < n) {
          work_tile_b[i][j] = b[row * n + col];
        } else {
          work_tile_b[i][j] = 0.0f;
        }
      }
    }

    // Compute partial product
    for (int i = 0; i < TILE_SIZE; i++) {
      for (int j = 0; j < TILE_SIZE; j++) {
        for (int l = 0; l < TILE_SIZE; l++) {
          work_tile_c[i][j] += work_tile_a[i][l] * work_tile_b[l][j];
        }
      }
    }
  }

  // Write output tile to C
  for (int i = 0; i < TILE_SIZE; i++) {
    for (int j = 0; j < TILE_SIZE; j++) {
      uint32_t row = tile_row + i;
      uint32_t col = tile_col + j;
      if (row < m && col < n) {
        c[row * n + col] = work_tile_c[i][j];
      }
    }
  }

#undef TILE_SIZE
  return 0;
}

// Matrix-vector multiplication: y = A * x
// A is M x N, x is N x 1, y is M x 1
static int matrix_vector_multiply_impl(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Get pointers to buffers
  const float* a = (const float*)dispatch_state->binding_ptrs[0];
  const float* x = (const float*)dispatch_state->binding_ptrs[1];
  float* y = (float*)dispatch_state->binding_ptrs[2];

  // Get dimensions from push constants
  // Layout: [m, n]
  uint32_t m = dispatch_state->constants[0];
  uint32_t n = dispatch_state->constants[1];

  // Get workgroup info (1D dispatch over rows)
  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;

  // Calculate rows this workgroup processes
  uint32_t rows_per_workgroup = (m + workgroup_count - 1) / workgroup_count;
  uint32_t row_start = workgroup_id * rows_per_workgroup;
  uint32_t row_end = row_start + rows_per_workgroup;
  if (row_end > m) row_end = m;

  // Perform matrix-vector multiplication
  for (uint32_t i = row_start; i < row_end; i++) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < n; j++) {
      sum += a[i * n + j] * x[j];
    }
    y[i] = sum;
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Library metadata
//===----------------------------------------------------------------------===//

// Version/metadata header
static const iree_hal_executable_library_header_t header = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "matrix_multiply_library",
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

// Entry point function table
static const iree_hal_executable_dispatch_v0_t entry_points[] = {
    matrix_multiply_impl,
    matrix_multiply_tiled_impl,
    matrix_vector_multiply_impl,
};

// Parameter metadata for matrix_multiply.
static const iree_hal_executable_dispatch_parameter_v0_t
    matrix_multiply_params[] = {
        {
            // Input matrix A.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 0,  // "a"
            .offset = 0,
        },
        {
            // Input matrix B.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 1,  // "b"
            .offset = 1,
        },
        {
            // Output matrix C.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 2,  // "c"
            .offset = 2,
        },
        {
            // Dimension m.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 3,  // "m"
            .offset = 0,
        },
        {
            // Dimension n.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 4,  // "n"
            .offset = 4,
        },
        {
            // Dimension k.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 5,  // "k"
            .offset = 8,
        },
};

// Parameter metadata for matrix_multiply_tiled.
static const iree_hal_executable_dispatch_parameter_v0_t
    matrix_multiply_tiled_params[] = {
        {
            // Input matrix A.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 0,  // "a"
            .offset = 0,
        },
        {
            // Input matrix B.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 1,  // "b"
            .offset = 1,
        },
        {
            // Output matrix C.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 2,  // "c"
            .offset = 2,
        },
        {
            // Dimension m.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 3,  // "m"
            .offset = 0,
        },
        {
            // Dimension n.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 4,  // "n"
            .offset = 4,
        },
        {
            // Dimension k.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 5,  // "k"
            .offset = 8,
        },
};

// Parameter metadata for matrix_vector_multiply.
static const iree_hal_executable_dispatch_parameter_v0_t
    matrix_vector_multiply_params[] = {
        {
            // Input matrix A.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 0,  // "a"
            .offset = 0,
        },
        {
            // Input vector x.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 6,  // "x"
            .offset = 1,
        },
        {
            // Output vector y.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 7,  // "y"
            .offset = 2,
        },
        {
            // Dimension m.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 3,  // "m"
            .offset = 0,
        },
        {
            // Dimension n.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 4,  // "n"
            .offset = 4,
        },
};

// Parameter table - one array per dispatch entry point.
static const iree_hal_executable_dispatch_parameter_v0_t* entry_params[] = {
    matrix_multiply_params,
    matrix_multiply_tiled_params,
    matrix_vector_multiply_params,
};

// String table for parameter names.
static const char* parameter_names[] = {
    "a", "b", "c", "m", "n", "k", "x", "y",
};

// Attributes for each dispatch function.
static const iree_hal_executable_dispatch_attrs_v0_t entry_attrs[] = {
    {
        // matrix_multiply
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 3,    // m, n, k
        .binding_count = 3,     // a, b, c
        .workgroup_size_x = 0,  // runtime specified
        .workgroup_size_y = 0,  // runtime specified
        .workgroup_size_z = 0,  // runtime specified
        .parameter_count =
            sizeof(matrix_multiply_params) / sizeof(matrix_multiply_params[0]),
    },
    {
        // matrix_multiply_tiled
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 3,  // 3 tiles of 16x16 floats = 3*256*4 = 3KB
        .constant_count = 3,      // m, n, k
        .binding_count = 3,       // a, b, c
        .workgroup_size_x = 0,    // runtime specified
        .workgroup_size_y = 0,    // runtime specified
        .workgroup_size_z = 0,    // runtime specified
        .parameter_count = sizeof(matrix_multiply_tiled_params) /
                           sizeof(matrix_multiply_tiled_params[0]),
    },
    {
        // matrix_vector_multiply
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 2,    // m, n
        .binding_count = 3,     // a, x, y
        .workgroup_size_x = 0,  // runtime specified
        .workgroup_size_y = 0,  // runtime specified
        .workgroup_size_z = 0,  // runtime specified
        .parameter_count = sizeof(matrix_vector_multiply_params) /
                           sizeof(matrix_vector_multiply_params[0]),
    },
};

// Names for each entry point (must match kernel names in GPU versions)
static const char* entry_point_names[] = {
    "matrix_multiply",
    "matrix_multiply_tiled",
    "matrix_vector_multiply",
};

// Optional tags for debugging
static const char* entry_point_tags[] = {
    "matmul_simple",
    "matmul_tiled",
    "matvec",
};

// Library descriptor.
static const iree_hal_executable_library_v0_t library = {
    .header = &header,
    .imports =
        {
            .count = 0,
            .symbols = NULL,
        },
    .exports =
        {
            .count = sizeof(entry_points) / sizeof(entry_points[0]),
            .ptrs = entry_points,
            .attrs = entry_attrs,
            .params = entry_params,
            .occupancy = NULL,  // no occupancy info provided
            .names = entry_point_names,
            .tags = entry_point_tags,
            .parameter_names = parameter_names,
            .source_locations = NULL,
            .stage_locations = NULL,
        },
    .constants =
        {
            .count = 0,
        },
    .sources =
        {
            .count = 0,
            .files = NULL,
        },
};

//===----------------------------------------------------------------------===//
// Library entry point
//===----------------------------------------------------------------------===//

// This is the main entry point for querying the library.
const iree_hal_executable_library_header_t** matrix_multiply_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  return max_version <= IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST
             ? (const iree_hal_executable_library_header_t**)&library
             : NULL;
}

// Export with the standard name for dynamic library loading
#ifdef __cplusplus
extern "C" {
#endif

const iree_hal_executable_library_header_t** iree_hal_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  return matrix_multiply_library_query(max_version, environment);
}

#ifdef __cplusplus
}
#endif
