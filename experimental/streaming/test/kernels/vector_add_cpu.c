// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CPU implementation of vector addition using IREE executable library interface

#include "iree/hal/local/executable_library.h"

//===----------------------------------------------------------------------===//
// Kernel implementations
//===----------------------------------------------------------------------===//

// Simple vector addition: C = A + B
static int vector_add_impl(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Get pointers to buffers.
  const float* a = (const float*)dispatch_state->binding_ptrs[0];
  const float* b = (const float*)dispatch_state->binding_ptrs[1];
  float* c = (float*)dispatch_state->binding_ptrs[2];
  uint32_t n = (uint32_t)dispatch_state->constants[0];

  // Get workgroup info for parallel execution.
  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;
  // TODO(benvanik): properly wire up workgroup size.
  uint32_t workgroup_size = dispatch_state->workgroup_size_x;
  (void)workgroup_size;

  // Calculate the range this workgroup should process.
  uint32_t total_items = n;
  uint32_t items_per_workgroup =
      (total_items + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > total_items) end_idx = total_items;

  // Perform the vector addition for this workgroup's range.
  for (uint32_t i = start_idx; i < end_idx; i++) {
    c[i] = a[i] + b[i];
  }

  return 0;
}

// Vector addition with raw pointers packed in constants: C = A + B
// This version packs pointers directly as constants instead of using bindings.
static int vector_add_raw_impl(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Get pointers from constants instead of bindings.
  // Layout: [a_ptr (64-bit, 2 constants), b_ptr (2 constants),
  //          c_ptr (2 constants), n (32-bit, 1 constant)]
  uint64_t a_ptr = ((uint64_t)dispatch_state->constants[1] << 32) |
                   (uint64_t)dispatch_state->constants[0];
  uint64_t b_ptr = ((uint64_t)dispatch_state->constants[3] << 32) |
                   (uint64_t)dispatch_state->constants[2];
  uint64_t c_ptr = ((uint64_t)dispatch_state->constants[5] << 32) |
                   (uint64_t)dispatch_state->constants[4];
  uint32_t n = (uint32_t)dispatch_state->constants[6];

  const float* a = (const float*)a_ptr;
  const float* b = (const float*)b_ptr;
  float* c = (float*)c_ptr;

  // Get workgroup info for parallel execution.
  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;

  // Calculate the range this workgroup should process.
  uint32_t total_items = n;
  uint32_t items_per_workgroup =
      (total_items + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > total_items) end_idx = total_items;

  // Perform the vector addition for this workgroup's range.
  for (uint32_t i = start_idx; i < end_idx; i++) {
    c[i] = a[i] + b[i];
  }

  return 0;
}

// Vector addition with scalar multiplication: C = alpha * A + beta * B
static int vector_add_scaled_impl(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Get pointers to buffers.
  const float* a = (const float*)dispatch_state->binding_ptrs[0];
  const float* b = (const float*)dispatch_state->binding_ptrs[1];
  float* c = (float*)dispatch_state->binding_ptrs[2];

  // Get parameters from push constants.
  // Assuming layout: [alpha, beta, n]
  float alpha = *(const float*)&dispatch_state->constants[0];
  float beta = *(const float*)&dispatch_state->constants[1];
  uint32_t n = (uint32_t)dispatch_state->constants[2];

  // Get workgroup info.
  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;

  // Calculate range.
  uint32_t items_per_workgroup = (n + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > n) end_idx = n;

  // Perform scaled vector addition.
  for (uint32_t i = start_idx; i < end_idx; i++) {
    c[i] = alpha * a[i] + beta * b[i];
  }

  return 0;
}

// Element-wise vector multiplication: C = A * B
static int vector_multiply_impl(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Get pointers to buffers.
  const float* a = (const float*)dispatch_state->binding_ptrs[0];
  const float* b = (const float*)dispatch_state->binding_ptrs[1];
  float* c = (float*)dispatch_state->binding_ptrs[2];
  uint32_t n = (uint32_t)dispatch_state->constants[0];

  // Get workgroup info.
  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;

  // Calculate range.
  uint32_t items_per_workgroup = (n + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > n) end_idx = n;

  // Perform element-wise multiplication.
  for (uint32_t i = start_idx; i < end_idx; i++) {
    c[i] = a[i] * b[i];
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Library metadata
//===----------------------------------------------------------------------===//

// Version/metadata header.
static const iree_hal_executable_library_header_t header = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "vector_add_library",
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

// Entry point function table.
static const iree_hal_executable_dispatch_v0_t entry_points[] = {
    vector_add_impl,
    vector_add_raw_impl,
    vector_add_scaled_impl,
    vector_multiply_impl,
};

// Parameter metadata for vector_add.
static const iree_hal_executable_dispatch_parameter_v0_t vector_add_params[] = {
    {
        // Input buffer a.
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 0,  // "a"
        .offset = 0,
    },
    {
        // Input buffer b.
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 1,  // "b"
        .offset = 8,
    },
    {
        // Output buffer c.
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 2,  // "c"
        .offset = 16,
    },
    {
        // Size n.
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
        .size = sizeof(uint32_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 3,  // "n"
        .offset = 0,
    },
};

// Parameter metadata for vector_add_raw.
// Uses BUFFER_PTR type to pack pointers directly into constants.
static const iree_hal_executable_dispatch_parameter_v0_t
    vector_add_raw_params[] = {
        {
            // Input buffer a pointer (packed as constants).
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR,
            .size = sizeof(uint64_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 0,    // "a"
            .offset = 0,  // Constants 0-1
        },
        {
            // Input buffer b pointer (packed as constants).
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR,
            .size = sizeof(uint64_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 1,    // "b"
            .offset = 8,  // Constants 2-3
        },
        {
            // Output buffer c pointer (packed as constants).
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR,
            .size = sizeof(uint64_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 2,     // "c"
            .offset = 16,  // Constants 4-5
        },
        {
            // Size n.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 3,     // "n"
            .offset = 24,  // Constant 6
        },
};

// Parameter metadata for vector_add_scaled.
static const iree_hal_executable_dispatch_parameter_v0_t
    vector_add_scaled_params[] = {
        {
            // Input buffer a.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 0,  // "a"
            .offset = 0,
        },
        {
            // Input buffer b.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 1,  // "b"
            .offset = 8,
        },
        {
            // Output buffer c.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 2,  // "c"
            .offset = 16,
        },
        {
            // Scale factor alpha.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(float),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 4,  // "alpha"
            .offset = 24,
        },
        {
            // Scale factor beta.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(float),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 5,  // "beta"
            .offset = 28,
        },
        {
            // Size n.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 3,  // "n"
            .offset = 32,
        },
};

// Parameter metadata for vector_multiply.
static const iree_hal_executable_dispatch_parameter_v0_t
    vector_multiply_params[] = {
        {
            // Input buffer a.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 0,  // "a"
            .offset = 0,
        },
        {
            // Input buffer b.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 1,  // "b"
            .offset = 8,
        },
        {
            // Output buffer c.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 2,  // "c"
            .offset = 16,
        },
        {
            // Size n.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(uint32_t),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 3,  // "n"
            .offset = 24,
        },
};

// Parameter table - one array per dispatch entry point.
static const iree_hal_executable_dispatch_parameter_v0_t* entry_params[] = {
    vector_add_params,
    vector_add_raw_params,
    vector_add_scaled_params,
    vector_multiply_params,
};

// String table for parameter names.
static const char* parameter_names[] = {
    "a", "b", "c", "n", "alpha", "beta",
};

// Attributes for each dispatch function.
static const iree_hal_executable_dispatch_attrs_v0_t entry_attrs[] = {
    {
        // vector_add
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 1,    // n (size)
        .binding_count = 3,     // a, b, c
        .workgroup_size_x = 0,  // runtime specified
        .workgroup_size_y = 0,  // runtime specified
        .workgroup_size_z = 0,  // runtime specified
        .parameter_count =
            sizeof(vector_add_params) / sizeof(vector_add_params[0]),
    },
    {
        // vector_add_raw
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 7,    // a_ptr (2), b_ptr (2), c_ptr (2), n (1)
        .binding_count = 0,     // No bindings, all pointers in constants
        .workgroup_size_x = 0,  // runtime specified
        .workgroup_size_y = 0,  // runtime specified
        .workgroup_size_z = 0,  // runtime specified
        .parameter_count =
            sizeof(vector_add_raw_params) / sizeof(vector_add_raw_params[0]),
    },
    {
        // vector_add_scaled
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 3,    // alpha, beta, n
        .binding_count = 3,     // a, b, c
        .workgroup_size_x = 0,  // runtime specified
        .workgroup_size_y = 0,  // runtime specified
        .workgroup_size_z = 0,  // runtime specified
        .parameter_count = sizeof(vector_add_scaled_params) /
                           sizeof(vector_add_scaled_params[0]),
    },
    {
        // vector_multiply
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 1,    // n (size)
        .binding_count = 3,     // a, b, c
        .workgroup_size_x = 0,  // runtime specified
        .workgroup_size_y = 0,  // runtime specified
        .workgroup_size_z = 0,  // runtime specified
        .parameter_count =
            sizeof(vector_multiply_params) / sizeof(vector_multiply_params[0]),
    },
};

// Names for each entry point (must match kernel names in GPU versions).
static const char* entry_point_names[] = {
    "vector_add",
    "vector_add_raw",
    "vector_add_scaled",
    "vector_multiply",
};

// Optional tags for debugging.
static const char* entry_point_tags[] = {
    "vector_add",
    "vector_add_raw",
    "vector_add_scaled",
    "vector_multiply",
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
// For static linking, this can be called directly.
// For dynamic libraries, this should be exported as:
// IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME ("iree_hal_executable_library_query")
const iree_hal_executable_library_header_t** vector_add_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  return max_version <= IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST
             ? (const iree_hal_executable_library_header_t**)&library
             : NULL;
}

#ifdef __cplusplus
extern "C" {
#endif

// Export with the standard name for dynamic library loading.
const iree_hal_executable_library_header_t** iree_hal_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  return vector_add_library_query(max_version, environment);
}

#ifdef __cplusplus
}
#endif
