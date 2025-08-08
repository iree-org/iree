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
  // DO NOT SUBMIT remove the double indirection? not possible?
  const uint64_t* params = (const uint64_t*)dispatch_state->constants;
  const float* a = *(const float**)params[0];
  const float* b = *(const float**)params[1];
  float* c = *(float**)params[2];
  uint32_t n = *(const uint32_t*)params[3];

  // Get workgroup info for parallel execution.
  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;
  uint32_t workgroup_size = dispatch_state->workgroup_size_x;

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
  uint32_t n = dispatch_state->constants[2];

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

  // Get size from push constants.
  uint32_t n =
      dispatch_state->constant_count > 0 ? dispatch_state->constants[0] : 0;
  if (n == 0) {
    n = dispatch_state->binding_lengths[0] / sizeof(float);
  }

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
    vector_add_scaled_impl,
    vector_multiply_impl,
};

// Attributes for each dispatch function.
static const iree_hal_executable_dispatch_attrs_v0_t entry_attrs[] = {
    {
        // vector_add
        .local_memory_pages = 0,
        .constant_count = 1,  // n (size)
        .binding_count = 3,   // a, b, c
    },
    {
        // vector_add_scaled
        .local_memory_pages = 0,
        .constant_count = 3,  // alpha, beta, n
        .binding_count = 3,   // a, b, c
    },
    {
        // vector_multiply
        .local_memory_pages = 0,
        .constant_count = 1,  // n (size)
        .binding_count = 3,   // a, b, c
    },
};

// Names for each entry point (must match kernel names in GPU versions).
static const char* entry_point_names[] = {
    "vector_add",
    "vector_add_scaled",
    "vector_multiply",
};

// Optional tags for debugging.
static const char* entry_point_tags[] = {
    "vector_add",
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
            .names = entry_point_names,
            .tags = entry_point_tags,
        },
    .constants =
        {
            .count = 0,
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
