// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CPU test kernels for benchmarking graph execution with real parameter metadata

#include "iree/hal/local/executable_library.h"

//===----------------------------------------------------------------------===//
// Kernel implementations
//===----------------------------------------------------------------------===//

// No-op kernel with no parameters - baseline overhead test.
static int nop_kernel(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Minimal work to prevent complete optimization.
  return 0;
}

// Simple addition with raw pointers in constants.
// Parameters: a_ptr (2 constants), b_ptr (2 constants), c_ptr (2 constants), n (1 constant)
static int simple_add_raw(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Extract 64-bit pointers from constants.
  uint64_t a_ptr = ((uint64_t)dispatch_state->constants[1] << 32) | 
                    (uint64_t)dispatch_state->constants[0];
  uint64_t b_ptr = ((uint64_t)dispatch_state->constants[3] << 32) | 
                    (uint64_t)dispatch_state->constants[2];
  uint64_t c_ptr = ((uint64_t)dispatch_state->constants[5] << 32) | 
                    (uint64_t)dispatch_state->constants[4];
  uint32_t n = dispatch_state->constants[6];
  
  const float* a = (const float*)a_ptr;
  const float* b = (const float*)b_ptr;
  float* c = (float*)c_ptr;

  // Simple parallel addition.
  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;
  uint32_t items_per_workgroup = (n + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > n) end_idx = n;

  for (uint32_t i = start_idx; i < end_idx; i++) {
    c[i] = a[i] + b[i];
  }
  return 0;
}

// Simple addition with HAL bindings.
// Parameters: a (binding 0), b (binding 1), c (binding 2), n (constant 0)
static int simple_add_hal(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  const float* a = (const float*)dispatch_state->binding_ptrs[0];
  const float* b = (const float*)dispatch_state->binding_ptrs[1];
  float* c = (float*)dispatch_state->binding_ptrs[2];
  uint32_t n = dispatch_state->constants[0];

  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;
  uint32_t items_per_workgroup = (n + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > n) end_idx = n;

  for (uint32_t i = start_idx; i < end_idx; i++) {
    c[i] = a[i] + b[i];
  }
  return 0;
}

// Many parameters using raw constants (20 total).
// Simulates kernels with many scalar parameters.
static int many_params_raw(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // Extract parameters from constants.
  uint64_t in_ptr = ((uint64_t)dispatch_state->constants[1] << 32) | 
                     (uint64_t)dispatch_state->constants[0];
  uint64_t out_ptr = ((uint64_t)dispatch_state->constants[3] << 32) | 
                      (uint64_t)dispatch_state->constants[2];
  
  // 16 scalar parameters.
  float scale = *(const float*)&dispatch_state->constants[4];
  float bias = *(const float*)&dispatch_state->constants[5];
  uint32_t n = dispatch_state->constants[6];
  uint32_t stride = dispatch_state->constants[7];
  float min_val = *(const float*)&dispatch_state->constants[8];
  float max_val = *(const float*)&dispatch_state->constants[9];
  uint32_t flags = dispatch_state->constants[10];
  uint32_t mode = dispatch_state->constants[11];
  float alpha = *(const float*)&dispatch_state->constants[12];
  float beta = *(const float*)&dispatch_state->constants[13];
  float gamma = *(const float*)&dispatch_state->constants[14];
  float delta = *(const float*)&dispatch_state->constants[15];
  uint32_t offset1 = dispatch_state->constants[16];
  uint32_t offset2 = dispatch_state->constants[17];
  uint32_t offset3 = dispatch_state->constants[18];
  uint32_t offset4 = dispatch_state->constants[19];

  const float* in = (const float*)in_ptr;
  float* out = (float*)out_ptr;

  // Use all parameters to prevent optimization.
  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;
  uint32_t items_per_workgroup = (n + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > n) end_idx = n;

  for (uint32_t i = start_idx; i < end_idx; i++) {
    float val = in[i * stride + offset1 % stride];
    val = val * scale + bias;
    val = val * alpha + beta * gamma - delta;
    val = (val < min_val) ? min_val : val;
    val = (val > max_val) ? max_val : val;
    if (flags & 1) val = -val;
    if (mode == 1) val = val * val;
    out[i + offset2 % n] = val;
    // Use remaining offsets to prevent dead code elimination.
    if (offset3 == offset4) out[0] += 0.0001f;
  }
  return 0;
}

// Many parameters using HAL mode (10 bindings + 10 constants).
static int many_params_hal(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // 10 buffer bindings.
  const float* buf0 = (const float*)dispatch_state->binding_ptrs[0];
  const float* buf1 = (const float*)dispatch_state->binding_ptrs[1];
  const float* buf2 = (const float*)dispatch_state->binding_ptrs[2];
  const float* buf3 = (const float*)dispatch_state->binding_ptrs[3];
  const float* buf4 = (const float*)dispatch_state->binding_ptrs[4];
  float* out0 = (float*)dispatch_state->binding_ptrs[5];
  float* out1 = (float*)dispatch_state->binding_ptrs[6];
  float* out2 = (float*)dispatch_state->binding_ptrs[7];
  float* out3 = (float*)dispatch_state->binding_ptrs[8];
  float* out4 = (float*)dispatch_state->binding_ptrs[9];
  
  // 10 scalar constants.
  uint32_t n = dispatch_state->constants[0];
  float weight0 = *(const float*)&dispatch_state->constants[1];
  float weight1 = *(const float*)&dispatch_state->constants[2];
  float weight2 = *(const float*)&dispatch_state->constants[3];
  float weight3 = *(const float*)&dispatch_state->constants[4];
  float weight4 = *(const float*)&dispatch_state->constants[5];
  uint32_t offset = dispatch_state->constants[6];
  uint32_t stride = dispatch_state->constants[7];
  uint32_t flags = dispatch_state->constants[8];
  uint32_t mode = dispatch_state->constants[9];

  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;
  uint32_t items_per_workgroup = (n + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > n) end_idx = n;

  // Weighted sum of inputs.
  for (uint32_t i = start_idx; i < end_idx; i++) {
    uint32_t idx = (i + offset) % n;
    float sum = buf0[idx] * weight0 + buf1[idx] * weight1 + 
                buf2[idx] * weight2 + buf3[idx] * weight3 + 
                buf4[idx] * weight4;
    if (flags & 1) sum = -sum;
    if (mode == 1) sum = sum * sum;
    out0[idx] = sum;
    out1[idx] = sum * 0.5f;
    out2[idx] = sum * 0.25f;
    out3[idx] = sum * 0.125f;
    out4[idx] = sum * 0.0625f;
  }
  return 0;
}

// Memory copy kernel for bandwidth testing.
static int memory_copy(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  const uint32_t* src = (const uint32_t*)dispatch_state->binding_ptrs[0];
  uint32_t* dst = (uint32_t*)dispatch_state->binding_ptrs[1];
  uint32_t n = dispatch_state->constants[0];  // Number of uint32_t elements.

  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;
  uint32_t items_per_workgroup = (n + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > n) end_idx = n;

  // Simple memory copy.
  for (uint32_t i = start_idx; i < end_idx; i++) {
    dst[i] = src[i];
  }
  return 0;
}

// Compute-intensive kernel for arithmetic throughput testing.
static int compute_intensive(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  const float* in = (const float*)dispatch_state->binding_ptrs[0];
  float* out = (float*)dispatch_state->binding_ptrs[1];
  uint32_t n = dispatch_state->constants[0];
  uint32_t iterations = dispatch_state->constants[1];  // Inner loop iterations.

  uint32_t workgroup_id = workgroup_state->workgroup_id_x;
  uint32_t workgroup_count = dispatch_state->workgroup_count_x;
  uint32_t items_per_workgroup = (n + workgroup_count - 1) / workgroup_count;
  uint32_t start_idx = workgroup_id * items_per_workgroup;
  uint32_t end_idx = start_idx + items_per_workgroup;
  if (end_idx > n) end_idx = n;

  // High arithmetic intensity.
  for (uint32_t i = start_idx; i < end_idx; i++) {
    float val = in[i];
    for (uint32_t j = 0; j < iterations; j++) {
      val = val * 1.1f - 0.1f;
      val = val * val * 0.999f;
      val = val + 0.001f;
    }
    out[i] = val;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Library metadata
//===----------------------------------------------------------------------===//

// Version/metadata header.
static const iree_hal_executable_library_header_t header = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "test_kernels_library",
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

// Entry point function table.
static const iree_hal_executable_dispatch_v0_t entry_points[] = {
    nop_kernel,
    simple_add_raw,
    simple_add_hal,
    many_params_raw,
    many_params_hal,
    memory_copy,
    compute_intensive,
};

// Parameter metadata for nop_kernel.
static const iree_hal_executable_dispatch_parameter_v0_t nop_kernel_params[] = {};

// Parameter metadata for simple_add_raw.
static const iree_hal_executable_dispatch_parameter_v0_t simple_add_raw_params[] = {
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 0,  // "a"
        .offset = 0,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 1,  // "b"
        .offset = 8,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 2,  // "c"
        .offset = 16,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
        .size = sizeof(uint32_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 3,  // "n"
        .offset = 24,
    },
};

// Parameter metadata for simple_add_hal.
static const iree_hal_executable_dispatch_parameter_v0_t simple_add_hal_params[] = {
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 0,  // "a"
        .offset = 0,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 1,  // "b"
        .offset = 8,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 2,  // "c"
        .offset = 16,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
        .size = sizeof(uint32_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 3,  // "n"
        .offset = 0,
    },
};

// Parameter metadata for many_params_raw - 20 total parameters.
static const iree_hal_executable_dispatch_parameter_v0_t many_params_raw_params[] = {
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR, .size = sizeof(uint64_t), .flags = 0, .name = 4, .offset = 0 },   // in_ptr
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR, .size = sizeof(uint64_t), .flags = 0, .name = 5, .offset = 8 },   // out_ptr
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 6, .offset = 16 },       // scale
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 7, .offset = 20 },       // bias
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 3, .offset = 24 },     // n
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 8, .offset = 28 },     // stride
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 9, .offset = 32 },       // min_val
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 10, .offset = 36 },      // max_val
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 11, .offset = 40 },    // flags
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 12, .offset = 44 },    // mode
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 13, .offset = 48 },      // alpha
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 14, .offset = 52 },      // beta
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 15, .offset = 56 },      // gamma
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 16, .offset = 60 },      // delta
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 17, .offset = 64 },    // offset1
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 18, .offset = 68 },    // offset2
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 19, .offset = 72 },    // offset3
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 20, .offset = 76 },    // offset4
};

// Parameter metadata for many_params_hal - 10 bindings + 10 constants.
static const iree_hal_executable_dispatch_parameter_v0_t many_params_hal_params[] = {
    // 10 bindings.
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 21, .offset = 0 },
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 22, .offset = 8 },
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 23, .offset = 16 },
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 24, .offset = 24 },
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 25, .offset = 32 },
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 26, .offset = 40 },
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 27, .offset = 48 },
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 28, .offset = 56 },
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 29, .offset = 64 },
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING, .size = sizeof(uint64_t), .flags = 0, .name = 30, .offset = 72 },
    // 10 constants.
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 3, .offset = 0 },     // n
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 31, .offset = 4 },      // weight0
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 32, .offset = 8 },      // weight1
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 33, .offset = 12 },     // weight2
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 34, .offset = 16 },     // weight3
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(float), .flags = 0, .name = 35, .offset = 20 },     // weight4
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 36, .offset = 24 },   // offset
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 8, .offset = 28 },    // stride
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 11, .offset = 32 },   // flags
    { .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT, .size = sizeof(uint32_t), .flags = 0, .name = 12, .offset = 36 },   // mode
};

// Parameter metadata for memory_copy.
static const iree_hal_executable_dispatch_parameter_v0_t memory_copy_params[] = {
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 37,  // "src"
        .offset = 0,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 38,  // "dst"
        .offset = 8,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
        .size = sizeof(uint32_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 3,  // "n"
        .offset = 0,
    },
};

// Parameter metadata for compute_intensive.
static const iree_hal_executable_dispatch_parameter_v0_t compute_intensive_params[] = {
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 39,  // "in"
        .offset = 0,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .size = sizeof(uint64_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 40,  // "out"
        .offset = 8,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
        .size = sizeof(uint32_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 3,  // "n"
        .offset = 0,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
        .size = sizeof(uint32_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 41,  // "iterations"
        .offset = 4,
    },
};

// Parameter table.
static const iree_hal_executable_dispatch_parameter_v0_t* entry_params[] = {
    nop_kernel_params,
    simple_add_raw_params,
    simple_add_hal_params,
    many_params_raw_params,
    many_params_hal_params,
    memory_copy_params,
    compute_intensive_params,
};

// String table for parameter names.
static const char* parameter_names[] = {
    "a", "b", "c", "n",                                                    // 0-3
    "in_ptr", "out_ptr", "scale", "bias",                                  // 4-7
    "stride", "min_val", "max_val", "flags",                               // 8-11
    "mode", "alpha", "beta", "gamma",                                      // 12-15
    "delta", "offset1", "offset2", "offset3",                              // 16-19
    "offset4", "buf0", "buf1", "buf2",                                     // 20-23
    "buf3", "buf4", "out0", "out1",                                        // 24-27
    "out2", "out3", "out4", "weight0",                                     // 28-31
    "weight1", "weight2", "weight3", "weight4",                            // 32-35
    "offset", "src", "dst", "in",                                          // 36-39
    "out", "iterations",                                                   // 40-41
};

// Attributes for each dispatch function.
static const iree_hal_executable_dispatch_attrs_v0_t entry_attrs[] = {
    {
        // nop_kernel
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 0,
        .binding_count = 0,
        .workgroup_size_x = 0,
        .workgroup_size_y = 0,
        .workgroup_size_z = 0,
        .parameter_count = 0,
    },
    {
        // simple_add_raw
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 7,  // 3 pointers (2 constants each) + n
        .binding_count = 0,
        .workgroup_size_x = 0,
        .workgroup_size_y = 0,
        .workgroup_size_z = 0,
        .parameter_count = sizeof(simple_add_raw_params) / sizeof(simple_add_raw_params[0]),
    },
    {
        // simple_add_hal
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 1,
        .binding_count = 3,
        .workgroup_size_x = 0,
        .workgroup_size_y = 0,
        .workgroup_size_z = 0,
        .parameter_count = sizeof(simple_add_hal_params) / sizeof(simple_add_hal_params[0]),
    },
    {
        // many_params_raw
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 20,  // All parameters in constants
        .binding_count = 0,
        .workgroup_size_x = 0,
        .workgroup_size_y = 0,
        .workgroup_size_z = 0,
        .parameter_count = sizeof(many_params_raw_params) / sizeof(many_params_raw_params[0]),
    },
    {
        // many_params_hal
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 10,
        .binding_count = 10,
        .workgroup_size_x = 0,
        .workgroup_size_y = 0,
        .workgroup_size_z = 0,
        .parameter_count = sizeof(many_params_hal_params) / sizeof(many_params_hal_params[0]),
    },
    {
        // memory_copy
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 1,
        .binding_count = 2,
        .workgroup_size_x = 0,
        .workgroup_size_y = 0,
        .workgroup_size_z = 0,
        .parameter_count = sizeof(memory_copy_params) / sizeof(memory_copy_params[0]),
    },
    {
        // compute_intensive
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 2,
        .binding_count = 2,
        .workgroup_size_x = 0,
        .workgroup_size_y = 0,
        .workgroup_size_z = 0,
        .parameter_count = sizeof(compute_intensive_params) / sizeof(compute_intensive_params[0]),
    },
};

// Names for each entry point.
static const char* entry_point_names[] = {
    "nop_kernel",
    "simple_add_raw",
    "simple_add_hal",
    "many_params_raw",
    "many_params_hal",
    "memory_copy",
    "compute_intensive",
};

// Optional tags for debugging.
static const char* entry_point_tags[] = {
    "nop",
    "simple_raw",
    "simple_hal",
    "many_raw",
    "many_hal",
    "memcpy",
    "compute",
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
            .occupancy = NULL,
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

const iree_hal_executable_library_header_t** test_kernels_library_query(
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
  return test_kernels_library_query(max_version, environment);
}

#ifdef __cplusplus
}
#endif