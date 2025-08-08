// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector addition kernel for testing CUDA HAL
// Compilable for AMDGPU, SPIR-V, and CPU targets

#ifdef __AMDGCN__
// AMDGPU specific attributes and intrinsics
#define KERNEL_ATTR __attribute__((amdgpu_kernel))
#define GET_GLOBAL_ID()              \
  __builtin_amdgcn_workitem_id_x() + \
      __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x()
#elif defined(__SPIRV__)
// SPIR-V specific attributes
#define KERNEL_ATTR __attribute__((kernel))
// For SPIR-V, we'd use OpenCL-style built-ins
extern unsigned get_global_id(unsigned dim);
#define GET_GLOBAL_ID() get_global_id(0)
#else
// CPU fallback - standard C
#define KERNEL_ATTR
#define GET_GLOBAL_ID() 0  // Will be set by host
#endif

// Simple vector addition: C = A + B
KERNEL_ATTR
void vector_add(const float* a, const float* b, float* c, unsigned int n) {
#ifdef __AMDGCN__
  unsigned int idx = GET_GLOBAL_ID();
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
#elif defined(__SPIRV__)
  unsigned int idx = GET_GLOBAL_ID();
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
#else
  // CPU version - simple loop (would be called from host)
  for (unsigned int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
#endif
}

// Vector addition with scalar multiplication: C = alpha * A + beta * B
KERNEL_ATTR
void vector_add_scaled(const float* a, const float* b, float* c, float alpha,
                       float beta, unsigned int n) {
#ifdef __AMDGCN__
  unsigned int idx = GET_GLOBAL_ID();
  if (idx < n) {
    c[idx] = alpha * a[idx] + beta * b[idx];
  }
#elif defined(__SPIRV__)
  unsigned int idx = GET_GLOBAL_ID();
  if (idx < n) {
    c[idx] = alpha * a[idx] + beta * b[idx];
  }
#else
  // CPU version
  for (unsigned int i = 0; i < n; i++) {
    c[i] = alpha * a[i] + beta * b[i];
  }
#endif
}

// Element-wise vector multiplication: C = A * B
KERNEL_ATTR
void vector_multiply(const float* a, const float* b, float* c, unsigned int n) {
#ifdef __AMDGCN__
  unsigned int idx = GET_GLOBAL_ID();
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
#elif defined(__SPIRV__)
  unsigned int idx = GET_GLOBAL_ID();
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
#else
  // CPU version
  for (unsigned int i = 0; i < n; i++) {
    c[i] = a[i] * b[i];
  }
#endif
}
