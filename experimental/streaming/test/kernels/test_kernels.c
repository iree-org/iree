// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GPU test kernel stubs for benchmarking - placeholder for future GPU implementations

// No-op kernel.
extern "C" __global__ void nop_kernel() {
  // Minimal work to prevent optimization.
}

// Simple addition with raw pointers.
extern "C" __global__ void simple_add_raw(
    float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// Simple addition with HAL bindings.
extern "C" __global__ void simple_add_hal(
    float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// Many parameters raw mode.
extern "C" __global__ void many_params_raw(
    float* in, float* out,
    float scale, float bias, int n, int stride,
    float min_val, float max_val, int flags, int mode,
    float alpha, float beta, float gamma, float delta,
    int offset1, int offset2, int offset3, int offset4) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx * stride + offset1 % stride];
    val = val * scale + bias;
    val = val * alpha + beta * gamma - delta;
    val = (val < min_val) ? min_val : val;
    val = (val > max_val) ? max_val : val;
    if (flags & 1) val = -val;
    if (mode == 1) val = val * val;
    out[idx + offset2 % n] = val;
    if (offset3 == offset4) out[0] += 0.0001f;
  }
}

// Many parameters HAL mode.
extern "C" __global__ void many_params_hal(
    float* buf0, float* buf1, float* buf2, float* buf3, float* buf4,
    float* out0, float* out1, float* out2, float* out3, float* out4,
    int n, float weight0, float weight1, float weight2, float weight3, float weight4,
    int offset, int stride, int flags, int mode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int i = (idx + offset) % n;
    float sum = buf0[i] * weight0 + buf1[i] * weight1 + 
                buf2[i] * weight2 + buf3[i] * weight3 + 
                buf4[i] * weight4;
    if (flags & 1) sum = -sum;
    if (mode == 1) sum = sum * sum;
    out0[i] = sum;
    out1[i] = sum * 0.5f;
    out2[i] = sum * 0.25f;
    out3[i] = sum * 0.125f;
    out4[i] = sum * 0.0625f;
  }
}

// Memory copy kernel.
extern "C" __global__ void memory_copy(
    unsigned int* src, unsigned int* dst, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

// Compute-intensive kernel.
extern "C" __global__ void compute_intensive(
    float* in, float* out, int n, int iterations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
    for (int j = 0; j < iterations; j++) {
      val = val * 1.1f - 0.1f;
      val = val * val * 0.999f;
      val = val + 0.001f;
    }
    out[idx] = val;
  }
}