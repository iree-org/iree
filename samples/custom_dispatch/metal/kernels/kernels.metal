// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernels following the IREE Metal ABI: the bindings of a dispatch are passed
// in an argument buffer at [[buffer(0)]] whose [[id(N)]] is the binding
// ordinal, and its constants as 32-bit values at [[buffer(3)]]. The
// threadgroup size comes from the export's workgroup_size.

#include <metal_stdlib>

using namespace metal;

// `ret = lhs * rhs`
//
// Conforms to ABI:
// #hal.pipeline.layout<constants = 1, bindings = [
//   #hal.pipeline.binding<storage_buffer, ReadOnly>,
//   #hal.pipeline.binding<storage_buffer, ReadOnly>,
//   #hal.pipeline.binding<storage_buffer>
// ]>
struct SimpleMulBindings {
  const device float* lhs [[id(0)]];
  const device float* rhs [[id(1)]];
  device float* ret [[id(2)]];
};

kernel void simple_mul(constant SimpleMulBindings& bindings [[buffer(0)]],
                       constant uint* constants [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
  uint dim = constants[0];
  if (tid < dim) {
    bindings.ret[tid] = bindings.lhs[tid] * bindings.rhs[tid];
  }
}

// `rhs *= lhs`
//
// Conforms to ABI:
// #hal.pipeline.layout<constants = 1, bindings = [
//   #hal.pipeline.binding<storage_buffer, ReadOnly>,
//   #hal.pipeline.binding<storage_buffer>
// ]>
struct SimpleMulInplaceBindings {
  const device float* lhs [[id(0)]];
  device float* rhs [[id(1)]];
};

kernel void simple_mul_inplace(
    constant SimpleMulInplaceBindings& bindings [[buffer(0)]],
    constant uint* constants [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint dim = constants[0];
  if (tid < dim) {
    bindings.rhs[tid] *= bindings.lhs[tid];
  }
}
