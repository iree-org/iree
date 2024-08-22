// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// `rhs *= lhs`
//
// Conforms to ABI:
// #hal.pipeline.layout<constants = 1, bindings = [
//   #hal.pipeline.binding<storage_buffer, ReadOnly>,
//   #hal.pipeline.binding<storage_buffer>
// ]>

#version 450

// Workgroup local size that factors into the host-side workgroup count math.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Binding0 { float binding0[]; };
layout(set = 0, binding = 1) buffer Binding1 { float binding1[]; };

layout(push_constant) uniform PushConstants {
  uint dim;
};

void main() {
  uint tid = gl_GlobalInvocationID.x;
  if (tid < dim) {
    binding1[tid] *= binding0[tid];
  }
}
