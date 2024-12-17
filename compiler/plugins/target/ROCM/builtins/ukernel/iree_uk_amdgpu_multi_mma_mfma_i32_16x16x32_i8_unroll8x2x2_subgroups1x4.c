// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/builtins/ukernel/common.h"

// Very naive kernel. TODO(bjacob):
// 1. Shared memory: can't allocate it within the microkernel (which is just a
//    helper device function, not the actual amdgpu_kernel). Need to get it
//    passed down here as a `T [[clang::address_space(3)]] *` parameter.
// 2. Better scheduling via either barrier intrinsics or inline assemby.
// 3. Subgroups1x4 being asymmetric is a historical accident... should be 2x2.
[[clang::always_inline]] void
iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8_unroll8x2x2_subgroups1x4(
    const int8_t *a_buffer, int64_t a_offset, const int8_t *b_buffer,
    int64_t b_offset, int32_t *c_buffer, int64_t c_offset, int64_t k_size) {
  int tid = __builtin_amdgcn_workitem_id_x();

  // Load existing accumulators.
  int32x4_t acc[8][2] = {{0}};
  int32x4_t *c_global = (int32x4_t *)(c_buffer + c_offset);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 2; ++j) {
      acc[i][j] = c_global[256 * (2 * i + j) + tid];
    }
  }

  // Arithmetic loop.
  const int64x2_t *a_global =
      (const int64x2_t *)(a_buffer + a_offset) + (tid % 64);
  const int64x2_t *b_global = (const int64x2_t *)(b_buffer + b_offset) + tid;
  for (int k_outer = 0; k_outer < k_size; ++k_outer) {
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
              a_global[64 * i][k], b_global[256 * j][k], acc[i][j], 0, 0, 0);
        }
      }
    }
    a_global += 512;
    b_global += 512;
  }

  // Store accumulators.
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 2; ++j) {
      c_global[256 * (2 * i + j) + tid] = acc[i][j];
    }
  }
}
