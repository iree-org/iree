// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/builtins/ukernel/common.h"

// Very naive kernel. TODO(bjacob):
// 1. Inlining: the `always_inline` attribute here is correctly preserved in
//    the bitcode, but isn't having the intended effect of inlining calls to
//    this function. Making that work is key as various function parameters
//    (e.g. `unroll_m`) are meant to be constants.
// 2. Shared memory: can't allocate it within the microkernel (which is just a
//    helper device function, not the actual amdgpu_kernel). Need to get it
//    passed down here as a `T [[clang::address_space(3)]] *` parameter.
// 3. Better scheduling via either barrier intrinsics or inline assemby.
// 4. Subgroups1x4 being asymmetric is a historical accident... should be 2x2.
[[clang::always_inline]] void iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8(
    const int8_t *a_buffer, int64_t a_offset, const int8_t *b_buffer,
    int64_t b_offset, int32_t *c_buffer, int64_t c_offset, int32_t k_size,
    int32_t unroll_m, int32_t subgroups_m, int32_t unroll_n,
    int32_t subgroups_n, int32_t unroll_k) {
  /*
    TODO(bjacob): reenable this once inlining works.
    // Load existing accumulators. This is a VLA, but should become fixed-size
    // once this function is inlined and unroll_* factors become constants.
    int32x4_t c[unroll_m][unroll_n];
  */
  // Load existing accumulators.
  if (unroll_m > 8 || unroll_n > 2) {
    __builtin_trap();
  }
  int32x4_t c[8][2];
  int32x4_t *c_global = (int32x4_t *)(c_buffer + c_offset);
  for (int m = 0; m < unroll_m; ++m) {
    for (int n = 0; n < unroll_n; ++n) {
      c[m][n] = c_global[64 * (m * unroll_n + n)];
    }
  }

  // Arithmetic loop.
  const int64_t *a_global = (const int64_t *)(a_buffer + a_offset);
  const int64_t *b_global = (const int64_t *)(b_buffer + b_offset);
  for (int k_outer = 0; k_outer < k_size; ++k_outer) {
    for (int m = 0; m < unroll_m; ++m) {
      for (int n = 0; n < unroll_n; ++n) {
        for (int k = 0; k < unroll_k; ++k) {
          c[m][n] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
              a_global[64 * unroll_k * m + k], b_global[64 * unroll_k * n + k],
              c[m][n], 0, 0, 0);
        }
      }
    }
    a_global += 64 * unroll_m * subgroups_m * unroll_k;
    b_global += 64 * unroll_n * subgroups_n * unroll_k;
  }

  // Store accumulators.
  for (int m = 0; m < unroll_m; ++m) {
    for (int n = 0; n < unroll_n; ++n) {
      c_global[64 * (m * unroll_n + n)] = c[m][n];
    }
  }
}
