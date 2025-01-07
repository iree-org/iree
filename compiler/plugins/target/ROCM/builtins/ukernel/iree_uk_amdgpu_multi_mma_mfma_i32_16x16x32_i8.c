// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/builtins/ukernel/common.h"

// Very naive kernel. TODO(bjacob):
// 1. Shared memory: can't allocate it within the microkernel (which is just a
//    helper device function, not the actual amdgpu_kernel). Need to get it
//    passed down here as a `int8_t LOCAL *` parameter.
// 2. Better scheduling via either barrier intrinsics or inline assemby.
[[clang::always_inline]] void iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8(
    const int8_t GLOBAL *a_buffer, int64_t a_offset,
    const int8_t GLOBAL *b_buffer, int64_t b_offset, int32_t GLOBAL *c_buffer,
    int64_t c_offset, int32_t k_size, int32_t unroll_m, int32_t subgroups_m,
    int32_t unroll_n, int32_t subgroups_n, int32_t unroll_k) {
  // Load existing accumulators. This VLA becomes a regular fixed-size array
  // after inlining into the caller where these values are constants.
  int32x4_t c[unroll_m][unroll_n];
  int32x4_t GLOBAL *c_ptr = (int32x4_t GLOBAL *)(c_buffer + c_offset);
  for (int m = 0; m < unroll_m; ++m) {
    for (int n = 0; n < unroll_n; ++n) {
      c[m][n] = c_ptr[64 * (m * unroll_n + n)];
    }
  }

  // Arithmetic loop.
  const int64_t GLOBAL *a_ptr = (const int64_t GLOBAL *)(a_buffer + a_offset);
  const int64_t GLOBAL *b_ptr = (const int64_t GLOBAL *)(b_buffer + b_offset);
  for (int k_outer = 0; k_outer < k_size; ++k_outer) {
    for (int m = 0; m < unroll_m; ++m) {
      for (int n = 0; n < unroll_n; ++n) {
        for (int k = 0; k < unroll_k; ++k) {
          c[m][n] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
              a_ptr[64 * unroll_k * m + k], b_ptr[64 * unroll_k * n + k],
              c[m][n], 0, 0, 0);
        }
      }
    }
    a_ptr += 64 * unroll_m * subgroups_m * unroll_k;
    b_ptr += 64 * unroll_n * subgroups_n * unroll_k;
  }

  // Store accumulators.
  for (int m = 0; m < unroll_m; ++m) {
    for (int n = 0; n < unroll_n; ++n) {
      c_ptr[64 * (m * unroll_n + n)] = c[m][n];
    }
  }
}
