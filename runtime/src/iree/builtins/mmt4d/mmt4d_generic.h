// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_MMT4D_MMT4D_GENERIC_H_
#define IREE_BUILTINS_MMT4D_MMT4D_GENERIC_H_

#include "mmt4d.h"

//===----------------------------------------------------------------------===//
// Reference implementation
//===----------------------------------------------------------------------===//
// Available for use within the library to cover code paths not yet implemented
// on a particular target architecture. When an architecture is not covered at
// all we export these as the public API in mmt4d_generic.c.
//
// Though these are intended for internal use they may be used in tests
// verifying that an architecture-specific versions matches the reference.

static int iree_mmt4d_example_matmul_f32_generic(
    const float* lhs, iree_mmt4d_size_t lhs_stride, const float* rhs,
    iree_mmt4d_size_t rhs_stride, float* IREE_RESTRICT out,
    iree_mmt4d_size_t out_stride, int32_t m, int32_t n, int32_t k, float alpha,
    float beta) {
  for (int32_t mi = 0; mi < m; ++mi) {
    for (int32_t mk = 0; mk < k; ++mk) {
      float apart = alpha * lhs[mi * lhs_stride + mk];
      for (int32_t mj = 0; mj < n; ++mj) {
        out[mi * out_stride + mj] += beta * apart * rhs[mk * rhs_stride + mj];
      }
    }
  }
  return 0;
}

#endif  // IREE_BUILTINS_MMT4D_MMT4D_GENERIC_H_
