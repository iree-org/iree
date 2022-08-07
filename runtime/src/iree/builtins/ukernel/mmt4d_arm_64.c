// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mmt4d.h"
#include "mmt4d_generic.h"

#if defined(IREE_UKERNEL_ARCH_ARM_64)

//===----------------------------------------------------------------------===//
// Target-specific queries
//===----------------------------------------------------------------------===//
// These are substituted with values from the compiler and must not be specified
// here in C before we generate the IR.

#if defined(IREE_UKERNEL_PLATFORM_EXAMPLE_FLAG)
// Set by command-line logic:
static const int iree_microkernels_platform_example_flag =
    IREE_UKERNEL_PLATFORM_EXAMPLE_FLAG;
#else
// Set by IREE AOT compiler:
extern int iree_microkernels_platform_example_flag;
#endif  // IREE_UKERNEL_PLATFORM_EXAMPLE_FLAG

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// This is just an example exporting the reference method.
// Implementations can do this when they don't otherwise have a code-path that
// covers a particular case and here we have no code-paths :)
IREE_UKERNEL_EXPORT int iree_mmt4d_example_matmul_f32(
    const float* lhs, iree_ukernel_size_t lhs_stride, const float* rhs,
    iree_ukernel_size_t rhs_stride, float* IREE_RESTRICT out,
    iree_ukernel_size_t out_stride, int32_t m, int32_t n, int32_t k,
    float alpha, float beta) {
  return iree_mmt4d_example_matmul_f32_generic(
      lhs, lhs_stride, rhs, rhs_stride, out, out_stride, m, n, k, alpha, beta);
}

#endif  // IREE_MMT4D_ARCH_ARM_64
