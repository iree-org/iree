// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mmt4d_generic.h"

#if defined(IREE_UKERNEL_ARCH_GENERIC_32) || \
    defined(IREE_UKERNEL_ARCH_GENERIC_64)

//===----------------------------------------------------------------------===//
// Target-specific queries
//===----------------------------------------------------------------------===//
// These are substituted with values from the compiler and must not be specified
// here in C before we generate the IR.
//
// NOTE: we can use this flag to change how/which reference implementations we
// use but should not use it to change the reference implementations themselves.
// This ensures that if multiple architectures reuse the generic methods the
// flags don't conflict.

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
// Exports this generic reference implementation as the primary API.
// Specific architectures may still share various parts of the reference
// implementation but will export their own API as primary.

IREE_UKERNEL_EXPORT int iree_mmt4d_example_matmul_f32(
    const float* lhs, iree_ukernel_size_t lhs_stride, const float* rhs,
    iree_ukernel_size_t rhs_stride, float* IREE_RESTRICT out,
    iree_ukernel_size_t out_stride, int32_t m, int32_t n, int32_t k,
    float alpha, float beta) {
  // We could check the flag here or branch off to different implementations
  // based on mnk or alpha/beta values.
  return iree_mmt4d_example_matmul_f32_generic(
      lhs, lhs_stride, rhs, rhs_stride, out, out_stride, m, n, k, alpha, beta);
}

#endif  // IREE_MMT4D_ARCH_GENERIC_*
