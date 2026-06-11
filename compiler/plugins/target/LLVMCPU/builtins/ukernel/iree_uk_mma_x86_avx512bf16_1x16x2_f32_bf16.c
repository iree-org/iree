// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <immintrin.h>
#include "common.h"

// Microkernel for `iree_codegen.inner_tiled` with
// `#iree_cpu.data_tiled_mma_layout<intrinsic =
//     MMA_X86_AVX512BF16_1x16x2_F32_BF16>`. Function name matches the
// intrinsic name verbatim (lowercased, with the `iree_uk_` prefix), in line
// with the AMDGPU C ukernel convention.
//
// The "inner K loop" the ukernel owns is the loop over the K *tiles* that
// sits *inside* the outer M/N loops; those outer M/N loops are tiled away by
// ordinary IREE tiling before this ukernel runs. The ukernel handles
// arbitrary positive `intrinsics_{m,n,k}` (passed as arguments and looped
// over); the loops fully unroll after the ukernel is inlined into its
// constant-`intrinsics_*` caller -- the bitcode-LTO equivalent of a C++
// template.
//
// ABI: each shaped operand is passed as (base pointer, element offset) so the
// caller doesn't need a GEP before the call; the accumulator additionally
// gets the element stride of its innermost cross-intrinsic (N) dimension.
//
// NOTE (seed scaffolding): this initial seed has a stub body. It exists so
// that the surrounding *framework* -- bitcode build, embedding,
// `hal.executable_object` injection, IR rewrite to `ukernel.generic` -- can
// be landed and lit-tested. A follow-up commit replaces the body with the
// `_mm512_dpbf16_ps`-based inner loop and adds an e2e matmul test for it.
IREE_UK_ALWAYS_INLINE
void iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16(
    const uint16_t *lhs_base, int64_t lhs_offset, const uint16_t *rhs_base,
    int64_t rhs_offset, float *acc_base, int64_t acc_offset, int64_t acc_stride,
    int32_t k_outer, int32_t intrinsics_m, int32_t intrinsics_n,
    int32_t intrinsics_k) {
  (void)lhs_base;
  (void)lhs_offset;
  (void)rhs_base;
  (void)rhs_offset;
  (void)acc_base;
  (void)acc_offset;
  (void)acc_stride;
  (void)k_outer;
  (void)intrinsics_m;
  (void)intrinsics_n;
  (void)intrinsics_k;
  // TODO(ukernels): real inner K loop using `_mm512_dpbf16_ps`, looping over
  // intrinsics_{m,n,k}.
}
