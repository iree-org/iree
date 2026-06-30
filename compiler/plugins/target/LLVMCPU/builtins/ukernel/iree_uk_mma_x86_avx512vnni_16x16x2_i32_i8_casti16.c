// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <immintrin.h>
#include "common.h"

// Microkernel for `iree_codegen.inner_tiled` with
// `#iree_cpu.data_tiled_mma_layout<intrinsic =
//     MMA_X86_AVX512VNNI_16x16x2_I32_I8_CASTI16>`. Function name matches the
// intrinsic name verbatim (lowercased, with the `iree_uk_` prefix), in line
// with the AMDGPU C ukernel convention.
//
// Implements the inner K-loop for the unrolled (intrinsics_m, intrinsics_n,
// intrinsics_k) tile built from the 16x16x2 i8 VNNI intrinsic via AVX-512
// VNNI `vpdpwssd`. The "CASTI16" in the MMA intrinsic name reflects that
// the s8 inputs are zero/sign-extended into i16 lanes before being fed to
// the 16-bit VNNI instruction; that cast is handled in the inner loop.
//
// `intrinsics_{m,n,k}` are passed as function arguments and so look like
// runtime values inside this translation unit, but the ukernel is always
// inlined into its caller (a bug otherwise) and the caller always passes
// the matching `DataTiledMMAAttr` constants. Together with post-inline IR
// optimization on the linked bitcode, the body specializes to specific
// compile-time `intrinsics_{m,n,k}` values at each call site.
//
// NOTE (seed scaffolding): this initial seed has a stub body. It exists so
// that the surrounding *framework* -- bitcode build, embedding,
// `hal.executable_object` injection, IR rewrite to `ukernel.generic` -- can
// be exercised end-to-end. A follow-up commit replaces the body with the
// `_mm512_dpwssd_epi32`-based inner loop and adds an e2e matmul test for
// it. This is the "practically useful" seed: i8x i8->i32 via VNNI is a
// workhorse for quantized inference, and codegen has a residual perf gap
// on this case.
// ABI matches the inner_tiled -> ukernel.generic lowering (see
// `iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16`): each shaped operand passed
// as (base, element offset) only (no strides; the ACC tile is contiguous),
// then the scalar `k_outer` / `intrinsics_{m,n,k}`.
IREE_UK_ALWAYS_INLINE
void iree_uk_mma_x86_avx512vnni_16x16x2_i32_i8_casti16(
    const void *lhs_base, int64_t lhs_offset, const void *rhs_base,
    int64_t rhs_offset, void *acc_base, int64_t acc_offset, int32_t k_outer,
    int32_t intrinsics_m, int32_t intrinsics_n, int32_t intrinsics_k) {
  (void)lhs_base;
  (void)lhs_offset;
  (void)rhs_base;
  (void)rhs_offset;
  (void)acc_base;
  (void)acc_offset;
  (void)k_outer;
  (void)intrinsics_m;
  (void)intrinsics_n;
  (void)intrinsics_k;
  // TODO(ukernels): real inner K loop using `_mm512_dpwssd_epi32` after
  // widening the s8 LHS/RHS halves to i16 lanes (loop over
  // intrinsics_{m,n,k} like the bf16 ukernel).
}
