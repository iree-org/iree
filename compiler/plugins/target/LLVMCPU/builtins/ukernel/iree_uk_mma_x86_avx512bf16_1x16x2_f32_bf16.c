// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <immintrin.h>
#include "common.h"

// Microkernel for `iree_codegen.inner_tiled` with
// `#iree_cpu.data_tiled_mma_layout<intrinsic =
//     MMA_X86_AVX512BF16_1x16x2_F32_BF16>`. See README.md for the framework
// design: the `iree_uk_<lowercased-intrinsic>` naming, why the
// `intrinsics_{m,n,k}` parameters fully specialize the body after inlining,
// and how the inner K loop relates to IREE tiling.
//
// Structure: accumulators stay in registers; an outer loop over the K *tiles*
// (`k_outer`) wraps the unrolled `(intrinsics_m, intrinsics_n, intrinsics_k)`
// grid (arbitrary positive values, fully unrolled at the inlined call site).
//
// ABI: each shaped operand is passed as (base pointer, element offset) so the
// caller doesn't need a GEP before the call (the offset is added here). No
// strides are passed: the ACC tile is contiguous, so the ukernel addresses
// each fragment from `intrinsics_{m,n}` and its fixed fragment size. Offsets
// are in units of the operand element type (bf16 for LHS/RHS, f32 for ACC).
//
// Data-tiled operand layout (matching the `DataTiledMMAAttr` swizzle):
//   - ACC: one `__m512` (= M0=1 x N0=16 f32) per (m, n) intrinsic, tightly
//     packed row-major over the (m, n) grid, so fragment (m, n) is at
//     `acc + (m * intrinsics_n + n) * 16`.
//   - LHS: per outer-K step, `intrinsics_m * intrinsics_k` units of 2 bf16
//     (= one M0=1 x K0=2 fragment = a 4-byte `vdpbf16ps` m_bcst unit),
//     ordered [m][k]; consecutive outer-K steps are contiguous.
//   - RHS: per outer-K step, `intrinsics_n * intrinsics_k` panels of 32 bf16
//     (= one N0=16 x K0=2 fragment = one `__m512`), ordered [n][k];
//     consecutive outer-K steps are contiguous.
IREE_UK_ALWAYS_INLINE
void iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16(
    const uint16_t *lhs_base, int64_t lhs_offset, const uint16_t *rhs_base,
    int64_t rhs_offset, float *acc_base, int64_t acc_offset, int32_t k_outer,
    int32_t intrinsics_m, int32_t intrinsics_n, int32_t intrinsics_k) {
  const uint16_t *lhs = lhs_base + lhs_offset;
  const float *rhs = (const float *)(rhs_base + rhs_offset);
  float *acc = acc_base + acc_offset;

  // The ACC tile is contiguous: each intrinsic's fragment is one __m512
  // (M0*N0 = 1*16 f32), so fragment (m, n) sits
  // `(m * intrinsics_n + n) * kAccFragElems` elements into `acc`.
  enum { kAccFragElems = 16 };

  // One accumulator register (1x16 f32) per (m, n) intrinsic. The VLA
  // dimensions are compile-time constants at the inlined call site, so this
  // lowers to a fixed register array.
  __m512 acc_regs[intrinsics_m][intrinsics_n];
  for (int32_t m = 0; m < intrinsics_m; ++m) {
    for (int32_t n = 0; n < intrinsics_n; ++n) {
      acc_regs[m][n] =
          _mm512_loadu_ps(acc + (m * intrinsics_n + n) * kAccFragElems);
    }
  }

  for (int32_t ko = 0; ko < k_outer; ++ko) {
    const uint16_t *lhs_block =
        lhs + (int64_t)ko * intrinsics_m * intrinsics_k * 2;
    const float *rhs_block =
        rhs + (int64_t)ko * intrinsics_n * intrinsics_k * 16;
    for (int32_t m = 0; m < intrinsics_m; ++m) {
      for (int32_t n = 0; n < intrinsics_n; ++n) {
        for (int32_t k = 0; k < intrinsics_k; ++k) {
          // LHS fragment: 2 bf16 (one M-row's K-pair) broadcast across the
          // 16 SIMD lanes via `set1_ps` (the splat shape `vdpbf16ps`'s
          // m_bcst variant pattern-matches). The bitcast to `__m512bh` is a
          // width-preserving no-op LLVM elides.
          __m512 lhs_bcast = _mm512_set1_ps(
              *(const float *)(lhs_block + (m * intrinsics_k + k) * 2));
          // RHS fragment: one (N=16 x K=2) bf16 panel = 16 f32.
          __m512 rhs_panel =
              _mm512_loadu_ps(rhs_block + (n * intrinsics_k + k) * 16);
          acc_regs[m][n] =
              _mm512_dpbf16_ps(acc_regs[m][n], *(const __m512bh *)&lhs_bcast,
                               *(const __m512bh *)&rhs_panel);
        }
      }
    }
  }

  for (int32_t m = 0; m < intrinsics_m; ++m) {
    for (int32_t n = 0; n < intrinsics_n; ++n) {
      _mm512_storeu_ps(acc + (m * intrinsics_n + n) * kAccFragElems,
                       acc_regs[m][n]);
    }
  }
}
