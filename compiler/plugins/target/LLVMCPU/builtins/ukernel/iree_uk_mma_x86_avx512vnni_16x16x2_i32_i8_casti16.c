// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <immintrin.h>
#include "common.h"

// Microkernel for `iree_codegen.inner_tiled` with
// `#iree_cpu.data_tiled_mma_layout<intrinsic =
//     MMA_X86_AVX512VNNI_16x16x2_I32_I8_CASTI16>`. See README.md for the
// framework design (naming, specialization-after-inlining, inner K loop).
//
// Structure: accumulators stay in registers; an outer loop over the K *tiles*
// (`k_outer`) wraps the unrolled `(intrinsics_m, intrinsics_n, intrinsics_k)`
// grid (arbitrary positive values, fully unrolled at the inlined call site).
//
// The "CASTI16" in the intrinsic name reflects that the s8 inputs are
// sign-extended to i16 lanes before being fed to the 16-bit VNNI instruction
// `vpdpwssd` (the 8-bit `vpdpbusd` would mishandle the s8 x s8 signedness);
// that widen happens once per panel in the inner loop.
//
// ABI: each shaped operand is passed as (base pointer, element offset) so the
// caller doesn't need a GEP before the call (the offset is added here). No
// strides are passed: the ACC tile is contiguous, so the ukernel addresses
// each fragment from `intrinsics_{m,n}` and its fixed fragment size. Offsets
// are in units of the operand element type (i8 for LHS/RHS, i32 for ACC).
//
// Per-intrinsic (16x16x2) tile, matching `lowerX86Avx512Vnni16x16x2I8` in
// IREECPUAttrs.cpp (the codegen path this ukernel must be bit-compatible
// with) and the `getIntrinsicSwizzle` data layout:
//   - LHS: one row-major 16x2 i8 panel (= 32 i8 = <32xi8>). Dword `x` of its
//     i16-widened form holds the (k0, k1) pair of LHS row `x`.
//   - RHS: one row-major 16x2 i8 panel (= 32 i8), same shape; dword `x` holds
//     column `x`'s (k0, k1) pair.
//   - ACC: one 16x16 i32 tile (= 256 i32) in the block-interleaved
//     (rlo, chi, rhi, clo) order, with row r = 4*rhi + rlo and column
//     c = 4*chi + clo. The 16 i32 at flat offset (4*rlo + chi)*16 are one
//     `vpdpwssd` accumulator, whose dword `4*rhi + clo` holds ACC element
//     (r, c). The (m, n) grid of 16x16 tiles is tightly packed, so fragment
//     (m, n) is at `acc + (m * intrinsics_n + n) * 256`.
IREE_UK_ALWAYS_INLINE
void iree_uk_mma_x86_avx512vnni_16x16x2_i32_i8_casti16(
    const void *lhs_base, int64_t lhs_offset, const void *rhs_base,
    int64_t rhs_offset, void *acc_base, int64_t acc_offset, int32_t k_outer,
    int32_t intrinsics_m, int32_t intrinsics_n, int32_t intrinsics_k) {
  const int8_t *lhs = (const int8_t *)lhs_base + lhs_offset;
  const int8_t *rhs = (const int8_t *)rhs_base + rhs_offset;
  int32_t *acc = (int32_t *)acc_base + acc_offset;

  // The ACC tile is contiguous: each intrinsic's fragment is one 16x16 i32
  // tile (256 i32), so fragment (m, n) sits
  // `(m * intrinsics_n + n) * kAccFragElems` elements into `acc`.
  enum { kAccFragElems = 256 };

  // 256 i32 (= 16 __m512i `vpdpwssd` accumulators) per (m, n) intrinsic. The
  // VLA dimensions are compile-time constants at the inlined call site, so
  // this lowers to a fixed register array.
  __m512i acc_regs[intrinsics_m][intrinsics_n][16];
  for (int32_t m = 0; m < intrinsics_m; ++m) {
    for (int32_t n = 0; n < intrinsics_n; ++n) {
      const int32_t *frag = acc + (m * intrinsics_n + n) * kAccFragElems;
      for (int c = 0; c < 16; ++c) {
        acc_regs[m][n][c] = _mm512_loadu_si512(frag + c * 16);
      }
    }
  }

  for (int32_t ko = 0; ko < k_outer; ++ko) {
    // Each (m, k) / (n, k) fragment is one 16x2 i8 panel = 32 i8.
    const int8_t *lhs_block =
        lhs + (int64_t)ko * intrinsics_m * intrinsics_k * 32;
    const int8_t *rhs_block =
        rhs + (int64_t)ko * intrinsics_n * intrinsics_k * 32;
    for (int32_t m = 0; m < intrinsics_m; ++m) {
      for (int32_t n = 0; n < intrinsics_n; ++n) {
        __m512i(*regs)[16] = &acc_regs[m][n];
        for (int32_t k = 0; k < intrinsics_k; ++k) {
          // Widen each i8 panel to i16 once (one `vpmovsxbw`); dword `x` then
          // holds the (k0, k1) pair of LHS row / RHS column `x`.
          __m512i lhs_i16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
              (const __m256i *)(lhs_block + (m * intrinsics_k + k) * 32)));
          __m512i rhs_i16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
              (const __m256i *)(rhs_block + (n * intrinsics_k + k) * 32)));
          // lhs_dup[rlo]: `vpshufd` broadcasting dword `4*lane + rlo` across
          // each 128-bit lane (lane L then holds LHS row 4*L + rlo).
          // rhs_bcast[chi]: `vbroadcasti32x4` of the 128-bit block of columns
          // [4*chi, 4*chi+4) to all 4 lanes. The shuffle immediates must be
          // compile-time constants (`s * 0x55` for s = 0..3), so the 4 cases
          // are spelled out rather than looped.
          __m512i lhs_dup[4] = {
              _mm512_shuffle_epi32(lhs_i16, (_MM_PERM_ENUM)0x00),
              _mm512_shuffle_epi32(lhs_i16, (_MM_PERM_ENUM)0x55),
              _mm512_shuffle_epi32(lhs_i16, (_MM_PERM_ENUM)0xAA),
              _mm512_shuffle_epi32(lhs_i16, (_MM_PERM_ENUM)0xFF),
          };
          __m512i rhs_bcast[4] = {
              _mm512_shuffle_i32x4(rhs_i16, rhs_i16, 0x00),
              _mm512_shuffle_i32x4(rhs_i16, rhs_i16, 0x55),
              _mm512_shuffle_i32x4(rhs_i16, rhs_i16, 0xAA),
              _mm512_shuffle_i32x4(rhs_i16, rhs_i16, 0xFF),
          };
          // 16 `vpdpwssd` over the 4x4 (rlo, chi) grid; accumulator (rlo, chi)
          // lives at flat offset (4*rlo + chi)*16.
          for (int rlo = 0; rlo < 4; ++rlo) {
            for (int chi = 0; chi < 4; ++chi) {
              int idx = 4 * rlo + chi;
              (*regs)[idx] = _mm512_dpwssd_epi32((*regs)[idx], lhs_dup[rlo],
                                                 rhs_bcast[chi]);
            }
          }
        }
      }
    }
  }

  for (int32_t m = 0; m < intrinsics_m; ++m) {
    for (int32_t n = 0; n < intrinsics_n; ++n) {
      int32_t *frag = acc + (m * intrinsics_n + n) * kAccFragElems;
      for (int c = 0; c < 16; ++c) {
        _mm512_storeu_si512(frag + c * 16, acc_regs[m][n][c]);
      }
    }
  }
}
