// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <immintrin.h>

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_internal.h"

#ifdef IREE_UK_COMPILER_CLANG
// This inline-asm function is a work-around for:
// 1. https://github.com/llvm/llvm-project/issues/68117
//    Summary: LLVM crash affecting Clang 16-17. Fixed in Clang 18.
// 2. https://github.com/llvm/llvm-project/issues/68810
//    Summary: performance regression in the generated code.
// 3. Passing lhs as `__m512` instead of the more proper `__m512bh`
//    works around https://github.com/llvm/llvm-project/issues/68149,
//    and passing it as-is as the asm operand instead of casting it in C
//    works around a crash in clang-16 in Red Hat specifically, not
//    reproducible in other clang-16.
static inline __m512 iree_mm512_dpbf16_ps_broadcast_rhs(
    __m512 acc, __m512 lhs, const iree_uk_uint16_t* rhs) {
  // Sorry about the crazy AT&T syntax with reversed operand order.
  // Couldn't figure how to use Intel syntax with inline asm operands.
  asm("vdpbf16ps %[rhs]%{1to16%}, %[lhs], %[acc]"
      : [acc] "+v"(acc)
      : [lhs] "v"(lhs), [rhs] "m"(*rhs)
      :);
  return acc;
}
#else
static inline __m512bh bitcast_16xf32_to_32xbf16(__m512 a) {
  return *(const __m512bh*)(&a);
}
static inline __m512 iree_mm512_dpbf16_ps_broadcast_rhs(
    __m512 acc, __m512 lhs, const iree_uk_uint16_t* rhs) {
  return _mm512_dpbf16_ps(
      acc, bitcast_16xf32_to_32xbf16(lhs),
      bitcast_16xf32_to_32xbf16(_mm512_set1_ps(*(const float*)rhs)));
}
#endif

static inline void
iree_uk_mmt4d_tile_bf16bf16f32_1x16x2_to_16x16x2_x86_64_avx512_bf16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  __m512 acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_loadu_ps(out_ptr + i * 16);
    }
  } else {
    for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_setzero_ps();
    }
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_loadu_ps(rhs_ptr);
    rhs_ptr += 32;
    for (int i = 0; i < M0; ++i) {
      acc[i] = iree_mm512_dpbf16_ps_broadcast_rhs(acc[i], rhs, lhs_ptr + 2 * i);
    }
    lhs_ptr += M0 * 2;
  }

  for (int i = 0; i < M0; ++i) {
    _mm512_storeu_ps(out_ptr + i * 16, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8_16(
    iree_uk_mmt4d_tile_bf16bf16f32_1x16x2_to_16x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_1x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_2x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_4x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_8x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_16x16x2_x86_64_avx512_bf16)
