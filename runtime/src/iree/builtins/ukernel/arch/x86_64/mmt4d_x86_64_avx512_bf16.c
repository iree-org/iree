// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_internal.h"

#if defined(IREE_UK_COMPILER_CLANG) && !defined(IREE_UK_COMPILER_MSVC)
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
#endif  // IREE_UK_COMPILER_CLANG

static inline void
iree_uk_mmt4d_tile_bf16bf16fXX_1x16x2_to_16x16x2_x86_64_avx512_bf16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, iree_uk_type_t acc_type, int M0) {
  IREE_UK_ASSERT(acc_type == IREE_UK_TYPE_FLOAT_32 ||
                 acc_type == IREE_UK_TYPE_BFLOAT_16);
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  __m512 acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < M0; ++i) {
        acc[i] = _mm512_loadu_ps(out_ptr + i * 16);
      }
    } else {
      iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < M0; ++i) {
        __m256i loaded = _mm256_loadu_si256((const __m256i*)(out_ptr + i * 16));
        acc[i] = _mm512_cvtpbh_ps(*(const __m256bh*)&loaded);
      }
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
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < M0; ++i) {
        _mm512_storeu_ps(out_ptr + i * 16, acc[i]);
      }
    } else {
      iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < M0; ++i) {
        __m256bh converted = _mm512_cvtneps_pbh(acc[i]);
        _mm256_storeu_si256((__m256i*)(out_ptr + i * 16),
                            *(const __m256i*)&converted);
      }
    }
  }
}

static inline void
iree_uk_mmt4d_tile_bf16bf16f32_1x16x2_to_16x16x2_x86_64_avx512_bf16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_bf16bf16fXX_1x16x2_to_16x16x2_x86_64_avx512_bf16(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_32, M0);
}

static inline void
iree_uk_mmt4d_tile_bf16bf16bf16_1x16x2_to_16x16x2_x86_64_avx512_bf16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_bf16bf16fXX_1x16x2_to_16x16x2_x86_64_avx512_bf16(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_BFLOAT_16, M0);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8_16(
    iree_uk_mmt4d_tile_bf16bf16f32_1x16x2_to_16x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_1x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_2x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_4x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_8x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_16x16x2_x86_64_avx512_bf16)

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8_16(
    iree_uk_mmt4d_tile_bf16bf16bf16_1x16x2_to_16x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16bf16_1x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16bf16_2x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16bf16_4x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16bf16_8x16x2_x86_64_avx512_bf16,
    iree_uk_mmt4d_tile_bf16bf16bf16_16x16x2_x86_64_avx512_bf16)
