// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_internal.h"

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_16x16x2_x86_64_avx512_vnni(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  __m512i acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_loadu_si512((__m512i*)(out_ptr + i * 16));
    }
  } else {
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_setzero_si512();
    }
  }

  for (int k = 0; k < params->K; ++k) {
    __m512i rhs =
        _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    rhs_ptr += 32;
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_dpwssd_epi32(acc[i], rhs,
                                   _mm512_cvtepi8_epi16(_mm256_set1_epi16(
                                       *(const iree_uk_int16_t*)(lhs_ptr))));
      lhs_ptr += 2;
    }
  }

  IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
    _mm512_storeu_si512((__m512i*)(out_ptr + i * 16), acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_x86_64_avx512_vnni, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s8s8s32_2x16x2_x86_64_avx512_vnni, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s8s8s32_4x16x2_x86_64_avx512_vnni, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s8s8s32_8x16x2_x86_64_avx512_vnni, 8)

void iree_uk_mmt4d_tile_s8s8s32_16x16x2_x86_64_avx512_vnni(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // acc[i][0] contains the 1st 128bits of row i, the 2nd 128bits of row (i+4),
  //           the 3rd 128bits of row (i+8), the 4th 128bits of row (i+C).
  // The other acc[i][j] are permutations of these 128bits groups.
  // This unusual layout is chosen so that the inner arithmetic loop only needs
  // to perform cheap shuffles within 128bit groups of lanes.
  __m512i acc[4][4];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    IREE_UK_UNROLL for (int i = 0; i < 4; ++i) {
      IREE_UK_UNROLL for (int j = 0; j < 4; ++j) {
        acc[i][j] = iree_uk_avx512_loadu_4x128_from_16x16xi32(
            out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
            4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4));
      }
    }
  } else {
    IREE_UK_UNROLL for (int i = 0; i < 4; ++i) {
      IREE_UK_UNROLL for (int j = 0; j < 4; ++j) {
        acc[i][j] = _mm512_setzero_si512();
      }
    }
  }

  __m512i idx_45670123CDEF89AB =
      _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
  __m512i idx_89ABCDEF01234567 =
      _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
  __m512i idx_CDEF89AB45670123 =
      _mm512_setr_epi32(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);

  for (int k = 0; k < params->K; ++k) {
    __m512i rhs_i16_perm[4];
    // rhs_i16_perm[0] is the rhs tile (2x8), sign-extended to i16.
    rhs_i16_perm[0] =
        _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    rhs_ptr += 32;
    // The other 3 rhs_i16_perm[i] are permutations of 128-bit groups of that.
    rhs_i16_perm[1] =
        _mm512_permutexvar_epi32(idx_45670123CDEF89AB, rhs_i16_perm[0]);
    rhs_i16_perm[2] =
        _mm512_permutexvar_epi32(idx_89ABCDEF01234567, rhs_i16_perm[0]);
    rhs_i16_perm[3] =
        _mm512_permutexvar_epi32(idx_CDEF89AB45670123, rhs_i16_perm[0]);
    // lhs_i16 is the lhs tile (M0x2), sign-extended to i16.
    __m512i lhs_i16 =
        _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)lhs_ptr));
    lhs_ptr += 32;
    // lhs_i16_dup4[i] is lanes of lhs_i16 shuffled as:
    // (i, i, i, i, i+4, i+4, i+4, i+4, i+8, i+8, i+8, i+C, i+C, i+C, i+C).
    __m512i lhs_i16_dup4[4];
    lhs_i16_dup4[0] = _mm512_shuffle_epi32(lhs_i16, 0 * 0x55);
    lhs_i16_dup4[1] = _mm512_shuffle_epi32(lhs_i16, 1 * 0x55);
    lhs_i16_dup4[2] = _mm512_shuffle_epi32(lhs_i16, 2 * 0x55);
    lhs_i16_dup4[3] = _mm512_shuffle_epi32(lhs_i16, 3 * 0x55);
    IREE_UK_UNROLL for (int i = 0; i < 4; ++i) {
      IREE_UK_UNROLL for (int j = 0; j < 4; ++j) {
        acc[i][j] =
            _mm512_dpwssd_epi32(acc[i][j], lhs_i16_dup4[i], rhs_i16_perm[j]);
      }
    }
  }

  IREE_UK_UNROLL for (int i = 0; i < 4; ++i) {
    IREE_UK_UNROLL for (int j = 0; j < 4; ++j) {
      iree_uk_avx512_storeu_4x128_to_16x16xi32(
          out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8, 4 * ((j + 2) % 4),
          i + 12, 4 * ((7 - j) % 4), acc[i][j]);
    }
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_vnni(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  __m512i acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_loadu_si512((__m512i*)(out_ptr + i * 16));
    }
  } else {
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_setzero_si512();
    }
  }

  for (int k = 0; k < params->K; ++k) {
    __m512i rhs = _mm512_loadu_si512((const __m512i*)rhs_ptr);
    rhs_ptr += 32;
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_dpwssd_epi32(
          acc[i], rhs, _mm512_set1_epi32(*(const iree_uk_int32_t*)lhs_ptr));
      lhs_ptr += 2;
    }
  }

  IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
    _mm512_storeu_si512((__m512i*)(out_ptr + i * 16), acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_x86_64_avx512_vnni, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_2x16x2_x86_64_avx512_vnni, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_4x16x2_x86_64_avx512_vnni, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_8x16x2_x86_64_avx512_vnni, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_16x16x2_x86_64_avx512_vnni, 16)

// The idea of this kernel is to split the LHS s16 values into high and low
// 8-bit components to be able to use _mm512_dpbusd_epi32.
//
// In itself, that doesn't reduce the number of arithmetic instructions: while
// each now computes a 4D dot-product instead of a 2D one as in
// _mm512_dpwssd_epi32, we now need twice more of them to do separately the
// high and low 8bit parts of the LHS s16 values.
//
// The real benefit is that this removes the need to extend RHS u4 values to
// s16. Since this is a vecmat kernel, the LHS is small and the RHS is big,
// so it matters to avoid RHS-processing work.
//
// It's not trivial how to use _mm512_dpbusd_epi32, with its quirky
// unsigned * signed semantics. We take advantage of the fact that our u4
// RHS values, when extended to u8, do not use the top bit -- so they are
// also interpretable as s8 values in place. So this is specific to RHS
// being less-than-8-bit values (it's not specific beyond that to 4bit).
// Meanwhile, when we split the LHS s16 values into high and low 8bit components
// the high 8bits are signed s8 and the low 8bit are unsigned u8. So, for each
// of the combinations of operands that we have to feed _mm512_dpbusd_epi32, we
// manage to find an operand order that accomodates the instruction's
// requirements on signednesses.
void iree_uk_mmt4d_tile_s16u4s32_1x32x8_x86_64_avx512_vnni(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // Accumulator shape: 1x32xs32, in 2 registers, each 1x16xs32.
  __m512i acc0, acc1;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    acc0 = _mm512_loadu_si512((const __m512i*)(out_ptr + 16 * 0));
    acc1 = _mm512_loadu_si512((const __m512i*)(out_ptr + 16 * 1));
  } else {
    acc0 = _mm512_setzero_si512();
    acc1 = _mm512_setzero_si512();
  }
  // Additional internal accumulators - acc{i}{j} will be folded into acc{i} at
  // the end of the loop.
  __m512i acc01 = _mm512_setzero_si512();
  __m512i acc02 = _mm512_setzero_si512();
  __m512i acc03 = _mm512_setzero_si512();
  __m512i acc11 = _mm512_setzero_si512();
  __m512i acc12 = _mm512_setzero_si512();
  __m512i acc13 = _mm512_setzero_si512();
  // Shuffle indices.
  const __m128i idx_0_mod_4 = _mm_set1_epi32(0x0c080400);
  const __m128i idx_1_mod_4 = _mm_set1_epi32(0x0d090501);
  const __m128i idx_2_mod_4 = _mm_set1_epi32(0x0e0a0602);
  const __m128i idx_3_mod_4 = _mm_set1_epi32(0x0f0b0703);
  const __m512i mask_0f = _mm512_set1_epi8(0x0f);
  for (int k = 0; k < params->K; ++k) {
    // Load 8xs16 LHS data.
    __m128i lhs = _mm_loadu_si128((const __m128i*)lhs_ptr);
    lhs_ptr += 8;
    // Extract the even/odd s16 lanes and within them, the low/high 8bit parts,
    // and broadcast into 512bit registers to multiply against RHS data.
    __m512i lhs_even_s16_low_u8 =
        _mm512_broadcastq_epi64(_mm_shuffle_epi8(lhs, idx_0_mod_4));
    __m512i lhs_even_s16_high_s8 =
        _mm512_broadcastq_epi64(_mm_shuffle_epi8(lhs, idx_1_mod_4));
    __m512i lhs_odd_s16_low_u8 =
        _mm512_broadcastq_epi64(_mm_shuffle_epi8(lhs, idx_2_mod_4));
    __m512i lhs_odd_s16_high_s8 =
        _mm512_broadcastq_epi64(_mm_shuffle_epi8(lhs, idx_3_mod_4));
    // Load 8x32xu4 RHS data, in 2 registers, each 8x16xu4.
    __m512i rhs0 = _mm512_loadu_si512((const __m512i*)(rhs_ptr + 64 * 0));
    __m512i rhs1 = _mm512_loadu_si512((const __m512i*)(rhs_ptr + 64 * 1));
    rhs_ptr += 128;
    // Extract the even/odd u4 lanes.
    __m512i rhs0_even_u4 = _mm512_and_si512(mask_0f, rhs0);
    __m512i rhs1_even_u4 = _mm512_and_si512(mask_0f, rhs1);
    __m512i rhs0_odd_u4 = _mm512_and_si512(mask_0f, _mm512_srli_epi16(rhs0, 4));
    __m512i rhs1_odd_u4 = _mm512_and_si512(mask_0f, _mm512_srli_epi16(rhs1, 4));
    // Arithmetic. See the comment at the top of this kernel for an explanation.
    // _mm512_dpbusd_epi32 takes an unsigned LHS and a signed RHS. The parameter
    // order in each call is adapted to that constraint.
    acc0 = _mm512_dpbusd_epi32(acc0, lhs_even_s16_low_u8, rhs0_even_u4);
    acc01 = _mm512_dpbusd_epi32(acc01, rhs0_even_u4, lhs_even_s16_high_s8);
    acc02 = _mm512_dpbusd_epi32(acc02, lhs_odd_s16_low_u8, rhs0_odd_u4);
    acc03 = _mm512_dpbusd_epi32(acc03, rhs0_odd_u4, lhs_odd_s16_high_s8);
    acc1 = _mm512_dpbusd_epi32(acc1, lhs_even_s16_low_u8, rhs1_even_u4);
    acc11 = _mm512_dpbusd_epi32(acc11, rhs1_even_u4, lhs_even_s16_high_s8);
    acc12 = _mm512_dpbusd_epi32(acc12, lhs_odd_s16_low_u8, rhs1_odd_u4);
    acc13 = _mm512_dpbusd_epi32(acc13, rhs1_odd_u4, lhs_odd_s16_high_s8);
  }

  // The accumulators that contain products against high 8bit parts of s16 LHS
  // values need to be left-shifted by 8 bits to account for that.
  acc01 = _mm512_slli_epi32(acc01, 8);
  acc03 = _mm512_slli_epi32(acc03, 8);
  acc11 = _mm512_slli_epi32(acc11, 8);
  acc13 = _mm512_slli_epi32(acc13, 8);

  // Add accumulators together.
  acc0 = _mm512_add_epi32(acc0, acc01);
  acc1 = _mm512_add_epi32(acc1, acc11);
  acc0 = _mm512_add_epi32(acc0, acc02);
  acc1 = _mm512_add_epi32(acc1, acc12);
  acc0 = _mm512_add_epi32(acc0, acc03);
  acc1 = _mm512_add_epi32(acc1, acc13);

  // Store.
  _mm512_storeu_si512((__m512i*)(out_ptr + 16 * 0), acc0);
  _mm512_storeu_si512((__m512i*)(out_ptr + 16 * 1), acc1);
}
