// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_internal.h"

static inline void
iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_16x16x2_x86_64_avx512_vnni(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // acc[i][0] contains the 1st 128bits of row i, the 2nd 128bits of row (i+4),
  //           the 3rd 128bits of row (i+8), the 4th 128bits of row (i+C).
  // The other acc[i][j] are permutations of these 128bits groups.
  // This unusual layout is chosen so that the inner arithmetic loop only needs
  // to perform cheap shuffles within 128bit groups of lanes.
  __m512i acc[4][4];
  const int imax = M0 <= 4 ? M0 : 4;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (M0 <= 8) {
          acc[i][j] = _mm512_castsi128_si512(
              _mm_loadu_si128((__m128i*)(out_ptr + i * 16 + j * 4)));
          if (M0 > 4) {
            acc[i][j] = _mm512_inserti32x4(
                acc[i][j],
                _mm_loadu_si128(
                    (__m128i*)(out_ptr + (i + 4) * 16 + ((5 - j) % 4) * 4)),
                1);
          }
        } else {
          acc[i][j] = iree_uk_avx512_loadu_4x128_from_16x16xi32(
              out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
              4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4));
        }
      }
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
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

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
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
    __m512i lhs_i16;
    if (M0 == 1) {
      lhs_i16 =
          _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_loadu_si16(lhs_ptr)));
      lhs_ptr += 2;
    } else if (M0 == 2) {
      lhs_i16 =
          _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_loadu_si32(lhs_ptr)));
      lhs_ptr += 4;
    } else if (M0 == 4) {
      lhs_i16 =
          _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_loadu_si64(lhs_ptr)));
      lhs_ptr += 8;
    } else if (M0 == 8) {
      lhs_i16 = _mm512_castsi256_si512(
          _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)lhs_ptr)));
      lhs_ptr += 16;
    } else {
      lhs_i16 =
          _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)lhs_ptr));
      lhs_ptr += 32;
    }
    // lhs_i16_dup4[i] is lanes of lhs_i16 shuffled as:
    // (i, i, i, i, i+4, i+4, i+4, i+4, i+8, i+8, i+8, i+C, i+C, i+C, i+C).
    __m512i lhs_i16_dup4[4];
    if (M0 >= 1) lhs_i16_dup4[0] = _mm512_shuffle_epi32(lhs_i16, 0 * 0x55);
    if (M0 >= 2) lhs_i16_dup4[1] = _mm512_shuffle_epi32(lhs_i16, 1 * 0x55);
    if (M0 >= 4) lhs_i16_dup4[2] = _mm512_shuffle_epi32(lhs_i16, 2 * 0x55);
    if (M0 >= 4) lhs_i16_dup4[3] = _mm512_shuffle_epi32(lhs_i16, 3 * 0x55);
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] =
            _mm512_dpwssd_epi32(acc[i][j], lhs_i16_dup4[i], rhs_i16_perm[j]);
      }
    }
  }

  for (int i = 0; i < imax; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (M0 <= 8) {
        _mm_storeu_si128((__m128i*)(out_ptr + i * 16 + j * 4),
                         _mm512_extracti32x4_epi32(acc[i][j], 0));
        if (M0 > 4) {
          _mm_storeu_si128(
              (__m128i*)(out_ptr + (i + 4) * 16 + ((5 - j) % 4) * 4),
              _mm512_extracti32x4_epi32(acc[i][j], 1));
        }
      } else {
        iree_uk_avx512_storeu_4x128_to_16x16xi32(
            out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
            4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4), acc[i][j]);
      }
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8_16(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s8s8s32_2x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s8s8s32_4x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s8s8s32_8x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s8s8s32_16x16x2_x86_64_avx512_vnni)

static inline void
iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_vnni(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // acc[i][0] contains the 1st 128bits of row i, the 2nd 128bits of row (i+4),
  //           the 3rd 128bits of row (i+8), the 4th 128bits of row (i+C).
  // The other acc[i][j] are permutations of these 128bits groups.
  // This unusual layout is chosen so that the inner arithmetic loop only needs
  // to perform cheap shuffles within 128bit groups of lanes.
  __m512i acc[4][4];
  const int imax = M0 <= 4 ? M0 : 4;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (M0 <= 8) {
          acc[i][j] = _mm512_castsi128_si512(
              _mm_loadu_si128((__m128i*)(out_ptr + i * 16 + j * 4)));
          if (M0 > 4) {
            acc[i][j] = _mm512_inserti32x4(
                acc[i][j],
                _mm_loadu_si128(
                    (__m128i*)(out_ptr + (i + 4) * 16 + ((5 - j) % 4) * 4)),
                1);
          }
        } else {
          acc[i][j] = iree_uk_avx512_loadu_4x128_from_16x16xi32(
              out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
              4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4));
        }
      }
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
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

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512i rhs_perm[4];
    // rhs_perm[0] is the rhs tile (2x8).
    rhs_perm[0] = _mm512_loadu_si512((const __m512i*)rhs_ptr);
    rhs_ptr += 32;
    // The other 3 rhs_perm[i] are permutations of 128-bit groups of that.
    rhs_perm[1] = _mm512_permutexvar_epi32(idx_45670123CDEF89AB, rhs_perm[0]);
    rhs_perm[2] = _mm512_permutexvar_epi32(idx_89ABCDEF01234567, rhs_perm[0]);
    rhs_perm[3] = _mm512_permutexvar_epi32(idx_CDEF89AB45670123, rhs_perm[0]);
    // lhs is the lhs tile (M0x2).
    __m512i lhs;
    if (M0 == 1) {
      lhs = _mm512_castsi128_si512(_mm_loadu_si32(lhs_ptr));
      lhs_ptr += 2;
    } else if (M0 == 2) {
      lhs = _mm512_castsi128_si512(_mm_loadu_si64(lhs_ptr));
      lhs_ptr += 4;
    } else if (M0 == 4) {
      lhs = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)lhs_ptr));
      lhs_ptr += 8;
    } else if (M0 == 8) {
      lhs = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)lhs_ptr));
      lhs_ptr += 16;
    } else {
      lhs = _mm512_loadu_si512((const __m512i*)lhs_ptr);
      lhs_ptr += 32;
    }
    // lhs_dup4[i] is lanes of lhs shuffled as:
    // (i, i, i, i, i+4, i+4, i+4, i+4, i+8, i+8, i+8, i+C, i+C, i+C, i+C).
    __m512i lhs_dup4[4];
    if (M0 >= 1) lhs_dup4[0] = _mm512_shuffle_epi32(lhs, 0 * 0x55);
    if (M0 >= 2) lhs_dup4[1] = _mm512_shuffle_epi32(lhs, 1 * 0x55);
    if (M0 >= 4) lhs_dup4[2] = _mm512_shuffle_epi32(lhs, 2 * 0x55);
    if (M0 >= 4) lhs_dup4[3] = _mm512_shuffle_epi32(lhs, 3 * 0x55);
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = _mm512_dpwssd_epi32(acc[i][j], lhs_dup4[i], rhs_perm[j]);
      }
    }
  }

  for (int i = 0; i < imax; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (M0 <= 8) {
        _mm_storeu_si128((__m128i*)(out_ptr + i * 16 + j * 4),
                         _mm512_extracti32x4_epi32(acc[i][j], 0));
        if (M0 > 4) {
          _mm_storeu_si128(
              (__m128i*)(out_ptr + (i + 4) * 16 + ((5 - j) % 4) * 4),
              _mm512_extracti32x4_epi32(acc[i][j], 1));
        }
      } else {
        iree_uk_avx512_storeu_4x128_to_16x16xi32(
            out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
            4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4), acc[i][j]);
      }
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8_16(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_2x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_4x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_8x16x2_x86_64_avx512_vnni,
    iree_uk_mmt4d_tile_s16s16s32_16x16x2_x86_64_avx512_vnni)

// This kernel is parametrized in N0, allowing N0==16 and N0==32.
// Performance on AMD Ryzen 9 7950X3D:
//   - with N0=16:  180 Gop/s
//   - with N0=32:  240 Gop/s
// So there's a nice reward for going extra large, but that's also a liability
// for vecmat shapes whose N dimension isn't a multiple of 32. Maybe we can
// keep both for now.
//
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
// of the combinations of operands that we have to feed _mm512_dpbusd_epi32,
// we manage to find an operand order that accomodates the instruction's
// requirements on signednesses.
static inline void
iree_uk_mmt4d_tile_s16u4s32_1x16x8_to_1x32x8_x86_64_avx512_vnni(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int N0) {
  IREE_UK_ASSERT(N0 >= 16 && N0 <= 32 && iree_uk_is_po2_u32(N0));
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // acc[4 * i] is the actual accumulator.
  // The other acc[4 * i + j] are only used internally in the accumulation loop.
  __m512i acc[8];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < N0 / 16; ++i) {
      acc[4 * i] = _mm512_loadu_si512((const __m512i*)(out_ptr + 16 * i));
    }
  } else {
    for (int i = 0; i < N0 / 16; ++i) {
      acc[4 * i] = _mm512_setzero_si512();
    }
  }
  for (int i = 0; i < N0 / 16; ++i) {
    for (int j = 1; j < 4; ++j) {
      acc[4 * i + j] = _mm512_setzero_si512();
    }
  }

  const __m128i idx_0_mod_4 = _mm_set1_epi32(0x0c080400);
  const __m128i idx_1_mod_4 = _mm_set1_epi32(0x0d090501);
  const __m128i idx_2_mod_4 = _mm_set1_epi32(0x0e0a0602);
  const __m128i idx_3_mod_4 = _mm_set1_epi32(0x0f0b0703);
  const __m512i mask_0f = _mm512_set1_epi8(0x0f);
  IREE_UK_ASSUME(params->K >= 1);
  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
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
    // Load 8x16xu4 RHS data.
    __m512i rhs[2];
    for (int i = 0; i < N0 / 16; ++i) {
      rhs[i] = _mm512_loadu_si512((const __m512i*)(rhs_ptr + 64 * i));
    }
    rhs_ptr += N0 * 4;
    // Extract the even/odd u4 lanes.
    __m512i rhs_even_u4[2];
    __m512i rhs_odd_u4[2];
    for (int i = 0; i < N0 / 16; ++i) {
      rhs_even_u4[i] = _mm512_and_si512(mask_0f, rhs[i]);
      rhs_odd_u4[i] = _mm512_and_si512(mask_0f, _mm512_srli_epi16(rhs[i], 4));
    }
    // Arithmetic. See the comment at the top of this kernel for an explanation.
    // _mm512_dpbusd_epi32 takes an unsigned LHS and a signed RHS. The parameter
    // order in each call is adapted to that constraint.
    for (int i = 0; i < N0 / 16; ++i) {
      acc[4 * i + 0] = _mm512_dpbusd_epi32(acc[4 * i + 0], lhs_even_s16_low_u8,
                                           rhs_even_u4[i]);
      acc[4 * i + 1] = _mm512_dpbusd_epi32(acc[4 * i + 1], rhs_even_u4[i],
                                           lhs_even_s16_high_s8);
      acc[4 * i + 2] = _mm512_dpbusd_epi32(acc[4 * i + 2], lhs_odd_s16_low_u8,
                                           rhs_odd_u4[i]);
      acc[4 * i + 3] = _mm512_dpbusd_epi32(acc[4 * i + 3], rhs_odd_u4[i],
                                           lhs_odd_s16_high_s8);
    }
  }

  // The accumulators that contain products against high 8bit parts of s16 LHS
  // values need to be left-shifted by 8 bits to account for that.
  for (int i = 0; i < N0 / 16; ++i) {
    acc[4 * i + 1] = _mm512_slli_epi32(acc[4 * i + 1], 8);
    acc[4 * i + 3] = _mm512_slli_epi32(acc[4 * i + 3], 8);
  }

  // Add accumulators together.
  for (int i = 0; i < N0 / 16; ++i) {
    for (int j = 1; j <= 3; ++j) {
      acc[4 * i + 0] = _mm512_add_epi32(acc[4 * i + 0], acc[4 * i + j]);
    }
  }

  for (int i = 0; i < N0 / 16; ++i) {
    _mm512_storeu_si512((__m512i*)(out_ptr + 16 * i), acc[4 * i]);
  }
}

void iree_uk_mmt4d_tile_s16u4s32_1x16x8_x86_64_avx512_vnni(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  iree_uk_mmt4d_tile_s16u4s32_1x16x8_to_1x32x8_x86_64_avx512_vnni(
      out_tile, lhs_panel, rhs_panel, params, 16);
}

void iree_uk_mmt4d_tile_s16u4s32_1x32x8_x86_64_avx512_vnni(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  iree_uk_mmt4d_tile_s16u4s32_1x16x8_to_1x32x8_x86_64_avx512_vnni(
      out_tile, lhs_panel, rhs_panel, params, 32);
}
