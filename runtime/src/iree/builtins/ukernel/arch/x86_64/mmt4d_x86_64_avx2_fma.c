// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_internal.h"

static inline void iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_x86_64_avx2_fma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  __m256 acc[8];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0; ++i) {
      acc[i] = _mm256_loadu_ps(out_ptr + i * 8);
    }
  } else {
    for (int i = 0; i < M0; ++i) {
      acc[i] = _mm256_setzero_ps();
    }
  }
  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m256 rhs = _mm256_loadu_ps(rhs_ptr);
    rhs_ptr += 8;
    for (int i = 0; i < M0; ++i) {
      acc[i] = _mm256_fmadd_ps(_mm256_broadcast_ss(lhs_ptr + i), rhs, acc[i]);
    }
    lhs_ptr += M0;
  }
  for (int i = 0; i < M0; ++i) {
    _mm256_storeu_ps(out_ptr + i * 8, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f32f32f32_2x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f32f32f32_4x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f32f32f32_8x8x1_x86_64_avx2_fma)

// Shared implementation for f16f16f16 and f16f16f32.
// In the f16f16f16 case, intermediate roundings are skipped. This function
// should only be used if IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS is set.
static inline void iree_uk_mmt4d_tile_f16f16fXX_1x8x1_to_8x8x1_x86_64_avx2_fma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, iree_uk_type_t acc_type, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  __m256 acc[8];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < M0; ++i) {
        acc[i] = _mm256_loadu_ps(out_ptr + i * 8);
      }
    } else {
      iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < M0; ++i) {
        acc[i] =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(out_ptr + i * 8)));
      }
    }
  } else {
    for (int i = 0; i < M0; ++i) {
      acc[i] = _mm256_setzero_ps();
    }
  }
  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m256 rhs = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)rhs_ptr));
    rhs_ptr += 8;
    for (int i = 0; i < M0; ++i) {
      acc[i] = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(lhs_ptr[i])), rhs,
                               acc[i]);
    }
    lhs_ptr += M0;
  }
  if (acc_type == IREE_UK_TYPE_FLOAT_32) {
    float* IREE_UK_RESTRICT out_ptr = out_tile;
    for (int i = 0; i < M0; ++i) {
      _mm256_storeu_ps(out_ptr + i * 8, acc[i]);
    }
  } else {
    iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
    for (int i = 0; i < M0; ++i) {
      _mm_storeu_si128((__m128i*)(out_ptr + i * 8),
                       _mm256_cvtps_ph(acc[i], _MM_FROUND_TO_NEAREST_INT));
    }
  }
}

static inline void iree_uk_mmt4d_tile_f16f16f32_1x8x1_to_8x8x1_x86_64_avx2_fma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1x8x1_to_8x8x1_x86_64_avx2_fma(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_32, M0);
}

static inline void iree_uk_mmt4d_tile_f16f16f16_1x8x1_to_8x8x1_x86_64_avx2_fma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1x8x1_to_8x8x1_x86_64_avx2_fma(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_16, M0);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f16f16f32_1x8x1_to_8x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f16f16f32_1x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f16f16f32_2x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f16f16f32_4x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f16f16f32_8x8x1_x86_64_avx2_fma)

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f16f16f16_1x8x1_to_8x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f16f16f16_1x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f16f16f16_2x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f16f16f16_4x8x1_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_f16f16f16_8x8x1_x86_64_avx2_fma)

static inline void iree_uk_mmt4d_tile_i8i8i32_1x8x2_to_8x8x2_x86_64_avx2_fma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 18 && iree_uk_is_po2_u32(M0));
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // acc[i][0] contains the 1st half of row i and the 2nd half of row (i+4).
  // acc[i][1] contains the 2nd half of row i and the 1st half of row (i+4).
  // This unusual layout is chosen so that the inner arithmetic loop only needs
  // to perform cheap shuffles within 128bit groups of lanes.
  __m256i acc[4][2];
  const int imax = M0 <= 4 ? M0 : 4;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (M0 <= 4) {
          acc[i][j] = _mm256_castsi128_si256(
              _mm_loadu_si128((__m128i*)(out_ptr + i * 8 + j * 4)));
        } else {
          acc[i][j] = iree_uk_avx_loadu_2x128(
              (__m128i*)(out_ptr + i * 8 + j * 4),
              (__m128i*)(out_ptr + (i + 4) * 8 + (1 - j) * 4));
        }
      }
    }
  } else {
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 2; ++j) {
        acc[i][j] = _mm256_setzero_si256();
      }
    }
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m256i rhs_i16_perm[2];
    // rhs_i16_perm[0] is the rhs tile (2x8), sign-extended to i16.
    rhs_i16_perm[0] =
        _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)rhs_ptr));
    rhs_ptr += 16;
    // rhs_i16_perm[1] is that with the halves swapped.
    rhs_i16_perm[1] =
        _mm256_permute2x128_si256(rhs_i16_perm[0], rhs_i16_perm[0], 0x01);
    // lhs_i16 is the lhs tile (M0x2), sign-extended to i16.
    __m256i lhs_i16;
    if (M0 == 1) {
      lhs_i16 = _mm256_cvtepi8_epi16(_mm_loadu_si16(lhs_ptr));
      lhs_ptr += 2;
    } else if (M0 == 2) {
      lhs_i16 = _mm256_cvtepi8_epi16(_mm_loadu_si32(lhs_ptr));
      lhs_ptr += 4;
    } else if (M0 == 4) {
      lhs_i16 = _mm256_cvtepi8_epi16(_mm_loadu_si64(lhs_ptr));
      lhs_ptr += 8;
    } else {
      lhs_i16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)lhs_ptr));
      lhs_ptr += 16;
    }
    // lhs_i16_dup4[i] is lanes of lhs_i16 shuffled as:
    // (i, i, i, i, i+4, i+4, i+4, i+4).
    __m256i lhs_i16_dup4[4];
    if (M0 >= 1) lhs_i16_dup4[0] = _mm256_shuffle_epi32(lhs_i16, 0 * 0x55);
    if (M0 >= 2) lhs_i16_dup4[1] = _mm256_shuffle_epi32(lhs_i16, 1 * 0x55);
    if (M0 >= 4) lhs_i16_dup4[2] = _mm256_shuffle_epi32(lhs_i16, 2 * 0x55);
    if (M0 >= 4) lhs_i16_dup4[3] = _mm256_shuffle_epi32(lhs_i16, 3 * 0x55);
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 2; ++j) {
        acc[i][j] = _mm256_add_epi32(
            acc[i][j], _mm256_madd_epi16(lhs_i16_dup4[i], rhs_i16_perm[j]));
      }
    }
  }

  for (int i = 0; i < imax; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (M0 <= 4) {
        _mm_storeu_si128((__m128i*)(out_ptr + i * 8 + j * 4),
                         _mm256_extracti128_si256(acc[i][j], 0));
      } else {
        iree_uk_avx_storeu_2x128(
            (__m128i*)(out_ptr + i * 8 + j * 4),
            (__m128i*)(out_ptr + (i + 4) * 8 + (1 - j) * 4), acc[i][j]);
      }
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_i8i8i32_1x8x2_to_8x8x2_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_i8i8i32_1x8x2_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_i8i8i32_2x8x2_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_i8i8i32_4x8x2_x86_64_avx2_fma,
    iree_uk_mmt4d_tile_i8i8i32_8x8x2_x86_64_avx2_fma)
