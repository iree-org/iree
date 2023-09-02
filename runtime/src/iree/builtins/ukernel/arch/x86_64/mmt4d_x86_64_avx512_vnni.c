// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_internal.h"

void iree_uk_mmt4d_tile_i8i8i32_16x16x2_x86_64_avx512_vnni(
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
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = iree_uk_avx512_loadu_4x128_from_16x16xi32(
            out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
            4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4));
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
    // rhs_i16_perm[0] is the rhs row, sign-extended to i16.
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
    // lhs_i16 is the lhs column, sign-extended to i16.
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

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] =
            _mm512_dpwssd_epi32(acc[i][j], lhs_i16_dup4[i], rhs_i16_perm[j]);
      }
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      iree_uk_avx512_storeu_4x128_to_16x16xi32(
          out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8, 4 * ((j + 2) % 4),
          i + 12, 4 * ((7 - j) % 4), acc[i][j]);
    }
  }
}
