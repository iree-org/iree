// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_internal.h"

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f32f32f32_1x16x1_to_16x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are motivated by benchmarking on
  // Skylake; their effect was a > 1.3x speedup on 1024x1024 matmuls. The
  // prefetch-ahead offset of 128*sizeof(float) in the loop was empirically
  // determined. Similar prefetches did not produce any benefit in other
  // kernels, even though they are very similar to this one.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_loadu_ps(out_ptr + i * 16);
    }
  } else {
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_setzero_ps();
    }
  }

  for (int k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_loadu_ps(rhs_ptr);
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_fmadd_ps(rhs, _mm512_set1_ps(lhs_ptr[i]), acc[i]);
    }
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += M0;
  }

  IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
    _mm512_storeu_ps(out_ptr + i * 16, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f32f32f32_1x16x1_x86_64_avx512_base, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f32f32f32_2x16x1_x86_64_avx512_base, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f32f32f32_4x16x1_x86_64_avx512_base, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f32f32f32_8x16x1_x86_64_avx512_base, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base, 16)

// Shared implementation for f16f16f16 and f16f16f32.
// In the f16f16f16 case, intermediate roundings are skipped. This function
// should only be used if IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS is set.
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f16f16fXX_1x16x1_to_16x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, iree_uk_type_t acc_type, int M0) {
  IREE_UK_ASSERT(acc_type == IREE_UK_TYPE_FLOAT_32 ||
                 acc_type == IREE_UK_TYPE_FLOAT_16);
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
        acc[i] = _mm512_loadu_ps(out_ptr + i * 16);
      }
    } else {
      iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
      IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
        acc[i] = _mm512_cvtph_ps(
            _mm256_loadu_si256((const __m256i*)(out_ptr + i * 16)));
      }
    }
  } else {
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_setzero_ps();
    }
  }

  for (int k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      acc[i] = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[i])),
                               rhs, acc[i]);
    }
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += M0;
  }
  if (acc_type == IREE_UK_TYPE_FLOAT_32) {
    float* IREE_UK_RESTRICT out_ptr = out_tile;
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      _mm512_storeu_ps(out_ptr + i * 16, acc[i]);
    }
  } else {
    iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
    IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
      _mm256_storeu_si256((__m256i*)(out_ptr + i * 16),
                          _mm512_cvtps_ph(acc[i], _MM_FROUND_TO_NEAREST_INT));
    }
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE
static inline void
iree_uk_mmt4d_tile_f16f16f32_1x16x1_to_16x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1x16x1_to_16x16x1_x86_64_avx512_base(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_32, M0);
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE
static inline void
iree_uk_mmt4d_tile_f16f16f16_1x16x1_to_16x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1x16x1_to_16x16x1_x86_64_avx512_base(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_16, M0);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f32_1x16x1_x86_64_avx512_base, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f32_2x16x1_x86_64_avx512_base, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f32_4x16x1_x86_64_avx512_base, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f32_8x16x1_x86_64_avx512_base, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f32_16x16x1_x86_64_avx512_base, 16)

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f16_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f16_1x16x1_x86_64_avx512_base, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f16_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f16_2x16x1_x86_64_avx512_base, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f16_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f16_4x16x1_x86_64_avx512_base, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f16_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f16_8x16x1_x86_64_avx512_base, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f16_1x16x1_to_16x16x1_x86_64_avx512_base,
    iree_uk_mmt4d_tile_f16f16f16_16x16x1_x86_64_avx512_base, 16)

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_8x16x2_x86_64_avx512_base(
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
      acc[i] = _mm512_add_epi32(
          acc[i], _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm256_set1_epi16(
                                        *(const iree_uk_int16_t*)(lhs_ptr))),
                                    rhs));
      lhs_ptr += 2;
    }
  }

  IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
    _mm512_storeu_si512((__m512i*)(out_ptr + i * 16), acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_8x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_x86_64_avx512_base, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_8x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s8s8s32_2x16x2_x86_64_avx512_base, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_8x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s8s8s32_4x16x2_x86_64_avx512_base, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_8x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s8s8s32_8x16x2_x86_64_avx512_base, 8)

void iree_uk_mmt4d_tile_s8s8s32_16x16x2_x86_64_avx512_base(
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
        acc[i][j] = _mm512_add_epi32(
            acc[i][j], _mm512_madd_epi16(lhs_i16_dup4[i], rhs_i16_perm[j]));
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
iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_base(
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
      acc[i] = _mm512_add_epi32(
          acc[i],
          _mm512_madd_epi16(_mm512_set1_epi32(*(const iree_uk_int32_t*)lhs_ptr),
                            rhs));
      lhs_ptr += 2;
    }
  }

  IREE_UK_UNROLL for (int i = 0; i < M0; ++i) {
    _mm512_storeu_si512((__m512i*)(out_ptr + i * 16), acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_x86_64_avx512_base, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_2x16x2_x86_64_avx512_base, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_4x16x2_x86_64_avx512_base, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_8x16x2_x86_64_avx512_base, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_16x16x2_x86_64_avx512_base, 16)
