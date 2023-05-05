// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <immintrin.h>

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/mmt4d.h"

void iree_uk_mmt4d_tile_f32f32f32_8x8x1_x86_64_avx2_fma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel, iree_uk_int32_t K,
    iree_uk_uint32_t flags, const iree_uk_mmt4d_params_t* params) {
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  __m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
  if (flags & IREE_UK_FLAG_ACCUMULATE) {
    acc0 = _mm256_loadu_ps(out_ptr + 0 * 8);
    acc1 = _mm256_loadu_ps(out_ptr + 1 * 8);
    acc2 = _mm256_loadu_ps(out_ptr + 2 * 8);
    acc3 = _mm256_loadu_ps(out_ptr + 3 * 8);
    acc4 = _mm256_loadu_ps(out_ptr + 4 * 8);
    acc5 = _mm256_loadu_ps(out_ptr + 5 * 8);
    acc6 = _mm256_loadu_ps(out_ptr + 6 * 8);
    acc7 = _mm256_loadu_ps(out_ptr + 7 * 8);
  } else {
    acc0 = _mm256_setzero_ps();
    acc1 = _mm256_setzero_ps();
    acc2 = _mm256_setzero_ps();
    acc3 = _mm256_setzero_ps();
    acc4 = _mm256_setzero_ps();
    acc5 = _mm256_setzero_ps();
    acc6 = _mm256_setzero_ps();
    acc7 = _mm256_setzero_ps();
  }
  for (iree_uk_int32_t k = 0; k < K; ++k) {
    __m256 rhs = _mm256_loadu_ps(rhs_ptr);
    rhs_ptr += 8;
    acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(lhs_ptr + 0), rhs, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(lhs_ptr + 1), rhs, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(lhs_ptr + 2), rhs, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(lhs_ptr + 3), rhs, acc3);
    acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(lhs_ptr + 4), rhs, acc4);
    acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(lhs_ptr + 5), rhs, acc5);
    acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(lhs_ptr + 6), rhs, acc6);
    acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(lhs_ptr + 7), rhs, acc7);
    lhs_ptr += 8;
  }
  _mm256_storeu_ps(out_ptr + 0 * 8, acc0);
  _mm256_storeu_ps(out_ptr + 1 * 8, acc1);
  _mm256_storeu_ps(out_ptr + 2 * 8, acc2);
  _mm256_storeu_ps(out_ptr + 3 * 8, acc3);
  _mm256_storeu_ps(out_ptr + 4 * 8, acc4);
  _mm256_storeu_ps(out_ptr + 5 * 8, acc5);
  _mm256_storeu_ps(out_ptr + 6 * 8, acc6);
  _mm256_storeu_ps(out_ptr + 7 * 8, acc7);
}

void iree_uk_mmt4d_tile_i8i8i32_8x8x2_x86_64_avx2_fma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel, iree_uk_int32_t K,
    iree_uk_uint32_t flags, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  __m256i acc_0_0123_4_4567;
  __m256i acc_0_4567_4_0123;
  __m256i acc_1_0123_5_4567;
  __m256i acc_1_4567_5_0123;
  __m256i acc_2_0123_6_4567;
  __m256i acc_2_4567_6_0123;
  __m256i acc_3_0123_7_4567;
  __m256i acc_3_4567_7_0123;

  if (flags & IREE_UK_FLAG_ACCUMULATE) {
    acc_0_0123_4_4567 = iree_uk_avx_loadu_2x128(
        (__m128i*)(out_ptr + 0 * 8 + 0), (__m128i*)(out_ptr + 4 * 8 + 4));
    acc_0_4567_4_0123 = iree_uk_avx_loadu_2x128(
        (__m128i*)(out_ptr + 0 * 8 + 4), (__m128i*)(out_ptr + 4 * 8 + 0));
    acc_1_0123_5_4567 = iree_uk_avx_loadu_2x128(
        (__m128i*)(out_ptr + 1 * 8 + 0), (__m128i*)(out_ptr + 5 * 8 + 4));
    acc_1_4567_5_0123 = iree_uk_avx_loadu_2x128(
        (__m128i*)(out_ptr + 1 * 8 + 4), (__m128i*)(out_ptr + 5 * 8 + 0));
    acc_2_0123_6_4567 = iree_uk_avx_loadu_2x128(
        (__m128i*)(out_ptr + 2 * 8 + 0), (__m128i*)(out_ptr + 6 * 8 + 4));
    acc_2_4567_6_0123 = iree_uk_avx_loadu_2x128(
        (__m128i*)(out_ptr + 2 * 8 + 4), (__m128i*)(out_ptr + 6 * 8 + 0));
    acc_3_0123_7_4567 = iree_uk_avx_loadu_2x128(
        (__m128i*)(out_ptr + 3 * 8 + 0), (__m128i*)(out_ptr + 7 * 8 + 4));
    acc_3_4567_7_0123 = iree_uk_avx_loadu_2x128(
        (__m128i*)(out_ptr + 3 * 8 + 4), (__m128i*)(out_ptr + 7 * 8 + 0));
  } else {
    acc_0_0123_4_4567 = _mm256_setzero_si256();
    acc_0_4567_4_0123 = _mm256_setzero_si256();
    acc_1_0123_5_4567 = _mm256_setzero_si256();
    acc_1_4567_5_0123 = _mm256_setzero_si256();
    acc_2_0123_6_4567 = _mm256_setzero_si256();
    acc_2_4567_6_0123 = _mm256_setzero_si256();
    acc_3_0123_7_4567 = _mm256_setzero_si256();
    acc_3_4567_7_0123 = _mm256_setzero_si256();
  }
  for (iree_uk_int32_t k = 0; k < K; ++k) {
    __m256i rhs_i16_01234567 =
        _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)rhs_ptr));
    rhs_ptr += 16;
    __m256i lhs_i16_01234567 =
        _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)lhs_ptr));
    lhs_ptr += 16;
    __m256i rhs_i16_45670123 =
        _mm256_permute2x128_si256(rhs_i16_01234567, rhs_i16_01234567, 0x01);
    __m256i lhs_i16_00004444 = _mm256_shuffle_epi32(lhs_i16_01234567, 0 * 0x55);
    __m256i lhs_i16_11115555 = _mm256_shuffle_epi32(lhs_i16_01234567, 1 * 0x55);
    __m256i lhs_i16_22226666 = _mm256_shuffle_epi32(lhs_i16_01234567, 2 * 0x55);
    __m256i lhs_i16_33337777 = _mm256_shuffle_epi32(lhs_i16_01234567, 3 * 0x55);

    acc_0_0123_4_4567 =
        _mm256_add_epi32(acc_0_0123_4_4567,
                         _mm256_madd_epi16(lhs_i16_00004444, rhs_i16_01234567));
    acc_0_4567_4_0123 =
        _mm256_add_epi32(acc_0_4567_4_0123,
                         _mm256_madd_epi16(lhs_i16_00004444, rhs_i16_45670123));
    acc_1_0123_5_4567 =
        _mm256_add_epi32(acc_1_0123_5_4567,
                         _mm256_madd_epi16(lhs_i16_11115555, rhs_i16_01234567));
    acc_1_4567_5_0123 =
        _mm256_add_epi32(acc_1_4567_5_0123,
                         _mm256_madd_epi16(lhs_i16_11115555, rhs_i16_45670123));
    acc_2_0123_6_4567 =
        _mm256_add_epi32(acc_2_0123_6_4567,
                         _mm256_madd_epi16(lhs_i16_22226666, rhs_i16_01234567));
    acc_2_4567_6_0123 =
        _mm256_add_epi32(acc_2_4567_6_0123,
                         _mm256_madd_epi16(lhs_i16_22226666, rhs_i16_45670123));
    acc_3_0123_7_4567 =
        _mm256_add_epi32(acc_3_0123_7_4567,
                         _mm256_madd_epi16(lhs_i16_33337777, rhs_i16_01234567));
    acc_3_4567_7_0123 =
        _mm256_add_epi32(acc_3_4567_7_0123,
                         _mm256_madd_epi16(lhs_i16_33337777, rhs_i16_45670123));
  }
  iree_uk_avx_storeu_2x128((__m128i*)(out_ptr + 0 * 8 + 0),
                           (__m128i*)(out_ptr + 4 * 8 + 4), acc_0_0123_4_4567);
  iree_uk_avx_storeu_2x128((__m128i*)(out_ptr + 0 * 8 + 4),
                           (__m128i*)(out_ptr + 4 * 8 + 0), acc_0_4567_4_0123);
  iree_uk_avx_storeu_2x128((__m128i*)(out_ptr + 1 * 8 + 0),
                           (__m128i*)(out_ptr + 5 * 8 + 4), acc_1_0123_5_4567);
  iree_uk_avx_storeu_2x128((__m128i*)(out_ptr + 1 * 8 + 4),
                           (__m128i*)(out_ptr + 5 * 8 + 0), acc_1_4567_5_0123);
  iree_uk_avx_storeu_2x128((__m128i*)(out_ptr + 2 * 8 + 0),
                           (__m128i*)(out_ptr + 6 * 8 + 4), acc_2_0123_6_4567);
  iree_uk_avx_storeu_2x128((__m128i*)(out_ptr + 2 * 8 + 4),
                           (__m128i*)(out_ptr + 6 * 8 + 0), acc_2_4567_6_0123);
  iree_uk_avx_storeu_2x128((__m128i*)(out_ptr + 3 * 8 + 0),
                           (__m128i*)(out_ptr + 7 * 8 + 4), acc_3_0123_7_4567);
  iree_uk_avx_storeu_2x128((__m128i*)(out_ptr + 3 * 8 + 4),
                           (__m128i*)(out_ptr + 7 * 8 + 0), acc_3_4567_7_0123);
}
