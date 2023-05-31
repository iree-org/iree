// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_X86_64_COMMON_X86_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_X86_64_COMMON_X86_64_H_

#include <immintrin.h>

#include "iree/builtins/ukernel/common.h"
#include "iree/schemas/cpu_data.h"

// We default to requiring Clang>=8 and GCC>=9 because that's what Ruy used to
// do; in my memory it's not just about what's officially supported but also
// about skipping over quirky support in earlier releases.
#if IREE_UK_COMPILER_CLANG_VERSION_AT_LEAST(8, 0) || \
    IREE_UK_COMPILER_GCC_VERSION_AT_LEAST(9, 0) ||   \
    IREE_UK_COMPILER_MSVC_VERSION_AT_LEAST(1910)  // MSVC 2017
#define IREE_UK_BUILD_X86_64_AVX2_FMA
static inline bool iree_uk_cpu_supports_avx2_fma(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(
      cpu_data[0], IREE_CPU_DATA0_X86_64_AVX2 | IREE_CPU_DATA0_X86_64_FMA);
}
#endif

#if IREE_UK_COMPILER_CLANG_VERSION_AT_LEAST(8, 0) || \
    IREE_UK_COMPILER_GCC_VERSION_AT_LEAST(9, 0) ||   \
    IREE_UK_COMPILER_MSVC_VERSION_AT_LEAST(1920)  // MSVC 2019
#define IREE_UK_BUILD_X86_64_AVX512_BASE
static inline bool iree_uk_cpu_supports_avx512_base(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_X86_64_AVX512F |
                                               IREE_CPU_DATA0_X86_64_AVX512BW |
                                               IREE_CPU_DATA0_X86_64_AVX512DQ |
                                               IREE_CPU_DATA0_X86_64_AVX512VL |
                                               IREE_CPU_DATA0_X86_64_AVX512CD);
}
#endif

// GCC 9 introduced AVX512VNNI: https://gcc.gnu.org/gcc-9/changes.html
#if IREE_UK_COMPILER_CLANG_VERSION_AT_LEAST(8, 0) || \
    IREE_UK_COMPILER_GCC_VERSION_AT_LEAST(9, 0) ||   \
    IREE_UK_COMPILER_MSVC_VERSION_AT_LEAST(1930)  // MSVC 2022
#define IREE_UK_BUILD_X86_64_AVX512_VNNI
static inline bool iree_uk_cpu_supports_avx512_vnni(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_cpu_supports_avx512_base(cpu_data) &&
         iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_X86_64_AVX512VNNI);
}
#endif

#if defined(__AVX2__)

static inline __m256i iree_uk_avx_loadu_2x128(const void* src0,
                                              const void* src1) {
  __m128i v128_0 = _mm_loadu_si128((const __m128i*)src0);
  __m128i v128_1 = _mm_loadu_si128((const __m128i*)src1);
  return _mm256_inserti128_si256(_mm256_castsi128_si256(v128_0), v128_1, 1);
}

static inline void iree_uk_avx_storeu_2x128(void* dst0, void* dst1,
                                            __m256i vec256) {
  __m128i v128_0 = _mm256_extracti128_si256(vec256, 0);
  __m128i v128_1 = _mm256_extracti128_si256(vec256, 1);
  _mm_storeu_si128((__m128i*)dst0, v128_0);
  _mm_storeu_si128((__m128i*)dst1, v128_1);
}

static inline void iree_uk_copy_8x32xi8_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  for (int i = 0; i < 8; ++i) {
    iree_uk_memcpy(out_ptr + i * out_stride, in_ptr + i * in_stride, 32);
  }
}

static inline void iree_uk_copy_16x64xi8_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  for (int i = 0; i < 16; ++i) {
    iree_uk_memcpy(out_ptr + i * out_stride, in_ptr + i * in_stride, 64);
  }
}

static inline __m256i iree_uk_avx2_load_8x4xi8_strided(
    const iree_uk_int8_t* src, iree_uk_index_t stride) {
  __m256i indices = _mm256_mullo_epi32(
      _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(stride));
  return _mm256_i32gather_epi32(src, indices, 1);
}

static inline __m128i iree_uk_avx2_load_8x2xi8_strided(
    const iree_uk_int8_t* src, iree_uk_index_t stride) {
  __m128i result = _mm_setzero_si128();
  const iree_uk_int16_t* src_i16 = (const iree_uk_int16_t*)src;
  result =
      _mm_insert_epi16(result, *(const iree_uk_int16_t*)(src + 0 * stride), 0);
  result =
      _mm_insert_epi16(result, *(const iree_uk_int16_t*)(src + 1 * stride), 1);
  result =
      _mm_insert_epi16(result, *(const iree_uk_int16_t*)(src + 2 * stride), 2);
  result =
      _mm_insert_epi16(result, *(const iree_uk_int16_t*)(src + 3 * stride), 3);
  result =
      _mm_insert_epi16(result, *(const iree_uk_int16_t*)(src + 4 * stride), 4);
  result =
      _mm_insert_epi16(result, *(const iree_uk_int16_t*)(src + 5 * stride), 5);
  result =
      _mm_insert_epi16(result, *(const iree_uk_int16_t*)(src + 6 * stride), 6);
  result =
      _mm_insert_epi16(result, *(const iree_uk_int16_t*)(src + 7 * stride), 7);
  return result;
}

static inline __m256i iree_uk_avx2_load_16x2xi8_strided(
    const iree_uk_int8_t* src, iree_uk_index_t stride) {
  __m256i result = _mm256_setzero_si256();
  const iree_uk_int16_t* src_i16 = (const iree_uk_int16_t*)src;
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 0 * stride), 0);
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 1 * stride), 1);
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 2 * stride), 2);
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 3 * stride), 3);
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 4 * stride), 4);
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 5 * stride), 5);
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 6 * stride), 6);
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 7 * stride), 7);
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 8 * stride), 8);
  result = _mm256_insert_epi16(result,
                               *(const iree_uk_int16_t*)(src + 9 * stride), 9);
  result = _mm256_insert_epi16(
      result, *(const iree_uk_int16_t*)(src + 10 * stride), 10);
  result = _mm256_insert_epi16(
      result, *(const iree_uk_int16_t*)(src + 11 * stride), 11);
  result = _mm256_insert_epi16(
      result, *(const iree_uk_int16_t*)(src + 12 * stride), 12);
  result = _mm256_insert_epi16(
      result, *(const iree_uk_int16_t*)(src + 13 * stride), 13);
  result = _mm256_insert_epi16(
      result, *(const iree_uk_int16_t*)(src + 14 * stride), 14);
  result = _mm256_insert_epi16(
      result, *(const iree_uk_int16_t*)(src + 15 * stride), 15);
  return result;
}

static inline void iree_uk_avx2_copy_8x4xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  __m256i in = iree_uk_avx2_load_8x4xi8_strided(in_ptr, in_stride);
  _mm256_storeu_si256((__m256i*)out_ptr, in);
}

static inline void iree_uk_avx2_copy_8x2xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  __m128i in = iree_uk_avx2_load_8x2xi8_strided(in_ptr, in_stride);
  _mm_storeu_si128((__m128i*)out_ptr, in);
}

static inline void iree_uk_avx2_copy_16x2xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  __m256i in = iree_uk_avx2_load_16x2xi8_strided(in_ptr, in_stride);
  _mm256_storeu_si256((__m256i*)out_ptr, in);
}

static inline void
iree_uk_avx2_copy_8x16xi8_tiled_1x4_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  __m256i r00004444 =
      iree_uk_avx_loadu_2x128(in_ptr + 0 * in_stride, in_ptr + 4 * in_stride);
  __m256i r11115555 =
      iree_uk_avx_loadu_2x128(in_ptr + 1 * in_stride, in_ptr + 5 * in_stride);
  __m256i r22226666 =
      iree_uk_avx_loadu_2x128(in_ptr + 2 * in_stride, in_ptr + 6 * in_stride);
  __m256i r33337777 =
      iree_uk_avx_loadu_2x128(in_ptr + 3 * in_stride, in_ptr + 7 * in_stride);
  __m256i r00224466_0 = _mm256_unpacklo_epi64(r00004444, r22226666);
  __m256i r00224466_1 = _mm256_unpackhi_epi64(r00004444, r22226666);
  __m256i r11335577_0 = _mm256_unpacklo_epi64(r11115555, r33337777);
  __m256i r11335577_1 = _mm256_unpackhi_epi64(r11115555, r33337777);
  __m256i r01014545_0 = _mm256_unpacklo_epi32(r00224466_0, r11335577_0);
  __m256i r01014545_1 = _mm256_unpacklo_epi32(r00224466_1, r11335577_1);
  __m256i r23236767_0 = _mm256_unpackhi_epi32(r00224466_0, r11335577_0);
  __m256i r23236767_1 = _mm256_unpackhi_epi32(r00224466_1, r11335577_1);
  __m256i r01234567_0 = _mm256_unpacklo_epi64(r01014545_0, r23236767_0);
  __m256i r01234567_1 = _mm256_unpackhi_epi64(r01014545_0, r23236767_0);
  __m256i r01234567_2 = _mm256_unpacklo_epi64(r01014545_1, r23236767_1);
  __m256i r01234567_3 = _mm256_unpackhi_epi64(r01014545_1, r23236767_1);
  _mm256_storeu_si256((__m256i*)(out_ptr + 0 * out_stride), r01234567_0);
  _mm256_storeu_si256((__m256i*)(out_ptr + 1 * out_stride), r01234567_1);
  _mm256_storeu_si256((__m256i*)(out_ptr + 2 * out_stride), r01234567_2);
  _mm256_storeu_si256((__m256i*)(out_ptr + 3 * out_stride), r01234567_3);
}

static inline void
iree_uk_avx2_copy_8x16xi8_tiled_1x2_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  __m256i r0000000044444444 =
      iree_uk_avx_loadu_2x128(in_ptr + 0 * in_stride, in_ptr + 4 * in_stride);
  __m256i r1111111155555555 =
      iree_uk_avx_loadu_2x128(in_ptr + 1 * in_stride, in_ptr + 5 * in_stride);
  __m256i r2222222266666666 =
      iree_uk_avx_loadu_2x128(in_ptr + 2 * in_stride, in_ptr + 6 * in_stride);
  __m256i r3333333377777777 =
      iree_uk_avx_loadu_2x128(in_ptr + 3 * in_stride, in_ptr + 7 * in_stride);
  __m256i r0000111144445555_0 =
      _mm256_unpacklo_epi64(r0000000044444444, r1111111155555555);
  __m256i r0000111144445555_1 =
      _mm256_unpackhi_epi64(r0000000044444444, r1111111155555555);
  __m256i r2222333366667777_0 =
      _mm256_unpacklo_epi64(r2222222266666666, r3333333377777777);
  __m256i r2222333366667777_1 =
      _mm256_unpackhi_epi64(r2222222266666666, r3333333377777777);
  __m256i r0022002244664466_0 =
      _mm256_unpacklo_epi32(r0000111144445555_0, r2222333366667777_0);
  __m256i r0022002244664466_1 =
      _mm256_unpacklo_epi32(r0000111144445555_1, r2222333366667777_1);
  __m256i r1133113355775577_0 =
      _mm256_unpackhi_epi32(r0000111144445555_0, r2222333366667777_0);
  __m256i r1133113355775577_1 =
      _mm256_unpackhi_epi32(r0000111144445555_1, r2222333366667777_1);
  __m256i r0101232345456767_0 =
      _mm256_unpacklo_epi16(r0022002244664466_0, r1133113355775577_0);
  __m256i r0101232345456767_1 =
      _mm256_unpackhi_epi16(r0022002244664466_0, r1133113355775577_0);
  __m256i r0101232345456767_2 =
      _mm256_unpacklo_epi16(r0022002244664466_1, r1133113355775577_1);
  __m256i r0101232345456767_3 =
      _mm256_unpackhi_epi16(r0022002244664466_1, r1133113355775577_1);
  __m256i r0123012345674567_0 = _mm256_shuffle_epi32(r0101232345456767_0, 0xD8);
  __m256i r0123012345674567_1 = _mm256_shuffle_epi32(r0101232345456767_1, 0xD8);
  __m256i r0123012345674567_2 = _mm256_shuffle_epi32(r0101232345456767_2, 0xD8);
  __m256i r0123012345674567_3 = _mm256_shuffle_epi32(r0101232345456767_3, 0xD8);
  __m256i r0123456701234567_0 =
      _mm256_permute4x64_epi64(r0123012345674567_0, 0xD8);
  __m256i r0123456701234567_1 =
      _mm256_permute4x64_epi64(r0123012345674567_1, 0xD8);
  __m256i r0123456701234567_2 =
      _mm256_permute4x64_epi64(r0123012345674567_2, 0xD8);
  __m256i r0123456701234567_3 =
      _mm256_permute4x64_epi64(r0123012345674567_3, 0xD8);
  iree_uk_avx_storeu_2x128(out_ptr + 0 * out_stride, out_ptr + 1 * out_stride,
                           r0123456701234567_0);
  iree_uk_avx_storeu_2x128(out_ptr + 2 * out_stride, out_ptr + 3 * out_stride,
                           r0123456701234567_1);
  iree_uk_avx_storeu_2x128(out_ptr + 4 * out_stride, out_ptr + 5 * out_stride,
                           r0123456701234567_2);
  iree_uk_avx_storeu_2x128(out_ptr + 6 * out_stride, out_ptr + 7 * out_stride,
                           r0123456701234567_3);
}

#if defined(__AVX512F__)

static inline __m512i iree_uk_avx512_loadu_4x128(const void* src0,
                                                 const void* src1,
                                                 const void* src2,
                                                 const void* src3) {
  __m128i v128_0 = _mm_loadu_si128((const __m128i*)src0);
  __m128i v128_1 = _mm_loadu_si128((const __m128i*)src1);
  __m128i v128_2 = _mm_loadu_si128((const __m128i*)src2);
  __m128i v128_3 = _mm_loadu_si128((const __m128i*)src3);
  __m512i result = _mm512_castsi128_si512(v128_0);
  result = _mm512_inserti32x4(result, v128_1, 1);
  result = _mm512_inserti32x4(result, v128_2, 2);
  result = _mm512_inserti32x4(result, v128_3, 3);
  return result;
}

static inline void iree_uk_avx512_storeu_4x128(void* dst0, void* dst1,
                                               void* dst2, void* dst3,
                                               __m512i vec512) {
  __m128i v128_0 = _mm512_extracti32x4_epi32(vec512, 0);
  __m128i v128_1 = _mm512_extracti32x4_epi32(vec512, 1);
  __m128i v128_2 = _mm512_extracti32x4_epi32(vec512, 2);
  __m128i v128_3 = _mm512_extracti32x4_epi32(vec512, 3);
  _mm_storeu_si128((__m128i*)dst0, v128_0);
  _mm_storeu_si128((__m128i*)dst1, v128_1);
  _mm_storeu_si128((__m128i*)dst2, v128_2);
  _mm_storeu_si128((__m128i*)dst3, v128_3);
}

static inline __m512i iree_uk_avx512_loadu_4x128_from_16x16xi32(
    const iree_uk_int32_t* src, int i0, int j0, int i1, int j1, int i2, int j2,
    int i3, int j3) {
  return iree_uk_avx512_loadu_4x128(src + i0 * 16 + j0, src + i1 * 16 + j1,
                                    src + i2 * 16 + j2, src + i3 * 16 + j3);
}

static inline void iree_uk_avx512_storeu_4x128_to_16x16xi32(
    iree_uk_int32_t* dst, int i0, int j0, int i1, int j1, int i2, int j2,
    int i3, int j3, __m512i vec512) {
  return iree_uk_avx512_storeu_4x128(dst + i0 * 16 + j0, dst + i1 * 16 + j1,
                                     dst + i2 * 16 + j2, dst + i3 * 16 + j3,
                                     vec512);
}

static inline __m512i iree_uk_avx512_load_16x4xi8_strided(
    const iree_uk_int8_t* src, iree_uk_index_t stride) {
  __m512i indices = _mm512_mullo_epi32(
      _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
      _mm512_set1_epi32(stride));
  return _mm512_i32gather_epi32(indices, src, 1);
}

static inline void iree_uk_avx512_copy_16x4xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  __m512i in = iree_uk_avx512_load_16x4xi8_strided(in_ptr, in_stride);
  _mm512_storeu_si512((__m512i*)out_ptr, in);
}

static inline void
iree_uk_avx512_copy_16x16xi8_tiled_1x4_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  __m512i r000044448888CCCC = iree_uk_avx512_loadu_4x128(
      in_ptr + 0 * in_stride, in_ptr + 4 * in_stride, in_ptr + 8 * in_stride,
      in_ptr + 12 * in_stride);
  __m512i r111155559999DDDD = iree_uk_avx512_loadu_4x128(
      in_ptr + 1 * in_stride, in_ptr + 5 * in_stride, in_ptr + 9 * in_stride,
      in_ptr + 13 * in_stride);
  __m512i r22226666AAAAEEEE = iree_uk_avx512_loadu_4x128(
      in_ptr + 2 * in_stride, in_ptr + 6 * in_stride, in_ptr + 10 * in_stride,
      in_ptr + 14 * in_stride);
  __m512i r33337777BBBBFFFF = iree_uk_avx512_loadu_4x128(
      in_ptr + 3 * in_stride, in_ptr + 7 * in_stride, in_ptr + 11 * in_stride,
      in_ptr + 15 * in_stride);
  __m512i r0022446688AACCEE_0 =
      _mm512_unpacklo_epi64(r000044448888CCCC, r22226666AAAAEEEE);
  __m512i r0022446688AACCEE_1 =
      _mm512_unpackhi_epi64(r000044448888CCCC, r22226666AAAAEEEE);
  __m512i r1133557799BBDDFF_0 =
      _mm512_unpacklo_epi64(r111155559999DDDD, r33337777BBBBFFFF);
  __m512i r1133557799BBDDFF_1 =
      _mm512_unpackhi_epi64(r111155559999DDDD, r33337777BBBBFFFF);
  __m512i r010145458989CDCD_0 =
      _mm512_unpacklo_epi32(r0022446688AACCEE_0, r1133557799BBDDFF_0);
  __m512i r010145458989CDCD_1 =
      _mm512_unpacklo_epi32(r0022446688AACCEE_1, r1133557799BBDDFF_1);
  __m512i r23236767ABABEFEF_0 =
      _mm512_unpackhi_epi32(r0022446688AACCEE_0, r1133557799BBDDFF_0);
  __m512i r23236767ABABEFEF_1 =
      _mm512_unpackhi_epi32(r0022446688AACCEE_1, r1133557799BBDDFF_1);
  __m512i r0123456789ABCDEF_0 =
      _mm512_unpacklo_epi64(r010145458989CDCD_0, r23236767ABABEFEF_0);
  __m512i r0123456789ABCDEF_1 =
      _mm512_unpackhi_epi64(r010145458989CDCD_0, r23236767ABABEFEF_0);
  __m512i r0123456789ABCDEF_2 =
      _mm512_unpacklo_epi64(r010145458989CDCD_1, r23236767ABABEFEF_1);
  __m512i r0123456789ABCDEF_3 =
      _mm512_unpackhi_epi64(r010145458989CDCD_1, r23236767ABABEFEF_1);
  _mm512_storeu_si512((__m512i*)(out_ptr + 0 * out_stride),
                      r0123456789ABCDEF_0);
  _mm512_storeu_si512((__m512i*)(out_ptr + 1 * out_stride),
                      r0123456789ABCDEF_1);
  _mm512_storeu_si512((__m512i*)(out_ptr + 2 * out_stride),
                      r0123456789ABCDEF_2);
  _mm512_storeu_si512((__m512i*)(out_ptr + 3 * out_stride),
                      r0123456789ABCDEF_3);
}

static inline void
iree_uk_avx512_copy_16x16xi8_tiled_1x2_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  __m512i r000044448888CCCC = iree_uk_avx512_loadu_4x128(
      in_ptr + 0 * in_stride, in_ptr + 4 * in_stride, in_ptr + 8 * in_stride,
      in_ptr + 12 * in_stride);
  __m512i r111155559999DDDD = iree_uk_avx512_loadu_4x128(
      in_ptr + 1 * in_stride, in_ptr + 5 * in_stride, in_ptr + 9 * in_stride,
      in_ptr + 13 * in_stride);
  __m512i r22226666AAAAEEEE = iree_uk_avx512_loadu_4x128(
      in_ptr + 2 * in_stride, in_ptr + 6 * in_stride, in_ptr + 10 * in_stride,
      in_ptr + 14 * in_stride);
  __m512i r33337777BBBBFFFF = iree_uk_avx512_loadu_4x128(
      in_ptr + 3 * in_stride, in_ptr + 7 * in_stride, in_ptr + 11 * in_stride,
      in_ptr + 15 * in_stride);
  __m512i r0000111144445555_0 =
      _mm512_unpacklo_epi64(r000044448888CCCC, r111155559999DDDD);
  __m512i r0000111144445555_1 =
      _mm512_unpackhi_epi64(r000044448888CCCC, r111155559999DDDD);
  __m512i r2222333366667777_0 =
      _mm512_unpacklo_epi64(r22226666AAAAEEEE, r33337777BBBBFFFF);
  __m512i r2222333366667777_1 =
      _mm512_unpackhi_epi64(r22226666AAAAEEEE, r33337777BBBBFFFF);
  __m512i r0022002244664466_0 =
      _mm512_unpacklo_epi32(r0000111144445555_0, r2222333366667777_0);
  __m512i r0022002244664466_1 =
      _mm512_unpacklo_epi32(r0000111144445555_1, r2222333366667777_1);
  __m512i r1133113355775577_0 =
      _mm512_unpackhi_epi32(r0000111144445555_0, r2222333366667777_0);
  __m512i r1133113355775577_1 =
      _mm512_unpackhi_epi32(r0000111144445555_1, r2222333366667777_1);
  __m512i r0101232345456767_0 =
      _mm512_unpacklo_epi16(r0022002244664466_0, r1133113355775577_0);
  __m512i r0101232345456767_1 =
      _mm512_unpackhi_epi16(r0022002244664466_0, r1133113355775577_0);
  __m512i r0101232345456767_2 =
      _mm512_unpacklo_epi16(r0022002244664466_1, r1133113355775577_1);
  __m512i r0101232345456767_3 =
      _mm512_unpackhi_epi16(r0022002244664466_1, r1133113355775577_1);
  __m512i r0123012345674567_0 = _mm512_shuffle_epi32(r0101232345456767_0, 0xD8);
  __m512i r0123012345674567_1 = _mm512_shuffle_epi32(r0101232345456767_1, 0xD8);
  __m512i r0123012345674567_2 = _mm512_shuffle_epi32(r0101232345456767_2, 0xD8);
  __m512i r0123012345674567_3 = _mm512_shuffle_epi32(r0101232345456767_3, 0xD8);
  __m512i r0123456701234567_0 =
      _mm512_permutex_epi64(r0123012345674567_0, 0xD8);
  __m512i r0123456701234567_1 =
      _mm512_permutex_epi64(r0123012345674567_1, 0xD8);
  __m512i r0123456701234567_2 =
      _mm512_permutex_epi64(r0123012345674567_2, 0xD8);
  __m512i r0123456701234567_3 =
      _mm512_permutex_epi64(r0123012345674567_3, 0xD8);
  iree_uk_avx512_storeu_4x128(
      out_ptr + 0 * out_stride, out_ptr + 1 * out_stride + 0,
      out_ptr + 0 * out_stride + 16, out_ptr + 1 * out_stride + 16,
      r0123456701234567_0);
  iree_uk_avx512_storeu_4x128(
      out_ptr + 2 * out_stride, out_ptr + 3 * out_stride + 0,
      out_ptr + 2 * out_stride + 16, out_ptr + 3 * out_stride + 16,
      r0123456701234567_1);
  iree_uk_avx512_storeu_4x128(
      out_ptr + 4 * out_stride, out_ptr + 5 * out_stride + 0,
      out_ptr + 4 * out_stride + 16, out_ptr + 5 * out_stride + 16,
      r0123456701234567_2);
  iree_uk_avx512_storeu_4x128(
      out_ptr + 6 * out_stride, out_ptr + 7 * out_stride + 0,
      out_ptr + 6 * out_stride + 16, out_ptr + 7 * out_stride + 16,
      r0123456701234567_3);
}

#endif  // defined (__AVX512F__)

#endif  // defined(__AVX2__)

#endif  // IREE_BUILTINS_UKERNEL_ARCH_X86_64_COMMON_X86_64_H_