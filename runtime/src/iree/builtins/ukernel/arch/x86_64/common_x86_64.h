// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <immintrin.h>

#include "iree/builtins/ukernel/arch/x86_64/config.h"
#include "iree/builtins/ukernel/common.h"
#include "iree/schemas/cpu_data.h"

static inline bool iree_uk_all_bits_set(const iree_uk_uint64_t val,
                                        const iree_uk_uint64_t required_bits) {
  return (val & required_bits) == required_bits;
}

static inline bool iree_uk_cpu_supports_avx2_fma(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(
      cpu_data[0], IREE_CPU_DATA0_X86_64_AVX2 | IREE_CPU_DATA0_X86_64_FMA);
}

static inline bool iree_uk_cpu_supports_avx512_base(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_X86_64_AVX512F |
                                               IREE_CPU_DATA0_X86_64_AVX512BW |
                                               IREE_CPU_DATA0_X86_64_AVX512DQ |
                                               IREE_CPU_DATA0_X86_64_AVX512VL |
                                               IREE_CPU_DATA0_X86_64_AVX512CD);
}

static inline bool iree_uk_cpu_supports_avx512_vnni(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_cpu_supports_avx512_base(cpu_data) &&
         iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_X86_64_AVX512VNNI);
}

static inline __m256i iree_uk_avx_loadu_2x128(const void* src0,
                                              const void* src1) {
  __m128i v128_0 = _mm_loadu_si128((const __m128i*)src0);
  __m128i v128_1 = _mm_loadu_si128((const __m128i*)src1);
  return _mm256_inserti128_si256(_mm256_castsi128_si256(v128_0), v128_1, 1);
}

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

static inline void iree_uk_avx_storeu_2x128(void* dst0, void* dst1,
                                            __m256i vec256) {
  __m128i v128_0 = _mm256_extracti128_si256(vec256, 0);
  __m128i v128_1 = _mm256_extracti128_si256(vec256, 1);
  _mm_storeu_si128((__m128i*)dst0, v128_0);
  _mm_storeu_si128((__m128i*)dst1, v128_1);
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
