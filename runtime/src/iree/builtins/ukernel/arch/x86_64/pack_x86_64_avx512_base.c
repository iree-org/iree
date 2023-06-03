// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/pack_internal.h"

#if defined(IREE_UK_BUILD_X86_64_AVX512_BASE)

void iree_uk_pack_tile_16x16_x32_x86_64_avx512_base_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 16);
  IREE_UK_ASSERT(tile_size1 == 16);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_copy_16x64xi8_strided_to_strided(out_ptr, in_ptr, 64,
                                             4 * in_stride0);
    out_ptr += 4 * out_stride1;
    in_ptr += 64;
  }
}

static void iree_uk_pack_tile_16x4_x8_x86_64_avx512_base_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 16);
  IREE_UK_ASSERT(tile_size1 == 4);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 >= 4; outer_size1 -= 4) {
    iree_uk_avx512_copy_16x16xi8_tiled_1x4_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride1, in_stride0);
    out_ptr += 4 * out_stride1;
    in_ptr += 16;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_avx512_copy_16x4xi8_strided_to_unstrided(out_ptr, in_ptr,
                                                     in_stride0);
    out_ptr += out_stride1;
    in_ptr += 4;
  }
}

void iree_uk_pack_tile_16x1_x32_x86_64_avx512_base_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 16);
  IREE_UK_ASSERT(tile_size1 == 1);
  iree_uk_pack_tile_16x4_x8_x86_64_avx512_base_direct(
      out_tile_ptr, in_tile_ptr, outer_size1, out_stride1 * 4, in_stride0 * 4,
      1, 16, 4);
}

void iree_uk_pack_tile_16x1_x32_x86_64_avx512_base_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 16);
  const iree_uk_int32_t* IREE_UK_RESTRICT in_tile_ptr_i32 = in_tile_ptr;
  iree_uk_int32_t* IREE_UK_RESTRICT out_tile_i32_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_memcpy(out_tile_i32_ptr, in_tile_ptr_i32, 64);
    out_tile_i32_ptr += out_stride1;
    in_tile_ptr_i32 += 16;
  }
}

void iree_uk_pack_tile_16x2_x8_x86_64_avx512_base_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 16);
  IREE_UK_ASSERT(tile_size1 == 2);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 >= 8; outer_size1 -= 8) {
    iree_uk_avx512_copy_16x16xi8_tiled_1x2_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride1, in_stride0);
    out_ptr += 8 * out_stride1;
    in_ptr += 16;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_avx2_copy_16x2xi8_strided_to_unstrided(out_ptr, in_ptr, in_stride0);
    out_ptr += out_stride1;
    in_ptr += 2;
  }
}

void iree_uk_pack_tile_16x2_x8_x86_64_avx512_base_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 2);
  IREE_UK_ASSERT(tile_size1 == 16);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  iree_uk_index_t outer_i1 = 0;
  for (; outer_i1 <= outer_size1 - 4; outer_i1 += 4) {
    __m512i in0 = _mm512_loadu_si512((const __m512i*)in_ptr);
    __m512i in1 = _mm512_loadu_si512((const __m512i*)(in_ptr + in_stride0));
    __m512i out0 = _mm512_unpacklo_epi8(in0, in1);
    __m512i out1 = _mm512_unpackhi_epi8(in0, in1);
    iree_uk_avx512_storeu_4x128(
        out_ptr + 0 * out_stride1, out_ptr + 1 * out_stride1,
        out_ptr + 2 * out_stride1, out_ptr + 3 * out_stride1, out0);
    iree_uk_avx512_storeu_4x128(
        out_ptr + 0 * out_stride1 + 16, out_ptr + 1 * out_stride1 + 16,
        out_ptr + 2 * out_stride1 + 16, out_ptr + 3 * out_stride1 + 16, out1);
    out_ptr += 4 * out_stride1;
    in_ptr += 64;
  }
  for (; outer_i1 < outer_size1; ++outer_i1) {
    __m256i in0 = _mm256_permute4x64_epi64(
        _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)in_ptr)), 0xD8);
    __m256i in1 = _mm256_permute4x64_epi64(
        _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i*)(in_ptr + in_stride0))),
        0xD8);
    __m256i out = _mm256_unpacklo_epi8(in0, in1);
    _mm256_storeu_si256((__m256i*)out_ptr, out);
    out_ptr += out_stride1;
    in_ptr += 16;
  }
}

#endif  // defined(IREE_UK_BUILD_X86_64_AVX512_BASE)
