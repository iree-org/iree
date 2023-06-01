// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/pack_internal.h"

#if defined(IREE_UK_BUILD_X86_64_AVX2_FMA)

void iree_uk_pack_tile_8x8_x32_x86_64_avx2_fma_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_copy_8x32xi8_strided_to_strided(out_ptr, in_ptr, 32,
                                            4 * in_stride0);
    out_ptr += 4 * out_stride1;
    in_ptr += 32;
  }
}

static void iree_uk_pack_tile_8x4_x8_x86_64_avx2_fma_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 4);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 >= 4; outer_size1 -= 4) {
    iree_uk_avx2_copy_8x16xi8_tiled_1x4_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride1, in_stride0);
    out_ptr += 4 * out_stride1;
    in_ptr += 16;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_avx2_copy_8x4xi8_strided_to_unstrided(out_ptr, in_ptr, in_stride0);
    out_ptr += out_stride1;
    in_ptr += 4;
  }
}

void iree_uk_pack_tile_8x1_x32_x86_64_avx2_fma_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 1);
  iree_uk_pack_tile_8x4_x8_x86_64_avx2_fma_direct(out_tile_ptr, in_tile_ptr,
                                                  outer_size1, out_stride1 * 4,
                                                  in_stride0 * 4, 1, 8, 4);
}

void iree_uk_pack_tile_8x1_x32_x86_64_avx2_fma_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int32_t* IREE_UK_RESTRICT in_tile_ptr_i32 = in_tile_ptr;
  iree_uk_int32_t* IREE_UK_RESTRICT out_tile_i32_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_memcpy(out_tile_i32_ptr, in_tile_ptr_i32, 32);
    out_tile_i32_ptr += out_stride1;
    in_tile_ptr_i32 += 8;
  }
}

void iree_uk_pack_tile_8x2_x8_x86_64_avx2_fma_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 2);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 >= 8; outer_size1 -= 8) {
    iree_uk_avx2_copy_8x16xi8_tiled_1x2_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride1, in_stride0);
    out_ptr += 8 * out_stride1;
    in_ptr += 16;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_avx2_copy_8x2xi8_strided_to_unstrided(out_ptr, in_ptr, in_stride0);
    out_ptr += out_stride1;
    in_ptr += 2;
  }
}

void iree_uk_pack_tile_8x2_x8_x86_64_avx2_fma_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 2);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  iree_uk_index_t outer_i1 = 0;
  for (; outer_i1 <= outer_size1 - 4; outer_i1 += 4) {
    __m256i in0 = _mm256_loadu_si256((const __m256i*)in_ptr);
    __m256i in1 = _mm256_loadu_si256((const __m256i*)(in_ptr + in_stride0));
    __m256i out0 = _mm256_unpacklo_epi8(in0, in1);
    __m256i out1 = _mm256_unpackhi_epi8(in0, in1);
    iree_uk_avx_storeu_2x128(out_ptr + 0 * out_stride1,
                             out_ptr + 2 * out_stride1, out0);
    iree_uk_avx_storeu_2x128(out_ptr + 1 * out_stride1,
                             out_ptr + 3 * out_stride1, out1);
    out_ptr += 4 * out_stride1;
    in_ptr += 32;
  }
  for (; outer_i1 < outer_size1; ++outer_i1) {
    __m128i in0 = _mm_loadu_si64(in_ptr);
    __m128i in1 = _mm_loadu_si64(in_ptr + in_stride0);
    __m128i out = _mm_unpacklo_epi8(in0, in1);
    _mm_storeu_si128((__m128i*)out_ptr, out);
    out_ptr += out_stride1;
    in_ptr += 8;
  }
}

#endif  // defined(IREE_UK_BUILD_X86_64_AVX2_FMA)
