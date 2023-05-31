// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/unpack_internal.h"

#if defined(IREE_UK_BUILD_X86_64_AVX2_FMA)

void iree_uk_unpack_tile_8x8_x32_x86_64_avx2_fma_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_copy_8x32xi8_strided_to_strided(out_ptr, in_ptr, 4 * out_stride0,
                                            32);
    out_ptr += 32;
    in_ptr += 4 * in_stride1;
  }
}

#endif  // defined(IREE_UK_BUILD_X86_64_AVX2_FMA)
