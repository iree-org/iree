// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/unpack_x86_64.h"

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"

IREE_UK_UNPACK_TILE_FUNC_DECL(
    iree_uk_unpack_tile_8x8_x32_x86_64_avx2_fma_direct)
IREE_UK_UNPACK_TILE_FUNC_DECL(
    iree_uk_unpack_tile_16x16_x32_x86_64_avx512_base_direct)

iree_uk_unpack_tile_func_t iree_uk_unpack_select_tile_func_x86_64(
    const iree_uk_unpack_params_t* params) {
  iree_uk_unpack_type_t unpack_type = iree_uk_unpack_type(params->flags);
  int esize = iree_uk_type_size(iree_uk_unpack_out_type(unpack_type));
  bool transpose = params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER;
  // Unpack is currently only used in practice with esize==4 and non-transpose.
  if (esize != 4 || transpose) return 0;
  if (params->in_size2 == 8 && params->in_size3 == 8) {
#ifdef IREE_UK_BUILD_X86_64_AVX2_FMA
    if (iree_uk_cpu_supports_avx2_fma(params->cpu_data)) {
      return iree_uk_unpack_tile_8x8_x32_x86_64_avx2_fma_direct;
    }
#endif
  } else if (params->in_size2 == 16 && params->in_size3 == 16) {
#ifdef IREE_UK_BUILD_X86_64_AVX512_BASE
    if (iree_uk_cpu_supports_avx512_base(params->cpu_data)) {
      return iree_uk_unpack_tile_16x16_x32_x86_64_avx512_base_direct;
    }
#endif
  }
  return 0;
}
