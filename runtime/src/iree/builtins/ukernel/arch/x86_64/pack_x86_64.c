// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/pack_x86_64.h"

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"

IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x8_x32_x86_64_avx2_fma_direct)
IREE_UK_PACK_TILE_FUNC_DECL(
    iree_uk_pack_tile_16x16_x32_x86_64_avx512_base_direct)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x1_x32_x86_64_avx2_fma_direct)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x1_x32_x86_64_avx2_fma_transpose)
IREE_UK_PACK_TILE_FUNC_DECL(
    iree_uk_pack_tile_16x1_x32_x86_64_avx512_base_direct)
IREE_UK_PACK_TILE_FUNC_DECL(
    iree_uk_pack_tile_16x1_x32_x86_64_avx512_base_transpose)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x2_x8_x86_64_avx2_fma_direct)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x2_x8_x86_64_avx2_fma_transpose)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_16x2_x8_x86_64_avx512_base_direct)
IREE_UK_PACK_TILE_FUNC_DECL(
    iree_uk_pack_tile_16x2_x8_x86_64_avx512_base_transpose)

static iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_x86_64_8x8_x32(
    const iree_uk_pack_params_t* params) {
#ifdef IREE_UK_BUILD_X86_64_AVX2_FMA
  if (iree_uk_cpu_supports_avx2_fma(params->cpu_data)) {
    bool transpose = params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
    return transpose ? 0 : iree_uk_pack_tile_8x8_x32_x86_64_avx2_fma_direct;
  }
#endif
  return 0;
}

static iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_x86_64_16x16_x32(
    const iree_uk_pack_params_t* params) {
#ifdef IREE_UK_BUILD_X86_64_AVX512_BASE
  if (iree_uk_cpu_supports_avx512_base(params->cpu_data)) {
    bool transpose = params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
    return transpose ? 0
                     : iree_uk_pack_tile_16x16_x32_x86_64_avx512_base_direct;
  }
#endif
  return 0;
}

static iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_x86_64_8x1_x32(
    const iree_uk_pack_params_t* params) {
#ifdef IREE_UK_BUILD_X86_64_AVX2_FMA
  if (iree_uk_cpu_supports_avx2_fma(params->cpu_data)) {
    bool transpose = params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
    return transpose ? iree_uk_pack_tile_8x1_x32_x86_64_avx2_fma_transpose
                     : iree_uk_pack_tile_8x1_x32_x86_64_avx2_fma_direct;
  }
#endif
  return 0;
}

static iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_x86_64_16x1_x32(
    const iree_uk_pack_params_t* params) {
#ifdef IREE_UK_BUILD_X86_64_AVX512_BASE
  if (iree_uk_cpu_supports_avx512_base(params->cpu_data)) {
    bool transpose = params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
    return transpose ? iree_uk_pack_tile_16x1_x32_x86_64_avx512_base_transpose
                     : iree_uk_pack_tile_16x1_x32_x86_64_avx512_base_direct;
  }
#endif
  return 0;
}

static iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_x86_64_8x2_x8(
    const iree_uk_pack_params_t* params) {
#ifdef IREE_UK_BUILD_X86_64_AVX2_FMA
  if (iree_uk_cpu_supports_avx2_fma(params->cpu_data)) {
    bool transpose = params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
    return transpose ? iree_uk_pack_tile_8x2_x8_x86_64_avx2_fma_transpose
                     : iree_uk_pack_tile_8x2_x8_x86_64_avx2_fma_direct;
  }
#endif
  return 0;
}

static iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_x86_64_16x2_x8(
    const iree_uk_pack_params_t* params) {
#ifdef IREE_UK_BUILD_X86_64_AVX512_BASE
  if (iree_uk_cpu_supports_avx512_base(params->cpu_data)) {
    bool transpose = params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
    return transpose ? iree_uk_pack_tile_16x2_x8_x86_64_avx512_base_transpose
                     : iree_uk_pack_tile_16x2_x8_x86_64_avx512_base_direct;
  }
#endif
  return 0;
}

iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_x86_64(
    const iree_uk_pack_params_t* params) {
  // At the moment, as sum-reductions are not yet part of pack ops,
  // no arithmetic whatsoever is being done here, so only the element type
  // size matters, not the type itself.
  iree_uk_pack_type_t pack_type = iree_uk_pack_type(params->flags);
  int esize = iree_uk_type_size(iree_uk_pack_out_type(pack_type));
  if (esize == 4 && params->out_size2 == 8 && params->out_size3 == 8) {
    return iree_uk_pack_select_tile_func_x86_64_8x8_x32(params);
  } else if (esize == 4 && params->out_size2 == 16 && params->out_size3 == 16) {
    return iree_uk_pack_select_tile_func_x86_64_16x16_x32(params);
  } else if (esize == 4 && params->out_size2 == 8 && params->out_size3 == 1) {
    return iree_uk_pack_select_tile_func_x86_64_8x1_x32(params);
  } else if (esize == 4 && params->out_size2 == 16 && params->out_size3 == 1) {
    return iree_uk_pack_select_tile_func_x86_64_16x1_x32(params);
  } else if (esize == 1 && params->out_size2 == 8 && params->out_size3 == 2) {
    return iree_uk_pack_select_tile_func_x86_64_8x2_x8(params);
  } else if (esize == 1 && params->out_size2 == 16 && params->out_size3 == 2) {
    return iree_uk_pack_select_tile_func_x86_64_16x2_x8(params);
  }
  return 0;
}
