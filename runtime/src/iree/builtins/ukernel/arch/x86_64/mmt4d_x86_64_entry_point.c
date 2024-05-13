// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_internal.h"

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arch(
    const iree_uk_mmt4d_params_t* params) {
  IREE_UK_ATTRIBUTE_UNUSED iree_uk_mmt4d_type_t mmt4d_type =
      iree_uk_mmt4d_type(params->flags);
  iree_uk_mmt4d_tile_func_t tile_func = 0;

#define IREE_UK_MMT4D_TILE_IMPL_x86_64(lhs, rhs, out, m0, n0, k0, suffix)         \
  if (mmt4d_type == iree_uk_mmt4d_type_##lhs##rhs##out && params->M0 == m0 &&     \
      params->N0 == n0 && params->K0 == k0 &&                                     \
      iree_uk_cpu_x86_64##suffix(params->cpu_data)) {                             \
    tile_func =                                                                   \
        iree_uk_mmt4d_tile_##lhs##rhs##out##_##m0##x##n0##x##k0##_x86_64##suffix; \
  }

#define IREE_UK_MMT4D_TILE_x86_64(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_x86_64(lhs, rhs, out, m0, n0, k0, )

#ifdef IREE_UK_BUILD_X86_64_AVX2_FMA
#define IREE_UK_MMT4D_TILE_x86_64_avx2_fma(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_x86_64(lhs, rhs, out, m0, n0, k0, _avx2_fma)
#else
#define IREE_UK_MMT4D_TILE_x86_64_avx2_fma(lhs, rhs, out, m0, n0, k0)
#endif

#ifdef IREE_UK_BUILD_X86_64_AVX512_BASE
#define IREE_UK_MMT4D_TILE_x86_64_avx512_base(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_x86_64(lhs, rhs, out, m0, n0, k0, _avx512_base)
#else
#define IREE_UK_MMT4D_TILE_x86_64_avx512_base(lhs, rhs, out, m0, n0, k0)
#endif

#ifdef IREE_UK_BUILD_X86_64_AVX512_VNNI
#define IREE_UK_MMT4D_TILE_x86_64_avx512_vnni(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_x86_64(lhs, rhs, out, m0, n0, k0, _avx512_vnni)
#else
#define IREE_UK_MMT4D_TILE_x86_64_avx512_vnni(lhs, rhs, out, m0, n0, k0)
#endif

#ifdef IREE_UK_BUILD_X86_64_AVX512_BF16
#define IREE_UK_MMT4D_TILE_x86_64_avx512_bf16(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_x86_64(lhs, rhs, out, m0, n0, k0, _avx512_bf16)
#else
#define IREE_UK_MMT4D_TILE_x86_64_avx512_bf16(lhs, rhs, out, m0, n0, k0)
#endif

#define IREE_UK_MMT4D_TILE(arch, lhs, rhs, out, m0, n0, k0, suffix) \
  IREE_UK_MMT4D_TILE_x86_64##suffix(lhs, rhs, out, m0, n0, k0)

#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_tiles.inl"

  return tile_func;
}
