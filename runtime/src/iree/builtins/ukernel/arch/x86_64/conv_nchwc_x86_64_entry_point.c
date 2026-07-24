// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/conv_nchwc_x86_64_internal.h"

iree_uk_conv_nchwc_tile_selection_t iree_uk_conv_nchwc_select_tile_func_arch(
    const iree_uk_conv_nchwc_params_t* params) {
  IREE_UK_ATTRIBUTE_UNUSED iree_uk_conv_nchwc_type_t type =
      iree_uk_conv_nchwc_type(params->flags);
  iree_uk_conv_nchwc_tile_selection_t selection = {0};

#define IREE_UK_CONV_NCHWC_TILE_IMPL_x86_64(LHS, RHS, OUT, OW_TILE, K0_TILE,                          \
                                            C0_TILE, SUFFIX)                                          \
  if (type == iree_uk_conv_nchwc_type_##LHS##RHS##OUT &&                                              \
      params->k0 == (K0_TILE) && params->c0 == (C0_TILE) &&                                           \
      iree_uk_cpu_x86_64##SUFFIX(params->cpu_data)) {                                                 \
    selection.tile_func =                                                                             \
        iree_uk_conv_nchwc_tile_##LHS##RHS##OUT##_##OW_TILE##x##K0_TILE##x##C0_TILE##_x86_64##SUFFIX; \
    selection.ow_tile = (OW_TILE);                                                                    \
  }

#ifdef IREE_UK_BUILD_X86_64_AVX512_BASE
#define IREE_UK_CONV_NCHWC_TILE_x86_64_avx512_base(LHS, RHS, OUT, OW_TILE, \
                                                   K0_TILE, C0_TILE)       \
  IREE_UK_CONV_NCHWC_TILE_IMPL_x86_64(LHS, RHS, OUT, OW_TILE, K0_TILE,     \
                                      C0_TILE, _avx512_base)
#else
#define IREE_UK_CONV_NCHWC_TILE_x86_64_avx512_base(LHS, RHS, OUT, OW_TILE, \
                                                   K0_TILE, C0_TILE)
#endif

#define IREE_UK_CONV_NCHWC_TILE(ARCH, LHS, RHS, OUT, OW_TILE, K0_TILE,    \
                                C0_TILE, SUFFIX)                          \
  IREE_UK_CONV_NCHWC_TILE_x86_64##SUFFIX(LHS, RHS, OUT, OW_TILE, K0_TILE, \
                                         C0_TILE)

#include "iree/builtins/ukernel/arch/x86_64/conv_nchwc_x86_64_tiles.inl"

  return selection;
}
