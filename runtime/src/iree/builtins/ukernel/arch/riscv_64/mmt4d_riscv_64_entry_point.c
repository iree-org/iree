// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arch(
    const iree_uk_mmt4d_params_t* params) {
  IREE_UK_ATTRIBUTE_UNUSED iree_uk_mmt4d_type_t mmt4d_type =
      iree_uk_mmt4d_type(params->flags);
  iree_uk_mmt4d_tile_func_t tile_func = 0;

#define IREE_UK_MMT4D_TILE_IMPL_riscv_64(lhs, rhs, out, m0, k0, suffix)         \
  if (mmt4d_type == iree_uk_mmt4d_type_##lhs##rhs##out && params->M0 == m0 &&   \
      params->K0 == k0 && iree_uk_cpu_riscv_64##suffix(params->cpu_data)) {     \
    tile_func =                                                                 \
        iree_uk_mmt4d_tile_##lhs##rhs##out##_##m0##xXXx##k0##_riscv_64##suffix; \
  }

#define IREE_UK_MMT4D_TILE_riscv_64(lhs, rhs, out, m0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_riscv_64(lhs, rhs, out, m0, k0, )

#ifdef IREE_UK_BUILD_RISCV_64_V
#define IREE_UK_MMT4D_TILE_riscv_64_v(lhs, rhs, out, m0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_riscv_64(lhs, rhs, out, m0, k0, _v)
#else
#define IREE_UK_MMT4D_TILE_riscv_64_v(lhs, rhs, out, m0, k0)
#endif

#define IREE_UK_MMT4D_TILE(arch, lhs, rhs, out, m0, k0, suffix) \
  IREE_UK_MMT4D_TILE_riscv_64##suffix(lhs, rhs, out, m0, k0)

#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_tiles.inl"

  return tile_func;
}
