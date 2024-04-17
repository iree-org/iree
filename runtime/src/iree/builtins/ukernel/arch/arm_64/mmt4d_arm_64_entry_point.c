// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arch(
    const iree_uk_mmt4d_params_t* params) {
  IREE_UK_ATTRIBUTE_UNUSED iree_uk_mmt4d_type_t mmt4d_type =
      iree_uk_mmt4d_type(params->flags);
  iree_uk_mmt4d_tile_func_t tile_func = 0;

#define IREE_UK_MMT4D_TILE_IMPL_arm_64(lhs, rhs, out, m0, n0, k0, suffix)         \
  if (mmt4d_type == iree_uk_mmt4d_type_##lhs##rhs##out && params->M0 == m0 &&     \
      params->N0 == n0 && params->K0 == k0 &&                                     \
      iree_uk_cpu_arm_64##suffix(params->cpu_data)) {                             \
    tile_func =                                                                   \
        iree_uk_mmt4d_tile_##lhs##rhs##out##_##m0##x##n0##x##k0##_arm_64##suffix; \
  }

#define IREE_UK_MMT4D_TILE_arm_64(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_arm_64(lhs, rhs, out, m0, n0, k0, )

#ifdef IREE_UK_BUILD_ARM_64_FULLFP16
#define IREE_UK_MMT4D_TILE_arm_64_fullfp16(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_arm_64(lhs, rhs, out, m0, n0, k0, _fullfp16)
#else
#define IREE_UK_MMT4D_TILE_arm_64_fullfp16(lhs, rhs, out, m0, n0, k0)
#endif

#ifdef IREE_UK_BUILD_ARM_64_FP16FML
#define IREE_UK_MMT4D_TILE_arm_64_fp16fml(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_arm_64(lhs, rhs, out, m0, n0, k0, _fp16fml)
#else
#define IREE_UK_MMT4D_TILE_arm_64_fp16fml(lhs, rhs, out, m0, n0, k0)
#endif

#ifdef IREE_UK_BUILD_ARM_64_BF16
#define IREE_UK_MMT4D_TILE_arm_64_bf16(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_arm_64(lhs, rhs, out, m0, n0, k0, _bf16)
#else
#define IREE_UK_MMT4D_TILE_arm_64_bf16(lhs, rhs, out, m0, n0, k0)
#endif

#ifdef IREE_UK_BUILD_ARM_64_DOTPROD
#define IREE_UK_MMT4D_TILE_arm_64_dotprod(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_arm_64(lhs, rhs, out, m0, n0, k0, _dotprod)
#else
#define IREE_UK_MMT4D_TILE_arm_64_dotprod(lhs, rhs, out, m0, n0, k0)
#endif

#ifdef IREE_UK_BUILD_ARM_64_I8MM
#define IREE_UK_MMT4D_TILE_arm_64_i8mm(lhs, rhs, out, m0, n0, k0) \
  IREE_UK_MMT4D_TILE_IMPL_arm_64(lhs, rhs, out, m0, n0, k0, _i8mm)
#else
#define IREE_UK_MMT4D_TILE_arm_64_i8mm(lhs, rhs, out, m0, n0, k0)
#endif

#define IREE_UK_MMT4D_TILE(arch, lhs, rhs, out, m0, n0, k0, suffix) \
  IREE_UK_MMT4D_TILE_arm_64##suffix(lhs, rhs, out, m0, n0, k0)

#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_tiles.inl"

  return tile_func;
}
