// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_ARM_64_MMT4D_TILE_ARM_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_ARM_64_MMT4D_TILE_ARM_64_H_

#include "iree/builtins/ukernel/mmt4d.h"

IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_f32f32f32_8x8x1_arm_64)
IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_i8i8i32_8x8x1_arm_64)
IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_i8i8i32_8x8x4_arm_64_dotprod)
IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_i8i8i32_8x8x8_arm_64_i8mm)

#endif  // IREE_BUILTINS_UKERNEL_ARCH_ARM_64_MMT4D_TILE_ARM_64_H_
