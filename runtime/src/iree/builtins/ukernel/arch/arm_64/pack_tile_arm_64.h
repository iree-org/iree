// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_ARM_64_PACK_TILE_ARM_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_ARM_64_PACK_TILE_ARM_64_H_

#include "iree/builtins/ukernel/pack.h"

IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x1_x32_arm_64_direct)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x1_x32_arm_64_transpose)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x1_x8_arm_64_direct)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x1_x8_arm_64_transpose)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x4_x8_arm_64_direct)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x4_x8_arm_64_transpose)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x8_x8_arm_64_direct)
IREE_UK_PACK_TILE_FUNC_DECL(iree_uk_pack_tile_8x8_x8_arm_64_transpose)

#endif  // IREE_BUILTINS_UKERNEL_ARCH_ARM_64_PACK_TILE_ARM_64_H_
