// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_MMT4D_RISCV_64_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_MMT4D_RISCV_64_INTERNAL_H_

#include "iree/builtins/ukernel/mmt4d_internal.h"

#define IREE_UK_MMT4D_TILE(ARCH, LHS, RHS, OUT, M0, K0, SUFFIX) \
  IREE_UK_MMT4D_TILE_FUNC_DECL(                                 \
      iree_uk_mmt4d_tile_##LHS##RHS##OUT##_##M0##xXXx##K0##_##ARCH##SUFFIX)

#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_tiles.inl"

#undef IREE_UK_MMT4D_TILE

#endif  // IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_MMT4D_RISCV_64_INTERNAL_H_
