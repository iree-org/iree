// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_X86_64_MMT4D_X86_64_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_ARCH_X86_64_MMT4D_X86_64_INTERNAL_H_

#include "iree/builtins/ukernel/mmt4d_internal.h"

#define IREE_UK_MMT4D_TILE(ARCH, LHS, RHS, OUT, M0, N0, K0, SUFFIX) \
  IREE_UK_MMT4D_TILE_FUNC_DECL(                                     \
      iree_uk_mmt4d_tile_##LHS##RHS##OUT##_##M0##x##N0##x##K0##_##ARCH##SUFFIX)

#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_tiles.inl"

#undef IREE_UK_MMT4D_TILE

#endif  // IREE_BUILTINS_UKERNEL_ARCH_X86_64_MMT4D_X86_64_INTERNAL_H_
