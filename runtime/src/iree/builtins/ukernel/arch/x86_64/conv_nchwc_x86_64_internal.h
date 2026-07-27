// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_X86_64_CONV_NCHWC_X86_64_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_ARCH_X86_64_CONV_NCHWC_X86_64_INTERNAL_H_

#include "iree/builtins/ukernel/conv_nchwc_internal.h"

#define IREE_UK_CONV_NCHWC_TILE(ARCH, LHS, RHS, OUT, OW_TILE, K0, C0, SUFFIX)         \
  void                                                                                \
  iree_uk_conv_nchwc_tile_##LHS##RHS##OUT##_##OW_TILE##x##K0##x##C0##_##ARCH##SUFFIX( \
      void* IREE_UK_RESTRICT output_panel,                                            \
      const void* IREE_UK_RESTRICT input_panel,                                       \
      const void* IREE_UK_RESTRICT filter_panel,                                      \
      const iree_uk_conv_nchwc_params_t* params);

#include "iree/builtins/ukernel/arch/x86_64/conv_nchwc_x86_64_tiles.inl"

#undef IREE_UK_CONV_NCHWC_TILE

#endif  // IREE_BUILTINS_UKERNEL_ARCH_X86_64_CONV_NCHWC_X86_64_INTERNAL_H_
