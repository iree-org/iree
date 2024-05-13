// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_X86_64_PACK_X86_64_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_ARCH_X86_64_PACK_X86_64_INTERNAL_H_

#include "iree/builtins/ukernel/pack_internal.h"

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
IREE_UK_PACK_TILE_FUNC_DECL(
    iree_uk_pack_tile_16x2_x16_x86_64_avx512_base_direct)
IREE_UK_PACK_TILE_FUNC_DECL(
    iree_uk_pack_tile_16x2_x16_x86_64_avx512_base_transpose)

#endif  // foIREE_BUILTINS_UKERNEL_ARCH_X86_64_PACK_X86_64_INTERNAL_H_