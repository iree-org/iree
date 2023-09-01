// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_X86_64_MMT4D_X86_64_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_ARCH_X86_64_MMT4D_X86_64_INTERNAL_H_

#include "iree/builtins/ukernel/mmt4d_internal.h"

IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_i8i8i32_8x8x2_x86_64_avx2_fma)
IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_f32f32f32_8x8x1_x86_64_avx2_fma)
IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_f16f16f32_8x8x1_x86_64_avx2_fma)
IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_f16f16f16_8x8x1_x86_64_avx2_fma)
IREE_UK_MMT4D_TILE_FUNC_DECL(
    iree_uk_mmt4d_tile_i8i8i32_16x16x2_x86_64_avx512_base)
IREE_UK_MMT4D_TILE_FUNC_DECL(
    iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base)
IREE_UK_MMT4D_TILE_FUNC_DECL(
    iree_uk_mmt4d_tile_f16f16f32_16x16x1_x86_64_avx512_base)
IREE_UK_MMT4D_TILE_FUNC_DECL(
    iree_uk_mmt4d_tile_f16f16f16_16x16x1_x86_64_avx512_base)

IREE_UK_MMT4D_TILE_FUNC_DECL(
    iree_uk_mmt4d_tile_i8i8i32_16x16x2_x86_64_avx512_vnni)

#endif  // foIREE_BUILTINS_UKERNEL_ARCH_X86_64_MMT4D_X86_64_INTERNAL_H_
