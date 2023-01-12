// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/query_tile_sizes_arch.h"

#if defined(IREE_UK_ARCH_ARM_64)
#include "iree/builtins/ukernel/arch/arm_64/query_tile_sizes_arm_64.h"
#endif

bool iree_uk_query_matmul_tile_sizes_arch(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_matmul_tile_sizes_t* out_matmul_tile_sizes) {
#if defined(IREE_UK_ARCH_ARM_64)
  return iree_uk_query_matmul_tile_sizes_arm_64(params, out_matmul_tile_sizes);
#endif
  return false;
}
