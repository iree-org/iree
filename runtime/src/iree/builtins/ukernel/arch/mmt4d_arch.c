// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/mmt4d_arch.h"

#if defined(IREE_UK_ARCH_ARM_64)
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64.h"
#endif

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arch(
    const iree_uk_mmt4d_params_t* params) {
#if defined(IREE_UK_ARCH_ARM_64)
  return iree_uk_mmt4d_select_tile_func_arm_64(params);
#endif
  return 0;
}
