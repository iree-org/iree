// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mmt4d_riscv_32_internal.h"

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_riscv_32_i8i8i32(
    const iree_uk_mmt4d_params_t* params) {
  if (params->M0 == 16 && params->N0 == 16 && params->K0 == 16) {
    return iree_uk_mmt4d_tile_i8i8i32_16x16x16_riscv_32;
  }
  return 0;
}

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arch(
    const iree_uk_mmt4d_params_t* params) {
  switch (iree_uk_mmt4d_type(params->flags)) {
    case iree_uk_mmt4d_type_i8i8i32:
      return iree_uk_mmt4d_select_tile_func_riscv_32_i8i8i32(params);
    default:
      return 0;
  }
}
