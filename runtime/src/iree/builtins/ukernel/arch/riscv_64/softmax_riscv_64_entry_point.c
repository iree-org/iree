// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64_entry_point.h"
#include "iree/builtins/ukernel/arch/riscv_64/softmax_riscv_64_internal.h"

iree_uk_softmax_tile_func_t iree_uk_softmax_select_tile_func_arch(
    const iree_uk_softmax_params_t* params) {
  return iree_uk_softmax_tile_riscv_64_f32_rvv;
}
