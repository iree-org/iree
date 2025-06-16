// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/unpack_riscv_64_internal.h"

iree_uk_unpack_tile_func_t iree_uk_unpack_select_tile_func_arch(
    const iree_uk_unpack_params_t* params) {
  // UnPack ukernels for riscv_64 have not been implemented yet
  // fallback to generic implementation
  return 0;
}
