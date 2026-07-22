// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// No arm_64-specific conv_nchwc tile function yet. Falls back to generic.
// TODO: implement arm_64 tile functions analogous to the +dotprod / +i8mm /
// +bf16 dispatch in mmt4d_arm_64_entry_point.c.
#include "iree/builtins/ukernel/conv_nchwc_internal.h"

iree_uk_conv_nchwc_tile_selection_t iree_uk_conv_nchwc_select_tile_func_arch(
    const iree_uk_conv_nchwc_params_t* params) {
  (void)params;
  iree_uk_conv_nchwc_tile_selection_t selection = {0};
  return selection;
}
