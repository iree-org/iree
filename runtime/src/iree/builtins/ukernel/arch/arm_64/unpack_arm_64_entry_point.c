// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/unpack_arm_64_internal.h"

iree_uk_unpack_tile_func_t iree_uk_unpack_select_tile_func_arch(
    const iree_uk_unpack_params_t* params) {
  iree_uk_unpack_type_t unpack_type = iree_uk_unpack_type(params->flags);
  int esize = iree_uk_type_size(iree_uk_unpack_out_type(unpack_type));
  bool transpose = params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER;
  // Unpack is currently only used in practice with esize==4 and non-transpose.
  if (esize != 4 || transpose) return 0;
  if (params->in_size2 == 8 && params->in_size3 == 8) {
    return iree_uk_unpack_tile_8x8_x32_arm_64_direct;
  }
  return 0;
}
