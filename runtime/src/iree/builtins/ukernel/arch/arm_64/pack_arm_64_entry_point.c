// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/pack_arm_64_internal.h"

iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_arch(
    const iree_uk_pack_params_t* params) {
  // At the moment, as sum-reductions are not yet part of pack ops,
  // no arithmetic whatsoever is being done here, so only the element type
  // size matters, not the type itself.
  iree_uk_pack_type_t pack_type = iree_uk_pack_type(params->flags);
  int esize = iree_uk_type_size(iree_uk_pack_out_type(pack_type));
  bool transpose = params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
  if (esize == 4 && params->out_size2 == 8 && params->out_size3 == 8) {
    // Currently only used for accumulators, which are never transposed.
    return transpose ? 0 : iree_uk_pack_tile_8x8_x32_arm_64_direct;
  } else if (esize == 4 && params->out_size2 == 8 && params->out_size3 == 1) {
    return transpose ? iree_uk_pack_tile_8x1_x32_arm_64_transpose
                     : iree_uk_pack_tile_8x1_x32_arm_64_direct;
  } else if (esize == 1 && params->out_size2 == 8 && params->out_size3 == 1) {
    return transpose ? iree_uk_pack_tile_8x1_x8_arm_64_transpose
                     : iree_uk_pack_tile_8x1_x8_arm_64_direct;
  } else if (esize == 1 && params->out_size2 == 8 && params->out_size3 == 4) {
    return transpose ? iree_uk_pack_tile_8x4_x8_arm_64_transpose
                     : iree_uk_pack_tile_8x4_x8_arm_64_direct;
  } else if (esize == 1 && params->out_size2 == 8 && params->out_size3 == 8) {
    return transpose ? iree_uk_pack_tile_8x8_x8_arm_64_transpose
                     : iree_uk_pack_tile_8x8_x8_arm_64_direct;
  }
  return 0;
}
