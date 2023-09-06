// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include "iree/builtins/ukernel/softmax.h"
#include "iree/builtins/ukernel/softmax_internal.h"

static void iree_uk_softmax_validate(const iree_uk_softmax_params_t* params) {
#ifdef IREE_UK_ENABLE_ASSERTS
  const iree_uk_int32_t allflags = IREE_UK_FLAG_SOFTMAX_TYPE_MASK;
  IREE_UK_ASSERT(!(params->flags & ~allflags));
  iree_uk_uint32_t flags_type = params->flags & IREE_UK_FLAG_SOFTMAX_TYPE_MASK;
  IREE_UK_ASSERT((flags_type == IREE_UK_FLAG_SOFTMAX_TYPE_F32));
#endif
}

static bool iree_uk_softmax_early(const iree_uk_softmax_params_t* params) {
  if (params->M == 0) {
    return true;
  }
  return false;
}

static void iree_uk_softmax_using_tile_func(const iree_uk_softmax_params_t *params,
                                            iree_uk_softmax_tile_func_t tile_func) {
  const float *src_buffer = params->src_buffer;
  float *dst_buffer = params->dst_buffer;
  iree_uk_int32_t M = params->M;
  iree_uk_int32_t N = params->N;

  tile_func(src_buffer, dst_buffer, M, N);
}

IREE_UK_EXPORT int iree_uk_softmax(const iree_uk_softmax_params_t* params) {
  iree_uk_softmax_validate(params);

  if (iree_uk_softmax_early(params)) return 0;

  iree_uk_softmax_tile_func_t tile_func = iree_uk_softmax_select_tile_func(params);
  iree_uk_softmax_using_tile_func(params, tile_func);

  return 0;
}
