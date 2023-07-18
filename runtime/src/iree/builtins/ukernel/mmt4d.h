// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_MMT4D_H_
#define IREE_BUILTINS_UKERNEL_MMT4D_H_

#include "iree/builtins/ukernel/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_uk_mmt4d_params_t {
  const void* lhs_buffer;
  iree_uk_index_t lhs_offset;
  iree_uk_index_t lhs_stride0;
  const void* rhs_buffer;
  iree_uk_index_t rhs_offset;
  iree_uk_index_t rhs_stride0;
  void* out_buffer;
  iree_uk_index_t out_offset;
  iree_uk_index_t out_stride0;
  iree_uk_index_t M;
  iree_uk_index_t N;
  iree_uk_index_t K;
  iree_uk_int32_t M0;
  iree_uk_int32_t N0;
  iree_uk_int32_t K0;
  iree_uk_uint32_t flags;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_mmt4d_params_t;

IREE_UK_EXPORT int iree_uk_mmt4d(const iree_uk_mmt4d_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_H_
