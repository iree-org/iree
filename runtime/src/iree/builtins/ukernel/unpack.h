// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_UNPACK_H_
#define IREE_BUILTINS_UKERNEL_UNPACK_H_

#include "iree/builtins/ukernel/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_uk_unpack_params_t {
  const void* in_buffer;
  iree_uk_ssize_t in_offset;
  iree_uk_ssize_t in_stride0;
  void* out_buffer;
  iree_uk_ssize_t out_offset;
  iree_uk_ssize_t out_stride0;
  iree_uk_ssize_t in_size0;
  iree_uk_ssize_t in_size1;
  iree_uk_ssize_t in_size2;
  iree_uk_ssize_t in_size3;
  iree_uk_ssize_t out_size0;
  iree_uk_ssize_t out_size1;
  iree_uk_uint32_t flags;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_unpack_params_t;

IREE_UK_EXPORT void iree_uk_unpack(const iree_uk_unpack_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_UNPACK_H_
