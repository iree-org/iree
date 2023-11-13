// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_UNPACK_H_
#define IREE_BUILTINS_UKERNEL_UNPACK_H_

#include "iree/builtins/ukernel/common.h"

// `unpack` microkernel. Currently only used in the VMVX backend, not used in
// the LLVMCPU backend, because codegen is thought to be good enough and because
// pack ops tend to get fused with many other ops, with substantial performance
// benefit outweighing the microkernel advantage.

typedef struct iree_uk_unpack_params_t {
  const void* in_buffer;
  iree_uk_index_t in_offset;
  iree_uk_index_t in_stride0;
  void* out_buffer;
  iree_uk_index_t out_offset;
  iree_uk_index_t out_stride0;
  iree_uk_index_t in_size0;
  iree_uk_index_t in_size1;
  iree_uk_index_t in_size2;
  iree_uk_index_t in_size3;
  iree_uk_index_t out_size0;
  iree_uk_index_t out_size1;
  iree_uk_uint32_t flags;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_unpack_params_t;

IREE_UK_EXPORT int iree_uk_unpack(const iree_uk_unpack_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_UNPACK_H_
