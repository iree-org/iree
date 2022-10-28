// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_PACK_TYPES_H_
#define IREE_BUILTINS_UKERNEL_PACK_TYPES_H_

#include <assert.h>

#include "iree/builtins/ukernel/common.h"

// Supported combinations of data types (order: IN, OUT).
enum iree_ukernel_pack_type_t {
  iree_ukernel_pack_type_f32f32,
  iree_ukernel_pack_type_i8i8,
  iree_ukernel_pack_type_i32i32,
};

typedef enum iree_ukernel_pack_type_t iree_ukernel_pack_type_t;

// Parameters for a pack operation.
struct iree_ukernel_pack_params_t {
  iree_ukernel_pack_type_t type;
  const void* in_buffer;
  void* out_buffer;
  iree_ukernel_ssize_t in_stride0;
  iree_ukernel_ssize_t out_stride0;
  iree_ukernel_ssize_t in_size0;
  iree_ukernel_ssize_t in_size1;
  iree_ukernel_ssize_t out_size0;
  iree_ukernel_ssize_t out_size1;
  iree_ukernel_ssize_t out_size2;
  iree_ukernel_ssize_t out_size3;
  const void* padding_value;
  iree_ukernel_uint32_t flags;
};

typedef struct iree_ukernel_pack_params_t iree_ukernel_pack_params_t;

static int iree_ukernel_pack_elem_size(iree_ukernel_pack_type_t type) {
  switch (type) {
    case iree_ukernel_pack_type_f32f32:
    case iree_ukernel_pack_type_i32i32:
      return 4;
    case iree_ukernel_pack_type_i8i8:
      return 1;
    default:
      assert(0 && "unknown type");
      return 0;
  }
}

#endif  // IREE_BUILTINS_UKERNEL_PACK_TYPES_H_
