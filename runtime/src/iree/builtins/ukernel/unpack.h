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

// TODO(benoitjacob): move to internal, user specifies type in flags.
typedef enum iree_uk_unpack_type_t {
  iree_uk_unpack_type_f32f32 = IREE_UK_TIE_2_TYPES_LITERAL(FLOAT_32, FLOAT_32),
  iree_uk_unpack_type_i8i8 = IREE_UK_TIE_2_TYPES_LITERAL(INT_8, INT_8),
  iree_uk_unpack_type_i32i32 = IREE_UK_TIE_2_TYPES_LITERAL(INT_32, INT_32),
} iree_uk_unpack_type_t;

typedef struct iree_uk_unpack_params_t {
  iree_uk_unpack_type_t type;
  iree_uk_uint32_t flags;
  iree_uk_ssize_t in_stride0;
  iree_uk_ssize_t out_stride0;
  iree_uk_ssize_t in_size0;
  iree_uk_ssize_t in_size1;
  iree_uk_ssize_t in_size2;
  iree_uk_ssize_t in_size3;
  iree_uk_ssize_t out_size0;
  iree_uk_ssize_t out_size1;
  const void* in_buffer;
  void* out_buffer;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_unpack_params_t;

IREE_UK_EXPORT void iree_uk_unpack(const iree_uk_unpack_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_UNPACK_H_
