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

struct iree_ukernel_mmt4d_f32f32f32_params_t {
  const float* lhs_buffer;
  const float* rhs_buffer;
  float* out_buffer;
  iree_ukernel_size_t lhs_stride;
  iree_ukernel_size_t rhs_stride;
  iree_ukernel_size_t out_stride;
  iree_ukernel_size_t M;
  iree_ukernel_size_t N;
  iree_ukernel_size_t K;
  int32_t M0;
  int32_t N0;
  int32_t K0;
  uint32_t flags;
};

struct iree_ukernel_mmt4d_i8i8i32_params_t {
  const int8_t* lhs_buffer;
  const int8_t* rhs_buffer;
  int32_t* out_buffer;
  iree_ukernel_size_t lhs_stride;
  iree_ukernel_size_t rhs_stride;
  iree_ukernel_size_t out_stride;
  iree_ukernel_size_t M;
  iree_ukernel_size_t N;
  iree_ukernel_size_t K;
  int32_t M0;
  int32_t N0;
  int32_t K0;
  uint32_t flags;
};

typedef struct iree_ukernel_mmt4d_f32f32f32_params_t
    iree_ukernel_mmt4d_f32f32f32_params_t;
typedef struct iree_ukernel_mmt4d_i8i8i32_params_t
    iree_ukernel_mmt4d_i8i8i32_params_t;

#define IREE_UKERNEL_MMT4D_ERROR_UNIMPLEMENTED 1
#define IREE_UKERNEL_MMT4D_ERROR_BAD_FLAGS 2

// TODO: move these flags to a header file shared with compiler/.
#define IREE_VMVX_MATMUL_FLAG_ACCUMULATE 1

IREE_UKERNEL_EXPORT int iree_ukernel_mmt4d_f32f32f32(
    const iree_ukernel_mmt4d_f32f32f32_params_t* params);
IREE_UKERNEL_EXPORT int iree_ukernel_mmt4d_i8i8i32(
    const iree_ukernel_mmt4d_i8i8i32_params_t* params);

IREE_UKERNEL_EXPORT const char* iree_ukernel_mmt4d_error_message(int retcode);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_H_
