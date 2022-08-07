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

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Example matmul tile. This should be removed after we have any real methods
// that can be used to demonstrate how this all fits together. The shape of the
// method is close to what we need (proper buffer types, using the
// iree_mmt4d_size_t type for strides, and taking the minimal required metadata)
// but what we pass is dependent on the what the compiler produces.
//
// Returns 0 if the parameters were valid and the operation was performed.
// Non-zero results will fail the entire submission or lose the device and
// should be used as if an abort() ("no correct execution is possible after
// this point").
IREE_UKERNEL_EXPORT int iree_mmt4d_example_matmul_f32(
    const float* lhs, iree_ukernel_size_t lhs_stride, const float* rhs,
    iree_ukernel_size_t rhs_stride, float* IREE_RESTRICT out,
    iree_ukernel_size_t out_stride, int32_t m, int32_t n, int32_t k,
    float alpha, float beta);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_H_
