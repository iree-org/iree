// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_SOFTMAX_H_
#define IREE_BUILTINS_UKERNEL_SOFTMAX_H_

#include "iree/builtins/ukernel/common.h"

#ifdef __cplusplus
extern "c" {
#endif  // __cplusplus
        //
typedef struct iree_uk_softmax_params_t {
        const void* src_buffer;
        iree_uk_index_t in_M;
        iree_uk_index_t in_N;
        void* out_buffer;
        iree_uk_index_t out_M;
        iree_uk_index_t out_N;

} iree_uk_softmax_params_t;

IREE_UK_EXPORT int iree_uk_softmax(const iree_uk_softmax_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_SOFTMAX_H_
