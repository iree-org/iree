// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_TOOLS_MMT4D_TEST_UTILS_H_
#define IREE_BUILTINS_UKERNEL_TOOLS_MMT4D_TEST_UTILS_H_

#include "iree/builtins/ukernel/mmt4d_types.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_ukernel_ssize_t iree_ukernel_mmt4d_lhs_buffer_size(
    const iree_ukernel_mmt4d_params_t* params);
iree_ukernel_ssize_t iree_ukernel_mmt4d_rhs_buffer_size(
    const iree_ukernel_mmt4d_params_t* params);
iree_ukernel_ssize_t iree_ukernel_mmt4d_out_buffer_size(
    const iree_ukernel_mmt4d_params_t* params);

struct iree_mmt4d_test_random_engine_t;
typedef struct iree_mmt4d_test_random_engine_t iree_mmt4d_test_random_engine_t;
iree_mmt4d_test_random_engine_t* iree_mmt4d_test_random_engine_create();
void iree_mmt4d_test_random_engine_destroy(iree_mmt4d_test_random_engine_t* e);
int iree_mmt4d_test_random_engine_get_0_or_1(
    iree_mmt4d_test_random_engine_t* e);
int iree_mmt4d_test_random_engine_get_between_minus16_and_plus15(
    iree_mmt4d_test_random_engine_t* e);

void write_random_buffer(void* buffer, iree_ukernel_ssize_t size_in_bytes,
                         iree_ukernel_type_t type,
                         iree_mmt4d_test_random_engine_t* engine);

const char* get_mmt4d_type_str(const iree_ukernel_mmt4d_params_t* params);
const char* get_cpu_features_str(const iree_ukernel_mmt4d_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BUILTINS_UKERNEL_TOOLS_MMT4D_TEST_UTILS_H_
