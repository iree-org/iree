// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_TOOLS_UKERNEL_TEST_UTILS_H_
#define IREE_BUILTINS_UKERNEL_TOOLS_UKERNEL_TEST_UTILS_H_

#include "iree/builtins/ukernel/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Helper to determine the length of test buffers to allocate.
iree_uk_ssize_t iree_uk_test_2d_buffer_length(iree_uk_type_t type,
                                              iree_uk_ssize_t size0,
                                              iree_uk_ssize_t size1);

bool iree_uk_test_2d_buffers_equal(const void* buf1, const void* buf2,
                                   iree_uk_type_t type, iree_uk_ssize_t size0,
                                   iree_uk_ssize_t size1,
                                   iree_uk_ssize_t stride0);

// Helpers to fill buffers with pseudorandom values. The main entry point here
// is iree_uk_test_write_random_buffer.
struct iree_uk_test_random_engine_t;
typedef struct iree_uk_test_random_engine_t iree_uk_test_random_engine_t;
iree_uk_test_random_engine_t* iree_uk_test_random_engine_create();
void iree_uk_test_random_engine_destroy(iree_uk_test_random_engine_t* e);
int iree_uk_test_random_engine_get_0_65535(iree_uk_test_random_engine_t* e);
int iree_uk_test_random_engine_get_0_1(iree_uk_test_random_engine_t* e);
int iree_uk_test_random_engine_get_minus16_plus15(
    iree_uk_test_random_engine_t* e);
void iree_uk_test_write_random_buffer(void* buffer,
                                      iree_uk_ssize_t size_in_bytes,
                                      iree_uk_type_t type,
                                      iree_uk_test_random_engine_t* engine);

// Helpers to stringify types and other ukernel parameters. They all work like
// snprintf: they take a buffer and buffer length, guarantee they will zero-
// terminate the string and won't write more than `length` bytes (so they will
// write at most `length - 1` characters before the terminating zero), and
// return the number of characters in the output (without the terminating zero).
int iree_uk_test_type_str(char* buf, int buf_length, const iree_uk_type_t type);
int iree_uk_test_type_pair_str(char* buf, int buf_length,
                               const iree_uk_type_pair_t pair);
int iree_uk_test_type_triple_str(char* buf, int buf_length,
                                 const iree_uk_type_triple_t triple);
int iree_uk_test_cpu_features_str(char* buf, int buf_length,
                                  const iree_uk_uint64_t* cpu_data,
                                  int cpu_data_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BUILTINS_UKERNEL_TOOLS_UKERNEL_TEST_UTILS_H_
