// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_TOOLS_UTIL_H_
#define IREE_BUILTINS_UKERNEL_TOOLS_UTIL_H_

#include "iree/builtins/ukernel/common.h"

// Helper to determine the length of test buffers to allocate.
iree_uk_ssize_t iree_uk_2d_buffer_length(iree_uk_type_t type,
                                         iree_uk_ssize_t size0,
                                         iree_uk_ssize_t size1);

bool iree_uk_2d_buffers_equal(const void* buf1, const void* buf2,
                              iree_uk_type_t type, iree_uk_ssize_t size0,
                              iree_uk_ssize_t size1, iree_uk_ssize_t stride0);

// Simple deterministic pseudorandom generator. Same as C++'s std::minstd_rand.
typedef struct iree_uk_random_engine_t {
  iree_uk_uint32_t state;
} iree_uk_random_engine_t;

static inline iree_uk_random_engine_t iree_uk_random_engine_init(void) {
  return (iree_uk_random_engine_t){.state = 1};
}

int iree_uk_random_engine_get_0_65535(iree_uk_random_engine_t* e);
int iree_uk_random_engine_get_0_1(iree_uk_random_engine_t* e);
int iree_uk_random_engine_get_minus16_plus15(iree_uk_random_engine_t* e);
void iree_uk_write_random_buffer(void* buffer, iree_uk_ssize_t size_in_bytes,
                                 iree_uk_type_t type,
                                 iree_uk_random_engine_t* engine);

// Helpers to stringify types and other ukernel parameters. They all work like
// snprintf: they take a buffer and buffer length, guarantee they will zero-
// terminate the string and won't write more than `length` bytes (so they will
// write at most `length - 1` characters before the terminating zero), and
// return the number of characters in the output (without the terminating zero).
int iree_uk_type_str(char* buf, int buf_length, const iree_uk_type_t type);
int iree_uk_type_pair_str(char* buf, int buf_length,
                          const iree_uk_type_pair_t pair);
int iree_uk_type_triple_str(char* buf, int buf_length,
                            const iree_uk_type_triple_t triple);

typedef struct iree_uk_cpu_features_list_t iree_uk_cpu_features_list_t;

iree_uk_cpu_features_list_t* iree_uk_cpu_features_list_create(int count, ...);
void iree_uk_cpu_features_list_destroy(iree_uk_cpu_features_list_t* list);
void iree_uk_cpu_features_list_append(iree_uk_cpu_features_list_t* list,
                                      int count, ...);
iree_uk_cpu_features_list_t* iree_uk_cpu_features_list_create_extend(
    iree_uk_cpu_features_list_t* list, int count, ...);
int iree_uk_cpu_features_list_size(const iree_uk_cpu_features_list_t* list);
const char* iree_uk_cpu_features_list_entry(
    const iree_uk_cpu_features_list_t* list, int index);
void iree_uk_cpu_features_list_set_name(iree_uk_cpu_features_list_t* list,
                                        const char* name);
const char* iree_uk_cpu_features_list_get_name(
    const iree_uk_cpu_features_list_t* list);

void iree_uk_make_cpu_data_for_features(
    const iree_uk_cpu_features_list_t* cpu_features,
    iree_uk_uint64_t* out_cpu_data_fields);

void iree_uk_initialize_cpu_once(void);

bool iree_uk_cpu_supports(const iree_uk_uint64_t* cpu_data_fields);

const char* iree_uk_cpu_first_unsupported_feature(
    const iree_uk_cpu_features_list_t* cpu_features);

typedef struct iree_uk_standard_cpu_features_t {
#if defined(IREE_UK_ARCH_ARM_64)
  iree_uk_cpu_features_list_t* dotprod;
  iree_uk_cpu_features_list_t* i8mm;
#elif defined(IREE_UK_ARCH_X86_64)
  iree_uk_cpu_features_list_t* avx2_fma;
  iree_uk_cpu_features_list_t* avx512_base;
  iree_uk_cpu_features_list_t* avx512_vnni;
#endif
} iree_uk_standard_cpu_features_t;

iree_uk_standard_cpu_features_t* iree_uk_standard_cpu_features_create(void);
void iree_uk_standard_cpu_features_destroy(
    iree_uk_standard_cpu_features_t* cpu);

#endif  // IREE_BUILTINS_UKERNEL_TOOLS_UTIL_H_
