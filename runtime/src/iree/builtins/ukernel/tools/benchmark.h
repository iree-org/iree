// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_TOOLS_BENCHMARK_H_
#define IREE_BUILTINS_UKERNEL_TOOLS_BENCHMARK_H_

#include "iree/builtins/ukernel/tools/util.h"
#include "iree/testing/benchmark.h"

// Struct for passing around benchmark user data
typedef struct iree_uk_benchmark_user_data_t iree_uk_benchmark_user_data_t;

// High level init/register/run/cleanup entry points. Used in main().
void iree_uk_benchmark_initialize(int* argc, char** argv);
void iree_uk_benchmark_register(
    const char* name,
    iree_status_t (*benchmark_func)(const iree_benchmark_def_t*,
                                    iree_benchmark_state_t*),
    const void* params, size_t params_size,
    const iree_uk_cpu_features_list_t* cpu_features);
void iree_uk_benchmark_run_and_cleanup(void);

// Like malloc, but any buffers allocated through this are freed by
// iree_uk_benchmark_run_and_cleanup. Used during benchmark registration to
// allocate buffers that will be accessed when the benchmark is run.
void* iree_uk_benchmark_static_alloc(size_t size);

// Accessors for iree_uk_benchmark_user_data_t. Used by benchmark payload funcs.
const void* iree_uk_benchmark_params(
    const iree_uk_benchmark_user_data_t* user_data);
const iree_uk_uint64_t* iree_uk_benchmark_cpu_data(
    const iree_uk_benchmark_user_data_t* user_data);
iree_uk_random_engine_t* iree_uk_benchmark_random_engine(
    const iree_uk_benchmark_user_data_t* user_data);

#endif  // IREE_BUILTINS_UKERNEL_TOOLS_BENCHMARK_H_
