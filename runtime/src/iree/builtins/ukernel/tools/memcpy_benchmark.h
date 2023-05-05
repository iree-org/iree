// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_TOOLS_MEMCPY_BENCHMARK_H_
#define IREE_BUILTINS_UKERNEL_TOOLS_MEMCPY_BENCHMARK_H_

#include <stdint.h>

void iree_uk_benchmark_register_memcpy(int64_t working_set_size,
                                       int64_t batch_min_traversal_size);

#endif  // IREE_BUILTINS_UKERNEL_TOOLS_MEMCPY_BENCHMARK_H_
