// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_H_

#include "iree/builtins/ukernel/common.h"
#include "iree/schemas/cpu_data.h"

#if defined(IREE_DEVICE_STANDALONE)
// Standalone builds (e.g. bitcode) use our own Clang, supporting everything.
#define IREE_UK_BUILD_RISCV_64_V
#else
// Compiling with the system toolchain. Include the configured header.
#include "iree/builtins/ukernel/arch/riscv_64/config_riscv_64.h"
#endif

static inline bool iree_uk_cpu_riscv_64(const iree_uk_uint64_t* cpu_data) {
  (void)cpu_data;
  return true;
}

static inline bool iree_uk_cpu_riscv_64_v(const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_RISCV_64_V);
}

#endif  // IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_H_
