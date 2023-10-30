// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_X86_64_COMMON_X86_64_ENTRY_POINT_H_
#define IREE_BUILTINS_UKERNEL_ARCH_X86_64_COMMON_X86_64_ENTRY_POINT_H_

#include "iree/builtins/ukernel/common.h"
#include "iree/schemas/cpu_data.h"

#if defined(IREE_DEVICE_STANDALONE)
// Standalone builds (e.g. bitcode) use our own Clang, supporting everything.
#define IREE_UK_BUILD_X86_64_AVX2_FMA
#define IREE_UK_BUILD_X86_64_AVX512_BASE
#define IREE_UK_BUILD_X86_64_AVX512_VNNI
#define IREE_UK_BUILD_X86_64_AVX512_BF16
#else  // IREE_DEVICE_STANDALONE
// Compiling with the system toolchain. Include the configured header.
#include "iree/builtins/ukernel/arch/x86_64/config_x86_64.h"
#endif  // IREE_DEVICE_STANDALONE

static inline bool iree_uk_cpu_supports_avx2_fma(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_X86_64_AVX2 |
                                               IREE_CPU_DATA0_X86_64_FMA |
                                               IREE_CPU_DATA0_X86_64_F16C);
}

static inline bool iree_uk_cpu_supports_avx512_base(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_cpu_supports_avx2_fma(cpu_data) &&
         iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_X86_64_AVX512F |
                                               IREE_CPU_DATA0_X86_64_AVX512BW |
                                               IREE_CPU_DATA0_X86_64_AVX512DQ |
                                               IREE_CPU_DATA0_X86_64_AVX512VL |
                                               IREE_CPU_DATA0_X86_64_AVX512CD);
}

static inline bool iree_uk_cpu_supports_avx512_vnni(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_cpu_supports_avx512_base(cpu_data) &&
         iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_X86_64_AVX512VNNI);
}

static inline bool iree_uk_cpu_supports_avx512_bf16(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_cpu_supports_avx512_base(cpu_data) &&
         iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_X86_64_AVX512BF16);
}

#endif  // IREE_BUILTINS_UKERNEL_ARCH_X86_64_COMMON_X86_64_ENTRY_POINT_H_
