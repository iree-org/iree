// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_ARM_64_COMMON_ARM_64_ENTRY_POINT_H_
#define IREE_BUILTINS_UKERNEL_ARCH_ARM_64_COMMON_ARM_64_ENTRY_POINT_H_

#include "iree/builtins/ukernel/common.h"
#include "iree/schemas/cpu_data.h"

#if defined(IREE_DEVICE_STANDALONE)
// Standalone builds (e.g. bitcode) use our own Clang, supporting everything.
#define IREE_UK_BUILD_ARM_64_FULLFP16
#define IREE_UK_BUILD_ARM_64_FP16FML
#define IREE_UK_BUILD_ARM_64_BF16
#define IREE_UK_BUILD_ARM_64_DOTPROD
#define IREE_UK_BUILD_ARM_64_I8MM
#else
// Compiling with the system toolchain. Include the configured header.
#include "iree/builtins/ukernel/arch/arm_64/config_arm_64.h"
#endif

#if defined(IREE_UK_BUILD_ARM_64_FULLFP16)
static inline bool iree_uk_cpu_supports_fp16(const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_FULLFP16);
}
#endif  // IREE_UK_BUILD_ARM_64_FULLFP16

#if defined(IREE_UK_BUILD_ARM_64_FP16FML)
static inline bool iree_uk_cpu_supports_fp16fml(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_FP16FML);
}
#endif  // IREE_UK_BUILD_ARM_64_FP16FML

#if defined(IREE_UK_BUILD_ARM_64_BF16)
static inline bool iree_uk_cpu_supports_bf16(const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_BF16);
}
#endif  // IREE_UK_BUILD_ARM_64_BF16

#if defined(IREE_UK_BUILD_ARM_64_DOTPROD)
static inline bool iree_uk_cpu_supports_dotprod(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_DOTPROD);
}
#endif  // IREE_UK_BUILD_ARM_64_DOTPROD

#if defined(IREE_UK_BUILD_ARM_64_I8MM)
static inline bool iree_uk_cpu_supports_i8mm(const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_I8MM);
}
#endif  // IREE_UK_BUILD_ARM_64_I8MM

#endif  // IREE_BUILTINS_UKERNEL_ARCH_ARM_64_COMMON_ARM_64_ENTRY_POINT_H_
