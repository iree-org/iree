// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"

int main(int argc, char *argv[]) {
  iree_cpu_initialize(iree_allocator_system());
  const uint64_t* cpu_data = iree_cpu_data_fields();

#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, llvm_name) \
  if (IREE_ARCH_ENUM == IREE_ARCH_ENUM_##arch) {                              \
    bool result = (cpu_data[field_index] & (1ull << bit_pos)) != 0;           \
    printf("%-20s %ld\n", llvm_name, result);                                 \
  }
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT

  return 0;
}
