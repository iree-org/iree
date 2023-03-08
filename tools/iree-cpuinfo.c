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

#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, llvm_name) \
  if (IREE_ARCH_ENUM == IREE_ARCH_ENUM_##arch) {                              \
    int64_t result = 0;                                                       \
    IREE_CHECK_OK(iree_cpu_lookup_data_by_key(IREE_SV(llvm_name), &result));  \
    printf("%-20s %ld\n", llvm_name, result);                                 \
  }
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT

  return 0;
}
