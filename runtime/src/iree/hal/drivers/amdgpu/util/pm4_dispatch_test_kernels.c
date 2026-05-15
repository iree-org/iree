// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define IREE_PM4_TEST_ATTRIBUTE_KERNEL \
  [[clang::amdgpu_kernel, gnu::visibility("protected")]]

typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

IREE_PM4_TEST_ATTRIBUTE_KERNEL void iree_hal_amdgpu_pm4_dispatch_test_store_a(
    uint32_t* target, uint32_t value) {
  target[0] = value;
}

IREE_PM4_TEST_ATTRIBUTE_KERNEL void iree_hal_amdgpu_pm4_dispatch_test_store_b(
    uint32_t* target, uint32_t value) {
  target[0] = value + 0x100u;
}

IREE_PM4_TEST_ATTRIBUTE_KERNEL void iree_hal_amdgpu_pm4_dispatch_test_read_add(
    uint32_t* source, uint32_t* target, uint32_t value) {
  target[0] = source[0] + value;
}

IREE_PM4_TEST_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_pm4_dispatch_test_patch_user_data(uint32_t* target_dwords,
                                                  uint32_t dword_offset,
                                                  uint64_t kernarg_address) {
  target_dwords[dword_offset + 0] = (uint32_t)kernarg_address;
  target_dwords[dword_offset + 1] = (uint32_t)(kernarg_address >> 32);
}
