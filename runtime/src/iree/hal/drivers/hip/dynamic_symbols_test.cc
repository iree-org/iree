// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hip/dynamic_symbols.h"

#include <iostream>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace hip {
namespace {

#define HIP_CHECK_ERRORS(expr)     \
  {                                \
    hipError_t status = expr;      \
    ASSERT_EQ(hipSuccess, status); \
  }

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_hip_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_hip_dynamic_symbols_initialize(
      iree_allocator_system(), &symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  int device_count = 0;
  HIP_CHECK_ERRORS(symbols.hipInit(0));
  HIP_CHECK_ERRORS(symbols.hipGetDeviceCount(&device_count));
  if (device_count > 0) {
    hipDevice_t device;
    HIP_CHECK_ERRORS(symbols.hipDeviceGet(&device, /*ordinal=*/0));
  }

  iree_hal_hip_dynamic_symbols_deinitialize(&symbols);
}

}  // namespace
}  // namespace hip
}  // namespace hal
}  // namespace iree
