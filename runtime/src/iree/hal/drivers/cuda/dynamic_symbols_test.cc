// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/dynamic_symbols.h"

#include <iostream>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace cuda {
namespace {

#define CUDA_CHECK_ERRORS(expr)      \
  {                                  \
    CUresult status = expr;          \
    ASSERT_EQ(CUDA_SUCCESS, status); \
  }

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_cuda_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_cuda_dynamic_symbols_initialize(
      iree_allocator_system(), &symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  int device_count = 0;
  CUDA_CHECK_ERRORS(symbols.cuInit(0));
  CUDA_CHECK_ERRORS(symbols.cuDeviceGetCount(&device_count));
  if (device_count > 0) {
    CUdevice device;
    CUDA_CHECK_ERRORS(symbols.cuDeviceGet(&device, /*ordinal=*/0));
  }

  iree_hal_cuda_dynamic_symbols_deinitialize(&symbols);
}

}  // namespace
}  // namespace cuda
}  // namespace hal
}  // namespace iree
