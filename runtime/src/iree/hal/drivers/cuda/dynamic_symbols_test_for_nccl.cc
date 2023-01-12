// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>

#include "iree/base/api.h"
#include "iree/hal/drivers/cuda/dynamic_symbols.h"
#include "iree/testing/gtest.h"
#if IREE_HAL_CUDA_NCCL_ENABLE
#include "nccl.h"
#endif  // IREE_HAL_CUDA_NCCL_ENABLE

namespace iree {
namespace hal {
namespace cuda {
namespace {

#define NCCL_CHECK_ERRORS(expr)     \
  {                                 \
    ncclResult_t status = expr;     \
    ASSERT_EQ(ncclSuccess, status); \
  }

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_cuda_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_cuda_dynamic_symbols_initialize(
      iree_allocator_system(), &symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_FAIL();
  }

  int nccl_version = 0;
  NCCL_CHECK_ERRORS(symbols.ncclGetVersion(&nccl_version));
  ASSERT_EQ(NCCL_VERSION_CODE, nccl_version);
  iree_hal_cuda_dynamic_symbols_deinitialize(&symbols);
}

}  // namespace
}  // namespace cuda
}  // namespace hal
}  // namespace iree
