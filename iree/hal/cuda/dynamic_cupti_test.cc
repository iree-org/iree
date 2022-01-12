// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace cuda {
namespace {

#define CUDE_CHECK_ERRORS(expr)           \
  {                                       \
    CUresult status = expr;               \
    IREE_ASSERT_EQ(CUDA_SUCCESS, status); \
  }

#define CUPTI_CHECK_ERRORS(expr)           \
  {                                        \
    CUptiResult status = expr;             \
    IREE_ASSERT_EQ(CUPTI_SUCCESS, status); \
  }

// Buffer request callack for CUPTI activity records.
void bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {}

// Buffer completed callack for CUPTI activity records.
void bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                     size_t size, size_t validSize) {}

TEST(DynamicCuptiTest, CreateFromSystemLoader) {
  iree_hal_cuda_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_cuda_dynamic_symbols_initialize(
      iree_allocator_system(), &symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  // Device timestamp querying
  uint64_t timestamp1 = 0;
  uint64_t timestamp2 = 0;
  CUPTI_CHECK_ERRORS(symbols.cuptiGetTimestamp(&timestamp1));
  CUPTI_CHECK_ERRORS(symbols.cuptiGetTimestamp(&timestamp2));
  IREE_ASSERT_GT(timestamp2, timestamp1);
  IREE_ASSERT_NE(timestamp1, 0);
  IREE_ASSERT_NE(timestamp2, 0);

  // Activity API record collection and callbacks
  CUPTI_CHECK_ERRORS(
      symbols.cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CHECK_ERRORS(
      symbols.cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  CUPTI_CHECK_ERRORS(symbols.cuptiActivityFlushAll(0));

  iree_hal_cuda_dynamic_symbols_deinitialize(&symbols);
}

}  // namespace
}  // namespace cuda
}  // namespace hal
}  // namespace iree
