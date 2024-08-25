// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/dynamic_symbols.h"

#include <iostream>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace hsa {
namespace {

#define HSA_CHECK_ERRORS(expr)             \
  {                                        \
    hsa_status_t status = expr;            \
    ASSERT_EQ(HSA_STATUS_SUCCESS, status); \
  }

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_hsa_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_hsa_dynamic_symbols_initialize(
      iree_allocator_system(), &symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  HSA_CHECK_ERRORS(symbols.hsa_init());
  HSA_CHECK_ERRORS(symbols.hsa_shut_down());

  iree_hal_hsa_dynamic_symbols_deinitialize(&symbols);
}

}  // namespace
}  // namespace hsa
}  // namespace hal
}  // namespace iree
