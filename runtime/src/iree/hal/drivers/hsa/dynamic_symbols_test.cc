// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hsa/dynamic_symbols.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace {

TEST(DynamicSymbolsTest, LoadSymbols) {
  iree_hal_hsa_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_hsa_dynamic_symbols_initialize(
      iree_allocator_system(), /*hsa_lib_search_path_count=*/0,
      /*hsa_lib_search_paths=*/nullptr, &symbols);

  if (iree_status_is_unavailable(status)) {
    GTEST_SKIP() << "HSA runtime library not available";
    iree_status_ignore(status);
    return;
  }

  ASSERT_TRUE(iree_status_is_ok(status))
      << iree_status_code_string(iree_status_code(status));

  // Verify some required symbols are loaded.
  EXPECT_NE(symbols.hsa_init, nullptr);
  EXPECT_NE(symbols.hsa_shut_down, nullptr);
  EXPECT_NE(symbols.hsa_iterate_agents, nullptr);
  EXPECT_NE(symbols.hsa_agent_get_info, nullptr);
  EXPECT_NE(symbols.hsa_queue_create, nullptr);
  EXPECT_NE(symbols.hsa_signal_create, nullptr);
  EXPECT_NE(symbols.hsa_amd_memory_pool_allocate, nullptr);

  iree_hal_hsa_dynamic_symbols_deinitialize(&symbols);
}

}  // namespace
