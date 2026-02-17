// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for proactor lifecycle operations.
//
// Tests create, retain, release, and capability queries. These are the most
// basic operations and should pass before any operation-specific tests.

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"

namespace iree::async::cts {

class LifecycleTest : public CtsTestBase<> {};

// Proactor was created successfully (SetUp didn't skip).
TEST_P(LifecycleTest, CreateSucceeds) {
  // If we get here, SetUp succeeded and proactor_ is valid.
  EXPECT_NE(proactor_, nullptr);
}

// Query capabilities returns a valid value (at least NONE).
TEST_P(LifecycleTest, QueryCapabilities) {
  iree_async_proactor_capabilities_t caps =
      iree_async_proactor_query_capabilities(proactor_);
  // Capabilities is a bitmask - any value is valid, including 0.
  // This test just verifies the call doesn't crash.
  (void)caps;
}

// Retain/release pair doesn't crash or leak.
TEST_P(LifecycleTest, RetainRelease) {
  // Retain bumps refcount.
  iree_async_proactor_retain(proactor_);

  // Release decrements but shouldn't destroy (SetUp holds a ref).
  iree_async_proactor_release(proactor_);

  // Proactor should still be usable.
  iree_async_proactor_capabilities_t caps =
      iree_async_proactor_query_capabilities(proactor_);
  (void)caps;
}

// Multiple retain/release cycles.
TEST_P(LifecycleTest, MultipleRetainRelease) {
  for (int i = 0; i < 10; ++i) {
    iree_async_proactor_retain(proactor_);
  }
  for (int i = 0; i < 10; ++i) {
    iree_async_proactor_release(proactor_);
  }

  // Still usable.
  EXPECT_NE(proactor_, nullptr);
}

// Poll with immediate timeout on idle proactor returns deadline exceeded.
TEST_P(LifecycleTest, PollIdleReturnsDeadlineExceeded) {
  iree_host_size_t completed = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_async_proactor_poll(
                            proactor_, iree_immediate_timeout(), &completed));
  EXPECT_EQ(completed, 0u);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(LifecycleTest);

}  // namespace iree::async::cts
