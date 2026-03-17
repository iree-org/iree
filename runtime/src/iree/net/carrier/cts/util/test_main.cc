// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry point for carrier CTS test binaries.
//
// Links together test suite libraries and backend libraries to form complete
// test binaries. The registry system handles instantiation:
//
//   1. Test suites register via CTS_REGISTER_TEST_SUITE() at static init.
//   2. Backends register via CtsRegistry::RegisterBackend() at static init.
//   3. This main() calls InstantiateAll() to create gtest instances.
//   4. RUN_ALL_TESTS() executes the tests.
//
// Example binary composition (BUILD.bazel):
//   cc_test(
//       name = "carrier_cts_loopback",
//       srcs = ["//runtime/src/iree/net/carrier/cts:test_main.cc"],
//       deps = [
//           "//runtime/src/iree/net/carrier/cts:all_tests",
//           "//runtime/src/iree/net/carrier/loopback/cts:backends",
//       ],
//   )

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/testing/gtest.h"

int main(int argc, char** argv) {
  // Instantiate all test suites for all registered backends BEFORE gtest init.
  // This must happen before InitGoogleTest() because gtest caches test
  // enumeration state during initialization.
  iree::net::carrier::cts::CtsRegistry::InstantiateAll();

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
