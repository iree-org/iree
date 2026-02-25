// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry point for HAL CTS test binaries.
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
//   iree_runtime_cc_test(
//       name = "core_tests",
//       srcs = ["//runtime/src/iree/hal/cts2/util:test_main.cc"],
//       deps = [
//           ":backends",
//           "//runtime/src/iree/hal/cts2/core:all_tests",
//           "//runtime/src/iree/hal/cts2/util:registry",
//           "//runtime/src/iree/testing:gtest",
//       ],
//   )

#include "iree/hal/cts2/util/registry.h"
#include "iree/testing/gtest.h"

int main(int argc, char** argv) {
  // Instantiate all test suites for all registered backends BEFORE gtest init.
  // This must happen before InitGoogleTest() because gtest caches test
  // enumeration state during initialization.
  iree::hal::cts::CtsRegistry::InstantiateAll();

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
