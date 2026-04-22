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
//       srcs = ["//runtime/src/iree/hal/cts/util:test_main.cc"],
//       deps = [
//           ":backends",
//           "//runtime/src/iree/hal/cts/core:all_tests",
//           "//runtime/src/iree/hal/cts/util:registry",
//           "//runtime/src/iree/testing:gtest",
//       ],
//   )

#include "iree/base/tooling/flags.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/testing/gtest.h"

#if defined(IREE_PLATFORM_APPLE)
#include <sys/resource.h>
#endif  // IREE_PLATFORM_APPLE

int main(int argc, char** argv) {
#if defined(IREE_PLATFORM_APPLE)
  // macOS defaults to a soft limit of 256 file descriptors per process, which
  // is too low for CTS tests that create multiple HAL devices (each device
  // needs pipe-based wait handles consuming 2 fds per handle). Raise to 4096
  // to match Linux's typical default.
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < 4096) {
    rl.rlim_cur = 4096;
    setrlimit(RLIMIT_NOFILE, &rl);
  }
#endif  // IREE_PLATFORM_APPLE

  // Pass IREE flags through before instantiating registered backends so backend
  // selection flags affect the generated gtest instances.
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);

  // Instantiate all test suites for all registered backends BEFORE gtest init.
  // This must happen before InitGoogleTest() because gtest caches test
  // enumeration state during initialization.
  iree::hal::cts::CtsRegistry::InstantiateAll();

  ::testing::InitGoogleTest(&argc, argv);

  // Register cleanup for cached backend resources (driver, device,
  // device_group, allocator). These are shared across all tests for a given
  // backend to avoid device churn, which causes reliability issues on GPU cloud
  // runners.
  ::testing::AddGlobalTestEnvironment(
      new iree::hal::cts::CtsBackendCacheEnvironment);

  return RUN_ALL_TESTS();
}
