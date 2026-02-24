// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS Registry for link-time test and benchmark composition.
//
// This registry enables test suites to be compiled once and linked against
// multiple backends, eliminating per-backend instantiation boilerplate.
//
// Architecture:
//   1. Test suites register themselves at static init time via
//      CTS_REGISTER_TEST_SUITE().
//   2. Backends register themselves at static init time via
//      CtsRegistry::RegisterBackend().
//   3. main() calls CtsRegistry::InstantiateAll() before RUN_ALL_TESTS().
//   4. InstantiateAll() creates gtest parameterized test instances for each
//      suite × backend pair that passes tag filtering.
//
// Tag filtering:
//   - required_tags: Backend must have ALL of these tags.
//   - excluded_tags: Backend must have NONE of these tags.
//
// Example test suite registration:
//   CTS_REGISTER_TEST_SUITE(SocketLifecycleTest);  // All backends
//   CTS_REGISTER_TEST_SUITE_WITH_TAGS(ZeroCopyTest, {"zerocopy"}, {});
//
// Example backend registration:
//   CtsRegistry::RegisterBackend({
//       "io_uring",
//       CreateIoUringBackend(),
//       {"zerocopy", "multishot", "registered_buffers"}
//   });

#ifndef IREE_ASYNC_CTS_UTIL_REGISTRY_H_
#define IREE_ASYNC_CTS_UTIL_REGISTRY_H_

#include <functional>
#include <string>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Backend factory (matches test_base.h)
//===----------------------------------------------------------------------===//

// Factory function that creates a proactor backend. Each backend's
// registration provides its own factory, allowing implementation-specific
// options beyond iree_async_proactor_options_default().
using ProactorFactory = std::function<iree::StatusOr<iree_async_proactor_t*>()>;

// Identifies a proactor backend for test parameterization.
// This matches the BackendInfo in test_base.h for compatibility.
struct BackendInfo {
  const char* name;  // Human-readable name.
  // Creates a proactor with backend-specific options.
  ProactorFactory factory;
};

// Returns human-readable test suffix from BackendInfo.
// Used as the generator for INSTANTIATE_TEST_SUITE_P.
struct BackendName {
  std::string operator()(
      const ::testing::TestParamInfo<BackendInfo>& info) const {
    return info.param.name;
  }
};

// Printer for BackendInfo in gtest assertions/failure messages.
// Without this, gtest prints the raw bytes of the struct.
inline void PrintTo(const BackendInfo& info, std::ostream* os) {
  *os << info.name;
}

//===----------------------------------------------------------------------===//
// Backend configuration with tags
//===----------------------------------------------------------------------===//

// Identifies a backend configuration for test instantiation.
// Extends BackendInfo with tags for filtering which tests apply.
struct BackendConfig {
  const char* name;               // "io_uring", "io_uring_no_zerocopy", etc.
  BackendInfo info;               // Factory function + capabilities.
  std::vector<std::string> tags;  // {"zerocopy", "multishot", ...}
};

//===----------------------------------------------------------------------===//
// Test suite registration
//===----------------------------------------------------------------------===//

// Function called for each matching backend to accumulate backends for a suite.
using TestSuiteAccumulator = std::function<void(const BackendConfig& config)>;

// Function called once after all backends are accumulated to create gtest
// instances.
using TestSuiteFinalizer = std::function<void(const char* file, int line)>;

// Metadata about a registered test suite.
struct TestSuiteInfo {
  const char* name;
  TestSuiteAccumulator accumulator;
  TestSuiteFinalizer finalizer;
  const char* file;
  int line;

  // Required tags: suite only instantiated for backends with ALL of these.
  std::vector<std::string> required_tags;

  // Excluded tags: suite skipped for backends with ANY of these.
  std::vector<std::string> excluded_tags;
};

//===----------------------------------------------------------------------===//
// Benchmark registration
//===----------------------------------------------------------------------===//

// Function that registers benchmarks for a given backend config.
using BenchmarkInstantiator = std::function<void(const BackendConfig& config)>;

// Metadata about a registered benchmark suite.
struct BenchmarkSuiteInfo {
  const char* name;
  BenchmarkInstantiator instantiator;
  std::vector<std::string> required_tags;
  std::vector<std::string> excluded_tags;
};

//===----------------------------------------------------------------------===//
// CTS Registry
//===----------------------------------------------------------------------===//

class CtsRegistry {
 public:
  //===--------------------------------------------------------------------===//
  // Registration (called at static init time)
  //===--------------------------------------------------------------------===//

  // Register a test suite. Called by CTS_REGISTER_TEST_SUITE macro.
  static void RegisterSuite(TestSuiteInfo info);

  // Register a benchmark suite. Called by CTS_REGISTER_BENCHMARK macro.
  static void RegisterBenchmark(BenchmarkSuiteInfo info);

  // Register a backend configuration. Called by backend factory files.
  static void RegisterBackend(BackendConfig config);

  //===--------------------------------------------------------------------===//
  // Instantiation (called from main, before RUN_ALL_TESTS)
  //===--------------------------------------------------------------------===//

  // Instantiate all suites × all backends, respecting tag filters.
  static void InstantiateAll();

  // Instantiate specific suite for specific backend (for debugging).
  static void Instantiate(const char* suite_name, const char* backend_name);

  //===--------------------------------------------------------------------===//
  // Introspection (for tooling, test listing)
  //===--------------------------------------------------------------------===//

  static std::vector<std::string> ListSuites();
  static std::vector<std::string> ListBackends();

 private:
  // Check if backend has all required tags and none of the excluded tags.
  static bool TagsMatch(const BackendConfig& backend,
                        const std::vector<std::string>& required,
                        const std::vector<std::string>& excluded);
};

//===----------------------------------------------------------------------===//
// Test Registration Helpers
//===----------------------------------------------------------------------===//

// The gtest AddTestSuiteInstantiation API requires function pointers, not
// capturing lambdas. Since we can't create unique function pointers at runtime,
// we use a single-instantiation approach: each test suite is instantiated once
// with all registered backends, and tag filtering happens via the suite's
// instantiator function which only calls AddTestSuiteInstantiation for
// matching backends.
//
// Each test suite has its own static generator that returns all backends
// that were registered for it.

namespace internal {

// Name generator function for test instantiation.
inline std::string GetBackendName(
    const ::testing::TestParamInfo<BackendInfo>& info) {
  return BackendName()(info);
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Test Registration Macros
//===----------------------------------------------------------------------===//

// Basic registration - suite runs against all backends.
#define CTS_REGISTER_TEST_SUITE(TestClass) \
  CTS_REGISTER_TEST_SUITE_WITH_TAGS(TestClass, {}, {})

// Registration with tag requirements.
// required_tags: backend must have ALL of these tags
// excluded_tags: backend must have NONE of these tags
//
// Each test suite has a static vector of BackendInfos that accumulates
// matching backends during InstantiateAll(). After all backends are added,
// the suite is instantiated once with ValuesIn() covering all backends.
#define CTS_REGISTER_TEST_SUITE_WITH_TAGS(TestClass, required_tags,        \
                                          excluded_tags)                   \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestClass);                \
  namespace {                                                              \
  struct TestClass##_Backends {                                            \
    static std::vector<::iree::async::cts::BackendInfo>& Get() {           \
      static std::vector<::iree::async::cts::BackendInfo> backends;        \
      return backends;                                                     \
    }                                                                      \
    static ::testing::internal::ParamGenerator<                            \
        ::iree::async::cts::BackendInfo>                                   \
    Generator() {                                                          \
      return ::testing::ValuesIn(Get());                                   \
    }                                                                      \
  };                                                                       \
  }                                                                        \
  static bool TestClass##_registered_ =                                    \
      (::iree::async::cts::CtsRegistry::RegisterSuite(                     \
           {#TestClass,                                                    \
            [](const ::iree::async::cts::BackendConfig& cfg) {             \
              TestClass##_Backends::Get().push_back(cfg.info);             \
            },                                                             \
            [](const char* file, int line) {                               \
              if (TestClass##_Backends::Get().empty()) return;             \
              ::testing::UnitTest::GetInstance()                           \
                  ->parameterized_test_registry()                          \
                  .GetTestSuitePatternHolder<TestClass>(                   \
                      #TestClass,                                          \
                      ::testing::internal::CodeLocation(file, line))       \
                  ->AddTestSuiteInstantiation(                             \
                      "CTS", &TestClass##_Backends::Generator,             \
                      &::iree::async::cts::internal::GetBackendName, file, \
                      line);                                               \
            },                                                             \
            __FILE__, __LINE__, required_tags, excluded_tags}),            \
       true)

//===----------------------------------------------------------------------===//
// Benchmark Suite Registration Macros
//===----------------------------------------------------------------------===//

// Basic benchmark suite registration - runs against all backends.
// The BenchmarkClass must have a static method:
//   static void RegisterBenchmarks(const char* prefix,
//                                   const ProactorFactory& factory);
#define CTS_REGISTER_BENCHMARK_SUITE(BenchmarkClass) \
  CTS_REGISTER_BENCHMARK_SUITE_WITH_TAGS(BenchmarkClass, {}, {})

// Benchmark suite registration with tag requirements.
// required_tags: backend must have ALL of these tags.
// excluded_tags: backend must have NONE of these tags.
#define CTS_REGISTER_BENCHMARK_SUITE_WITH_TAGS(BenchmarkClass, required_tags, \
                                               excluded_tags)                 \
  static bool BenchmarkClass##_benchmark_registered_ =                        \
      (::iree::async::cts::CtsRegistry::RegisterBenchmark(                    \
           {#BenchmarkClass,                                                  \
            [](const ::iree::async::cts::BackendConfig& cfg) {                \
              BenchmarkClass::RegisterBenchmarks(cfg.name, cfg.info.factory); \
            },                                                                \
            required_tags, excluded_tags}),                                   \
       true)

}  // namespace iree::async::cts

#endif  // IREE_ASYNC_CTS_UTIL_REGISTRY_H_
