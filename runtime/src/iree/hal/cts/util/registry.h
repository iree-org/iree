// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS Registry for link-time HAL test composition.
//
// This registry enables HAL test suites to be compiled once and linked against
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
//   CTS_REGISTER_TEST_SUITE(AllocatorTest);  // All backends, once each.
//   CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CopyBufferTest);  //
//   Direct+indirect. CTS_REGISTER_EXECUTABLE_TEST_SUITE(ExecutableTest);  //
//   Per (device, format).
//   CTS_REGISTER_EXECUTABLE_COMMAND_BUFFER_TEST_SUITE(DispatchTest);  // All
//   axes.
//
// Example backend registration:
//   CtsRegistry::RegisterBackend({
//       "local_task",
//       {.name = "local_task", .factory = CreateLocalTaskDevice},
//       {"events", "indirect"},
//       {{"vmvx", "vmvx-bytecode-fb", GetVmvxExecutableData}},
//   });

#ifndef IREE_HAL_CTS_UTIL_REGISTRY_H_
#define IREE_HAL_CTS_UTIL_REGISTRY_H_

#include <functional>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"

namespace iree::hal::cts {

//===----------------------------------------------------------------------===//
// Backend factory
//===----------------------------------------------------------------------===//

// Factory function that creates a HAL driver and device pair.
//
// The shared CTS fixture owns |create_params| storage and keeps all borrowed
// resources in it alive for the lifetime of the returned device. Factories must
// pass it through to driver device creation unchanged unless the backend has a
// documented driver-specific extension chain to add.
//
// Returns:
//   iree_ok_status(): Success. Both out params populated with retained refs.
//   IREE_STATUS_UNAVAILABLE: Backend not present on this system. Tests skip.
//   Any other error: Creation failed. Tests fail.
//
// The caller takes ownership of both returned objects and must release them.
using DeviceFactory = std::function<iree_status_t(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device)>;

// Function that returns pre-compiled executable data for a given file name.
// Used by dispatch tests to load backend-specific device code.
// Returns an empty span if the file is not found.
using ExecutableDataFn =
    iree_const_byte_span_t (*)(iree_string_view_t file_name);

//===----------------------------------------------------------------------===//
// Executable format
//===----------------------------------------------------------------------===//

// A pre-compiled executable format available for a backend.
// Backends declare the formats they support; executable test suites expand
// parameterizations across all available formats. This keeps non-executable
// tests (allocator, semaphore, etc.) running once per device while executable
// tests run once per (device, format) pair.
struct ExecutableFormat {
  std::string name;    // Suffix appended to backend name: "vmvx", "llvm_cpu".
  const char* format;  // Format string: "vmvx-bytecode-fb".
  ExecutableDataFn data_fn;  // Lookup function for compiled binaries.
};

//===----------------------------------------------------------------------===//
// Recording mode
//===----------------------------------------------------------------------===//

// Command buffer recording mode for parameterized command buffer tests.
// CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE creates instantiations for both modes.
enum class RecordingMode {
  kDirect,    // Inline buffer references, binding_capacity = 0.
  kIndirect,  // Binding table slots, binding_capacity > 0.
};

//===----------------------------------------------------------------------===//
// Test exclusions and expected failures
//===----------------------------------------------------------------------===//

// Permanent categorical exclusion. The backend fundamentally cannot support
// this test (e.g., no file I/O, no indirect command buffers). These entries
// should never be removed — they document inherent backend limitations.
//
// Pattern is matched against "TestClass.TestMethod" using
// iree_string_view_match_pattern() glob syntax (* and ?).
struct TestUnsupported {
  std::string pattern;  // Glob: "TestClass.Method" or "TestClass.*"
  std::string reason;   // Why this will never work.
};

// Temporary expected failure. The test should eventually pass but currently
// doesn't due to incomplete implementation. These shrink over time as the
// backend matures.
//
// Pattern is matched against "TestClass.TestMethod" using
// iree_string_view_match_pattern() glob syntax (* and ?).
//
// In verify mode (IREE_CTS_VERIFY_XFAILS=1), xfail tests run instead of
// being skipped, and unexpected passes (XPASS) are flagged as test failures
// so stale entries can be detected and removed.
struct TestExpectedFailure {
  std::string pattern;  // Glob: "TestClass.Method" or "TestClass.*"
  std::string reason;   // What's broken and what the fix requires.
};

//===----------------------------------------------------------------------===//
// Backend info (test parameterization type)
//===----------------------------------------------------------------------===//

// Identifies a HAL backend for test parameterization.
// Tests receive this via GetParam() and use it to create devices and load
// executables.
struct BackendInfo {
  std::string name;                            // Human-readable backend name.
  DeviceFactory factory;                       // Creates driver + device.
  const char* executable_format = nullptr;     // E.g., "vmvx-bytecode-fb".
  ExecutableDataFn executable_data = nullptr;  // Compiled executable lookup.
  RecordingMode recording_mode = RecordingMode::kDirect;
  std::vector<TestUnsupported> unsupported_tests;
  std::vector<TestExpectedFailure> expected_failures;
};

// Returns human-readable test suffix from BackendInfo.
// Used as the generator for INSTANTIATE_TEST_SUITE_P.
struct BackendName {
  std::string operator()(
      const ::testing::TestParamInfo<BackendInfo>& info) const {
    return info.param.name;
  }
};

// Printer for BackendInfo in gtest assertions and failure messages.
// Without this, gtest prints the raw bytes of the struct.
inline void PrintTo(const BackendInfo& info, std::ostream* os) {
  *os << info.name;
}

//===----------------------------------------------------------------------===//
// Backend configuration with tags
//===----------------------------------------------------------------------===//

// Identifies a backend configuration for test instantiation.
// Extends BackendInfo with tags for filtering which test suites apply.
struct BackendConfig {
  const char* name;               // "local_task", etc.
  BackendInfo info;               // Factory + capabilities.
  std::vector<std::string> tags;  // {"events", "indirect", ...}
  std::vector<ExecutableFormat> executable_formats;  // Available formats.
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
// CTS Registry
//===----------------------------------------------------------------------===//

class CtsRegistry {
 public:
  //===--------------------------------------------------------------------===//
  // Registration (called at static init time)
  //===--------------------------------------------------------------------===//

  // Register a test suite. Called by CTS_REGISTER_TEST_SUITE macro.
  static void RegisterSuite(TestSuiteInfo info);

  // Register a backend configuration. Called by backend factory files.
  static void RegisterBackend(BackendConfig config);

  // Register an executable format for an already-registered (or
  // not-yet-registered) backend. Formats are stored in a pending list and
  // merged into their backends at InstantiateAll() time, so static init
  // ordering between RegisterBackend() and RegisterExecutableFormat() does
  // not matter.
  static void RegisterExecutableFormat(const char* backend_name,
                                       ExecutableFormat format);

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
// required_tags: backend must have ALL of these tags.
// excluded_tags: backend must have NONE of these tags.
//
// Each test suite has a static vector of BackendInfos that accumulates
// matching backends during InstantiateAll(). After all backends are added,
// the suite is instantiated once with ValuesIn() covering all backends.
#define CTS_REGISTER_TEST_SUITE_WITH_TAGS(TestClass, required_tags,           \
                                          excluded_tags)                      \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestClass);                   \
  namespace {                                                                 \
  struct TestClass##_Backends {                                               \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  static bool TestClass##_registered_ =                                       \
      (::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass,                                                       \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              TestClass##_Backends::Get().push_back(cfg.info);                \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_Backends::Get().empty()) return;                \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS", &TestClass##_Backends::Generator,                \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__, __LINE__, required_tags, excluded_tags}),               \
       true)

// Dual-mode registration for command buffer tests.
//
// Creates two gtest instantiation sets from a single test class:
//   - CTS/TestClass.* parameterized with direct-mode BackendInfos
//   - CTS_Indirect/TestClass.* parameterized with indirect-mode BackendInfos
//     (only for backends with the "indirect" tag)
//
// Tests call recording_mode() to get the current mode and create command
// buffers accordingly (direct uses inline buffer references, indirect uses
// binding table slots).
#define CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(TestClass)                     \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestClass);                   \
  namespace {                                                                 \
  struct TestClass##_DirectBackends {                                         \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  struct TestClass##_IndirectBackends {                                       \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  static bool TestClass##_registered_ =                                       \
      (::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass,                                                       \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              auto info = cfg.info;                                           \
              info.recording_mode = ::iree::hal::cts::RecordingMode::kDirect; \
              TestClass##_DirectBackends::Get().push_back(std::move(info));   \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_DirectBackends::Get().empty()) return;          \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS", &TestClass##_DirectBackends::Generator,          \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {},                                                               \
            {}}),                                                             \
       ::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass "_Indirect",                                           \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              auto info = cfg.info;                                           \
              info.recording_mode =                                           \
                  ::iree::hal::cts::RecordingMode::kIndirect;                 \
              TestClass##_IndirectBackends::Get().push_back(std::move(info)); \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_IndirectBackends::Get().empty()) return;        \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS_Indirect",                                         \
                      &TestClass##_IndirectBackends::Generator,               \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {"indirect"},                                                     \
            {}}),                                                             \
       true)

// Executable test registration - suite runs once per (device, format) pair.
//
// Backends with no executable_formats are silently skipped (the accumulator
// pushes nothing, and GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST handles
// the empty suite). The "executables" tag is unnecessary — format availability
// is structural, not string-based.
//
// For each matching backend, the accumulator iterates executable_formats and
// pushes one BackendInfo per format with name = "backend_format",
// executable_format and executable_data populated from the ExecutableFormat.
#define CTS_REGISTER_EXECUTABLE_TEST_SUITE(TestClass)                         \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestClass);                   \
  namespace {                                                                 \
  struct TestClass##_Backends {                                               \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  static bool TestClass##_registered_ =                                       \
      (::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass,                                                       \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              for (const auto& fmt : cfg.executable_formats) {                \
                auto info = cfg.info;                                         \
                info.name = std::string(cfg.name) + "_" + fmt.name;           \
                info.executable_format = fmt.format;                          \
                info.executable_data = fmt.data_fn;                           \
                TestClass##_Backends::Get().push_back(std::move(info));       \
              }                                                               \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_Backends::Get().empty()) return;                \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS", &TestClass##_Backends::Generator,                \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {},                                                               \
            {}}),                                                             \
       true)

// Executable command buffer test registration - suite runs once per
// (device, format, recording_mode) triple.
//
// Combines executable format expansion with direct/indirect recording mode
// expansion. Creates two gtest instantiation sets:
//   - CTS/TestClass.* with direct-mode BackendInfos for each format
//   - CTS_Indirect/TestClass.* with indirect-mode BackendInfos for each format
//     (only for backends with the "indirect" tag)
#define CTS_REGISTER_EXECUTABLE_COMMAND_BUFFER_TEST_SUITE(TestClass)          \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestClass);                   \
  namespace {                                                                 \
  struct TestClass##_DirectBackends {                                         \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  struct TestClass##_IndirectBackends {                                       \
    static std::vector<::iree::hal::cts::BackendInfo>& Get() {                \
      static std::vector<::iree::hal::cts::BackendInfo> backends;             \
      return backends;                                                        \
    }                                                                         \
    static ::testing::internal::ParamGenerator<::iree::hal::cts::BackendInfo> \
    Generator() {                                                             \
      return ::testing::ValuesIn(Get());                                      \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  static bool TestClass##_registered_ =                                       \
      (::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass,                                                       \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              for (const auto& fmt : cfg.executable_formats) {                \
                auto info = cfg.info;                                         \
                info.name = std::string(cfg.name) + "_" + fmt.name;           \
                info.executable_format = fmt.format;                          \
                info.executable_data = fmt.data_fn;                           \
                info.recording_mode =                                         \
                    ::iree::hal::cts::RecordingMode::kDirect;                 \
                TestClass##_DirectBackends::Get().push_back(std::move(info)); \
              }                                                               \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_DirectBackends::Get().empty()) return;          \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS", &TestClass##_DirectBackends::Generator,          \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {},                                                               \
            {}}),                                                             \
       ::iree::hal::cts::CtsRegistry::RegisterSuite(                          \
           {#TestClass "_Indirect",                                           \
            [](const ::iree::hal::cts::BackendConfig& cfg) {                  \
              for (const auto& fmt : cfg.executable_formats) {                \
                auto info = cfg.info;                                         \
                info.name = std::string(cfg.name) + "_" + fmt.name;           \
                info.executable_format = fmt.format;                          \
                info.executable_data = fmt.data_fn;                           \
                info.recording_mode =                                         \
                    ::iree::hal::cts::RecordingMode::kIndirect;               \
                TestClass##_IndirectBackends::Get().push_back(                \
                    std::move(info));                                         \
              }                                                               \
            },                                                                \
            [](const char* file, int line) {                                  \
              if (TestClass##_IndirectBackends::Get().empty()) return;        \
              ::testing::UnitTest::GetInstance()                              \
                  ->parameterized_test_registry()                             \
                  .GetTestSuitePatternHolder<TestClass>(                      \
                      #TestClass,                                             \
                      ::testing::internal::CodeLocation(file, line))          \
                  ->AddTestSuiteInstantiation(                                \
                      "CTS_Indirect",                                         \
                      &TestClass##_IndirectBackends::Generator,               \
                      &::iree::hal::cts::internal::GetBackendName, file,      \
                      line);                                                  \
            },                                                                \
            __FILE__,                                                         \
            __LINE__,                                                         \
            {"indirect"},                                                     \
            {}}),                                                             \
       true)

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_UTIL_REGISTRY_H_
