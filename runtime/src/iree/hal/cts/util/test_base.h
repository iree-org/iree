// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base class for HAL conformance tests.
//
// Provides the test fixture and helper utilities shared by all HAL CTS tests.
// Test logic lives in separate .cc files (core/allocator_test.cc, etc.)
// which use CTS_REGISTER_TEST_SUITE() for self-registration.
//
// Key design:
//   - Shared driver/device per backend via cache (no GPU device churn)
//   - Per-test resource isolation (semaphores, buffers, command buffers)
//   - Capability-gated GTEST_SKIP instead of external EXCLUDED_TESTS lists
//   - Link-time composition: test suites + backends linked together

#ifndef IREE_HAL_CTS_UTIL_TEST_BASE_H_
#define IREE_HAL_CTS_UTIL_TEST_BASE_H_

#include <cstdlib>
#include <map>
#include <string>
#include <string_view>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

//===----------------------------------------------------------------------===//
// Backend resource cache
//===----------------------------------------------------------------------===//

// Cached backend resources shared across all tests for a given backend.
// GPU backends cannot create/destroy devices per test — cloud GPU runners
// have reliability issues when devices are churned. CPU backends also benefit
// from avoiding redundant device creation overhead.
//
// Resources are created on first access and held until program exit, when
// the CtsBackendCacheEnvironment releases them in the correct order.
struct CachedBackendResources {
  iree_hal_driver_t* driver = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  iree_hal_device_t* device = nullptr;
  iree_hal_allocator_t* allocator = nullptr;
  bool unavailable = false;  // Factory returned UNAVAILABLE.
};

inline std::map<std::string, CachedBackendResources>& GetBackendCache() {
  static std::map<std::string, CachedBackendResources> cache;
  return cache;
}

// GTest environment that releases all cached backend resources at program exit.
// Must be registered via ::testing::AddGlobalTestEnvironment() in test_main.cc.
class CtsBackendCacheEnvironment : public ::testing::Environment {
 public:
  void TearDown() override {
    for (auto& [name, resources] : GetBackendCache()) {
      iree_hal_allocator_release(resources.allocator);
      iree_hal_device_release(resources.device);
      // Release device group after device — device holds a raw pointer to the
      // group's embedded topology.
      iree_hal_device_group_release(resources.device_group);
      iree_hal_driver_release(resources.driver);
    }
    GetBackendCache().clear();
  }
};

// Base class for all HAL CTS tests. Parameterized on BackendInfo.
// Creates a fresh driver + device in SetUp(), releases in TearDown().
template <typename BaseType = ::testing::TestWithParam<BackendInfo>>
class CtsTestBase : public BaseType {
 protected:
  void SetUp() override {
    const BackendInfo& backend = this->GetParam();
    std::string test_identity = ExtractTestIdentity();

    // Check unsupported tests first (permanent categorical exclusion).
    for (const auto& entry : backend.unsupported_tests) {
      iree_string_view_t pattern =
          iree_make_cstring_view(entry.pattern.c_str());
      iree_string_view_t value = iree_make_cstring_view(test_identity.c_str());
      if (iree_string_view_match_pattern(value, pattern)) {
        GTEST_SKIP() << "Unsupported on '" << backend.name
                     << "': " << entry.reason;
        return;
      }
    }

    // Check expected failures (temporary xfail).
    for (const auto& entry : backend.expected_failures) {
      iree_string_view_t pattern =
          iree_make_cstring_view(entry.pattern.c_str());
      iree_string_view_t value = iree_make_cstring_view(test_identity.c_str());
      if (iree_string_view_match_pattern(value, pattern)) {
        if (VerifyXfailsEnabled()) {
          // Verify mode: run the test to detect XPASS.
          xfail_active_ = true;
          xfail_identity_ = test_identity;
          xfail_reason_ = entry.reason;
          break;
        }
        GTEST_SKIP() << "Expected failure (xfail) on '" << backend.name
                     << "': " << entry.reason;
        return;
      }
    }

    // Get or create cached backend resources. GPU backends cannot
    // create/destroy devices per test (cloud runners have reliability issues
    // with device churn). CPU backends also benefit from avoiding redundant
    // creation overhead.
    auto& cached = GetBackendCache()[backend.name];
    if (!cached.device && !cached.unavailable) {
      iree_hal_driver_t* driver = nullptr;
      iree_hal_device_t* device = nullptr;
      iree_status_t status = backend.factory(&driver, &device);
      if (iree_status_is_unavailable(status)) {
        iree_status_ignore(status);
        cached.unavailable = true;
      } else {
        IREE_ASSERT_OK(status);
        cached.driver = driver;
        cached.device = device;
        IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
            device, iree_allocator_system(), &cached.device_group));
        cached.allocator = iree_hal_device_allocator(device);
        iree_hal_allocator_retain(cached.allocator);
      }
    }
    if (cached.unavailable) {
      GTEST_SKIP() << "Backend '" << backend.name
                   << "' unavailable on this system";
      return;
    }

    // Retain cached resources for this test's use. The cache holds its own
    // refs; per-test refs are released in TearDown.
    driver_ = cached.driver;
    iree_hal_driver_retain(driver_);
    device_group_ = cached.device_group;
    iree_hal_device_group_retain(device_group_);
    device_ = cached.device;
    iree_hal_device_retain(device_);
    device_allocator_ = cached.allocator;
    iree_hal_allocator_retain(device_allocator_);
  }

  void TearDown() override {
    // XPASS detection: if this test was expected to fail but passed, flag it
    // so the stale xfail entry can be removed.
    if (xfail_active_ && !this->HasFailure() && !this->IsSkipped()) {
      ADD_FAILURE() << "XPASS: '" << xfail_identity_
                    << "' was expected to fail but passed. Remove the xfail "
                    << "entry: " << xfail_reason_;
    }
    xfail_active_ = false;

    // Release per-test refs. The cache still holds its own refs — these
    // releases just decrement the refcount, they don't destroy anything.
    iree_hal_allocator_release(device_allocator_);
    device_allocator_ = nullptr;
    iree_hal_device_release(device_);
    device_ = nullptr;
    iree_hal_device_group_release(device_group_);
    device_group_ = nullptr;
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
  }

  // Returns the recording mode for the current test parameterization.
  // Command buffer tests registered via CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE
  // will see either kDirect or kIndirect depending on the instantiation.
  RecordingMode recording_mode() const {
    return this->GetParam().recording_mode;
  }

  // Returns the executable format string for the current backend, or nullptr
  // if the backend does not provide pre-compiled executables.
  const char* executable_format() const {
    return this->GetParam().executable_format;
  }

  // Returns pre-compiled executable data for the given file name.
  // Returns an empty span if the backend has no executable data function or
  // the file is not found.
  iree_const_byte_span_t executable_data(iree_string_view_t file_name) const {
    ExecutableDataFn data_fn = this->GetParam().executable_data;
    if (!data_fn) return iree_const_byte_span_empty();
    return data_fn(file_name);
  }

  //===--------------------------------------------------------------------===//
  // Buffer helpers
  //===--------------------------------------------------------------------===//

  void CreateUninitializedDeviceBuffer(iree_device_size_t buffer_size,
                                       iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage =
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                      buffer_size, &buffer));
    *out_buffer = buffer;
  }

  void CreateZeroedDeviceBuffer(iree_device_size_t buffer_size,
                                iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                      buffer_size, &buffer));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(buffer, 0, IREE_HAL_WHOLE_BUFFER));
    *out_buffer = buffer;
  }

  void CreateDeviceBufferWithData(const void* source_data,
                                  iree_device_size_t buffer_size,
                                  iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage =
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                      buffer_size, &buffer));
    IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
        device_, source_data, buffer, /*target_offset=*/0, buffer_size,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    *out_buffer = buffer;
  }

  template <typename PatternType>
  void CreateFilledDeviceBuffer(iree_device_size_t buffer_size,
                                PatternType pattern,
                                iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                      buffer_size, &buffer));
    IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, IREE_HAL_WHOLE_BUFFER,
                                            &pattern, sizeof(pattern)));
    *out_buffer = buffer;
  }

  //===--------------------------------------------------------------------===//
  // Command buffer and semaphore helpers
  //===--------------------------------------------------------------------===//

  // Submits |command_buffer| to the device and waits for completion.
  iree_status_t SubmitCommandBufferAndWait(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table =
          iree_hal_buffer_binding_table_empty()) {
    iree_hal_semaphore_list_t wait_semaphores = iree_hal_semaphore_list_empty();

    iree_hal_semaphore_t* signal_semaphore = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
    uint64_t target_payload_value = 1ull;
    iree_hal_semaphore_list_t signal_semaphores = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&target_payload_value,
    };

    iree_status_t status = iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
        signal_semaphores, command_buffer, binding_table,
        IREE_HAL_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(signal_semaphore, target_payload_value,
                                       iree_infinite_timeout(),
                                       IREE_HAL_WAIT_FLAG_DEFAULT);
    }

    iree_hal_semaphore_release(signal_semaphore);
    return status;
  }

  // NOTE: These helpers use IREE_EXPECT_OK (non-fatal) because IREE_ASSERT_OK
  // expands to `return;` which is incompatible with non-void return types.
  // If creation fails, the EXPECT records the failure and the null return
  // crashes at the call site with a clear backtrace.

  iree_hal_command_buffer_t* CreateEmptyCommandBuffer(
      iree_host_size_t binding_capacity = 0) {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        binding_capacity, &command_buffer));
    IREE_EXPECT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_EXPECT_OK(iree_hal_command_buffer_end(command_buffer));
    return command_buffer;
  }

  iree_hal_semaphore_t* CreateSemaphore() {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_EXPECT_OK(
        iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    return semaphore;
  }

  void CheckSemaphoreValue(iree_hal_semaphore_t* semaphore,
                           uint64_t expected_value) {
    uint64_t value;
    IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore, &value));
    EXPECT_EQ(expected_value, value);
  }

  // Checks that |a| carries the same error identity as |b|: same status code,
  // and (when status annotations are available) the annotation message of |b|
  // appears within |a|'s formatted representation.
  // Takes ownership of both statuses.
  //
  // With IREE_STATUS_MODE=0 (runtime_small), statuses carry no annotations or
  // source locations — the code check alone is the meaningful verification.
  // With higher modes, iree_status_clone captures a new stack trace from the
  // clone call site rather than preserving the original's, so we strip stack
  // trace payloads before checking string containment.
  void CheckStatusContains(iree_status_t a, iree_status_t b) {
    EXPECT_EQ(iree_status_code(a), iree_status_code(b));
#if IREE_STATUS_FEATURES != 0
    iree_allocator_t allocator = iree_allocator_system();
    char* a_string = nullptr;
    iree_host_size_t a_string_length = 0;
    ASSERT_TRUE(
        iree_status_to_string(a, &allocator, &a_string, &a_string_length));
    char* b_string = nullptr;
    iree_host_size_t b_string_length = 0;
    ASSERT_TRUE(
        iree_status_to_string(b, &allocator, &b_string, &b_string_length));
    // Strip stack trace payloads ("; stack:\n...") from both strings so
    // that differing clone-site traces don't cause spurious failures.
    std::string_view a_view(a_string, a_string_length);
    std::string_view b_view(b_string, b_string_length);
    auto a_stack = a_view.find("; stack:");
    auto b_stack = b_view.find("; stack:");
    if (a_stack != std::string_view::npos) a_view = a_view.substr(0, a_stack);
    if (b_stack != std::string_view::npos) b_view = b_view.substr(0, b_stack);
    EXPECT_NE(a_view.find(b_view), std::string_view::npos)
        << "  a: " << a_view << "\n  b: " << b_view;
    iree_allocator_free(allocator, a_string);
    iree_allocator_free(allocator, b_string);
#endif  // IREE_STATUS_FEATURES != 0
    iree_status_ignore(a);
    iree_status_ignore(b);
  }

  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_group_t* device_group_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* device_allocator_ = nullptr;

 private:
  // Extracts the canonical test identity "TestClass.TestMethod" from GTest's
  // parameterized test naming. Strips the CTS/CTS_Indirect prefix from the
  // suite name and the backend suffix from the test name.
  //
  // GTest naming for parameterized tests:
  //   test_suite_name() = "CTS/DispatchTest"     → "DispatchTest"
  //   name()            = "DispatchAbs/local_task_vmvx" → "DispatchAbs"
  //   Result: "DispatchTest.DispatchAbs"
  std::string ExtractTestIdentity() {
    const ::testing::TestInfo* test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = test_info->test_suite_name();
    std::string test_name = test_info->name();

    // Strip prefix: "CTS/DispatchTest" → "DispatchTest"
    auto slash_pos = suite_name.find('/');
    if (slash_pos != std::string::npos) {
      suite_name = suite_name.substr(slash_pos + 1);
    }

    // Strip suffix: "DispatchAbs/local_task_vmvx" → "DispatchAbs"
    slash_pos = test_name.find('/');
    if (slash_pos != std::string::npos) {
      test_name = test_name.substr(0, slash_pos);
    }

    return suite_name + "." + test_name;
  }

  // Returns true if IREE_CTS_VERIFY_XFAILS=1 is set, enabling verify mode
  // where xfail tests run instead of being skipped and XPASS is flagged.
  static bool VerifyXfailsEnabled() {
    const char* value = std::getenv("IREE_CTS_VERIFY_XFAILS");
    return value && std::string(value) == "1";
  }

  bool xfail_active_ = false;
  std::string xfail_identity_;
  std::string xfail_reason_;
};

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_UTIL_TEST_BASE_H_
