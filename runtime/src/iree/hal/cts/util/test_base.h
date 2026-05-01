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
//   - RAII wrappers (Ref<T>, SemaphoreList) eliminate manual release calls

#ifndef IREE_HAL_CTS_UTIL_TEST_BASE_H_
#define IREE_HAL_CTS_UTIL_TEST_BASE_H_

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

//===----------------------------------------------------------------------===//
// Ref<T> — move-only RAII wrapper for HAL objects
//===----------------------------------------------------------------------===//

// Traits that map HAL types to their release functions.
// Add a specialization for each HAL type used in CTS tests.
template <typename T>
struct HalTraits;

#define CTS_HAL_TRAITS(type, release_fn)                \
  template <>                                           \
  struct HalTraits<type> {                              \
    static void release(type* ptr) { release_fn(ptr); } \
  }

CTS_HAL_TRAITS(iree_hal_buffer_t, iree_hal_buffer_release);
CTS_HAL_TRAITS(iree_hal_command_buffer_t, iree_hal_command_buffer_release);
CTS_HAL_TRAITS(iree_hal_semaphore_t, iree_hal_semaphore_release);
CTS_HAL_TRAITS(iree_hal_executable_t, iree_hal_executable_release);
CTS_HAL_TRAITS(iree_hal_executable_cache_t, iree_hal_executable_cache_release);
CTS_HAL_TRAITS(iree_hal_file_t, iree_hal_file_release);
CTS_HAL_TRAITS(iree_hal_fence_t, iree_hal_fence_release);
CTS_HAL_TRAITS(iree_hal_pool_t, iree_hal_pool_release);

#undef CTS_HAL_TRAITS

// Move-only RAII wrapper for HAL objects. Calls the type-appropriate release
// function on destruction. Eliminates manual release boilerplate in tests.
//
// Usage:
//   Ref<iree_hal_buffer_t> buffer;
//   CreateZeroedDeviceBuffer(1024, buffer.out());
//   // buffer auto-released when scope exits
template <typename T>
class Ref {
 public:
  Ref() = default;
  explicit Ref(T* ptr) : ptr_(ptr) {}
  ~Ref() {
    if (ptr_) HalTraits<T>::release(ptr_);
  }

  Ref(Ref&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Ref& operator=(Ref&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;

  T* get() const { return ptr_; }
  operator T*() const { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }

  // For passing to creation functions that take T** out-parameters.
  T** out() { return &ptr_; }

  // Releases ownership without calling release. Returns the raw pointer.
  T* release() {
    T* p = ptr_;
    ptr_ = nullptr;
    return p;
  }

  // Releases the current object and takes ownership of |p|.
  void reset(T* p = nullptr) {
    if (ptr_) HalTraits<T>::release(ptr_);
    ptr_ = p;
  }

 private:
  T* ptr_ = nullptr;
};

//===----------------------------------------------------------------------===//
// SemaphoreList — RAII wrapper for iree_hal_semaphore_list_t
//===----------------------------------------------------------------------===//

// Creates semaphores with specified initial values and holds the desired
// payload values for wait/signal operations. Manages semaphore lifetime
// via retain/release.
struct SemaphoreList {
  SemaphoreList() = default;
  SemaphoreList(iree_hal_device_t* device, std::vector<uint64_t> initial_values,
                std::vector<uint64_t> desired_values) {
    for (size_t i = 0; i < initial_values.size(); ++i) {
      iree_hal_semaphore_t* semaphore = NULL;
      IREE_EXPECT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_AFFINITY_ANY, initial_values[i],
          IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
      semaphores.push_back(semaphore);
    }
    payload_values = desired_values;
    assert(semaphores.size() == payload_values.size());
  }

  SemaphoreList(const iree_hal_semaphore_list_t& list) {
    semaphores.reserve(list.count);
    payload_values.reserve(list.count);
    for (iree_host_size_t i = 0; i < list.count; ++i) {
      semaphores.push_back(list.semaphores[i]);
      payload_values.push_back(list.payload_values[i]);
    }
    iree_hal_semaphore_list_retain(*this);
  }

  SemaphoreList(const SemaphoreList& other) {
    semaphores = other.semaphores;
    payload_values = other.payload_values;
    iree_hal_semaphore_list_retain(*this);
  }

  SemaphoreList& operator=(const SemaphoreList& other) {
    if (this != &other) {
      iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
      semaphores = other.semaphores;
      payload_values = other.payload_values;
      iree_hal_semaphore_list_retain(*this);
    }
    return *this;
  }

  SemaphoreList(SemaphoreList&& other) noexcept
      : semaphores(std::move(other.semaphores)),
        payload_values(std::move(other.payload_values)) {
    other.semaphores.clear();
    other.payload_values.clear();
  }

  SemaphoreList& operator=(SemaphoreList&& other) noexcept {
    if (this != &other) {
      iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
      semaphores = std::move(other.semaphores);
      payload_values = std::move(other.payload_values);
      other.semaphores.clear();
      other.payload_values.clear();
    }
    return *this;
  }

  ~SemaphoreList() {
    iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
  }

  operator iree_hal_semaphore_list_t() {
    iree_hal_semaphore_list_t list;
    list.count = semaphores.size();
    list.semaphores = semaphores.data();
    list.payload_values = payload_values.data();
    return list;
  }

  std::vector<iree_hal_semaphore_t*> semaphores;
  std::vector<uint64_t> payload_values;
};

//===----------------------------------------------------------------------===//
// Backend resource cache
//===----------------------------------------------------------------------===//

class DeviceCreateContext {
 public:
  DeviceCreateContext();
  ~DeviceCreateContext();

  DeviceCreateContext(const DeviceCreateContext&) = delete;
  DeviceCreateContext& operator=(const DeviceCreateContext&) = delete;

  iree_status_t Initialize(iree_allocator_t host_allocator);
  void Deinitialize();

  const iree_hal_device_create_params_t* params() const;
  iree_async_frontier_tracker_t* frontier_tracker() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

// Cached backend resources shared across all tests for a given backend.
// GPU backends cannot create/destroy devices per test — cloud GPU runners
// have reliability issues when devices are churned. CPU backends also benefit
// from avoiding redundant device creation overhead.
//
// Resources are created on first access and held until program exit, when
// the CtsBackendCacheEnvironment releases them in the correct order.
struct CachedBackendResources {
  DeviceCreateContext create_context;
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
      resources.create_context.Deinitialize();
    }
    GetBackendCache().clear();
  }
};

//===----------------------------------------------------------------------===//
// Deterministic test data helpers
//===----------------------------------------------------------------------===//

// Returns a byte vector of the requested |length| filled with a deterministic
// pseudo-random sequence. Reproducible across runs and test backends so copy
// tests can compare against a known pattern without introducing RNG state.
inline std::vector<uint8_t> MakeDeterministicBytes(iree_device_size_t length) {
  std::vector<uint8_t> data(length);
  for (iree_device_size_t i = 0; i < length; ++i) {
    data[i] = static_cast<uint8_t>((i * 131 + (i >> 7) * 17 + 0x5A) & 0xFF);
  }
  return data;
}

// Returns the byte at |fill_offset| of a repeating |pattern| of
// |pattern_length| bytes. Used by fill tests to build the expected byte
// sequence for a filled range.
inline uint8_t FillPatternByte(uint32_t pattern,
                               iree_host_size_t pattern_length,
                               iree_device_size_t fill_offset) {
  return static_cast<uint8_t>(
      (pattern >> (8 * (fill_offset % pattern_length))) & 0xFF);
}

// Returns a byte vector of |buffer_size| bytes where
// [target_offset, target_offset + fill_length) is filled with a repeating
// |pattern| of |pattern_length| bytes and the rest is zero. Used by fill
// tests to build expected buffer contents after a fill operation.
inline std::vector<uint8_t> MakeFilledBytes(iree_device_size_t buffer_size,
                                            iree_device_size_t target_offset,
                                            iree_device_size_t fill_length,
                                            uint32_t pattern,
                                            iree_host_size_t pattern_length) {
  std::vector<uint8_t> expected(buffer_size, 0);
  for (iree_device_size_t i = 0; i < fill_length; ++i) {
    expected[target_offset + i] = FillPatternByte(pattern, pattern_length, i);
  }
  return expected;
}

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
      iree_status_t status =
          cached.create_context.Initialize(iree_allocator_system());
      if (iree_status_is_ok(status)) {
        status =
            backend.factory(cached.create_context.params(), &driver, &device);
      }
      if (iree_status_is_unavailable(status)) {
        iree_status_ignore(status);
        cached.unavailable = true;
        cached.create_context.Deinitialize();
      } else if (!iree_status_is_ok(status)) {
        cached.create_context.Deinitialize();
        IREE_ASSERT_OK(status);
      } else {
        cached.driver = driver;
        cached.device = device;
        IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
            device, cached.create_context.frontier_tracker(),
            iree_allocator_system(), &cached.device_group));
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

  iree_status_t CreateUninitializedDeviceBuffer(
      iree_device_size_t buffer_size, iree_hal_buffer_t** out_buffer) {
    *out_buffer = nullptr;
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        device_allocator_, params, buffer_size, &buffer));
    *out_buffer = buffer;
    return iree_ok_status();
  }

  iree_status_t CreateZeroedDeviceBuffer(iree_device_size_t buffer_size,
                                         iree_hal_buffer_t** out_buffer) {
    *out_buffer = nullptr;
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* buffer = nullptr;
    iree_status_t status = iree_hal_allocator_allocate_buffer(
        device_allocator_, params, buffer_size, &buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_buffer_map_zero(buffer, 0, IREE_HAL_WHOLE_BUFFER);
    }
    if (iree_status_is_ok(status)) {
      *out_buffer = buffer;
    } else {
      iree_hal_buffer_release(buffer);
    }
    return status;
  }

  iree_status_t CreateDeviceBufferWithData(const void* source_data,
                                           iree_device_size_t buffer_size,
                                           iree_hal_buffer_t** out_buffer) {
    *out_buffer = nullptr;
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* buffer = nullptr;
    iree_status_t status = iree_hal_allocator_allocate_buffer(
        device_allocator_, params, buffer_size, &buffer);
    if (iree_status_is_ok(status)) {
      SemaphoreList empty_wait;
      SemaphoreList upload_signal(device_, {0}, {1});
      status = iree_hal_device_queue_update(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, upload_signal,
          source_data, /*source_offset=*/0, buffer, /*target_offset=*/0,
          buffer_size, IREE_HAL_UPDATE_FLAG_NONE);
      if (iree_status_is_ok(status)) {
        status = iree_hal_semaphore_list_wait(
            upload_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
      }
    }
    if (iree_status_is_ok(status)) {
      *out_buffer = buffer;
    } else {
      iree_hal_buffer_release(buffer);
    }
    return status;
  }

  template <typename PatternType>
  iree_status_t CreateFilledDeviceBuffer(iree_device_size_t buffer_size,
                                         PatternType pattern,
                                         iree_hal_buffer_t** out_buffer) {
    *out_buffer = nullptr;
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* buffer = nullptr;
    iree_status_t status = iree_hal_allocator_allocate_buffer(
        device_allocator_, params, buffer_size, &buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_buffer_map_fill(buffer, 0, IREE_HAL_WHOLE_BUFFER,
                                        &pattern, sizeof(pattern));
    }
    if (iree_status_is_ok(status)) {
      *out_buffer = buffer;
    } else {
      iree_hal_buffer_release(buffer);
    }
    return status;
  }

  //===--------------------------------------------------------------------===//
  // Data readback helpers
  //===--------------------------------------------------------------------===//

  // Reads buffer contents back to host as a vector of T.
  // Reads from |offset| to end of buffer by default.
  template <typename T>
  std::vector<T> ReadBufferData(iree_hal_buffer_t* buffer,
                                iree_device_size_t offset = 0) {
    iree_device_size_t byte_length =
        iree_hal_buffer_byte_length(buffer) - offset;
    std::vector<T> data(byte_length / sizeof(T));
    std::vector<uint8_t> bytes = ReadBufferBytes(buffer, offset, byte_length);
    if (bytes.size() == byte_length) {
      std::memcpy(data.data(), bytes.data(), byte_length);
    }
    return data;
  }

  // Reads a specific byte range from a buffer.
  std::vector<uint8_t> ReadBufferBytes(iree_hal_buffer_t* buffer,
                                       iree_device_size_t offset,
                                       iree_device_size_t length) {
    std::vector<uint8_t> data(length);
    if (iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
        iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                          IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED)) {
      IREE_EXPECT_OK(
          iree_hal_buffer_map_read(buffer, offset, data.data(), length));
      return data;
    }

    iree_io_file_handle_t* handle = nullptr;
    IREE_EXPECT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(data.data(), length),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &handle));
    if (!handle) return data;

    iree_hal_file_t* file = nullptr;
    IREE_EXPECT_OK(iree_hal_file_import(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_WRITE,
        handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
    iree_io_file_handle_release(handle);
    if (!file) return data;

    SemaphoreList empty_wait;
    SemaphoreList write_signal(device_, {0}, {1});
    IREE_EXPECT_OK(iree_hal_device_queue_write(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, write_signal, buffer,
        offset, file, /*target_offset=*/0, length, IREE_HAL_WRITE_FLAG_NONE));
    IREE_EXPECT_OK(iree_hal_semaphore_list_wait(
        write_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    iree_hal_file_release(file);
    return data;
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
                                       IREE_ASYNC_WAIT_FLAG_NONE);
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
