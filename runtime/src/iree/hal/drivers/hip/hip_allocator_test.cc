// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HIP-specific regression test for the hipHostUnregister lifecycle fix.
//
// Background: when a HOST_ALLOCATION is imported via
// iree_hal_allocator_import_buffer the HIP allocator calls hipHostRegister to
// pin the memory.  The allocator must call hipHostUnregister on buffer destroy
// to release that registration.  Before the fix, destroying a buffer imported
// with a null release callback silently skipped the unregister call.
//
// CTS generic tests verify functional correctness (re-import succeeds), but
// on AMD ROCm hipHostRegister is idempotent, so double-register doesn't fail
// and those tests pass even without the fix.
//
// This test uses hipHostUnregister as a white-box oracle: after the buffer is
// destroyed, calling hipHostUnregister on the original host pointer should
// return hipErrorHostMemoryNotRegistered — meaning the fix already called it.
// Without the fix the pointer is still registered and the call returns
// hipSuccess instead.

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/registration/driver_module.h"
#include "iree/testing/gtest.h"

namespace iree::hal::hip {
namespace {

// Fixture that initializes HIP symbols and creates a HAL device/allocator once
// per test.  Tests are skipped gracefully if HIP is unavailable.
class HipAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_status_t status = iree_hal_hip_dynamic_symbols_initialize(
        iree_allocator_system(), /*hip_lib_search_path_count=*/0,
        /*hip_lib_search_paths=*/nullptr, &syms_);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "HIP symbols not available";
    }
    syms_initialized_ = true;

    status =
        iree_hal_hip_driver_module_register(iree_hal_driver_registry_default());
    if (iree_status_is_already_exists(status)) {
      iree_status_ignore(status);
      status = iree_ok_status();
    }
    ASSERT_TRUE(iree_status_is_ok(status));

    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("hip"),
        iree_allocator_system(), &driver_);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "HIP driver not available";
    }

    const iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    status = iree_hal_driver_create_default_device(
        driver_, &create_params, iree_allocator_system(), &device_);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "No HIP device available";
    }
  }

  void TearDown() override {
    iree_hal_device_release(device_);
    device_ = nullptr;
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
    if (syms_initialized_) {
      iree_hal_hip_dynamic_symbols_deinitialize(&syms_);
      syms_initialized_ = false;
    }
  }

  iree_hal_hip_dynamic_symbols_t syms_ = {};
  bool syms_initialized_ = false;
  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
};

// After importing a HOST_ALLOCATION and releasing the buffer, the HIP
// allocator must have called hipHostUnregister.  We verify this by attempting
// a manual hipHostUnregister on the same pointer: if the fix fired it returns
// hipErrorHostMemoryNotRegistered; without the fix it returns hipSuccess.
TEST_F(HipAllocatorTest, ImportHostAllocationUnregistersOnDestroy) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  constexpr iree_device_size_t kSize = 1024;
  iree_hal_buffer_params_t compat_params = params;
  iree_device_size_t compat_size = kSize;
  if (!iree_all_bits_set(
          iree_hal_allocator_query_buffer_compatibility(
              allocator, params, kSize, &compat_params, &compat_size),
          IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
    GTEST_SKIP() << "Allocator does not support importing host allocations";
  }

  void* host_ptr = nullptr;
  ASSERT_TRUE(iree_status_is_ok(iree_allocator_malloc_aligned(
      iree_allocator_system(), kSize, /*min_alignment=*/64, /*offset=*/0,
      &host_ptr)));

  iree_hal_external_buffer_t ext = {};
  ext.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  ext.size = kSize;
  ext.handle.host_allocation.ptr = host_ptr;

  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_TRUE(iree_status_is_ok(iree_hal_allocator_import_buffer(
      allocator, params, &ext, iree_hal_buffer_release_callback_null(),
      &buffer)));
  ASSERT_NE(nullptr, buffer);

  iree_hal_buffer_release(buffer);
  buffer = nullptr;

  // Oracle: hipHostUnregister should fail because the fix already called it
  // on buffer destroy.  On the buggy path the pointer is still registered and
  // this returns hipSuccess instead.
  hipError_t result = syms_.hipHostUnregister(host_ptr);
  EXPECT_EQ(hipErrorHostMemoryNotRegistered, result)
      << "Expected hipErrorHostMemoryNotRegistered — the fix should have "
         "called hipHostUnregister on buffer destroy, but got: "
      << syms_.hipGetErrorName(result);

  iree_allocator_free_aligned(iree_allocator_system(), host_ptr);
}

// Same oracle test for the non-null caller callback path.
TEST_F(HipAllocatorTest, ImportHostAllocationWithCallbackUnregistersOnDestroy) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  constexpr iree_device_size_t kSize = 1024;
  iree_hal_buffer_params_t compat_params = params;
  iree_device_size_t compat_size = kSize;
  if (!iree_all_bits_set(
          iree_hal_allocator_query_buffer_compatibility(
              allocator, params, kSize, &compat_params, &compat_size),
          IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
    GTEST_SKIP() << "Allocator does not support importing host allocations";
  }

  void* host_ptr = nullptr;
  ASSERT_TRUE(iree_status_is_ok(iree_allocator_malloc_aligned(
      iree_allocator_system(), kSize, /*min_alignment=*/64, /*offset=*/0,
      &host_ptr)));

  int release_count = 0;
  iree_hal_buffer_release_callback_t callback = {};
  callback.fn = [](void* user_data, iree_hal_buffer_t*) {
    ++*static_cast<int*>(user_data);
  };
  callback.user_data = &release_count;

  iree_hal_external_buffer_t ext = {};
  ext.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  ext.size = kSize;
  ext.handle.host_allocation.ptr = host_ptr;

  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_TRUE(iree_status_is_ok(iree_hal_allocator_import_buffer(
      allocator, params, &ext, callback, &buffer)));
  ASSERT_NE(nullptr, buffer);

  iree_hal_buffer_release(buffer);
  buffer = nullptr;

  // Caller callback must have fired exactly once via the thunk chain.
  EXPECT_EQ(1, release_count);

  // Oracle: hipHostUnregister must have been called (pointer now unregistered).
  hipError_t result = syms_.hipHostUnregister(host_ptr);
  EXPECT_EQ(hipErrorHostMemoryNotRegistered, result)
      << "Expected hipErrorHostMemoryNotRegistered — the fix should have "
         "called hipHostUnregister on buffer destroy, but got: "
      << syms_.hipGetErrorName(result);

  iree_allocator_free_aligned(iree_allocator_system(), host_ptr);
}

}  // namespace
}  // namespace iree::hal::hip
