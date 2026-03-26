// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

namespace {
constexpr iree_device_size_t kAllocationSize = 1024;

// Release callback that counts how many times it has been invoked.
// Used to verify that chained caller callbacks fire exactly once.
static void CountingReleaseCallback(void* user_data, iree_hal_buffer_t*) {
  ++*static_cast<int*>(user_data);
}
}  // namespace

class AllocatorTest : public CtsTestBase<> {};

// All allocators must support some baseline capabilities.
//
// Certain capabilities or configurations are optional and may vary between
// driver implementations or target devices, such as:
//   IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL
//   IREE_HAL_BUFFER_USAGE_MAPPING
TEST_P(AllocatorTest, BaselineBufferCompatibility) {
  // Need at least one way to get data between the host and device.
  iree_hal_buffer_params_t host_local_params = {0};
  host_local_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_local_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_device_size_t host_local_allocation_size = 0;
  iree_hal_buffer_compatibility_t transfer_compatibility_host =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_, host_local_params, kAllocationSize,
          &host_local_params, &host_local_allocation_size);
  EXPECT_GE(host_local_allocation_size, kAllocationSize);

  iree_hal_buffer_params_t device_local_params = {0};
  device_local_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  device_local_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_device_size_t device_allocation_size = 0;
  iree_hal_buffer_compatibility_t transfer_compatibility_device =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_, device_local_params, kAllocationSize,
          &device_local_params, &device_allocation_size);
  EXPECT_GE(device_allocation_size, kAllocationSize);

  iree_hal_buffer_compatibility_t required_transfer_compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  EXPECT_TRUE(iree_all_bits_set(transfer_compatibility_host,
                                required_transfer_compatibility) ||
              iree_all_bits_set(transfer_compatibility_device,
                                required_transfer_compatibility));

  // Need to be able to use some type of buffer as dispatch inputs or outputs.
  iree_hal_buffer_params_t dispatch_params = {0};
  dispatch_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  dispatch_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  iree_device_size_t dispatch_allocation_size = 0;
  iree_hal_buffer_compatibility_t dispatch_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_, dispatch_params, kAllocationSize, &dispatch_params,
          &dispatch_allocation_size);
  EXPECT_GE(dispatch_allocation_size, kAllocationSize);
  EXPECT_TRUE(
      iree_all_bits_set(dispatch_compatibility,
                        IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                            IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH));
}

TEST_P(AllocatorTest, AllocateBuffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                    kAllocationSize, &buffer));

  // At a minimum, the requested memory type should be respected.
  // Additional bits may be optionally set depending on the allocator.
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_memory_type(buffer), params.type));
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer), params.usage));
  EXPECT_GE(iree_hal_buffer_allocation_size(buffer),
            kAllocationSize);  // Larger is okay.

  iree_hal_buffer_release(buffer);
}

// While empty allocations aren't particularly useful, they can occur in
// practice so we should at least be able to create them without errors.
TEST_P(AllocatorTest, AllocateEmptyBuffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, params, /*allocation_size=*/0, &buffer));

  iree_hal_buffer_release(buffer);
}

// Regression test for hipHostUnregister/cuMemHostUnregister leak:
// When a HOST_ALLOCATION is imported with a null release callback, the
// backend allocator must still call hipHostUnregister/cuMemHostUnregister on
// destroy.  The old code only invoked the stored release_callback, so a null
// callback meant unregister was never called, leaking the pinned registration.
//
// The re-import postcondition (second import must succeed) is a reliable
// oracle on backends where double-register fails.  On AMD ROCm, hipHostRegister
// is idempotent so the oracle does not fire there; see HipAllocatorTest in
// hip_allocator_test.cc for a HIP-specific regression check.
TEST_P(AllocatorTest, ImportHostAllocationNullCallback) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_device_size_t compat_size = kAllocationSize;
  iree_hal_buffer_params_t compat_params = params;
  if (!iree_all_bits_set(iree_hal_allocator_query_buffer_compatibility(
                             device_allocator_, params, kAllocationSize,
                             &compat_params, &compat_size),
                         IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
    GTEST_SKIP() << "Allocator does not support importing host allocations";
  }

  // Use the portable aligned allocator to satisfy backend alignment
  // requirements (64 bytes on HIP/CUDA/local-task).
  void* host_ptr = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc_aligned(
      iree_allocator_system(), kAllocationSize, /*min_alignment=*/64,
      /*offset=*/0, &host_ptr));

  iree_hal_external_buffer_t ext = {};
  ext.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  ext.size = kAllocationSize;
  ext.handle.host_allocation.ptr = host_ptr;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
      device_allocator_, params, &ext, iree_hal_buffer_release_callback_null(),
      &buffer));
  ASSERT_NE(nullptr, buffer);

  iree_hal_buffer_release(buffer);
  buffer = nullptr;

  // Re-import the same pointer.  On backends where double-register fails, this
  // would fail without the fix because the pointer is still registered.
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
      device_allocator_, params, &ext, iree_hal_buffer_release_callback_null(),
      &buffer));
  ASSERT_NE(nullptr, buffer);

  iree_hal_buffer_release(buffer);
  iree_allocator_free_aligned(iree_allocator_system(), host_ptr);
}

// Regression test for the chained-callback path: when a HOST_ALLOCATION is
// imported with a non-null release callback, both backend cleanup
// (hipHostUnregister/cuMemHostUnregister) and the caller callback must fire.
// The caller callback must be invoked exactly once.  The re-import
// postcondition is a reliable oracle on backends where double-register fails;
// see HipAllocatorTest in hip_allocator_test.cc for a HIP-specific check.
TEST_P(AllocatorTest, ImportHostAllocationWithCallback) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_device_size_t compat_size = kAllocationSize;
  iree_hal_buffer_params_t compat_params = params;
  if (!iree_all_bits_set(iree_hal_allocator_query_buffer_compatibility(
                             device_allocator_, params, kAllocationSize,
                             &compat_params, &compat_size),
                         IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
    GTEST_SKIP() << "Allocator does not support importing host allocations";
  }

  void* host_ptr = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc_aligned(
      iree_allocator_system(), kAllocationSize, /*min_alignment=*/64,
      /*offset=*/0, &host_ptr));

  int release_count = 0;
  iree_hal_buffer_release_callback_t callback = {};
  callback.fn = CountingReleaseCallback;
  callback.user_data = &release_count;

  iree_hal_external_buffer_t ext = {};
  ext.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  ext.size = kAllocationSize;
  ext.handle.host_allocation.ptr = host_ptr;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(device_allocator_, params,
                                                  &ext, callback, &buffer));
  ASSERT_NE(nullptr, buffer);
  EXPECT_EQ(0, release_count);

  iree_hal_buffer_release(buffer);
  buffer = nullptr;

  // Caller callback must have been invoked exactly once (via the thunk chain).
  EXPECT_EQ(1, release_count);

  // Re-import to verify that the unregister fired even when a non-null caller
  // callback was provided.  On backends where double-register fails, this would
  // fail without the fix because the pointer is still registered.
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
      device_allocator_, params, &ext, iree_hal_buffer_release_callback_null(),
      &buffer));
  ASSERT_NE(nullptr, buffer);

  iree_hal_buffer_release(buffer);
  iree_allocator_free_aligned(iree_allocator_system(), host_ptr);
}

CTS_REGISTER_TEST_SUITE(AllocatorTest);

}  // namespace iree::hal::cts
