// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class AllocatorTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  class TestLogicalDevice {
   public:
    ~TestLogicalDevice() {
      iree_hal_device_release(base_device_);
      iree_hal_device_group_release(device_group_);
    }

    iree_status_t Initialize(const iree_hal_amdgpu_libhsa_t* libhsa,
                             const iree_hal_amdgpu_topology_t* topology,
                             iree_allocator_t host_allocator) {
      iree_hal_amdgpu_logical_device_options_t options;
      iree_hal_amdgpu_logical_device_options_initialize(&options);
      IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
          IREE_SV("amdgpu"), &options, libhsa, topology,
          create_context_.params(), host_allocator, &base_device_));
      return iree_hal_device_group_create_from_device(
          base_device_, create_context_.frontier_tracker(), host_allocator,
          &device_group_);
    }

    iree_hal_allocator_t* allocator() const {
      return iree_hal_device_allocator(base_device_);
    }

   private:
    // Creation context supplying the proactor pool and frontier tracker.
    iree::hal::cts::DeviceCreateContext create_context_;

    // Test-owned device reference released before the topology-owning group.
    iree_hal_device_t* base_device_ = NULL;

    // Device group that owns the topology assigned to |base_device_|.
    iree_hal_device_group_t* device_group_ = NULL;
  };

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t AllocatorTest::host_allocator_;
iree_hal_amdgpu_libhsa_t AllocatorTest::libhsa_;
iree_hal_amdgpu_topology_t AllocatorTest::topology_;

TEST_F(AllocatorTest, QueryMemoryHeapsReportsHsaLimits) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_host_size_t heap_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_allocator_query_memory_heaps(
                            test_device.allocator(),
                            /*capacity=*/0, /*heaps=*/NULL, &heap_count));
  ASSERT_EQ(heap_count, 3u);

  std::array<iree_hal_allocator_memory_heap_t, 3> heaps;
  IREE_ASSERT_OK(iree_hal_allocator_query_memory_heaps(
      test_device.allocator(), heaps.size(), heaps.data(), &heap_count));
  ASSERT_EQ(heap_count, heaps.size());

  for (const auto& heap : heaps) {
    EXPECT_NE(heap.max_allocation_size, 0u);
    EXPECT_NE(heap.max_allocation_size, ~(iree_device_size_t)0);
    EXPECT_NE(heap.min_alignment, 0u);
    EXPECT_TRUE(iree_device_size_is_power_of_two(heap.min_alignment));
  }
}

TEST_F(AllocatorTest, OversizedAllocationIsRejectedByCompatibility) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  std::array<iree_hal_allocator_memory_heap_t, 3> heaps;
  iree_host_size_t heap_count = 0;
  IREE_ASSERT_OK(iree_hal_allocator_query_memory_heaps(
      test_device.allocator(), heaps.size(), heaps.data(), &heap_count));
  ASSERT_EQ(heap_count, heaps.size());
  ASSERT_LT(heaps[0].max_allocation_size, ~(iree_device_size_t)0);

  iree_device_size_t oversized_allocation_size = 0;
  ASSERT_TRUE(iree_device_size_checked_add(heaps[0].max_allocation_size, 1,
                                           &oversized_allocation_size));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_params_t resolved_params = {0};
  iree_device_size_t resolved_allocation_size = 0;
  const iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), params, oversized_allocation_size,
          &resolved_params, &resolved_allocation_size);
  EXPECT_EQ(compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);
}

TEST_F(AllocatorTest, DeviceLocalHostVisibleMemoryIsLowPerformance) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH |
                 IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;

  iree_hal_buffer_params_t resolved_params = {0};
  iree_device_size_t resolved_allocation_size = 0;
  const iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), params, /*allocation_size=*/4096,
          &resolved_params, &resolved_allocation_size);
  EXPECT_TRUE(iree_all_bits_set(
      compatibility, IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                         IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
                         IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH |
                         IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE));
  EXPECT_TRUE(iree_all_bits_set(resolved_params.type,
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

TEST_F(AllocatorTest, OverAlignedAllocationIsRejected) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  std::array<iree_hal_allocator_memory_heap_t, 3> heaps;
  iree_host_size_t heap_count = 0;
  IREE_ASSERT_OK(iree_hal_allocator_query_memory_heaps(
      test_device.allocator(), heaps.size(), heaps.data(), &heap_count));
  ASSERT_EQ(heap_count, heaps.size());

  const iree_device_size_t over_alignment =
      ~(iree_device_size_t)0 ^ (~(iree_device_size_t)0 >> 1);
  ASSERT_TRUE(iree_device_size_is_power_of_two(over_alignment));
  ASSERT_GT(over_alignment, heaps[0].min_alignment);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.min_alignment = over_alignment;

  iree_hal_buffer_params_t resolved_params = {0};
  iree_device_size_t resolved_allocation_size = 0;
  const iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), params, /*allocation_size=*/1,
          &resolved_params, &resolved_allocation_size);
  EXPECT_EQ(compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);

  iree_hal_buffer_t* buffer = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_allocate_buffer(test_device.allocator(), params,
                                         /*allocation_size=*/1, &buffer));
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(AllocatorTest, UnsupportedExternalBufferImportsFailLoud) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  std::array<iree_hal_external_buffer_type_t, 3> unsupported_types = {
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32,
  };
  for (iree_hal_external_buffer_type_t unsupported_type : unsupported_types) {
    iree_hal_external_buffer_t external_buffer = {};
    external_buffer.type = unsupported_type;
    external_buffer.size = 4096;

    iree_hal_buffer_t* buffer = NULL;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNIMPLEMENTED,
        iree_hal_allocator_import_buffer(
            test_device.allocator(), params, &external_buffer,
            iree_hal_buffer_release_callback_null(), &buffer));
    EXPECT_EQ(buffer, nullptr);
  }
}

TEST_F(AllocatorTest, ExternalBufferExportFailsLoud) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      test_device.allocator(), params, /*allocation_size=*/4096, &buffer));

  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_allocator_export_buffer(
          test_device.allocator(), buffer,
          IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32,
          IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(external_buffer.type, IREE_HAL_EXTERNAL_BUFFER_TYPE_NONE);
  EXPECT_EQ(external_buffer.size, 0u);

  iree_hal_buffer_release(buffer);
}

}  // namespace
}  // namespace iree::hal::amdgpu
