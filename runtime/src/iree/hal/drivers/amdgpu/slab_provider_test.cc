// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class SlabProviderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_ignore(status);
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

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t SlabProviderTest::host_allocator_;
iree_hal_amdgpu_libhsa_t SlabProviderTest::libhsa_;
iree_hal_amdgpu_topology_t SlabProviderTest::topology_;

TEST_F(SlabProviderTest, DefaultPhysicalDevicePoolMaterializesMappedBuffer) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  iree::hal::cts::DeviceCreateContext create_context;
  IREE_ASSERT_OK(create_context.Initialize(host_allocator_));

  iree_hal_device_t* base_device = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_logical_device_create(
      IREE_SV("amdgpu"), &options, &libhsa_, &topology_,
      create_context.params(), host_allocator_, &base_device));

  auto* device = (iree_hal_amdgpu_logical_device_t*)base_device;
  ASSERT_GE(device->physical_device_count, 1u);
  iree_hal_pool_t* default_pool = device->physical_devices[0]->default_pool;
  ASSERT_NE(default_pool, nullptr);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      default_pool, params, /*allocation_size=*/128,
      /*requester_frontier=*/NULL, iree_make_timeout_ms(0), &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 128u);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 128u);
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      /*local_byte_offset=*/0, IREE_HAL_WHOLE_BUFFER, &mapping));
  ASSERT_EQ(mapping.contents.data_length, 128u);
  memset(mapping.contents.data, 0xA5, mapping.contents.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
  iree_hal_device_release(base_device);
}

}  // namespace
}  // namespace iree::hal::amdgpu
