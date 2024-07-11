// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_COMMAND_BUFFER_COPY_BUFFER_TEST_H_
#define IREE_HAL_CTS_COMMAND_BUFFER_COPY_BUFFER_TEST_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

namespace {
constexpr iree_device_size_t kDefaultAllocationSize = 1024;
}  // namespace

class CommandBufferCopyBufferTest : public CTSTestBase<> {};

TEST_F(CommandBufferCopyBufferTest, CopyWholeBuffer) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  uint8_t i8_val = 0x54;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), i8_val, kDefaultAllocationSize);

  // Create and fill a host buffer.
  iree_hal_buffer_params_t host_params = {0};
  host_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                      IREE_HAL_BUFFER_USAGE_TRANSFER |
                      IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* host_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, host_params, kDefaultAllocationSize, &host_buffer));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device_, reference_buffer.data(), host_buffer, 0, reference_buffer.size(),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  // Create a device buffer.
  iree_hal_buffer_params_t device_params = {0};
  device_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  device_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* device_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, /*source_ref=*/
      iree_hal_make_buffer_ref(host_buffer, 0, kDefaultAllocationSize),
      /*target_ref=*/
      iree_hal_make_buffer_ref(device_buffer, 0, kDefaultAllocationSize)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0,
      /*target_buffer=*/actual_data.data(),
      /*data_length=*/kDefaultAllocationSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

TEST_F(CommandBufferCopyBufferTest, CopySubBuffer) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  iree_hal_buffer_params_t device_params = {0};
  device_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  device_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* device_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer));

  uint8_t i8_val = 0x88;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data() + 8, i8_val,
              kDefaultAllocationSize / 2 - 4);

  // Create another host buffer with a smaller size.
  iree_hal_buffer_params_t host_params = {0};
  host_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                      IREE_HAL_BUFFER_USAGE_TRANSFER |
                      IREE_HAL_BUFFER_USAGE_MAPPING;
  std::vector<uint8_t> host_buffer_data(kDefaultAllocationSize, i8_val);
  iree_hal_buffer_t* host_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, host_params, host_buffer_data.size() / 2,
      &host_buffer));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device_, host_buffer_data.data(), host_buffer, 0,
      host_buffer_data.size() / 2, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  // Copy the host buffer to the device buffer; zero fill the untouched bytes.
  uint8_t zero_val = 0x0;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(device_buffer, /*target_offset=*/0,
                               /*length=*/8),
      &zero_val, /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(/*source_buffer=*/host_buffer,
                               /*source_offset=*/4,
                               /*length=*/kDefaultAllocationSize / 2 - 4),
      iree_hal_make_buffer_ref(/*target_buffer=*/device_buffer,
                               /*target_offset=*/8,
                               /*length=*/kDefaultAllocationSize / 2 - 4)));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(
          device_buffer,
          /*target_offset=*/8 + kDefaultAllocationSize / 2 - 4,
          /*length=*/kDefaultAllocationSize -
              (8 + kDefaultAllocationSize / 2 - 4)),
      &zero_val,
      /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0,
      /*target_buffer=*/actual_data.data(),
      /*data_length=*/kDefaultAllocationSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

TEST_F(CommandBufferCopyBufferTest, CopySubBufferIndirect) {
  const int kHostBufferSlot = 0;
  const int kDeviceBufferSlot = 1;
  const int kBindingSlotCount = 2;
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      kBindingSlotCount, &command_buffer));

  iree_hal_buffer_params_t device_params = {0};
  device_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  device_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* device_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer));

  uint8_t i8_val = 0x88;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data() + 8, i8_val,
              kDefaultAllocationSize / 2 - 4);

  // Create another host buffer with a smaller size.
  iree_hal_buffer_params_t host_params = {0};
  host_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                      IREE_HAL_BUFFER_USAGE_TRANSFER |
                      IREE_HAL_BUFFER_USAGE_MAPPING;
  std::vector<uint8_t> host_buffer_data(kDefaultAllocationSize, i8_val);
  iree_hal_buffer_t* host_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, host_params, host_buffer_data.size() / 2,
      &host_buffer));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device_, host_buffer_data.data(), host_buffer, 0,
      host_buffer_data.size() / 2, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  // Copy the host buffer to the device buffer; zero fill the untouched bytes.
  uint8_t zero_val = 0x0;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(kDeviceBufferSlot, /*offset=*/0,
                                        /*length=*/8),
      &zero_val, /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(
          kHostBufferSlot,
          /*offset=*/4,
          /*length=*/kDefaultAllocationSize / 2 - 4),
      iree_hal_make_indirect_buffer_ref(
          kDeviceBufferSlot,
          /*offset=*/8,
          /*length=*/kDefaultAllocationSize / 2 - 4)));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(
          kDeviceBufferSlot,
          /*target_offset=*/8 + kDefaultAllocationSize / 2 - 4,
          /*length=*/kDefaultAllocationSize -
              (8 + kDefaultAllocationSize / 2 - 4)),
      &zero_val,
      /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_buffer_binding_t bindings[] = {
      /*kHostBufferSlot=*/{host_buffer, 0, IREE_WHOLE_BUFFER},
      /*kDeviceBufferSlot=*/{device_buffer, 0, IREE_WHOLE_BUFFER},
  };
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer,
                                            iree_hal_buffer_binding_table_t{
                                                IREE_ARRAYSIZE(bindings),
                                                bindings,
                                            }));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0,
      /*target_buffer=*/actual_data.data(),
      /*data_length=*/kDefaultAllocationSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_COMMAND_BUFFER_COPY_BUFFER_TEST_H_
