// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/cts2/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

namespace {
constexpr iree_device_size_t kDefaultAllocationSize = 1024;
constexpr iree_host_size_t kHostBufferSlot = 0;
constexpr iree_host_size_t kDeviceBufferSlot = 1;
constexpr iree_host_size_t kBindingSlotCount = 2;
}  // namespace

class CommandBufferCopyBufferTest : public CtsTestBase<> {
 protected:
  // Returns a buffer ref using either direct or indirect mode.
  // In direct mode the slot is ignored; in indirect mode the buffer is ignored.
  iree_hal_buffer_ref_t MakeRef(iree_hal_buffer_t* buffer,
                                iree_host_size_t slot,
                                iree_device_size_t offset,
                                iree_device_size_t length) {
    if (recording_mode() == RecordingMode::kIndirect) {
      return iree_hal_make_indirect_buffer_ref(slot, offset, length);
    }
    return iree_hal_make_buffer_ref(buffer, offset, length);
  }

  iree_host_size_t binding_capacity() {
    return recording_mode() == RecordingMode::kIndirect ? kBindingSlotCount : 0;
  }

  iree_status_t SubmitWithBindings(iree_hal_command_buffer_t* command_buffer,
                                   iree_hal_buffer_t* host_buffer,
                                   iree_hal_buffer_t* device_buffer) {
    if (recording_mode() == RecordingMode::kIndirect) {
      const iree_hal_buffer_binding_t bindings[] = {
          /*kHostBufferSlot=*/{host_buffer, 0, IREE_HAL_WHOLE_BUFFER},
          /*kDeviceBufferSlot=*/{device_buffer, 0, IREE_HAL_WHOLE_BUFFER},
      };
      return SubmitCommandBufferAndWait(
          command_buffer,
          iree_hal_buffer_binding_table_t{IREE_ARRAYSIZE(bindings), bindings});
    }
    return SubmitCommandBufferAndWait(command_buffer);
  }
};

TEST_P(CommandBufferCopyBufferTest, CopyWholeBuffer) {
  uint8_t i8_val = 0x54;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize, i8_val);

  // Create and fill a host buffer.
  iree_hal_buffer_params_t host_params = {0};
  host_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                      IREE_HAL_BUFFER_USAGE_TRANSFER |
                      IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* host_buffer = NULL;
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
  iree_hal_buffer_t* device_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer));

  // Copy the host buffer to the device buffer.
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity(), &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      MakeRef(host_buffer, kHostBufferSlot, 0, kDefaultAllocationSize),
      MakeRef(device_buffer, kDeviceBufferSlot, 0, kDefaultAllocationSize),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(
      SubmitWithBindings(command_buffer, host_buffer, device_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0, actual_data.data(),
      kDefaultAllocationSize, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

TEST_P(CommandBufferCopyBufferTest, CopySubBuffer) {
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
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize, 0);
  std::memset(reference_buffer.data() + 8, i8_val,
              kDefaultAllocationSize / 2 - 4);

  // Create a host buffer with a smaller size.
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

  // Zero-fill, copy a sub-region, then zero-fill the remainder.
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity(), &command_buffer));
  uint8_t zero_val = 0x0;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      MakeRef(device_buffer, kDeviceBufferSlot, /*offset=*/0, /*length=*/8),
      &zero_val, /*pattern_length=*/sizeof(zero_val), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      MakeRef(host_buffer, kHostBufferSlot, /*offset=*/4,
              /*length=*/kDefaultAllocationSize / 2 - 4),
      MakeRef(device_buffer, kDeviceBufferSlot, /*offset=*/8,
              /*length=*/kDefaultAllocationSize / 2 - 4),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      MakeRef(device_buffer, kDeviceBufferSlot,
              /*offset=*/8 + kDefaultAllocationSize / 2 - 4,
              /*length=*/kDefaultAllocationSize -
                  (8 + kDefaultAllocationSize / 2 - 4)),
      &zero_val, /*pattern_length=*/sizeof(zero_val), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(
      SubmitWithBindings(command_buffer, host_buffer, device_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0, actual_data.data(),
      kDefaultAllocationSize, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CommandBufferCopyBufferTest);

}  // namespace iree::hal::cts
