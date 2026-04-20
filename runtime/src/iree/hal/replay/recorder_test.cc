// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder.h"

#include <cstdint>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/replay/file.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_device_t* CreateMockDevice(const char* identifier) {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = iree_make_cstring_view(identifier);
  iree_hal_device_t* device = nullptr;
  IREE_CHECK_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));
  return device;
}

static iree_hal_device_group_t* CreateMockDeviceGroup() {
  iree_hal_device_t* device_a = CreateMockDevice("mock");
  iree_hal_device_t* device_b = CreateMockDevice("mock");

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_async_frontier_tracker_release(frontier_tracker);
  IREE_CHECK_OK(iree_hal_device_group_builder_add_device(&builder, device_a));
  IREE_CHECK_OK(iree_hal_device_group_builder_add_device(&builder, device_b));

  iree_hal_device_group_t* group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
  return group;
}

TEST(ReplayRecorderTest, WrapDeviceGroupRecordsOrderedDeviceOperations) {
  std::vector<uint8_t> storage(16384, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_recorder_t* recorder = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_recorder_create(
      file_handle, nullptr, iree_allocator_system(), &recorder));
  iree_io_file_handle_release(file_handle);

  iree_hal_device_group_t* source_group = CreateMockDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_device_capabilities_t capabilities;
  IREE_ASSERT_OK(
      iree_hal_device_query_capabilities(wrapped_device, &capabilities));

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_ASSERT_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  ASSERT_LE(file_header.file_length, storage.size());
  iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      storage.data(), (iree_host_size_t)file_header.file_length);

  uint64_t expected_sequence_ordinal = 0;
  iree_host_size_t session_record_count = 0;
  iree_host_size_t device_object_record_count = 0;
  iree_host_size_t assign_topology_record_count = 0;
  iree_host_size_t query_capabilities_record_count = 0;
  while (offset < file_header.file_length) {
    iree_hal_replay_file_record_t record;
    IREE_ASSERT_OK(iree_hal_replay_file_parse_record(file_contents, offset,
                                                     &record, &offset));
    EXPECT_EQ(expected_sequence_ordinal++, record.header.sequence_ordinal);
    switch (record.header.record_type) {
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION:
        ++session_record_count;
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT:
        if (record.header.object_type == IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE) {
          ++device_object_record_count;
        }
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION:
        if (record.header.operation_code ==
            IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO) {
          ++assign_topology_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_CAPABILITIES) {
          ++query_capabilities_record_count;
        }
        EXPECT_EQ((uint32_t)IREE_STATUS_OK, record.header.status_code);
        break;
      default:
        break;
    }
  }

  EXPECT_EQ(1u, session_record_count);
  EXPECT_EQ(2u, device_object_record_count);
  EXPECT_EQ(2u, assign_topology_record_count);
  EXPECT_EQ(1u, query_capabilities_record_count);
}

}  // namespace
