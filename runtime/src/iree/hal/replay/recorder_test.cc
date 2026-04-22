// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
#include <stdlib.h>
#include <unistd.h>
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/replay/file_reader.h"
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

static iree_hal_device_group_t* CreateDeviceGroup(iree_host_size_t device_count,
                                                  iree_hal_device_t** devices) {
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_async_frontier_tracker_release(frontier_tracker);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    IREE_CHECK_OK(
        iree_hal_device_group_builder_add_device(&builder, devices[i]));
  }

  iree_hal_device_group_t* group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));
  return group;
}

static iree_hal_device_group_t* CreateMockDeviceGroup() {
  iree_hal_device_t* device_a = CreateMockDevice("mock");
  iree_hal_device_t* device_b = CreateMockDevice("mock");
  iree_hal_device_t* devices[] = {device_a, device_b};

  iree_hal_device_group_t* group = CreateDeviceGroup(2, devices);
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
  return group;
}

static iree_hal_device_t* CreateSyncDevice(const char* identifier) {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_allocator_t* device_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      iree_make_cstring_view("local"), iree_allocator_system(),
      iree_allocator_system(), &device_allocator));

  iree_hal_sync_device_params_t sync_params;
  iree_hal_sync_device_params_initialize(&sync_params);
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_sync_device_create(
      iree_make_cstring_view(identifier), &sync_params, &create_params,
      /*loader_count=*/0, /*loaders=*/nullptr, device_allocator,
      iree_allocator_system(), &device);
  iree_hal_allocator_release(device_allocator);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);
  return device;
}

static iree_hal_device_group_t* CreateSyncDeviceGroup() {
  iree_hal_device_t* device = CreateSyncDevice("local-sync");
  iree_hal_device_t* devices[] = {device};
  iree_hal_device_group_t* group = CreateDeviceGroup(1, devices);
  iree_hal_device_release(device);
  return group;
}

static iree_hal_replay_recorder_t* CreateHostAllocationRecorder(
    std::vector<uint8_t>* storage,
    const iree_hal_replay_recorder_options_t* options = nullptr) {
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage->data(), storage->size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_recorder_t* recorder = nullptr;
  IREE_CHECK_OK(iree_hal_replay_recorder_create(
      file_handle, options, iree_allocator_system(), &recorder));
  iree_io_file_handle_release(file_handle);
  return recorder;
}

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
class ScopedTempFile {
 public:
  explicit ScopedTempFile(iree_const_byte_span_t contents) {
    char path_template[] = "/tmp/iree_hal_replay_recorder_file_XXXXXX";
    const int fd = mkstemp(path_template);
    EXPECT_GE(fd, 0);
    if (fd < 0) return;
    iree_host_size_t total_written = 0;
    while (total_written < contents.data_length) {
      ssize_t written = write(fd, contents.data + total_written,
                              contents.data_length - total_written);
      if (written < 0 && errno == EINTR) continue;
      EXPECT_GT(written, 0);
      if (written <= 0) break;
      total_written += (iree_host_size_t)written;
    }
    EXPECT_EQ(contents.data_length, total_written);
    EXPECT_EQ(0, close(fd));
    path_ = path_template;
  }

  ~ScopedTempFile() {
    if (!path_.empty()) {
      unlink(path_.c_str());
    }
  }

  iree_string_view_t path_view() const {
    return iree_make_string_view(path_.data(), path_.size());
  }

 private:
  std::string path_;
};
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)

struct ReplayRecordSummary {
  iree_host_size_t session_record_count = 0;
  iree_host_size_t device_object_record_count = 0;
  iree_host_size_t allocator_object_record_count = 0;
  iree_host_size_t buffer_object_record_count = 0;
  iree_host_size_t command_buffer_object_record_count = 0;
  iree_host_size_t semaphore_object_record_count = 0;
  iree_host_size_t file_object_record_count = 0;
  iree_host_size_t external_file_object_count = 0;
  iree_host_size_t inline_file_object_count = 0;
  uint64_t inline_file_reference_length = 0;
  iree_host_size_t assign_topology_record_count = 0;
  iree_host_size_t query_capabilities_record_count = 0;
  iree_host_size_t allocate_buffer_record_count = 0;
  iree_host_size_t import_buffer_record_count = 0;
  iree_host_size_t buffer_map_range_record_count = 0;
  iree_host_size_t buffer_flush_range_record_count = 0;
  iree_host_size_t buffer_unmap_range_record_count = 0;
  iree_host_size_t queue_execute_record_count = 0;
  iree_host_size_t unsupported_import_buffer_record_count = 0;
  iree_host_size_t unsupported_export_buffer_record_count = 0;
  iree_host_size_t unsupported_host_call_record_count = 0;
  iree_host_size_t buffer_range_data_payload_count = 0;
  iree_host_size_t import_buffer_payload_count = 0;
  uint64_t import_buffer_captured_data_length = 0;
  iree_host_size_t device_queue_execute_payload_count = 0;
  iree_host_size_t semaphore_object_payload_count = 0;
};

static ReplayRecordSummary ParseReplayRecordSummary(
    const std::vector<uint8_t>& storage) {
  ReplayRecordSummary summary;

  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  EXPECT_LE(file_header.file_length, storage.size());
  iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      storage.data(), (iree_host_size_t)file_header.file_length);

  uint64_t expected_sequence_ordinal = 0;
  while (offset < file_header.file_length) {
    iree_hal_replay_file_record_t record;
    IREE_CHECK_OK(iree_hal_replay_file_parse_record(file_contents, offset,
                                                    &record, &offset));
    EXPECT_EQ(expected_sequence_ordinal++, record.header.sequence_ordinal);
    switch (record.header.record_type) {
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION:
        ++summary.session_record_count;
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT:
        if (record.header.object_type == IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE) {
          ++summary.device_object_record_count;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR) {
          ++summary.allocator_object_record_count;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER) {
          ++summary.buffer_object_record_count;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER) {
          ++summary.command_buffer_object_record_count;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE) {
          ++summary.semaphore_object_record_count;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_FILE) {
          ++summary.file_object_record_count;
          EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT,
                    record.header.payload_type);
          iree_hal_replay_file_object_payload_t file_payload;
          if (record.payload.data_length < sizeof(file_payload)) {
            ADD_FAILURE() << "file object payload is short";
            return summary;
          }
          memcpy(&file_payload, record.payload.data, sizeof(file_payload));
          if (file_payload.reference_length > IREE_HOST_SIZE_MAX ||
              sizeof(file_payload) +
                      (iree_host_size_t)file_payload.reference_length !=
                  record.payload.data_length) {
            ADD_FAILURE() << "file object reference length mismatch";
            return summary;
          }
          if (file_payload.reference_type ==
              IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH) {
            ++summary.external_file_object_count;
          } else if (file_payload.reference_type ==
                     IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_INLINE_BYTES) {
            ++summary.inline_file_object_count;
            summary.inline_file_reference_length +=
                file_payload.reference_length;
          }
        }
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION:
        if (record.header.operation_code ==
            IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO) {
          ++summary.assign_topology_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_CAPABILITIES) {
          ++summary.query_capabilities_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER) {
          ++summary.allocate_buffer_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER) {
          ++summary.import_buffer_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE) {
          ++summary.buffer_map_range_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE) {
          ++summary.buffer_flush_range_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE) {
          ++summary.buffer_unmap_range_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE) {
          ++summary.queue_execute_record_count;
        }
        if (record.header.payload_type ==
            IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA) {
          ++summary.buffer_range_data_payload_count;
        } else if (record.header.payload_type ==
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_IMPORT_BUFFER) {
          ++summary.import_buffer_payload_count;
          iree_hal_replay_allocator_import_buffer_payload_t import_payload;
          if (record.payload.data_length < sizeof(import_payload)) {
            ADD_FAILURE() << "import buffer payload is short";
            return summary;
          }
          memcpy(&import_payload, record.payload.data, sizeof(import_payload));
          summary.import_buffer_captured_data_length +=
              import_payload.data_length;
        } else if (record.header.payload_type ==
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE) {
          ++summary.device_queue_execute_payload_count;
        } else if (record.header.payload_type ==
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT) {
          ++summary.semaphore_object_payload_count;
        }
        EXPECT_EQ((uint32_t)IREE_STATUS_OK, record.header.status_code);
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_UNSUPPORTED:
        if (record.header.operation_code ==
            IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER) {
          ++summary.unsupported_import_buffer_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_EXPORT_BUFFER) {
          ++summary.unsupported_export_buffer_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_HOST_CALL) {
          ++summary.unsupported_host_call_record_count;
        }
        EXPECT_EQ((uint32_t)IREE_STATUS_OK, record.header.status_code);
        break;
      default:
        break;
    }
  }

  return summary;
}

static iree_status_t CountHostCall(void* user_data, const uint64_t args[4],
                                   iree_hal_host_call_context_t* context) {
  (void)args;
  (void)context;
  ++*(int*)user_data;
  return iree_ok_status();
}

TEST(ReplayRecorderTest, WrapDeviceGroupRecordsOrderedDeviceOperations) {
  std::vector<uint8_t> storage(16384, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

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

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.session_record_count);
  EXPECT_EQ(2u, summary.device_object_record_count);
  EXPECT_EQ(2u, summary.assign_topology_record_count);
  EXPECT_EQ(1u, summary.query_capabilities_record_count);
}

TEST(ReplayRecorderTest, WrappedDeviceRecordsHostCallAsUnsupported) {
  std::vector<uint8_t> storage(16384, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  int call_count = 0;
  iree_hal_host_call_t call =
      iree_hal_make_host_call(CountHostCall, &call_count);
  const uint64_t args[4] = {0, 1, 2, 3};
  IREE_ASSERT_OK(iree_hal_device_queue_host_call(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(), call,
      args, IREE_HAL_HOST_CALL_FLAG_NONE));
  EXPECT_EQ(1, call_count);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.unsupported_host_call_record_count);
}

TEST(ReplayRecorderTest,
     WrappedAllocatorSnapshotsHostAllocationImportsAndMarksExportsUnsupported) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_SHARING_EXPORT;

  uint8_t imported_storage[16] = {0};
  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  external_buffer.size = sizeof(imported_storage);
  external_buffer.handle.host_allocation.ptr = imported_storage;
  iree_hal_buffer_t* imported_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
      allocator, params, &external_buffer,
      iree_hal_buffer_release_callback_null(), &imported_buffer));

  iree_hal_buffer_t* allocated_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, sizeof(imported_storage), &allocated_buffer));
  iree_hal_external_buffer_t exported_buffer = {};
  IREE_ASSERT_OK(iree_hal_allocator_export_buffer(
      allocator, allocated_buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &exported_buffer));
  EXPECT_EQ(IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
            exported_buffer.type);

  iree_hal_buffer_release(allocated_buffer);
  iree_hal_buffer_release(imported_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.import_buffer_record_count);
  EXPECT_EQ(1u, summary.import_buffer_payload_count);
  EXPECT_EQ(sizeof(imported_storage),
            summary.import_buffer_captured_data_length);
  EXPECT_EQ(0u, summary.unsupported_import_buffer_record_count);
  EXPECT_EQ(1u, summary.unsupported_export_buffer_record_count);
  EXPECT_EQ(2u, summary.buffer_object_record_count);
}

TEST(ReplayRecorderTest, WrappedAllocatorRecordsBuffersAndMapping) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           0, 16, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_WRITE, 0, 16, &span));
  const uint8_t contents[16] = {
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
  };
  std::memcpy(span.data, contents, sizeof(contents));
  IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(&mapping, 0, 16));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.session_record_count);
  EXPECT_EQ(1u, summary.device_object_record_count);
  EXPECT_EQ(1u, summary.allocator_object_record_count);
  EXPECT_EQ(1u, summary.buffer_object_record_count);
  EXPECT_EQ(1u, summary.allocate_buffer_record_count);
  EXPECT_EQ(1u, summary.buffer_map_range_record_count);
  EXPECT_EQ(1u, summary.buffer_flush_range_record_count);
  EXPECT_EQ(1u, summary.buffer_unmap_range_record_count);
  EXPECT_EQ(2u, summary.buffer_range_data_payload_count);
}

TEST(ReplayRecorderTest, PersistentWriteMapsFailLoud) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));

  iree_hal_buffer_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
                                IREE_HAL_MEMORY_ACCESS_WRITE, 0, 16, &mapping));
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
}

TEST(ReplayRecorderTest, ExternalFileFailPolicyRejectsFdBackedFiles) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[4] = {0x00, 0x01, 0x02, 0x03};
  ScopedTempFile source_file(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_options_t options =
      iree_hal_replay_recorder_options_default();
  options.external_file_policy =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_FAIL;
  iree_hal_replay_recorder_t* recorder =
      CreateHostAllocationRecorder(&storage, &options);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));
  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
      source_file.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_file_import(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
                           IREE_HAL_MEMORY_ACCESS_READ, file_handle,
                           IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  EXPECT_EQ(nullptr, file);
  iree_io_file_handle_release(file_handle);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayRecorderTest, ExternalFileCaptureAllPolicyEmbedsFdBackedFiles) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[4] = {0x00, 0x01, 0x02, 0x03};
  ScopedTempFile source_file(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_options_t options =
      iree_hal_replay_recorder_options_default();
  options.external_file_policy =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_CAPTURE_ALL;
  iree_hal_replay_recorder_t* recorder =
      CreateHostAllocationRecorder(&storage, &options);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));
  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
      source_file.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.file_object_record_count);
  EXPECT_EQ(0u, summary.external_file_object_count);
  EXPECT_EQ(1u, summary.inline_file_object_count);
  EXPECT_EQ(sizeof(file_contents), summary.inline_file_reference_length);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayRecorderTest, WrappedDeviceRecordsQueueExecuteSemaphores) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      wrapped_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_semaphore_t* wait_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &wait_semaphore));
  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));

  iree_hal_semaphore_t* wait_semaphores[] = {wait_semaphore};
  uint64_t wait_values[] = {0};
  iree_hal_semaphore_list_t wait_list = {
      IREE_ARRAYSIZE(wait_semaphores),
      wait_semaphores,
      wait_values,
  };
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_values[] = {1};
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      signal_values,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_command_buffer_release(command_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.command_buffer_object_record_count);
  EXPECT_EQ(2u, summary.semaphore_object_record_count);
  EXPECT_EQ(2u, summary.semaphore_object_payload_count);
  EXPECT_EQ(1u, summary.queue_execute_record_count);
  EXPECT_EQ(1u, summary.device_queue_execute_payload_count);
}

}  // namespace
