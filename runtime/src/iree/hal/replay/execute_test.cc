// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
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
#include "iree/hal/replay/recorder.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::testing::HasSubstr;

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

static iree_hal_device_group_t* CreateDeviceGroup(
    iree_hal_device_t* const* devices, iree_host_size_t device_count) {
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

static iree_hal_device_group_t* CreateSyncDeviceGroup() {
  iree_hal_device_t* device = CreateSyncDevice("local-sync");
  iree_hal_device_t* devices[] = {device};
  iree_hal_device_group_t* group =
      CreateDeviceGroup(devices, IREE_ARRAYSIZE(devices));
  iree_hal_device_release(device);
  return group;
}

static iree_hal_device_group_t* CreateMockExecutableDeviceGroup() {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = iree_make_cstring_view("mock-executable-device");
  options.executable_cache_enabled = true;

  iree_hal_device_t* device = nullptr;
  IREE_CHECK_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));
  iree_hal_device_t* devices[] = {device};
  iree_hal_device_group_t* group =
      CreateDeviceGroup(devices, IREE_ARRAYSIZE(devices));
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

static iree_const_byte_span_t GetCapturedFileContents(
    const std::vector<uint8_t>& storage) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  EXPECT_LE(file_header.file_length, storage.size());
  return iree_make_const_byte_span(storage.data(),
                                   (iree_host_size_t)file_header.file_length);
}

static iree_status_t NoopHostCall(void* user_data, const uint64_t args[4],
                                  iree_hal_host_call_context_t* context) {
  (void)user_data;
  (void)args;
  (void)context;
  return iree_ok_status();
}

static std::vector<uint8_t> MakeMockExecutableData(uint8_t constant_count,
                                                   uint8_t binding_count,
                                                   uint8_t workgroup_size_x) {
  std::vector<uint8_t> data(12, 0);
  const uint32_t export_count = 1;
  std::memcpy(data.data(), &export_count, sizeof(export_count));
  data[4] = constant_count;
  data[5] = binding_count;
  data[6] = 0;  // flags
  data[7] = workgroup_size_x;
  data[8] = 1;
  data[9] = 1;
  return data;
}

static void CaptureMockExecutablePrepare(iree_const_byte_span_t executable_data,
                                         std::vector<uint8_t>* storage) {
  iree_hal_replay_recorder_t* recorder =
      CreateHostAllocationRecorder(storage, nullptr);

  iree_hal_device_group_t* source_group = CreateMockExecutableDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      wrapped_device, iree_make_cstring_view("mock-cache"), &executable_cache));

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format =
      iree_make_cstring_view("mock-executable");
  executable_params.executable_data = executable_data;

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
  iree_hal_replay_recorder_release(recorder);
}

typedef struct TestExecutableSubstitutionState {
  iree_string_view_t source;
  iree_string_view_t executable_format;
  iree_const_byte_span_t executable_data;
  iree_host_size_t invocation_count;
  iree_hal_replay_object_id_t executable_id;
} TestExecutableSubstitutionState;

static iree_status_t TestExecutableSubstitutionCallback(
    void* user_data,
    const iree_hal_replay_executable_substitution_request_t* request,
    iree_hal_replay_executable_substitution_t* out_substitution) {
  TestExecutableSubstitutionState* state =
      (TestExecutableSubstitutionState*)user_data;
  ++state->invocation_count;
  state->executable_id = request->executable_id;
  EXPECT_EQ(request->device_id, 1u);
  EXPECT_EQ(request->executable_cache_id, 2u);
  EXPECT_EQ(request->executable_id, 3u);
  EXPECT_EQ(std::string_view(request->captured_params->executable_format.data,
                             request->captured_params->executable_format.size),
            std::string_view("mock-executable"));
  memset(out_substitution, 0, sizeof(*out_substitution));
  out_substitution->substitute = true;
  out_substitution->source = state->source;
  out_substitution->executable_format = state->executable_format;
  out_substitution->executable_data = state->executable_data;
  return iree_ok_status();
}

TEST(ReplayExecuteTest, SubstitutesRecordedExecutablePayload) {
  std::vector<uint8_t> captured_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/3,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> replacement_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/3,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> storage(32768, 0);
  CaptureMockExecutablePrepare(
      iree_make_const_byte_span(captured_data.data(), captured_data.size()),
      &storage);

  TestExecutableSubstitutionState substitution_state = {
      /*.source=*/iree_make_cstring_view("replacement.mock"),
      /*.executable_format=*/iree_make_cstring_view("mock-executable"),
      /*.executable_data=*/
      iree_make_const_byte_span(replacement_data.data(),
                                replacement_data.size()),
      /*.invocation_count=*/0,
      /*.executable_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.executable_substitution_callback.fn =
      TestExecutableSubstitutionCallback;
  options.executable_substitution_callback.user_data = &substitution_state;

  iree_hal_device_group_t* replay_group = CreateMockExecutableDeviceGroup();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  EXPECT_EQ(substitution_state.invocation_count, 1u);
  EXPECT_EQ(substitution_state.executable_id, 3u);
  iree_hal_device_group_release(replay_group);
}

TEST(ReplayExecuteTest, RejectsExecutableSubstitutionAbiMismatch) {
  std::vector<uint8_t> captured_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/3,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> replacement_data =
      MakeMockExecutableData(/*constant_count=*/2, /*binding_count=*/4,
                             /*workgroup_size_x=*/4);
  std::vector<uint8_t> storage(32768, 0);
  CaptureMockExecutablePrepare(
      iree_make_const_byte_span(captured_data.data(), captured_data.size()),
      &storage);

  TestExecutableSubstitutionState substitution_state = {
      /*.source=*/iree_make_cstring_view("replacement.mock"),
      /*.executable_format=*/iree_make_cstring_view("mock-executable"),
      /*.executable_data=*/
      iree_make_const_byte_span(replacement_data.data(),
                                replacement_data.size()),
      /*.invocation_count=*/0,
      /*.executable_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
  };
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.executable_substitution_callback.fn =
      TestExecutableSubstitutionCallback;
  options.executable_substitution_callback.user_data = &substitution_state;

  iree_hal_device_group_t* replay_group = CreateMockExecutableDeviceGroup();
  iree_status_t status = iree_hal_replay_execute_file(
      GetCapturedFileContents(storage), replay_group, &options,
      iree_allocator_system());
  auto owned_status = ::iree::internal::ConsumeForTest(status);
  EXPECT_THAT(owned_status,
              ::iree::testing::status::StatusIs(static_cast<::iree::StatusCode>(
                  IREE_STATUS_FAILED_PRECONDITION)));
  EXPECT_THAT(owned_status.ToString(), HasSubstr("ABI mismatch"));
  EXPECT_THAT(owned_status.ToString(), HasSubstr("executable 3"));
  EXPECT_EQ(substitution_state.invocation_count, 1u);
  iree_hal_device_group_release(replay_group);
}

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
class ScopedTempFile {
 public:
  explicit ScopedTempFile(iree_const_byte_span_t contents) {
    char path_template[] = "/tmp/iree_hal_replay_file_XXXXXX";
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

  const std::string& path_string() const { return path_; }

  void RenameToUniquePath() {
    char path_template[] = "/tmp/iree_hal_replay_file_XXXXXX";
    const int fd = mkstemp(path_template);
    EXPECT_GE(fd, 0);
    if (fd < 0) return;
    EXPECT_EQ(0, close(fd));
    EXPECT_EQ(0, unlink(path_template));
    EXPECT_EQ(0, rename(path_.c_str(), path_template));
    path_ = path_template;
  }

 private:
  std::string path_;
};

static void CaptureFdBackedQueueRead(
    iree_string_view_t source_path,
    const iree_hal_replay_recorder_options_t* recorder_options,
    std::vector<uint8_t>* storage) {
  iree_hal_replay_recorder_t* recorder =
      CreateHostAllocationRecorder(storage, recorder_options);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS, source_path,
      iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      &signal_value,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, file, /*source_offset=*/4,
      target_buffer, /*target_offset=*/0, /*length=*/16,
      IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
  iree_hal_replay_recorder_release(recorder);
}
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)

TEST(ReplayExecuteTest, ExecutesRecordedMappedBufferWrite) {
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
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedHostAllocationFileRead) {
  uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(file_contents, sizeof(file_contents)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      &signal_value,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, file, /*source_offset=*/4,
      target_buffer, /*target_offset=*/0, /*length=*/16,
      IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedFdBackedQueueRead) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  ScopedTempFile source_file(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
      source_file.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      &signal_value,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, file, /*source_offset=*/4,
      target_buffer, /*target_offset=*/0, /*length=*/16,
      IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, ExecutesRemappedFdBackedQueueRead) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  ScopedTempFile source_file(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));
  std::string captured_path = source_file.path_string();

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
      source_file.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      &signal_value,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, file, /*source_offset=*/4,
      target_buffer, /*target_offset=*/0, /*length=*/16,
      IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  source_file.RenameToUniquePath();
  iree_hal_replay_file_path_remap_t file_path_remap = {
      iree_make_string_view(captured_path.data(), captured_path.size()),
      source_file.path_view(),
  };
  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.file_path_remap_count = 1;
  options.file_path_remaps = &file_path_remap;
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, CopiedFdBackedQueueReadFailsIdentityValidation) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  ScopedTempFile source_file(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));
  ScopedTempFile copied_file(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  CaptureFdBackedQueueRead(source_file.path_view(),
                           /*recorder_options=*/nullptr, &storage);

  iree_hal_replay_file_path_remap_t file_path_remap = {
      source_file.path_view(),
      copied_file.path_view(),
  };
  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.file_path_remap_count = 1;
  options.file_path_remaps = &file_path_remap;
  iree_status_t status = iree_hal_replay_execute_file(
      GetCapturedFileContents(storage), replay_group, &options,
      iree_allocator_system());
  auto owned_status = ::iree::internal::ConsumeForTest(status);
  EXPECT_THAT(owned_status,
              ::iree::testing::status::StatusIs(static_cast<::iree::StatusCode>(
                  IREE_STATUS_FAILED_PRECONDITION)));
  EXPECT_THAT(owned_status.ToString(),
              HasSubstr("--device_replay_file_validation=digest"));
  EXPECT_THAT(owned_status.ToString(), HasSubstr("--replay_file_remap"));
  iree_hal_device_group_release(replay_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, ExecutesDigestValidatedCopiedFdBackedQueueRead) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  ScopedTempFile source_file(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));
  ScopedTempFile copied_file(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_options_t recorder_options =
      iree_hal_replay_recorder_options_default();
  recorder_options.external_file_validation =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_CONTENT_DIGEST;
  CaptureFdBackedQueueRead(source_file.path_view(), &recorder_options,
                           &storage);

  iree_hal_replay_file_path_remap_t file_path_remap = {
      source_file.path_view(),
      copied_file.path_view(),
  };
  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.file_path_remap_count = 1;
  options.file_path_remaps = &file_path_remap;
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, DigestValidatedFdBackedQueueReadRejectsWrongBytes) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[32] = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
      0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
  };
  uint8_t wrong_file_contents[32];
  std::memcpy(wrong_file_contents, file_contents, sizeof(file_contents));
  wrong_file_contents[7] ^= 0xFF;
  ScopedTempFile source_file(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));
  ScopedTempFile wrong_file(iree_make_const_byte_span(
      wrong_file_contents, sizeof(wrong_file_contents)));

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_options_t recorder_options =
      iree_hal_replay_recorder_options_default();
  recorder_options.external_file_validation =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_CONTENT_DIGEST;
  CaptureFdBackedQueueRead(source_file.path_view(), &recorder_options,
                           &storage);

  iree_hal_replay_file_path_remap_t file_path_remap = {
      source_file.path_view(),
      wrong_file.path_view(),
  };
  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.file_path_remap_count = 1;
  options.file_path_remaps = &file_path_remap;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            &options, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayExecuteTest, ExecutesRecordedQueueAlloca) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_values[] = {1};
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      signal_values,
  };

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, /*pool=*/nullptr, params,
      16, IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer);
  iree_hal_semaphore_release(signal_semaphore);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, RejectsUnsupportedHostCallRecord) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_host_call_t call =
      iree_hal_make_host_call(NoopHostCall, /*user_data=*/nullptr);
  const uint64_t args[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_device_queue_host_call(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(), call,
      args, IREE_HAL_HOST_CALL_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            nullptr, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedQueueTransfersAndDealloca) {
  std::vector<uint8_t> storage(65536, 0);
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
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

  iree_hal_buffer_t* source_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 32,
                                                    &source_buffer));
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 32,
                                                    &target_buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};

  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  const uint8_t update_data[12] = {
      0xF0, 0xF1, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0xF2, 0xF3,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, update_data,
      /*source_offset=*/2, source_buffer, /*target_offset=*/0, /*length=*/8,
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t fill_pattern = 0xA5A5A5A5u;
  signal_value = 2;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, target_buffer,
      /*target_offset=*/0, /*length=*/16, &fill_pattern, sizeof(fill_pattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  signal_value = 3;
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, source_buffer,
      /*source_offset=*/0, target_buffer, /*target_offset=*/8, /*length=*/8,
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_t* transient_buffer = nullptr;
  signal_value = 4;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, /*pool=*/nullptr, params,
      16, IREE_HAL_ALLOCA_FLAG_NONE, &transient_buffer));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint64_t wait_value = 4;
  iree_hal_semaphore_list_t wait_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &wait_value,
  };
  signal_value = 5;
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(transient_buffer);
  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(source_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedQueueBarrier) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};

  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, /*command_buffer=*/nullptr,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedCommandBufferTransfers) {
  std::vector<uint8_t> storage(65536, 0);
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
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 32, &buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      wrapped_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  uint32_t fill_pattern = 0xCDCDCDCDu;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, iree_hal_make_buffer_ref(buffer, 0, 16), &fill_pattern,
      sizeof(fill_pattern), IREE_HAL_FILL_FLAG_NONE));
  const uint8_t update_data[8] = {
      0xE0, 0x20, 0x21, 0x22, 0x23, 0xE1, 0xE2, 0xE3,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, update_data, /*source_offset=*/1,
      iree_hal_make_buffer_ref(buffer, 4, 4), IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedIndirectCommandBufferBindings) {
  std::vector<uint8_t> storage(65536, 0);
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
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  iree_hal_buffer_t* source_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &source_buffer));
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, params, 16,
                                                    &target_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      wrapped_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, iree_hal_make_buffer_ref(source_buffer, 0, 16),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, 0, 16),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  iree_hal_buffer_binding_t binding = {
      target_buffer,
      0,
      16,
  };
  iree_hal_buffer_binding_table_t binding_table = {
      1,
      &binding,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(source_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayExecuteTest, ExecutesRecordedCommandBufferEvents) {
  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateSyncDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_hal_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_hal_event_create(wrapped_device,
                                       IREE_HAL_QUEUE_AFFINITY_ANY,
                                       IREE_HAL_EVENT_FLAG_NONE, &event));

  iree_hal_command_buffer_t* signal_command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      wrapped_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &signal_command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(signal_command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_signal_event(
      signal_command_buffer, event, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(signal_command_buffer));

  iree_hal_command_buffer_t* wait_command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      wrapped_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &wait_command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(wait_command_buffer));
  const iree_hal_event_t* events[] = {event};
  IREE_ASSERT_OK(iree_hal_command_buffer_wait_events(
      wait_command_buffer, IREE_ARRAYSIZE(events), events,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_reset_event(
      wait_command_buffer, event, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(wait_command_buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, signal_command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  uint64_t wait_value = 1;
  iree_hal_semaphore_list_t wait_list = {
      IREE_ARRAYSIZE(semaphores),
      semaphores,
      &wait_value,
  };
  signal_value = 2;
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      wait_command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(wait_command_buffer);
  iree_hal_command_buffer_release(signal_command_buffer);
  iree_hal_event_release(event);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  iree_hal_device_group_t* replay_group = CreateSyncDeviceGroup();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_EXPECT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

}  // namespace
