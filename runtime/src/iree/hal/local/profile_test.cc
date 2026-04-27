// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/profile.h"

#include <cstdint>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/local/local_executable.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::local {
namespace {

struct RecordingProfileSink {
  // HAL resource header for the sink.
  iree_hal_resource_t resource;

  // Number of begin-session callbacks observed.
  int begin_count = 0;

  // Number of write callbacks observed.
  int write_count = 0;

  // Number of end-session callbacks observed.
  int end_count = 0;

  // Status code returned from begin_session, or OK for success.
  iree_status_code_t fail_begin_session_status_code = IREE_STATUS_OK;

  // Content type whose write callback should fail, or empty when disabled.
  iree_string_view_t fail_write_content_type = iree_string_view_empty();

  // Number of matching write callbacks that should fail.
  int fail_write_remaining = 0;

  // Status code returned from matching write callbacks.
  iree_status_code_t fail_write_status_code = IREE_STATUS_OK;

  // Status code returned from end_session, or OK for success.
  iree_status_code_t fail_end_session_status_code = IREE_STATUS_OK;

  // Session identifier observed at begin.
  uint64_t session_id = 0;

  // Status code observed by the most recent end_session callback.
  iree_status_code_t observed_end_session_status_code = IREE_STATUS_OK;

  // Device records copied from metadata chunks.
  std::vector<iree_hal_profile_device_record_t> device_records;

  // Queue records copied from metadata chunks.
  std::vector<iree_hal_profile_queue_record_t> queue_records;

  // Executable records copied from metadata chunks.
  std::vector<iree_hal_profile_executable_record_t> executable_records;

  // Executable export records copied from metadata chunks.
  std::vector<iree_hal_profile_executable_export_record_t>
      executable_export_records;

  // Executable export names copied from trailing packed record data.
  std::vector<std::string> executable_export_names;

  // Queue event records copied from data chunks.
  std::vector<iree_hal_profile_queue_event_t> queue_events;

  // Host execution event records copied from data chunks.
  std::vector<iree_hal_profile_host_execution_event_t> host_execution_events;

  // Memory event records copied from data chunks.
  std::vector<iree_hal_profile_memory_event_t> memory_events;

  // Dropped queue event records reported by truncated chunks.
  uint64_t dropped_queue_event_count = 0;

  // Dropped host execution event records reported by truncated chunks.
  uint64_t dropped_host_execution_event_count = 0;

  // Dropped memory event records reported by truncated chunks.
  uint64_t dropped_memory_event_count = 0;
};

static RecordingProfileSink* RecordingProfileSinkCast(
    iree_hal_profile_sink_t* sink) {
  return reinterpret_cast<RecordingProfileSink*>(sink);
}

static void RecordingProfileSinkDestroy(iree_hal_profile_sink_t* sink) {
  (void)sink;
}

static iree_status_t RecordingProfileSinkBeginSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  RecordingProfileSink* test_sink = RecordingProfileSinkCast(sink);
  EXPECT_EQ(0, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  EXPECT_TRUE(iree_string_view_equal(metadata->content_type,
                                     IREE_HAL_PROFILE_CONTENT_TYPE_SESSION));
  ++test_sink->begin_count;
  test_sink->session_id = metadata->session_id;
  if (test_sink->fail_begin_session_status_code != IREE_STATUS_OK) {
    return iree_make_status(test_sink->fail_begin_session_status_code,
                            "injected profile sink begin_session failure");
  }
  return iree_ok_status();
}

template <typename T>
static iree_status_t CopyProfileRecords(iree_const_byte_span_t iovec,
                                        std::vector<T>* out_records) {
  if (iovec.data_length % sizeof(T) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "misaligned profile record chunk");
  }
  const auto* records = reinterpret_cast<const T*>(iovec.data);
  const iree_host_size_t record_count = iovec.data_length / sizeof(T);
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    EXPECT_EQ(sizeof(T), records[i].record_length);
    out_records->push_back(records[i]);
  }
  return iree_ok_status();
}

static iree_status_t CopyExecutableExportProfileRecords(
    iree_const_byte_span_t iovec,
    std::vector<iree_hal_profile_executable_export_record_t>* out_records,
    std::vector<std::string>* out_names) {
  iree_host_size_t offset = 0;
  while (offset < iovec.data_length) {
    const iree_host_size_t remaining_length = iovec.data_length - offset;
    if (remaining_length <
        sizeof(iree_hal_profile_executable_export_record_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "truncated executable export record");
    }
    const auto* record =
        reinterpret_cast<const iree_hal_profile_executable_export_record_t*>(
            iovec.data + offset);
    if (record->record_length < sizeof(*record) ||
        record->record_length > remaining_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid executable export record length");
    }
    if (record->name_length > record->record_length - sizeof(*record)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid executable export name length");
    }
    out_records->push_back(*record);
    out_names->emplace_back(
        reinterpret_cast<const char*>(iovec.data + offset + sizeof(*record)),
        record->name_length);
    offset += record->record_length;
  }
  return iree_ok_status();
}

static iree_status_t RecordingProfileSinkWrite(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  RecordingProfileSink* test_sink = RecordingProfileSinkCast(sink);
  EXPECT_EQ(1, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  EXPECT_EQ(test_sink->session_id, metadata->session_id);
  ++test_sink->write_count;

  if (test_sink->fail_write_remaining != 0 &&
      iree_string_view_equal(metadata->content_type,
                             test_sink->fail_write_content_type)) {
    --test_sink->fail_write_remaining;
    return iree_make_status(test_sink->fail_write_status_code,
                            "injected profile sink write failure");
  }

  if (metadata->dropped_record_count != 0) {
    EXPECT_TRUE(iree_all_bits_set(metadata->flags,
                                  IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED));
  }
  if (iovec_count > 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected at most two profile chunk iovecs");
  }

  if (iree_string_view_equal(metadata->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
    for (iree_host_size_t i = 0; i < iovec_count; ++i) {
      IREE_RETURN_IF_ERROR(
          CopyProfileRecords(iovecs[i], &test_sink->device_records));
    }
    return iree_ok_status();
  }
  if (iree_string_view_equal(metadata->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    for (iree_host_size_t i = 0; i < iovec_count; ++i) {
      IREE_RETURN_IF_ERROR(
          CopyProfileRecords(iovecs[i], &test_sink->queue_records));
    }
    return iree_ok_status();
  }
  if (iree_string_view_equal(metadata->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    for (iree_host_size_t i = 0; i < iovec_count; ++i) {
      IREE_RETURN_IF_ERROR(
          CopyProfileRecords(iovecs[i], &test_sink->executable_records));
    }
    return iree_ok_status();
  }
  if (iree_string_view_equal(
          metadata->content_type,
          IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    for (iree_host_size_t i = 0; i < iovec_count; ++i) {
      IREE_RETURN_IF_ERROR(CopyExecutableExportProfileRecords(
          iovecs[i], &test_sink->executable_export_records,
          &test_sink->executable_export_names));
    }
    return iree_ok_status();
  }
  if (iree_string_view_equal(metadata->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    test_sink->dropped_queue_event_count += metadata->dropped_record_count;
    for (iree_host_size_t i = 0; i < iovec_count; ++i) {
      IREE_RETURN_IF_ERROR(
          CopyProfileRecords(iovecs[i], &test_sink->queue_events));
    }
    return iree_ok_status();
  }
  if (iree_string_view_equal(
          metadata->content_type,
          IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS)) {
    test_sink->dropped_host_execution_event_count +=
        metadata->dropped_record_count;
    for (iree_host_size_t i = 0; i < iovec_count; ++i) {
      IREE_RETURN_IF_ERROR(
          CopyProfileRecords(iovecs[i], &test_sink->host_execution_events));
    }
    return iree_ok_status();
  }
  if (iree_string_view_equal(metadata->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    test_sink->dropped_memory_event_count += metadata->dropped_record_count;
    for (iree_host_size_t i = 0; i < iovec_count; ++i) {
      IREE_RETURN_IF_ERROR(
          CopyProfileRecords(iovecs[i], &test_sink->memory_events));
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unexpected profile chunk content type");
}

static iree_status_t RecordingProfileSinkEndSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  RecordingProfileSink* test_sink = RecordingProfileSinkCast(sink);
  EXPECT_EQ(1, test_sink->begin_count);
  EXPECT_TRUE(iree_string_view_equal(metadata->content_type,
                                     IREE_HAL_PROFILE_CONTENT_TYPE_SESSION));
  ++test_sink->end_count;
  test_sink->observed_end_session_status_code = session_status_code;
  if (test_sink->fail_end_session_status_code != IREE_STATUS_OK) {
    return iree_make_status(test_sink->fail_end_session_status_code,
                            "injected profile sink end_session failure");
  }
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t kRecordingProfileSinkVTable = {
    /*.destroy=*/RecordingProfileSinkDestroy,
    /*.begin_session=*/RecordingProfileSinkBeginSession,
    /*.write=*/RecordingProfileSinkWrite,
    /*.end_session=*/RecordingProfileSinkEndSession,
};

static void RecordingProfileSinkInitialize(RecordingProfileSink* sink) {
  iree_hal_resource_initialize(&kRecordingProfileSinkVTable, &sink->resource);
}

static iree_hal_profile_sink_t* RecordingProfileSinkAsBase(
    RecordingProfileSink* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

static iree_hal_profile_device_record_t MakeDeviceRecord(uint32_t queue_count) {
  iree_hal_profile_device_record_t record =
      iree_hal_profile_device_record_default();
  record.physical_device_ordinal = 0;
  record.queue_count = queue_count;
  return record;
}

static iree_hal_profile_queue_record_t MakeQueueRecord(uint32_t queue_ordinal) {
  iree_hal_profile_queue_record_t record =
      iree_hal_profile_queue_record_default();
  record.physical_device_ordinal = 0;
  record.queue_ordinal = queue_ordinal;
  record.stream_id = queue_ordinal + 1;
  return record;
}

typedef struct FakeLocalExecutable {
  iree_hal_local_executable_t base;
} FakeLocalExecutable;

static void FakeLocalExecutableDestroy(iree_hal_executable_t* executable) {
  (void)executable;
}

static iree_host_size_t FakeLocalExecutableExportCount(
    iree_hal_executable_t* executable) {
  (void)executable;
  return 2;
}

static iree_status_t FakeLocalExecutableExportInfo(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  (void)executable;
  memset(out_info, 0, sizeof(*out_info));
  switch (export_ordinal) {
    case 0:
      out_info->name = IREE_SV("dispatch_a");
      out_info->constant_count = 1;
      out_info->binding_count = 2;
      out_info->parameter_count = 3;
      out_info->workgroup_size[0] = 4;
      out_info->workgroup_size[1] = 5;
      out_info->workgroup_size[2] = 6;
      return iree_ok_status();
    case 1:
      out_info->name = IREE_SV("dispatch_b");
      out_info->workgroup_size[0] = 1;
      out_info->workgroup_size[1] = 1;
      out_info->workgroup_size[2] = 1;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "export ordinal out of range");
  }
}

static iree_status_t FakeLocalExecutableExportParameters(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  (void)executable;
  (void)export_ordinal;
  (void)capacity;
  (void)out_parameters;
  return iree_ok_status();
}

static iree_status_t FakeLocalExecutableLookupExportByName(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  (void)executable;
  if (iree_string_view_equal(name, IREE_SV("dispatch_a"))) {
    *out_export_ordinal = 0;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("dispatch_b"))) {
    *out_export_ordinal = 1;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "export not found");
}

static iree_status_t FakeLocalExecutableIssueCall(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  (void)executable;
  (void)ordinal;
  (void)dispatch_state;
  (void)workgroup_state;
  (void)worker_id;
  return iree_ok_status();
}

static const iree_hal_local_executable_vtable_t kFakeLocalExecutableVTable = {
    {
        FakeLocalExecutableDestroy,
        FakeLocalExecutableExportCount,
        FakeLocalExecutableExportInfo,
        FakeLocalExecutableExportParameters,
        FakeLocalExecutableLookupExportByName,
    },
    FakeLocalExecutableIssueCall,
};

class LocalProfileRecorderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    RecordingProfileSinkInitialize(&sink_);
    device_record_ = MakeDeviceRecord(1);
    queue_record_ = MakeQueueRecord(0);
    recorder_options_.name = IREE_SV("local-test");
    recorder_options_.session_id = 42;
    recorder_options_.device_record_count = 1;
    recorder_options_.device_records = &device_record_;
    recorder_options_.queue_record_count = 1;
    recorder_options_.queue_records = &queue_record_;
    recorder_options_.queue_event_capacity = 4;
    recorder_options_.host_execution_event_capacity = 4;
    recorder_options_.memory_event_capacity = 4;
  }

  void TearDown() override {
    if (recorder_) {
      IREE_EXPECT_OK(iree_hal_local_profile_recorder_end(recorder_));
      iree_hal_local_profile_recorder_destroy(recorder_);
    }
  }

  iree_hal_device_profiling_options_t MakeProfilingOptions(
      iree_hal_device_profiling_data_families_t data_families) {
    iree_hal_device_profiling_options_t options = {0};
    options.data_families = data_families;
    options.sink = RecordingProfileSinkAsBase(&sink_);
    return options;
  }

  iree_status_t Create(
      iree_hal_device_profiling_data_families_t data_families) {
    iree_hal_device_profiling_options_t options =
        MakeProfilingOptions(data_families);
    return iree_hal_local_profile_recorder_create(
        &recorder_options_, &options, iree_allocator_system(), &recorder_);
  }

  iree_hal_local_profile_queue_scope_t QueueScope() {
    iree_hal_local_profile_queue_scope_t scope =
        iree_hal_local_profile_queue_scope_default();
    scope.physical_device_ordinal = queue_record_.physical_device_ordinal;
    scope.queue_ordinal = queue_record_.queue_ordinal;
    scope.stream_id = queue_record_.stream_id;
    return scope;
  }

  RecordingProfileSink sink_;
  iree_hal_profile_device_record_t device_record_;
  iree_hal_profile_queue_record_t queue_record_;
  iree_hal_local_profile_recorder_options_t recorder_options_ = {};
  iree_hal_local_profile_recorder_t* recorder_ = nullptr;
};

TEST_F(LocalProfileRecorderTest, NoneCreatesNoRecorder) {
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_NONE));
  EXPECT_EQ(nullptr, recorder_);
  EXPECT_EQ(0, sink_.begin_count);
}

TEST_F(LocalProfileRecorderTest, BeginWritesMetadata) {
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS));
  EXPECT_NE(nullptr, recorder_);
  EXPECT_EQ(1, sink_.begin_count);
  EXPECT_EQ(2, sink_.write_count);
  ASSERT_EQ(1u, sink_.device_records.size());
  ASSERT_EQ(1u, sink_.queue_records.size());
  EXPECT_EQ(0u, sink_.device_records[0].physical_device_ordinal);
  EXPECT_EQ(0u, sink_.queue_records[0].queue_ordinal);
}

TEST_F(LocalProfileRecorderTest, RejectsUnsupportedDataFamily) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      Create(IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS));
}

TEST_F(LocalProfileRecorderTest, RejectsCaptureFilter) {
  iree_hal_device_profiling_options_t options =
      MakeProfilingOptions(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS);
  options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL;
  options.capture_filter.queue_ordinal = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_local_profile_recorder_create(
          &recorder_options_, &options, iree_allocator_system(), &recorder_));
}

TEST_F(LocalProfileRecorderTest, RecordsExecutableMetadataOnce) {
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA));

  FakeLocalExecutable executable;
  iree_hal_local_executable_initialize(
      &kFakeLocalExecutableVTable, iree_allocator_system(), &executable.base);
  iree_hal_executable_t* base_executable =
      reinterpret_cast<iree_hal_executable_t*>(&executable.base);

  IREE_EXPECT_OK(iree_hal_local_profile_recorder_record_executable(
      recorder_, base_executable));
  IREE_EXPECT_OK(iree_hal_local_profile_recorder_record_executable(
      recorder_, base_executable));

  ASSERT_EQ(1u, sink_.executable_records.size());
  ASSERT_EQ(2u, sink_.executable_export_records.size());
  EXPECT_EQ(2u, sink_.executable_records[0].export_count);
  EXPECT_EQ(sink_.executable_records[0].executable_id,
            sink_.executable_export_records[0].executable_id);
  EXPECT_EQ("dispatch_a", sink_.executable_export_names[0]);
  EXPECT_EQ("dispatch_b", sink_.executable_export_names[1]);

  iree_hal_local_executable_deinitialize(&executable.base);
}

TEST_F(LocalProfileRecorderTest, HostExecutionEnablesExecutableMetadata) {
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS));
  EXPECT_TRUE(iree_hal_local_profile_recorder_is_enabled(
      recorder_, IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA));

  FakeLocalExecutable executable;
  iree_hal_local_executable_initialize(
      &kFakeLocalExecutableVTable, iree_allocator_system(), &executable.base);
  iree_hal_executable_t* base_executable =
      reinterpret_cast<iree_hal_executable_t*>(&executable.base);

  IREE_EXPECT_OK(iree_hal_local_profile_recorder_record_executable(
      recorder_, base_executable));

  ASSERT_EQ(1u, sink_.executable_records.size());
  ASSERT_EQ(2u, sink_.executable_export_records.size());
  EXPECT_EQ("dispatch_a", sink_.executable_export_names[0]);
  EXPECT_EQ("dispatch_b", sink_.executable_export_names[1]);

  iree_hal_local_executable_deinitialize(&executable.base);
}

TEST_F(LocalProfileRecorderTest, SinkBeginFailurePropagates) {
  sink_.fail_begin_session_status_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Create(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS));
  EXPECT_EQ(1, sink_.begin_count);
  EXPECT_EQ(0, sink_.end_count);
}

TEST_F(LocalProfileRecorderTest, MetadataWriteFailureEndsSession) {
  sink_.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES;
  sink_.fail_write_remaining = 1;
  sink_.fail_write_status_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        Create(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS));
  EXPECT_EQ(1, sink_.begin_count);
  EXPECT_EQ(1, sink_.end_count);
  EXPECT_EQ(IREE_STATUS_UNAVAILABLE, sink_.observed_end_session_status_code);
}

TEST_F(LocalProfileRecorderTest, AppendsAndFlushesQueueAndHostEvents) {
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                        IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS));

  iree_hal_local_profile_queue_event_info_t queue_info =
      iree_hal_local_profile_queue_event_info_default();
  queue_info.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  queue_info.dependency_strategy =
      IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE;
  queue_info.scope = QueueScope();
  queue_info.host_time_ns = 100;
  queue_info.ready_host_time_ns = 105;
  queue_info.submission_id = 7;
  queue_info.wait_count = 1;
  queue_info.signal_count = 1;
  queue_info.operation_count = 1;
  uint64_t queue_event_id = 0;
  iree_hal_local_profile_recorder_append_queue_event(recorder_, &queue_info,
                                                     &queue_event_id);
  EXPECT_NE(0u, queue_event_id);

  iree_hal_local_profile_host_execution_event_info_t host_info =
      iree_hal_local_profile_host_execution_event_info_default();
  host_info.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  host_info.status_code = IREE_STATUS_OK;
  host_info.scope = QueueScope();
  host_info.submission_id = queue_info.submission_id;
  host_info.executable_id = 5;
  host_info.export_ordinal = 3;
  host_info.workgroup_count[0] = 4;
  host_info.workgroup_size[0] = 16;
  host_info.start_host_time_ns = 110;
  host_info.end_host_time_ns = 140;
  host_info.operation_count = 1;
  uint64_t host_event_id = 0;
  iree_hal_local_profile_recorder_append_host_execution_event(
      recorder_, &host_info, &host_event_id);
  EXPECT_NE(0u, host_event_id);

  IREE_EXPECT_OK(iree_hal_local_profile_recorder_flush(recorder_));
  ASSERT_EQ(1u, sink_.queue_events.size());
  ASSERT_EQ(1u, sink_.host_execution_events.size());
  EXPECT_EQ(queue_event_id, sink_.queue_events[0].event_id);
  EXPECT_EQ(100, sink_.queue_events[0].host_time_ns);
  EXPECT_EQ(105, sink_.queue_events[0].ready_host_time_ns);
  EXPECT_EQ(host_event_id, sink_.host_execution_events[0].event_id);
}

TEST_F(LocalProfileRecorderTest, AppendsAndFlushesMemoryEvents) {
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS));

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA;
  event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
  event.result = IREE_STATUS_OK;
  event.host_time_ns = 100;
  event.allocation_id = 5;
  event.pool_id = 6;
  event.backing_id = 7;
  event.submission_id = 8;
  event.physical_device_ordinal = QueueScope().physical_device_ordinal;
  event.queue_ordinal = QueueScope().queue_ordinal;
  event.memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  event.buffer_usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  event.offset = 64;
  event.length = 256;
  event.alignment = 16;

  uint64_t event_id = 0;
  iree_hal_local_profile_recorder_append_memory_event(recorder_, &event,
                                                      &event_id);
  EXPECT_NE(0u, event_id);

  IREE_EXPECT_OK(iree_hal_local_profile_recorder_flush(recorder_));
  ASSERT_EQ(1u, sink_.memory_events.size());
  const iree_hal_profile_memory_event_t& recorded_event =
      sink_.memory_events[0];
  EXPECT_EQ(event_id, recorded_event.event_id);
  EXPECT_EQ(IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
            recorded_event.type);
  EXPECT_EQ(100, recorded_event.host_time_ns);
  EXPECT_EQ(5u, recorded_event.allocation_id);
  EXPECT_EQ(8u, recorded_event.submission_id);
  EXPECT_EQ(256u, recorded_event.length);
}

TEST_F(LocalProfileRecorderTest, FullQueueRingReportsTruncatedChunk) {
  recorder_options_.queue_event_capacity = 1;
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS));

  iree_hal_local_profile_queue_event_info_t queue_info =
      iree_hal_local_profile_queue_event_info_default();
  queue_info.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER;
  queue_info.scope = QueueScope();
  uint64_t captured_event_id = 0;
  iree_hal_local_profile_recorder_append_queue_event(recorder_, &queue_info,
                                                     &captured_event_id);
  EXPECT_NE(0u, captured_event_id);

  uint64_t dropped_event_id = 0;
  iree_hal_local_profile_recorder_append_queue_event(recorder_, &queue_info,
                                                     &dropped_event_id);
  EXPECT_EQ(0u, dropped_event_id);

  IREE_EXPECT_OK(iree_hal_local_profile_recorder_flush(recorder_));
  ASSERT_EQ(1u, sink_.queue_events.size());
  EXPECT_EQ(captured_event_id, sink_.queue_events[0].event_id);
  EXPECT_EQ(1u, sink_.dropped_queue_event_count);
}

TEST_F(LocalProfileRecorderTest, FlushesWrappedQueueRing) {
  recorder_options_.queue_event_capacity = 2;
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS));

  iree_hal_local_profile_queue_event_info_t queue_info =
      iree_hal_local_profile_queue_event_info_default();
  queue_info.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER;
  queue_info.scope = QueueScope();

  uint64_t first_event_id = 0;
  iree_hal_local_profile_recorder_append_queue_event(recorder_, &queue_info,
                                                     &first_event_id);
  IREE_EXPECT_OK(iree_hal_local_profile_recorder_flush(recorder_));
  ASSERT_EQ(1u, sink_.queue_events.size());

  uint64_t second_event_id = 0;
  iree_hal_local_profile_recorder_append_queue_event(recorder_, &queue_info,
                                                     &second_event_id);
  uint64_t third_event_id = 0;
  iree_hal_local_profile_recorder_append_queue_event(recorder_, &queue_info,
                                                     &third_event_id);
  IREE_EXPECT_OK(iree_hal_local_profile_recorder_flush(recorder_));

  ASSERT_EQ(3u, sink_.queue_events.size());
  EXPECT_EQ(first_event_id, sink_.queue_events[0].event_id);
  EXPECT_EQ(second_event_id, sink_.queue_events[1].event_id);
  EXPECT_EQ(third_event_id, sink_.queue_events[2].event_id);
  EXPECT_EQ(0u, sink_.dropped_queue_event_count);
}

TEST_F(LocalProfileRecorderTest, FlushFailurePreservesPendingRecords) {
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS));
  iree_hal_local_profile_queue_event_info_t queue_info =
      iree_hal_local_profile_queue_event_info_default();
  queue_info.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER;
  queue_info.scope = QueueScope();
  iree_hal_local_profile_recorder_append_queue_event(recorder_, &queue_info,
                                                     nullptr);

  sink_.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
  sink_.fail_write_remaining = 1;
  sink_.fail_write_status_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_local_profile_recorder_flush(recorder_));

  IREE_EXPECT_OK(iree_hal_local_profile_recorder_flush(recorder_));
  EXPECT_EQ(1u, sink_.queue_events.size());
}

TEST_F(LocalProfileRecorderTest, EndFailurePropagates) {
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS));
  sink_.fail_end_session_status_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_local_profile_recorder_end(recorder_));
  EXPECT_EQ(1, sink_.end_count);
}

}  // namespace
}  // namespace iree::hal::local
