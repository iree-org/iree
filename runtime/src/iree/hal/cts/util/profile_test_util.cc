// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/profile_test_util.h"

#include <algorithm>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

DeviceProfilingScope::~DeviceProfilingScope() {
  if (is_active_) {
    IREE_EXPECT_OK(iree_hal_device_profiling_end(device_));
  }
}

iree_status_t DeviceProfilingScope::Begin(
    iree_hal_device_profiling_data_families_t data_families,
    iree_hal_profile_sink_t* sink) {
  iree_hal_device_profiling_options_t options = {0};
  options.data_families = data_families;
  options.sink = sink;
  return Begin(&options);
}

iree_status_t DeviceProfilingScope::Begin(
    const iree_hal_device_profiling_options_t* options) {
  iree_status_t status = iree_hal_device_profiling_begin(device_, options);
  if (iree_status_is_ok(status)) {
    is_active_ = true;
  }
  return status;
}

iree_status_t DeviceProfilingScope::End() {
  if (!is_active_) return iree_ok_status();
  is_active_ = false;
  return iree_hal_device_profiling_end(device_);
}

static TestProfileSink* TestProfileSinkCast(iree_hal_profile_sink_t* sink) {
  return reinterpret_cast<TestProfileSink*>(sink);
}

static void TestProfileSinkDestroy(iree_hal_profile_sink_t* sink) {
  (void)sink;
}

static iree_status_t TestProfileSinkBeginSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  TestProfileSink* test_sink = TestProfileSinkCast(sink);
  EXPECT_EQ(0, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  EXPECT_TRUE(iree_string_view_equal(metadata->content_type,
                                     IREE_HAL_PROFILE_CONTENT_TYPE_SESSION));
  test_sink->begin_count = 1;
  test_sink->session_id = metadata->session_id;
  return iree_ok_status();
}

static iree_status_t TestProfileSinkWrite(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  TestProfileSink* test_sink = TestProfileSinkCast(sink);
  EXPECT_EQ(1, test_sink->begin_count);
  if (test_sink->end_count != 0) {
    test_sink->write_after_end = true;
  }
  EXPECT_EQ(test_sink->session_id, metadata->session_id);

  if (iovec_count == 0) return iree_ok_status();
  if (iovec_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected one profile chunk iovec");
  }

  if (iree_string_view_equal(metadata->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
    EXPECT_FALSE(test_sink->saw_queue_metadata);
    EXPECT_EQ(0u,
              iovecs[0].data_length % sizeof(iree_hal_profile_device_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_device_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_device_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_device_record_t),
                records[i].record_length);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_GT(records[i].queue_count, 0u);
    }
    test_sink->saw_device_metadata = true;
    ++test_sink->device_metadata_count;
  } else if (iree_string_view_equal(metadata->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    EXPECT_TRUE(test_sink->saw_device_metadata);
    EXPECT_EQ(0u,
              iovecs[0].data_length % sizeof(iree_hal_profile_queue_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_queue_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_queue_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_queue_record_t),
                records[i].record_length);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
    }
    test_sink->saw_queue_metadata = true;
    ++test_sink->queue_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    EXPECT_TRUE(test_sink->saw_device_metadata);
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_executable_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_executable_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_executable_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_executable_record_t),
                records[i].record_length);
      EXPECT_NE(0u, records[i].executable_id);
      EXPECT_GT(records[i].export_count, 0u);
      test_sink->executable_ids.push_back(records[i].executable_id);
    }
    ++test_sink->executable_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    iree_host_size_t payload_offset = 0;
    while (payload_offset < iovecs[0].data_length) {
      if (iovecs[0].data_length - payload_offset <
          sizeof(iree_hal_profile_executable_export_record_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "truncated executable export profile record");
      }
      iree_hal_profile_executable_export_record_t record;
      memcpy(&record, iovecs[0].data + payload_offset, sizeof(record));
      if (record.record_length < sizeof(record) ||
          record.record_length > iovecs[0].data_length - payload_offset) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid executable export profile record");
      }
      EXPECT_NE(0u, record.executable_id);
      EXPECT_NE(UINT32_MAX, record.export_ordinal);
      EXPECT_EQ(record.name_length,
                record.record_length - (uint32_t)sizeof(record));
      test_sink->export_record_executable_ids.push_back(record.executable_id);
      payload_offset += record.record_length;
    }
    ++test_sink->executable_export_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    EXPECT_TRUE(test_sink->saw_device_metadata);
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_clock_correlation_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_clock_correlation_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length /
        sizeof(iree_hal_profile_clock_correlation_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_clock_correlation_record_t),
                records[i].record_length);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(0u, records[i].sample_id);
      EXPECT_TRUE(iree_all_bits_set(
          records[i].flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_SYSTEM_TIMESTAMP |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET));
      EXPECT_NE(0u, records[i].device_tick);
      EXPECT_NE(0u, records[i].host_cpu_timestamp_ns);
      EXPECT_NE(0u, records[i].host_system_timestamp);
      EXPECT_NE(0u, records[i].host_system_frequency_hz);
      EXPECT_LE(records[i].host_time_begin_ns, records[i].host_time_end_ns);
    }
    test_sink->clock_correlations.insert(test_sink->clock_correlations.end(),
                                         records, records + record_count);
    ++test_sink->clock_correlation_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    EXPECT_TRUE(test_sink->saw_device_metadata);
    EXPECT_NE(UINT32_MAX, metadata->physical_device_ordinal);
    EXPECT_EQ(
        0u, iovecs[0].data_length % sizeof(iree_hal_profile_dispatch_event_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_dispatch_event_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_dispatch_event_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_dispatch_event_t),
                records[i].record_length);
      EXPECT_EQ(test_sink->expected_dispatch_flags, records[i].flags);
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0u, records[i].executable_id);
      EXPECT_EQ(UINT32_MAX, records[i].command_index);
      if (!test_sink->executable_ids.empty() ||
          !test_sink->export_record_executable_ids.empty()) {
        EXPECT_NE(0u, records[i].executable_id);
        EXPECT_NE(test_sink->executable_ids.end(),
                  std::find(test_sink->executable_ids.begin(),
                            test_sink->executable_ids.end(),
                            records[i].executable_id));
        EXPECT_NE(test_sink->export_record_executable_ids.end(),
                  std::find(test_sink->export_record_executable_ids.begin(),
                            test_sink->export_record_executable_ids.end(),
                            records[i].executable_id));
      }
      EXPECT_EQ(0u, records[i].export_ordinal);
      if (test_sink->validate_dispatch_workgroup_count) {
        EXPECT_EQ(test_sink->expected_workgroup_count[0],
                  records[i].workgroup_count[0]);
        EXPECT_EQ(test_sink->expected_workgroup_count[1],
                  records[i].workgroup_count[1]);
        EXPECT_EQ(test_sink->expected_workgroup_count[2],
                  records[i].workgroup_count[2]);
      }
      EXPECT_NE(0u, records[i].workgroup_size[0]);
      EXPECT_NE(0u, records[i].start_tick);
      EXPECT_NE(0u, records[i].end_tick);
      EXPECT_GE(records[i].end_tick, records[i].start_tick);
    }
    test_sink->dispatch_events.insert(test_sink->dispatch_events.end(), records,
                                      records + record_count);
    test_sink->dispatch_event_physical_device_ordinals.insert(
        test_sink->dispatch_event_physical_device_ordinals.end(), record_count,
        metadata->physical_device_ordinal);
    ++test_sink->dispatch_event_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    EXPECT_TRUE(test_sink->saw_device_metadata);
    EXPECT_TRUE(test_sink->saw_queue_metadata);
    EXPECT_EQ(0u,
              iovecs[0].data_length % sizeof(iree_hal_profile_queue_event_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_queue_event_t*>(iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_queue_event_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_queue_event_t),
                records[i].record_length);
      EXPECT_NE(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE, records[i].type);
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0u, records[i].submission_id);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
      if (records[i].ready_host_time_ns != 0) {
        EXPECT_GE(records[i].ready_host_time_ns, records[i].host_time_ns);
      }
    }
    test_sink->queue_events.insert(test_sink->queue_events.end(), records,
                                   records + record_count);
    ++test_sink->queue_event_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS)) {
    EXPECT_TRUE(test_sink->saw_device_metadata);
    EXPECT_TRUE(test_sink->saw_queue_metadata);
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_host_execution_event_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_host_execution_event_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_host_execution_event_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_host_execution_event_t),
                records[i].record_length);
      EXPECT_NE(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE, records[i].type);
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0u, records[i].submission_id);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
      EXPECT_GE(records[i].end_host_time_ns, records[i].start_host_time_ns);
    }
    test_sink->host_execution_events.insert(
        test_sink->host_execution_events.end(), records,
        records + record_count);
    ++test_sink->host_execution_event_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
    EXPECT_TRUE(test_sink->saw_device_metadata);
    EXPECT_TRUE(test_sink->saw_queue_metadata);
    EXPECT_NE(UINT32_MAX, metadata->physical_device_ordinal);
    EXPECT_NE(UINT32_MAX, metadata->queue_ordinal);
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_queue_device_event_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_queue_device_event_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_queue_device_event_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_queue_device_event_t),
                records[i].record_length);
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0u, records[i].submission_id);
      EXPECT_NE(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE, records[i].type);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
      EXPECT_NE(0u, records[i].start_tick);
      EXPECT_NE(0u, records[i].end_tick);
      EXPECT_GE(records[i].end_tick, records[i].start_tick);
    }
    test_sink->queue_device_events.insert(test_sink->queue_device_events.end(),
                                          records, records + record_count);
    ++test_sink->queue_device_event_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS)) {
    EXPECT_TRUE(test_sink->saw_device_metadata);
    EXPECT_TRUE(test_sink->saw_queue_metadata);
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_event_relationship_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_event_relationship_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length /
        sizeof(iree_hal_profile_event_relationship_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_event_relationship_record_t),
                records[i].record_length);
      EXPECT_NE(0u, records[i].relationship_id);
      EXPECT_NE(0u, records[i].source_id);
      EXPECT_NE(0u, records[i].target_id);
    }
    test_sink->event_relationships.insert(test_sink->event_relationships.end(),
                                          records, records + record_count);
    ++test_sink->event_relationship_count;
  }

  return iree_ok_status();
}

static iree_status_t TestProfileSinkEndSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  TestProfileSink* test_sink = TestProfileSinkCast(sink);
  EXPECT_EQ(1, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  EXPECT_TRUE(iree_string_view_equal(metadata->content_type,
                                     IREE_HAL_PROFILE_CONTENT_TYPE_SESSION));
  EXPECT_EQ(test_sink->session_id, metadata->session_id);
  EXPECT_EQ(IREE_STATUS_OK, session_status_code);
  test_sink->end_count = 1;
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t kTestProfileSinkVTable = {
    /*.destroy=*/TestProfileSinkDestroy,
    /*.begin_session=*/TestProfileSinkBeginSession,
    /*.write=*/TestProfileSinkWrite,
    /*.end_session=*/TestProfileSinkEndSession,
};

void TestProfileSinkInitialize(TestProfileSink* sink) {
  iree_hal_resource_initialize(&kTestProfileSinkVTable, &sink->resource);
}

iree_hal_profile_sink_t* TestProfileSinkAsBase(TestProfileSink* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

bool IsProfilingUnsupported(iree_status_t status) {
  return iree_status_is_unimplemented(status) ||
         iree_status_is_invalid_argument(status);
}

void ExpectDispatchEventsWithinClockCorrelationRange(
    const TestProfileSink& sink) {
  ASSERT_GE(sink.clock_correlations.size(), 2u);
  ASSERT_EQ(sink.dispatch_events.size(),
            sink.dispatch_event_physical_device_ordinals.size());
  for (iree_host_size_t event_index = 0;
       event_index < sink.dispatch_events.size(); ++event_index) {
    const uint32_t physical_device_ordinal =
        sink.dispatch_event_physical_device_ordinals[event_index];
    uint64_t min_device_tick = UINT64_MAX;
    uint64_t max_device_tick = 0;
    for (const iree_hal_profile_clock_correlation_record_t& correlation :
         sink.clock_correlations) {
      if (correlation.physical_device_ordinal != physical_device_ordinal ||
          !iree_any_bit_set(
              correlation.flags,
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK)) {
        continue;
      }
      min_device_tick = std::min(min_device_tick, correlation.device_tick);
      max_device_tick = std::max(max_device_tick, correlation.device_tick);
    }
    ASSERT_NE(UINT64_MAX, min_device_tick);
    ASSERT_NE(0u, max_device_tick);
    ASSERT_LT(min_device_tick, max_device_tick);
    EXPECT_GE(sink.dispatch_events[event_index].start_tick, min_device_tick);
    EXPECT_LE(sink.dispatch_events[event_index].end_tick, max_device_tick);
  }
}

void ExpectQueueDeviceEventsWithinClockCorrelationRange(
    const TestProfileSink& sink) {
  ASSERT_GE(sink.clock_correlations.size(), 2u);
  for (const iree_hal_profile_queue_device_event_t& event :
       sink.queue_device_events) {
    uint64_t min_device_tick = UINT64_MAX;
    uint64_t max_device_tick = 0;
    for (const iree_hal_profile_clock_correlation_record_t& correlation :
         sink.clock_correlations) {
      if (correlation.physical_device_ordinal !=
              event.physical_device_ordinal ||
          !iree_any_bit_set(
              correlation.flags,
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK)) {
        continue;
      }
      min_device_tick = std::min(min_device_tick, correlation.device_tick);
      max_device_tick = std::max(max_device_tick, correlation.device_tick);
    }
    ASSERT_NE(UINT64_MAX, min_device_tick);
    ASSERT_NE(0u, max_device_tick);
    ASSERT_LT(min_device_tick, max_device_tick);
    EXPECT_GE(event.start_tick, min_device_tick);
    EXPECT_LE(event.end_tick, max_device_tick);
  }
}

}  // namespace iree::hal::cts
