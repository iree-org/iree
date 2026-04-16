// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HAL queue_dispatch operations.

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class QueueDispatchTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(executable_format());
    executable_params.executable_data = executable_data(iree_make_cstring_view(
        "command_buffer_dispatch_constants_bindings_test.bin"));

    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

static void MakeScaleAndOffsetBindings(iree_hal_buffer_t* input_buffer,
                                       iree_hal_buffer_t* output_buffer,
                                       iree_hal_buffer_ref_t binding_refs[2]) {
  binding_refs[0] = iree_hal_make_buffer_ref(
      input_buffer, /*offset=*/0, iree_hal_buffer_byte_length(input_buffer));
  binding_refs[1] = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
}

class DeviceProfilingScope {
 public:
  explicit DeviceProfilingScope(iree_hal_device_t* device) : device_(device) {}

  ~DeviceProfilingScope() {
    if (is_active_) {
      IREE_EXPECT_OK(iree_hal_device_profiling_end(device_));
    }
  }

  iree_status_t Begin(iree_hal_device_profiling_mode_t mode,
                      iree_hal_profile_sink_t* sink = nullptr) {
    iree_hal_device_profiling_options_t options = {0};
    options.mode = mode;
    options.sink = sink;
    iree_status_t status = iree_hal_device_profiling_begin(device_, &options);
    if (iree_status_is_ok(status)) {
      is_active_ = true;
    }
    return status;
  }

  iree_status_t End() {
    if (!is_active_) return iree_ok_status();
    is_active_ = false;
    return iree_hal_device_profiling_end(device_);
  }

 private:
  // Device whose profiling session is active.
  iree_hal_device_t* device_ = nullptr;
  // True when |device_| has an active profiling session owned by this scope.
  bool is_active_ = false;
};

struct TestProfileSink {
  // HAL resource header for the profile sink.
  iree_hal_resource_t resource;

  // Number of session begin notifications observed.
  int begin_count = 0;
  // Number of session end notifications observed.
  int end_count = 0;
  // Number of device metadata chunks observed.
  int device_metadata_count = 0;
  // Number of queue metadata chunks observed.
  int queue_metadata_count = 0;
  // Number of clock correlation chunks observed.
  int clock_correlation_count = 0;
  // Number of dispatch event chunks observed.
  int dispatch_event_count = 0;
  // Clock correlation records copied from CLOCK_CORRELATIONS chunks.
  std::vector<iree_hal_profile_clock_correlation_record_t> clock_correlations;
  // Dispatch event records copied from DISPATCH_EVENTS chunks.
  std::vector<iree_hal_profile_dispatch_event_t> dispatch_events;
  // Physical device ordinals for entries in |dispatch_events|.
  std::vector<uint32_t> dispatch_event_physical_device_ordinals;
  // Dispatch event flags expected for every event record.
  iree_hal_profile_dispatch_event_flags_t expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_NONE;
  // True if dispatch event workgroup counts should be checked.
  bool validate_dispatch_workgroup_count = true;
  // Expected dispatch event workgroup counts when validated.
  uint32_t expected_workgroup_count[3] = {1, 1, 1};
  // True after the device metadata chunk has been observed.
  bool saw_device_metadata = false;
  // True after the queue metadata chunk has been observed.
  bool saw_queue_metadata = false;
  // True if the backend writes after ending the profiling session.
  bool write_after_end = false;
  // Session identifier observed at begin and expected on later callbacks.
  uint64_t session_id = 0;
};

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
  EXPECT_EQ(0, test_sink->end_count);
  if (test_sink->end_count != 0) test_sink->write_after_end = true;
  EXPECT_EQ(test_sink->session_id, metadata->session_id);
  if (iovec_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected exactly one profile chunk iovec");
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
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    EXPECT_TRUE(test_sink->saw_device_metadata);
    EXPECT_TRUE(test_sink->saw_queue_metadata);
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
    EXPECT_TRUE(test_sink->saw_queue_metadata);
    EXPECT_NE(UINT32_MAX, metadata->physical_device_ordinal);
    EXPECT_NE(UINT32_MAX, metadata->queue_ordinal);
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
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0u, records[i].submission_id);
      EXPECT_EQ(test_sink->expected_dispatch_flags, records[i].flags);
      EXPECT_EQ(UINT32_MAX, records[i].command_index);
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

static void TestProfileSinkInitialize(TestProfileSink* sink) {
  iree_hal_resource_initialize(&kTestProfileSinkVTable, &sink->resource);
}

static iree_hal_profile_sink_t* TestProfileSinkAsBase(TestProfileSink* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

static void ExpectDispatchEventsWithinClockCorrelationRange(
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

static bool IsProfilingUnsupported(iree_status_t status) {
  return iree_status_is_unimplemented(status) ||
         iree_status_is_invalid_argument(status);
}

// Dispatches scale_and_offset directly on the queue:
// output[i] = input[i] * scale + offset.
TEST_P(QueueDispatchTest, DispatchWithConstantsAndBindings) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(input_data.data(),
                               input_data.size() * sizeof(input_data[0]),
                               input_buffer.out());
  }

  Ref<iree_hal_buffer_t> output_buffer;
  CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out());

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));
}

// Profiling must not perturb direct queue dispatch semantics or completion.
TEST_P(QueueDispatchTest, DispatchWithConstantsAndBindingsWhileProfiling) {
  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  DeviceProfilingScope profiling(device_);
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS,
                      TestProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling mode unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(input_data.data(),
                               input_data.size() * sizeof(input_data[0]),
                               input_buffer.out());
  }

  Ref<iree_hal_buffer_t> output_buffer;
  CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out());

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());
  if (sink.begin_count == 0) {
    EXPECT_EQ(0, sink.end_count);
    EXPECT_EQ(0, sink.device_metadata_count);
    EXPECT_EQ(0, sink.queue_metadata_count);
    EXPECT_EQ(0, sink.clock_correlation_count);
    EXPECT_EQ(0, sink.dispatch_event_count);
    EXPECT_FALSE(sink.saw_device_metadata);
    EXPECT_FALSE(sink.saw_queue_metadata);
    EXPECT_FALSE(sink.write_after_end);
  } else {
    EXPECT_EQ(1, sink.begin_count);
    EXPECT_EQ(1, sink.end_count);
    EXPECT_EQ(1, sink.device_metadata_count);
    EXPECT_EQ(1, sink.queue_metadata_count);
    EXPECT_GE(sink.clock_correlation_count, 3);
    EXPECT_GE(sink.dispatch_event_count, 1);
    EXPECT_TRUE(sink.saw_device_metadata);
    EXPECT_TRUE(sink.saw_queue_metadata);
    EXPECT_FALSE(sink.write_after_end);
    ExpectDispatchEventsWithinClockCorrelationRange(sink);
  }
}

// Zero-workgroup dispatches are no-ops that still participate in semaphore
// ordering and signal completion.
TEST_P(QueueDispatchTest, NoopDispatchSignalsAndDoesNotTouchBuffers) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(input_data.data(),
                               input_data.size() * sizeof(input_data[0]),
                               input_buffer.out());
  }

  Ref<iree_hal_buffer_t> output_buffer;
  CreateFilledDeviceBuffer(4 * sizeof(uint32_t), uint32_t{99},
                           output_buffer.out());

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(0, 0, 0), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{99, 99, 99, 99}));
}

// Deferred zero-workgroup dispatches still wait for their dependency before
// signaling completion, but never execute the kernel body.
TEST_P(QueueDispatchTest, DeferredNoopDispatch) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(input_data.data(),
                               input_data.size() * sizeof(input_data[0]),
                               input_buffer.out());
  }

  Ref<iree_hal_buffer_t> output_buffer;
  CreateFilledDeviceBuffer(4 * sizeof(uint32_t), uint32_t{99},
                           output_buffer.out());

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList dispatch_wait(device_, {0}, {1});
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, dispatch_wait, dispatch_signal,
      executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(0, 0, 0), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));

  uint64_t dispatch_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(dispatch_signal.semaphores[0], &dispatch_value));
  EXPECT_EQ(0u, dispatch_value);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(dispatch_wait.semaphores[0], 1,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{99, 99, 99, 99}));
}

// Wait-before-signal dispatches must be queued without head-of-line blocking
// and replayed after the wait semaphore advances.
TEST_P(QueueDispatchTest, DeferredWaitBeforeSignalDispatch) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(input_data.data(),
                               input_data.size() * sizeof(input_data[0]),
                               input_buffer.out());
  }

  Ref<iree_hal_buffer_t> output_buffer;
  CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out());

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList dispatch_wait(device_, {0}, {1});
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, dispatch_wait, dispatch_signal,
      executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));

  uint64_t dispatch_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(dispatch_signal.semaphores[0], &dispatch_value));
  EXPECT_EQ(0u, dispatch_value);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(dispatch_wait.semaphores[0], 1,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(QueueDispatchTest);

class QueueDispatchIndirectParametersTest : public CtsTestBase<> {
 protected:
  static constexpr iree_host_size_t kDispatchedWorkgroupCount = 4;
  static constexpr iree_host_size_t kOutputElementCount = 32;
  static constexpr iree_device_size_t kOutputByteLength =
      kOutputElementCount * sizeof(uint32_t);
  static constexpr iree_device_size_t kParameterByteLength =
      3 * sizeof(uint32_t);
  static constexpr uint32_t kSentinelValue = 0xCDCDCDCDu;

  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(executable_format());
    executable_params.executable_data = executable_data(iree_make_cstring_view(
        "command_buffer_dispatch_multi_workgroup_test.bin"));

    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  void CreateIndirectParameterBuffer(iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
                   IREE_HAL_BUFFER_USAGE_TRANSFER;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        device_allocator_, params, kParameterByteLength, out_buffer));
  }

  void RunIndirectQueueDispatch(
      iree_hal_dispatch_flags_t flags,
      iree_device_size_t parameter_ref_length = kParameterByteLength) {
    Ref<iree_hal_buffer_t> output_buffer;
    CreateFilledDeviceBuffer(kOutputByteLength, kSentinelValue,
                             output_buffer.out());

    Ref<iree_hal_buffer_t> parameter_buffer;
    CreateIndirectParameterBuffer(parameter_buffer.out());

    const uint32_t parameter_data[3] = {
        kDispatchedWorkgroupCount,
        1,
        1,
    };
    SemaphoreList update_signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_update(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, update_signal,
        parameter_data, /*source_offset=*/0, parameter_buffer,
        /*target_offset=*/0, sizeof(parameter_data),
        IREE_HAL_UPDATE_FLAG_NONE));

    iree_hal_buffer_ref_t binding_refs[1] = {
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 kOutputByteLength),
    };
    iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/IREE_ARRAYSIZE(binding_refs),
        /*.values=*/binding_refs,
    };

    iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
        /*workgroup_count_x=*/kOutputElementCount, /*workgroup_count_y=*/1,
        /*workgroup_count_z=*/1);
    config.workgroup_count_ref = iree_hal_make_buffer_ref(
        parameter_buffer, /*offset=*/0, parameter_ref_length);

    SemaphoreList dispatch_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, update_signal, dispatch_signal,
        executable_, /*export_ordinal=*/0, config, iree_const_byte_span_empty(),
        bindings, flags));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
    std::vector<uint32_t> expected(kOutputElementCount, kSentinelValue);
    std::iota(expected.begin(), expected.begin() + kDispatchedWorkgroupCount,
              0u);
    EXPECT_THAT(output_data, ContainerEq(expected));
  }

  void RunIndirectQueueDispatchWhileProfiling(iree_hal_dispatch_flags_t flags) {
    TestProfileSink sink = {};
    TestProfileSinkInitialize(&sink);
    sink.expected_dispatch_flags =
        IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS;
    sink.validate_dispatch_workgroup_count = false;

    DeviceProfilingScope profiling(device_);
    iree_status_t profiling_status =
        profiling.Begin(IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS,
                        TestProfileSinkAsBase(&sink));
    if (IsProfilingUnsupported(profiling_status)) {
      iree_status_free(profiling_status);
      GTEST_SKIP() << "device profiling mode unsupported by backend";
    }
    IREE_ASSERT_OK(profiling_status);

    RunIndirectQueueDispatch(flags);

    IREE_ASSERT_OK(profiling.End());
    if (sink.begin_count == 0) {
      EXPECT_EQ(0, sink.end_count);
      EXPECT_EQ(0, sink.device_metadata_count);
      EXPECT_EQ(0, sink.queue_metadata_count);
      EXPECT_EQ(0, sink.clock_correlation_count);
      EXPECT_EQ(0, sink.dispatch_event_count);
      EXPECT_FALSE(sink.saw_device_metadata);
      EXPECT_FALSE(sink.saw_queue_metadata);
      EXPECT_FALSE(sink.write_after_end);
    } else {
      EXPECT_EQ(1, sink.begin_count);
      EXPECT_EQ(1, sink.end_count);
      EXPECT_EQ(1, sink.device_metadata_count);
      EXPECT_EQ(1, sink.queue_metadata_count);
      EXPECT_GE(sink.clock_correlation_count, 2);
      EXPECT_GE(sink.dispatch_event_count, 1);
      EXPECT_TRUE(sink.saw_device_metadata);
      EXPECT_TRUE(sink.saw_queue_metadata);
      EXPECT_FALSE(sink.write_after_end);
      ExpectDispatchEventsWithinClockCorrelationRange(sink);
    }
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(QueueDispatchIndirectParametersTest, StaticParameters) {
  RunIndirectQueueDispatch(IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
}

TEST_P(QueueDispatchIndirectParametersTest, DynamicParameters) {
  RunIndirectQueueDispatch(IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
}

TEST_P(QueueDispatchIndirectParametersTest, StaticParametersWhileProfiling) {
  RunIndirectQueueDispatchWhileProfiling(
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
}

TEST_P(QueueDispatchIndirectParametersTest, DynamicParametersWhileProfiling) {
  RunIndirectQueueDispatchWhileProfiling(
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
}

TEST_P(QueueDispatchIndirectParametersTest, WholeBufferParameterRef) {
  RunIndirectQueueDispatch(IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                           IREE_HAL_WHOLE_BUFFER);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(QueueDispatchIndirectParametersTest);

}  // namespace iree::hal::cts
