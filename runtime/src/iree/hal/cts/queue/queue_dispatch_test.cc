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
#include "iree/hal/cts/util/profile_test_util.h"
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

// Dispatches scale_and_offset directly on the queue:
// output[i] = input[i] * scale + offset.
TEST_P(QueueDispatchTest, DispatchWithConstantsAndBindings) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

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

// Borrowed resource lifetimes are an optimization hint for callers that keep
// resources live until dispatch completion. The observable dispatch behavior is
// identical to the default retained mode.
TEST_P(QueueDispatchTest, DispatchWithBorrowedResourceLifetimes) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

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
      IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES));
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
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                      TestProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

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

// HAL-native CPU profiling should produce host queue and execution records for
// the direct dispatch submission.
TEST_P(QueueDispatchTest, DispatchHostQueueEventProfiling) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  DeviceProfilingScope profiling(device_);
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS,
                      TestProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "host queue profiling unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

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
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_GE(sink.queue_event_count, 1);
  EXPECT_GE(sink.host_execution_event_count, 1);
  EXPECT_EQ(0, sink.dispatch_event_count);
  EXPECT_EQ(0, sink.queue_device_event_count);
  EXPECT_TRUE(sink.saw_device_metadata);
  EXPECT_TRUE(sink.saw_queue_metadata);
  EXPECT_FALSE(sink.write_after_end);

  auto queue_event_it = std::find_if(
      sink.queue_events.begin(), sink.queue_events.end(),
      [](const iree_hal_profile_queue_event_t& event) {
        return event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
      });
  ASSERT_NE(sink.queue_events.end(), queue_event_it);
  EXPECT_EQ(1u, queue_event_it->operation_count);

  auto host_event_it = std::find_if(
      sink.host_execution_events.begin(), sink.host_execution_events.end(),
      [](const iree_hal_profile_host_execution_event_t& event) {
        return event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
      });
  ASSERT_NE(sink.host_execution_events.end(), host_event_it);
  EXPECT_EQ(IREE_STATUS_OK, host_event_it->status_code);
  EXPECT_EQ(0u, host_event_it->export_ordinal);
  EXPECT_EQ(queue_event_it->submission_id, host_event_it->submission_id);
  EXPECT_GE(host_event_it->end_host_time_ns, host_event_it->start_host_time_ns);
}

// Device queue event profiling should produce one device-domain span for the
// direct dispatch submission without also requiring dispatch-event capture.
TEST_P(QueueDispatchTest, DispatchDeviceQueueEventProfiling) {
  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  DeviceProfilingScope profiling(device_);
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
                      TestProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device queue profiling unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

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
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_GE(sink.clock_correlation_count, 3);
  EXPECT_EQ(0, sink.dispatch_event_count);
  EXPECT_TRUE(sink.dispatch_events.empty());
  EXPECT_GE(sink.queue_device_event_count, 1);
  ASSERT_EQ(1u, sink.queue_device_events.size());
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH,
            sink.queue_device_events[0].type);
  EXPECT_EQ(1u, sink.queue_device_events[0].operation_count);
  EXPECT_TRUE(sink.saw_device_metadata);
  EXPECT_TRUE(sink.saw_queue_metadata);
  EXPECT_FALSE(sink.write_after_end);
  ExpectQueueDeviceEventsWithinClockCorrelationRange(sink);
}

// Capture filters must not perturb direct queue dispatch semantics or
// completion when no dispatch event matches.
TEST_P(QueueDispatchTest, DispatchProfileFilterCanSkipDirectDispatchEvents) {
  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  profiling_options.sink = TestProfileSinkAsBase(&sink);
  profiling_options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN;
  profiling_options.capture_filter.executable_export_pattern =
      IREE_SV("iree-hal-cts-never-matches-*");
  DeviceProfilingScope profiling(device_);
  iree_status_t profiling_status = profiling.Begin(&profiling_options);
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

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
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(0, sink.dispatch_event_count);
  EXPECT_TRUE(sink.dispatch_events.empty());
  EXPECT_FALSE(sink.write_after_end);
}

// Zero-workgroup dispatches are no-ops that still participate in semaphore
// ordering and signal completion.
TEST_P(QueueDispatchTest, NoopDispatchSignalsAndDoesNotTouchBuffers) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateFilledDeviceBuffer(4 * sizeof(uint32_t), uint32_t{99},
                                          output_buffer.out()));

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
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateFilledDeviceBuffer(4 * sizeof(uint32_t), uint32_t{99},
                                          output_buffer.out()));

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
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out()));

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

  iree_status_t CreateIndirectParameterBuffer(iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
                   IREE_HAL_BUFFER_USAGE_TRANSFER;
    return iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                              kParameterByteLength, out_buffer);
  }

  void RunIndirectQueueDispatch(
      iree_hal_dispatch_flags_t flags,
      iree_device_size_t parameter_ref_length = kParameterByteLength) {
    Ref<iree_hal_buffer_t> output_buffer;
    IREE_ASSERT_OK(CreateFilledDeviceBuffer(kOutputByteLength, kSentinelValue,
                                            output_buffer.out()));

    Ref<iree_hal_buffer_t> parameter_buffer;
    IREE_ASSERT_OK(CreateIndirectParameterBuffer(parameter_buffer.out()));

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
        profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                            IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
                        TestProfileSinkAsBase(&sink));
    if (IsProfilingUnsupported(profiling_status)) {
      iree_status_free(profiling_status);
      GTEST_SKIP() << "device profiling data family unsupported by backend";
    }
    IREE_ASSERT_OK(profiling_status);

    RunIndirectQueueDispatch(flags);

    IREE_ASSERT_OK(profiling.End());
    EXPECT_EQ(1, sink.begin_count);
    EXPECT_EQ(1, sink.end_count);
    EXPECT_EQ(1, sink.device_metadata_count);
    EXPECT_EQ(1, sink.queue_metadata_count);
    EXPECT_GE(sink.clock_correlation_count, 2);
    EXPECT_GE(sink.dispatch_event_count, 1);
    EXPECT_GE(sink.queue_device_event_count, 1);
    const iree_host_size_t dispatch_queue_device_event_count = std::count_if(
        sink.queue_device_events.begin(), sink.queue_device_events.end(),
        [](const iree_hal_profile_queue_device_event_t& event) {
          return event.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
        });
    EXPECT_GE(dispatch_queue_device_event_count, 1u);
    EXPECT_TRUE(sink.saw_device_metadata);
    EXPECT_TRUE(sink.saw_queue_metadata);
    EXPECT_FALSE(sink.write_after_end);
    ExpectDispatchEventsWithinClockCorrelationRange(sink);
    ExpectQueueDeviceEventsWithinClockCorrelationRange(sink);
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
