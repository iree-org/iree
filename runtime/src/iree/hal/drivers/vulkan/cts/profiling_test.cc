// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>

#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class VulkanProfilingTest : public CtsTestBase<> {};

TEST_P(VulkanProfilingTest, QueueEventsRecordNativeTransferSubmissions) {
  constexpr iree_device_size_t kBufferSize = 128;

  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, source.out()));
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, target.out()));

  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  DeviceProfilingScope profiling(device_);
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 TestProfileSinkAsBase(&sink)));

  SemaphoreList empty_wait;
  SemaphoreList fill_signal(device_, {0}, {1});
  uint32_t pattern = 0xA5A5A5A5u;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, fill_signal, source, 0,
      kBufferSize, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, fill_signal, copy_signal, source, 0,
      target, 0, kBufferSize, IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      copy_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_GE(sink.queue_event_count, 1);
  EXPECT_EQ(0, sink.host_execution_event_count);
  EXPECT_EQ(0, sink.dispatch_event_count);
  EXPECT_EQ(0, sink.queue_device_event_count);
  EXPECT_TRUE(sink.saw_device_metadata);
  EXPECT_TRUE(sink.saw_queue_metadata);
  EXPECT_FALSE(sink.write_after_end);

  auto find_queue_event = [&](iree_hal_profile_queue_event_type_t type) {
    return std::find_if(sink.queue_events.begin(), sink.queue_events.end(),
                        [type](const iree_hal_profile_queue_event_t& event) {
                          return event.type == type;
                        });
  };

  auto fill_it = find_queue_event(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL);
  ASSERT_NE(sink.queue_events.end(), fill_it);
  EXPECT_EQ(kBufferSize, fill_it->payload_length);
  EXPECT_EQ(1u, fill_it->operation_count);
  EXPECT_NE(0u, fill_it->submission_id);
  EXPECT_GE(fill_it->ready_host_time_ns, fill_it->host_time_ns);

  auto copy_it = find_queue_event(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY);
  ASSERT_NE(sink.queue_events.end(), copy_it);
  EXPECT_EQ(kBufferSize, copy_it->payload_length);
  EXPECT_EQ(1u, copy_it->operation_count);
  EXPECT_NE(0u, copy_it->submission_id);
  EXPECT_GE(copy_it->ready_host_time_ns, copy_it->host_time_ns);
}

CTS_REGISTER_TEST_SUITE(VulkanProfilingTest);

}  // namespace iree::hal::cts
