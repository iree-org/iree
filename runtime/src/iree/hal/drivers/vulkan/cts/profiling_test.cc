// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class VulkanProfilingTest : public CtsTestBase<> {
 protected:
  iree_status_t CreateScaleAndOffsetExecutable(
      Ref<iree_hal_executable_cache_t>& executable_cache,
      Ref<iree_hal_executable_t>& executable) {
    IREE_RETURN_IF_ERROR(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), executable_cache.out()));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(executable_format());
    executable_params.executable_data = executable_data(iree_make_cstring_view(
        "command_buffer_dispatch_constants_bindings_test.bin"));

    if (!iree_hal_executable_cache_can_prepare_format(
            executable_cache.get(), executable_params.caching_mode,
            executable_params.executable_format)) {
      return iree_ok_status();
    }

    return iree_hal_executable_cache_prepare_executable(
        executable_cache.get(), &executable_params, executable.out());
  }
};

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

TEST_P(VulkanProfilingTest,
       LightweightStatisticsRecordQueueAllocaMemoryEvents) {
  constexpr iree_device_size_t kBufferSize = 1024;

  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  iree_hal_device_profiling_options_t options = {0};
  options.flags = IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
  options.sink = TestProfileSinkAsBase(&sink);

  DeviceProfilingScope profiling(device_);
  IREE_ASSERT_OK(profiling.Begin(&options));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  iree_hal_buffer_t* raw_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
      /*pool=*/NULL, params, kBufferSize, IREE_HAL_ALLOCA_FLAG_NONE,
      &raw_buffer));
  buffer.reset(raw_buffer);

  SemaphoreList dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
      buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_GE(sink.queue_event_count, 1);
  EXPECT_GE(sink.memory_event_count, 1);
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
  auto find_memory_event = [&](iree_hal_profile_memory_event_type_t type) {
    return std::find_if(sink.memory_events.begin(), sink.memory_events.end(),
                        [type](const iree_hal_profile_memory_event_t& event) {
                          return event.type == type;
                        });
  };

  auto queue_alloca_it =
      find_queue_event(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA);
  ASSERT_NE(sink.queue_events.end(), queue_alloca_it);
  EXPECT_EQ(kBufferSize, queue_alloca_it->payload_length);
  EXPECT_NE(0u, queue_alloca_it->allocation_id);

  auto queue_dealloca_it =
      find_queue_event(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA);
  ASSERT_NE(sink.queue_events.end(), queue_dealloca_it);
  EXPECT_EQ(queue_alloca_it->allocation_id, queue_dealloca_it->allocation_id);

  auto memory_alloca_it =
      find_memory_event(IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA);
  ASSERT_NE(sink.memory_events.end(), memory_alloca_it);
  EXPECT_EQ(queue_alloca_it->allocation_id, memory_alloca_it->allocation_id);
  EXPECT_EQ(kBufferSize, memory_alloca_it->length);
  EXPECT_NE(0u, memory_alloca_it->submission_id);

  auto memory_dealloca_it =
      find_memory_event(IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA);
  ASSERT_NE(sink.memory_events.end(), memory_dealloca_it);
  EXPECT_EQ(queue_alloca_it->allocation_id, memory_dealloca_it->allocation_id);
  EXPECT_EQ(kBufferSize, memory_dealloca_it->length);
  EXPECT_NE(0u, memory_dealloca_it->submission_id);
}

TEST_P(VulkanProfilingTest, ExecutableMetadataRecordsDirectDispatchExports) {
  Ref<iree_hal_executable_cache_t> executable_cache;
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(CreateScaleAndOffsetExecutable(executable_cache, executable));
  if (!executable) {
    GTEST_SKIP() << "descriptor executable format is disabled on this device";
  }

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

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer.get(), /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer.get())),
      iree_hal_make_buffer_ref(
          output_buffer.get(), /*offset=*/0,
          iree_hal_buffer_byte_length(output_buffer.get())),
  };
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  DeviceProfilingScope profiling(device_);
  IREE_ASSERT_OK(
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
                      TestProfileSinkAsBase(&sink)));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable.get(), /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  ASSERT_EQ(4u, output_data.size());
  EXPECT_EQ(13u, output_data[0]);
  EXPECT_EQ(16u, output_data[1]);
  EXPECT_EQ(19u, output_data[2]);
  EXPECT_EQ(22u, output_data[3]);

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_EQ(1, sink.executable_metadata_count);
  EXPECT_EQ(1, sink.executable_export_metadata_count);
  EXPECT_EQ(0, sink.queue_event_count);
  EXPECT_EQ(0, sink.memory_event_count);
  EXPECT_FALSE(sink.executable_ids.empty());
  EXPECT_FALSE(sink.export_record_executable_ids.empty());
  EXPECT_EQ(sink.executable_ids[0], sink.export_record_executable_ids[0]);
  EXPECT_TRUE(sink.saw_device_metadata);
  EXPECT_TRUE(sink.saw_queue_metadata);
  EXPECT_FALSE(sink.write_after_end);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(VulkanProfilingTest);

}  // namespace iree::hal::cts
