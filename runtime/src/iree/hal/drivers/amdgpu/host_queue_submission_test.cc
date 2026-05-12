// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#include <cstdint>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/host_queue_waits.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

constexpr uint32_t kNoHarvestPacketOffset = UINT32_MAX;

class HostQueueSubmissionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueueSubmissionTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueSubmissionTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueSubmissionTest::topology_;

class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(base_device_);
    iree_hal_device_group_release(device_group_);
  }

  iree_status_t Initialize(
      const iree_hal_amdgpu_logical_device_options_t* options,
      const iree_hal_amdgpu_libhsa_t* libhsa,
      const iree_hal_amdgpu_topology_t* topology,
      iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), options, libhsa, topology, create_context_.params(),
        host_allocator, &base_device_));
    return iree_hal_device_group_create_from_device(
        base_device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_);
  }

  iree_hal_amdgpu_host_queue_t* first_host_queue() const {
    iree_hal_amdgpu_logical_device_t* logical_device =
        (iree_hal_amdgpu_logical_device_t*)base_device_;
    if (logical_device->physical_device_count == 0) return NULL;
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[0];
    if (physical_device->host_queue_count == 0) return NULL;
    return &physical_device->host_queues[0];
  }

  iree_hal_device_t* base_device() const { return base_device_; }

 private:
  // Creation context supplying the proactor pool and frontier tracker.
  iree::hal::cts::DeviceCreateContext create_context_;

  // Test-owned device reference released before the topology-owning group.
  iree_hal_device_t* base_device_ = NULL;

  // Device group that owns the topology assigned to |base_device_|.
  iree_hal_device_group_t* device_group_ = NULL;
};

class HostQueueHsaProfilingScope {
 public:
  explicit HostQueueHsaProfilingScope(iree_hal_amdgpu_host_queue_t* queue)
      : queue_(queue) {}

  ~HostQueueHsaProfilingScope() {
    if (is_enabled_) {
      IREE_EXPECT_OK(
          iree_hal_amdgpu_host_queue_set_hsa_profiling_enabled(queue_, false));
    }
  }

  iree_status_t Enable() {
    iree_status_t status =
        iree_hal_amdgpu_host_queue_set_hsa_profiling_enabled(queue_, true);
    if (iree_status_is_ok(status)) {
      is_enabled_ = true;
    }
    return status;
  }

 private:
  // Host queue whose HSA profiling mode is enabled for the current test.
  iree_hal_amdgpu_host_queue_t* queue_;

  // True once |queue_| HSA profiling has been enabled and must be disabled.
  bool is_enabled_ = false;
};

struct NoopProfileSink {
  // HAL resource header for the sink.
  iree_hal_resource_t resource;
};

static void NoopProfileSinkDestroy(iree_hal_profile_sink_t* sink) {
  (void)sink;
}

static iree_status_t NoopProfileSinkBeginSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  (void)sink;
  (void)metadata;
  return iree_ok_status();
}

static iree_status_t NoopProfileSinkWrite(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  (void)sink;
  (void)metadata;
  (void)iovec_count;
  (void)iovecs;
  return iree_ok_status();
}

static iree_status_t NoopProfileSinkEndSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)sink;
  (void)metadata;
  (void)session_status_code;
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t kNoopProfileSinkVTable = {
    /*.destroy=*/NoopProfileSinkDestroy,
    /*.begin_session=*/NoopProfileSinkBeginSession,
    /*.write=*/NoopProfileSinkWrite,
    /*.end_session=*/NoopProfileSinkEndSession,
};

static void NoopProfileSinkInitialize(NoopProfileSink* sink) {
  iree_hal_resource_initialize(&kNoopProfileSinkVTable, &sink->resource);
}

static iree_hal_profile_sink_t* NoopProfileSinkAsBase(NoopProfileSink* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

typedef struct DispatchSubmissionPlanCase {
  // Number of wait-barrier packets preceding the dispatch payload.
  uint8_t barrier_count;
  // Whether a dispatch profiling event is reserved for the submission.
  bool reserve_dispatch_event;
  // Whether a queue-device event is reserved for the submission.
  bool reserve_queue_device_event;
  // Expected total AQL packets reserved for the submission.
  uint32_t expected_packet_count;
  // Expected dispatch packet offset from the first reserved packet.
  uint32_t expected_dispatch_packet_offset;
  // Expected harvest packet offset, or kNoHarvestPacketOffset when absent.
  uint32_t expected_harvest_packet_offset;
  // True when the dispatch packet should signal queue completion directly.
  bool expect_dispatch_completion_signal;
} DispatchSubmissionPlanCase;

static void ExpectDispatchSubmissionPlan(
    iree_hal_amdgpu_host_queue_t* queue,
    const DispatchSubmissionPlanCase& plan_case) {
  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  resolution.barrier_count = plan_case.barrier_count;
  const iree_hal_semaphore_list_t empty_signal_list = {0};
  iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events = {0};
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_queue_event_info = {
      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH,
      .operation_count = 1,
  };
  iree_hal_amdgpu_host_queue_set_profile_flags(
      queue, plan_case.reserve_queue_device_event
                 ? IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_QUEUE_DEVICE_EVENTS
                 : IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_NONE);

  bool is_ready = false;
  iree_hal_amdgpu_host_queue_dispatch_submission_t submission = {};
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status = iree_ok_status();
  if (plan_case.reserve_dispatch_event) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
        queue, /*event_count=*/1, &profile_events);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
        queue, &resolution, empty_signal_list,
        /*operation_resource_count=*/0, /*kernarg_block_count=*/1,
        profile_events, &profile_queue_event_info, &is_ready, &submission);
  }
  if (iree_status_is_ok(status) && is_ready) {
    EXPECT_EQ(plan_case.expected_packet_count, submission.kernel.packet_count);
    EXPECT_EQ(submission.kernel.first_packet_id +
                  plan_case.expected_dispatch_packet_offset,
              submission.dispatch_packet_id);
    EXPECT_EQ(iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring,
                                              submission.dispatch_packet_id),
              submission.dispatch_slot);

    const bool has_harvest_packet =
        plan_case.expected_harvest_packet_offset != kNoHarvestPacketOffset;
    if (has_harvest_packet) {
      EXPECT_EQ(
          iree_hal_amdgpu_aql_ring_packet(
              &queue->aql_ring, submission.kernel.first_packet_id +
                                    plan_case.expected_harvest_packet_offset),
          submission.profile_harvest_slot);
    } else {
      EXPECT_EQ(NULL, submission.profile_harvest_slot);
    }

    const bool has_dispatch_completion_signal =
        submission.dispatch_completion_signal.handle != 0;
    EXPECT_EQ(plan_case.expect_dispatch_completion_signal,
              has_dispatch_completion_signal);

    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, submission.profile_queue_device_events);
    iree_hal_amdgpu_host_queue_fail_kernel_submission(queue,
                                                      &submission.kernel);
  }
  if (profile_events.event_count != 0) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  IREE_EXPECT_OK(status);
  EXPECT_TRUE(is_ready);
}

typedef struct Pm4IbSubmissionPlanCase {
  // Number of wait-barrier packets preceding the PM4-IB payload.
  uint8_t barrier_count;
  // Whether a queue-device event is reserved for the submission.
  bool reserve_queue_device_event;
  // Expected total AQL packets reserved for the submission.
  uint32_t expected_packet_count;
  // Expected PM4-IB packet offset from the first reserved packet.
  uint32_t expected_pm4_ib_packet_offset;
} Pm4IbSubmissionPlanCase;

static void ExpectPm4IbSubmissionPlan(
    iree_hal_amdgpu_host_queue_t* queue,
    const Pm4IbSubmissionPlanCase& plan_case) {
  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  resolution.barrier_count = plan_case.barrier_count;
  const iree_hal_semaphore_list_t empty_signal_list = {0};
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_queue_event_info = {
      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE,
      .operation_count = 1,
  };
  iree_hal_amdgpu_host_queue_set_profile_flags(
      queue, plan_case.reserve_queue_device_event
                 ? IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_QUEUE_DEVICE_EVENTS
                 : IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_NONE);

  bool is_ready = false;
  iree_hal_amdgpu_host_queue_pm4_ib_submission_t submission = {};
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status = iree_hal_amdgpu_host_queue_try_begin_pm4_ib_submission(
      queue, &resolution, empty_signal_list,
      /*operation_resource_count=*/0, &profile_queue_event_info, &is_ready,
      &submission);
  if (iree_status_is_ok(status) && is_ready) {
    EXPECT_EQ(plan_case.expected_packet_count, submission.kernel.packet_count);
    EXPECT_EQ(
        iree_hal_amdgpu_aql_ring_packet(
            &queue->aql_ring, submission.kernel.first_packet_id +
                                  plan_case.expected_pm4_ib_packet_offset),
        submission.pm4_ib_packet_slot);
    EXPECT_EQ(&queue->pm4_ib_slots[(submission.kernel.first_packet_id +
                                    plan_case.expected_pm4_ib_packet_offset) &
                                   queue->aql_ring.mask],
              submission.pm4_ib_slot);
    if (plan_case.reserve_queue_device_event) {
      EXPECT_EQ(1u, submission.profile_queue_device_events.event_count);
      EXPECT_EQ(IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT,
                iree_hal_amdgpu_pm4_ib_builder_dword_count(
                    &submission.pm4_ib_builder));
      EXPECT_EQ(submission.pm4_ib_slot->dwords[0],
                iree_hal_amdgpu_pm4_make_header(
                    IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                    IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
    } else {
      EXPECT_EQ(0u, submission.profile_queue_device_events.event_count);
      EXPECT_EQ(0u, iree_hal_amdgpu_pm4_ib_builder_dword_count(
                        &submission.pm4_ib_builder));
    }

    iree_hal_amdgpu_host_queue_fail_pm4_ib_submission(queue, &submission);
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  IREE_EXPECT_OK(status);
  EXPECT_TRUE(is_ready);
}

static bool HostQueueSupportsQueueDeviceProfiling(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
      queue->pm4_timestamp_strategy);
}

TEST_F(HostQueueSubmissionTest, DispatchPacketAccountingCombinations) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  HostQueueHsaProfilingScope profiling_scope(queue);
  IREE_ASSERT_OK(profiling_scope.Enable());

  const DispatchSubmissionPlanCase cases[] = {
      {
          /*barrier_count=*/0,
          /*reserve_dispatch_event=*/false,
          /*reserve_queue_device_event=*/false,
          /*expected_packet_count=*/1,
          /*expected_dispatch_packet_offset=*/0,
          /*expected_harvest_packet_offset=*/kNoHarvestPacketOffset,
          /*expect_dispatch_completion_signal=*/true,
      },
      {
          /*barrier_count=*/2,
          /*reserve_dispatch_event=*/false,
          /*reserve_queue_device_event=*/false,
          /*expected_packet_count=*/3,
          /*expected_dispatch_packet_offset=*/2,
          /*expected_harvest_packet_offset=*/kNoHarvestPacketOffset,
          /*expect_dispatch_completion_signal=*/true,
      },
      {
          /*barrier_count=*/0,
          /*reserve_dispatch_event=*/false,
          /*reserve_queue_device_event=*/true,
          /*expected_packet_count=*/3,
          /*expected_dispatch_packet_offset=*/1,
          /*expected_harvest_packet_offset=*/kNoHarvestPacketOffset,
          /*expect_dispatch_completion_signal=*/false,
      },
      {
          /*barrier_count=*/0,
          /*reserve_dispatch_event=*/true,
          /*reserve_queue_device_event=*/false,
          /*expected_packet_count=*/2,
          /*expected_dispatch_packet_offset=*/0,
          /*expected_harvest_packet_offset=*/1,
          /*expect_dispatch_completion_signal=*/true,
      },
      {
          /*barrier_count=*/2,
          /*reserve_dispatch_event=*/true,
          /*reserve_queue_device_event=*/false,
          /*expected_packet_count=*/4,
          /*expected_dispatch_packet_offset=*/2,
          /*expected_harvest_packet_offset=*/3,
          /*expect_dispatch_completion_signal=*/true,
      },
      {
          /*barrier_count=*/0,
          /*reserve_dispatch_event=*/true,
          /*reserve_queue_device_event=*/true,
          /*expected_packet_count=*/4,
          /*expected_dispatch_packet_offset=*/1,
          /*expected_harvest_packet_offset=*/2,
          /*expect_dispatch_completion_signal=*/true,
      },
  };
  for (const DispatchSubmissionPlanCase& plan_case : cases) {
    if (plan_case.reserve_queue_device_event &&
        !HostQueueSupportsQueueDeviceProfiling(queue)) {
      GTEST_SKIP() << "queue device profiling is not supported";
    }
    ExpectDispatchSubmissionPlan(queue, plan_case);
  }
}

TEST_F(HostQueueSubmissionTest, LightweightStatisticsAvoidDeviceTimestamps) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  NoopProfileSink sink = {};
  NoopProfileSinkInitialize(&sink);

  iree_hal_device_profiling_options_t profiling_options = {};
  profiling_options.flags =
      IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
  profiling_options.sink = NoopProfileSinkAsBase(&sink);
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(test_device.base_device(),
                                                 &profiling_options));

  iree_hal_amdgpu_logical_device_t* logical_device =
      reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(
          test_device.base_device());
  EXPECT_EQ(logical_device->profiling.options.flags,
            IREE_HAL_DEVICE_PROFILING_FLAG_NONE);
  EXPECT_EQ(logical_device->profiling.options.data_families,
            IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA);
  EXPECT_TRUE(queue->profiling.queue_events_enabled);
  EXPECT_FALSE(queue->profiling.queue_device_events_enabled);
  EXPECT_FALSE(queue->profiling.dispatch_profiling_enabled);

  IREE_EXPECT_OK(iree_hal_device_profiling_end(test_device.base_device()));
}

TEST_F(HostQueueSubmissionTest, Pm4IbPacketAccountingCombinations) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);
  if (!queue->pm4_ib_slots) {
    GTEST_SKIP() << "PM4 IB slots are not available";
  }

  HostQueueHsaProfilingScope profiling_scope(queue);
  IREE_ASSERT_OK(profiling_scope.Enable());

  const Pm4IbSubmissionPlanCase cases[] = {
      {
          /*barrier_count=*/0,
          /*reserve_queue_device_event=*/false,
          /*expected_packet_count=*/1,
          /*expected_pm4_ib_packet_offset=*/0,
      },
      {
          /*barrier_count=*/2,
          /*reserve_queue_device_event=*/false,
          /*expected_packet_count=*/3,
          /*expected_pm4_ib_packet_offset=*/2,
      },
      {
          /*barrier_count=*/0,
          /*reserve_queue_device_event=*/true,
          /*expected_packet_count=*/1,
          /*expected_pm4_ib_packet_offset=*/0,
      },
      {
          /*barrier_count=*/2,
          /*reserve_queue_device_event=*/true,
          /*expected_packet_count=*/3,
          /*expected_pm4_ib_packet_offset=*/2,
      },
  };
  for (const Pm4IbSubmissionPlanCase& plan_case : cases) {
    if (plan_case.reserve_queue_device_event &&
        !HostQueueSupportsQueueDeviceProfiling(queue)) {
      GTEST_SKIP() << "queue device profiling is not supported";
    }
    ExpectPm4IbSubmissionPlan(queue, plan_case);
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
