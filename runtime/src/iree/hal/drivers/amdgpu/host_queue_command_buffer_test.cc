// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "runtime/src/iree/hal/drivers/amdgpu/cts/testdata_amdgpu.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;

class HostQueueCommandBufferTest : public ::testing::Test {
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

iree_allocator_t HostQueueCommandBufferTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueCommandBufferTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueCommandBufferTest::topology_;

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

  iree_hal_device_t* base_device() const { return base_device_; }

  iree_hal_allocator_t* allocator() const {
    return iree_hal_device_allocator(base_device_);
  }

  iree_hal_amdgpu_logical_device_t* logical_device() const {
    return (iree_hal_amdgpu_logical_device_t*)base_device_;
  }

  iree_hal_amdgpu_host_queue_t* first_host_queue() const {
    iree_hal_amdgpu_logical_device_t* logical_device = this->logical_device();
    if (logical_device->physical_device_count == 0) return NULL;
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[0];
    if (physical_device->host_queue_count == 0) return NULL;
    return &physical_device->host_queues[0];
  }

 private:
  // Creation context supplying the proactor pool and frontier tracker.
  iree::hal::cts::DeviceCreateContext create_context_;

  // Test-owned device reference released before the topology-owning group.
  iree_hal_device_t* base_device_ = NULL;

  // Device group that owns the topology assigned to |base_device_|.
  iree_hal_device_group_t* device_group_ = NULL;
};

static iree_status_t CreateHostVisibleTransferBuffer(
    iree_hal_allocator_t* allocator, iree_device_size_t buffer_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_allocate_buffer(allocator, params, buffer_size,
                                            out_buffer);
}

static iree_status_t CreateHostVisibleDispatchBuffer(
    iree_hal_allocator_t* allocator, iree_device_size_t buffer_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                 IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_allocate_buffer(allocator, params, buffer_size,
                                            out_buffer);
}

static iree_status_t CreateHostVisibleIndirectParameterBuffer(
    iree_hal_allocator_t* allocator, iree_device_size_t buffer_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
                 IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_allocate_buffer(allocator, params, buffer_size,
                                            out_buffer);
}

static iree_const_byte_span_t FindCtsExecutableData(
    iree_string_view_t file_name) {
  const iree_file_toc_t* toc = iree_cts_testdata_amdgpu_create();
  for (iree_host_size_t i = 0; toc[i].name != nullptr; ++i) {
    if (iree_string_view_equal(file_name,
                               iree_make_cstring_view(toc[i].name))) {
      return iree_make_const_byte_span(
          reinterpret_cast<const uint8_t*>(toc[i].data), toc[i].size);
    }
  }
  return iree_const_byte_span_empty();
}

static iree_status_t LoadCtsExecutable(
    iree_hal_device_t* device, iree_string_view_t file_name,
    iree_hal_executable_cache_t** out_executable_cache,
    iree_hal_executable_t** out_executable) {
  *out_executable_cache = NULL;
  *out_executable = NULL;

  iree_const_byte_span_t executable_data = FindCtsExecutableData(file_name);
  if (IREE_UNLIKELY(executable_data.data_length == 0)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AMDGPU CTS executable not found");
  }

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  iree_status_t status = iree_hal_executable_cache_create(
      device, iree_make_cstring_view("default"), &executable_cache);

  char executable_format[128] = {0};
  iree_host_size_t inferred_size = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_cache_infer_format(
        executable_cache, IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA,
        executable_data, IREE_ARRAYSIZE(executable_format), executable_format,
        &inferred_size);
  }
  (void)inferred_size;

  if (iree_status_is_ok(status)) {
    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(executable_format);
    executable_params.executable_data = executable_data;
    status = iree_hal_executable_cache_prepare_executable(
        executable_cache, &executable_params, &executable);
  }

  if (iree_status_is_ok(status)) {
    *out_executable_cache = executable_cache;
    *out_executable = executable;
  } else {
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
  }
  return status;
}

static iree_status_t QueueTransientTransferBuffer(
    iree_hal_device_t* device, const iree_hal_semaphore_list_t signal_list,
    iree_device_size_t buffer_size, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  return iree_hal_device_queue_alloca(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                      iree_hal_semaphore_list_empty(),
                                      signal_list,
                                      /*pool=*/NULL, params, buffer_size,
                                      IREE_HAL_ALLOCA_FLAG_NONE, out_buffer);
}

static iree_status_t EnqueueRawBlockingBarrier(
    iree_hal_amdgpu_host_queue_t* queue, hsa_signal_t blocker_signal) {
  const uint64_t packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&queue->aql_ring, /*count=*/1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  const hsa_signal_t dep_signals[1] = {blocker_signal};
  const uint16_t header = iree_hal_amdgpu_aql_emit_barrier_and(
      &packet->barrier_and, dep_signals, IREE_ARRAYSIZE(dep_signals),
      iree_hal_amdgpu_aql_packet_control_barrier_system(),
      iree_hsa_signal_null());
  iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring, packet_id);
  return iree_ok_status();
}

static bool HostQueueHasPostDrainAction(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->post_drain_mutex);
  const bool has_action = queue->post_drain_head != NULL;
  iree_slim_mutex_unlock(&queue->post_drain_mutex);
  return has_action;
}

static iree_status_t CreateSemaphore(iree_hal_device_t* device,
                                     iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_semaphore);
}

static iree_status_t SubmitProfiledQueueFill(TestLogicalDevice* test_device) {
  Ref<iree_hal_buffer_t> target_buffer;
  IREE_RETURN_IF_ERROR(CreateHostVisibleTransferBuffer(
      test_device->allocator(), sizeof(uint32_t), target_buffer.out()));

  Ref<iree_hal_semaphore_t> signal;
  IREE_RETURN_IF_ERROR(
      CreateSemaphore(test_device->base_device(), signal.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  const iree_hal_semaphore_list_t signal_list = {
      /*count=*/1,
      /*semaphores=*/&signal_ptr,
      /*payload_values=*/&signal_value,
  };
  const uint32_t pattern = 0xA11CA7E5u;
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
      test_device->base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, target_buffer,
      /*target_offset=*/0, sizeof(pattern), &pattern, sizeof(pattern),
      IREE_HAL_FILL_FLAG_NONE));
  return iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

class DeviceProfilingScope {
 public:
  explicit DeviceProfilingScope(iree_hal_device_t* device) : device_(device) {}

  ~DeviceProfilingScope() {
    if (is_active_) {
      IREE_EXPECT_OK(iree_hal_device_profiling_end(device_));
    }
  }

  iree_status_t Begin(iree_hal_device_profiling_data_families_t data_families,
                      iree_hal_profile_sink_t* sink) {
    iree_hal_device_profiling_options_t options = {0};
    options.data_families = data_families;
    options.sink = sink;
    return Begin(&options);
  }

  iree_status_t Begin(const iree_hal_device_profiling_options_t* options) {
    iree_status_t status = iree_hal_device_profiling_begin(device_, options);
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

struct CommandBufferProfileSink {
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

  // Number of executable metadata chunks observed.
  int executable_metadata_count = 0;

  // Number of executable export metadata chunks observed.
  int executable_export_metadata_count = 0;

  // Number of command-buffer metadata chunks observed.
  int command_buffer_metadata_count = 0;

  // Number of command-operation metadata chunks observed.
  int command_operation_metadata_count = 0;

  // Number of clock correlation chunks observed.
  int clock_correlation_count = 0;

  // Number of host queue event chunks observed.
  int queue_event_count = 0;

  // Number of device queue event chunks observed.
  int queue_device_event_count = 0;

  // Number of memory event chunks observed.
  int memory_event_count = 0;

  // Number of event relationship chunks observed.
  int relationship_count = 0;

  // Number of counter set metadata chunks observed.
  int counter_set_metadata_count = 0;

  // Number of counter metadata chunks observed.
  int counter_metadata_count = 0;

  // Number of counter sample chunks observed.
  int counter_sample_count = 0;

  // Number of chunks marked truncated by the producer.
  int truncated_chunk_count = 0;

  // Total dropped records reported by truncated chunks.
  uint64_t dropped_record_count = 0;

  // Dropped queue event records reported by QUEUE_EVENTS chunks.
  uint64_t queue_event_dropped_record_count = 0;

  // Dropped memory event records reported by MEMORY_EVENTS chunks.
  uint64_t memory_event_dropped_record_count = 0;

  // Executable identifiers copied from EXECUTABLES chunks.
  std::vector<uint64_t> executable_ids;

  // Executable identifiers copied from EXECUTABLE_EXPORTS chunks.
  std::vector<uint64_t> executable_export_ids;

  // Command-buffer identifiers copied from COMMAND_BUFFERS chunks.
  std::vector<uint64_t> command_buffer_ids;

  // Command operations copied from COMMAND_OPERATIONS chunks.
  std::vector<iree_hal_profile_command_operation_record_t> command_operations;

  // Clock correlation records copied from CLOCK_CORRELATIONS chunks.
  std::vector<iree_hal_profile_clock_correlation_record_t> clock_correlations;

  // Host queue events copied from QUEUE_EVENTS chunks.
  std::vector<iree_hal_profile_queue_event_t> queue_events;

  // Device queue events copied from QUEUE_DEVICE_EVENTS chunks.
  std::vector<iree_hal_profile_queue_device_event_t> queue_device_events;

  // Memory events copied from MEMORY_EVENTS chunks.
  std::vector<iree_hal_profile_memory_event_t> memory_events;

  // Event relationships copied from EVENT_RELATIONSHIPS chunks.
  std::vector<iree_hal_profile_event_relationship_record_t> event_relationships;

  // Dispatch events copied from DISPATCH_EVENTS chunks.
  std::vector<iree_hal_profile_dispatch_event_t> dispatch_events;

  // Counter sample records copied from COUNTER_SAMPLES chunks.
  std::vector<iree_hal_profile_counter_sample_record_t> counter_samples;

  // Counter sample values copied from COUNTER_SAMPLES chunks.
  std::vector<uint64_t> counter_sample_values;

  // Counter set metadata records copied from COUNTER_SETS chunks.
  std::vector<iree_hal_profile_counter_set_record_t> counter_set_records;

  // Counter metadata records copied from COUNTERS chunks.
  std::vector<iree_hal_profile_counter_record_t> counter_records;

  // Physical device ordinals for entries in |dispatch_events|.
  std::vector<uint32_t> dispatch_event_physical_device_ordinals;

  // Session identifier observed at begin and expected on later callbacks.
  uint64_t session_id = 0;

  // True if the backend writes after ending the profiling session.
  bool write_after_end = false;

  // Status code returned from begin_session, or OK for success.
  iree_status_code_t fail_begin_session_status_code = IREE_STATUS_OK;

  // Content type whose write callback should fail, or empty when disabled.
  iree_string_view_t fail_write_content_type = {nullptr, 0};

  // Number of matching write callbacks that should fail.
  int fail_write_remaining = 0;

  // Status code returned from matching write callbacks.
  iree_status_code_t fail_write_status_code = IREE_STATUS_OK;

  // Expected session status code passed to end_session.
  iree_status_code_t expected_end_session_status_code = IREE_STATUS_OK;

  // Status code observed by the most recent end_session callback.
  iree_status_code_t observed_end_session_status_code = IREE_STATUS_OK;

  // Status code returned from end_session, or OK for success.
  iree_status_code_t fail_end_session_status_code = IREE_STATUS_OK;
};

static CommandBufferProfileSink* CommandBufferProfileSinkCast(
    iree_hal_profile_sink_t* sink) {
  return reinterpret_cast<CommandBufferProfileSink*>(sink);
}

static void CommandBufferProfileSinkDestroy(iree_hal_profile_sink_t* sink) {
  (void)sink;
}

static iree_status_t CommandBufferProfileSinkBeginSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  CommandBufferProfileSink* test_sink = CommandBufferProfileSinkCast(sink);
  EXPECT_EQ(0, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  EXPECT_TRUE(iree_string_view_equal(metadata->content_type,
                                     IREE_HAL_PROFILE_CONTENT_TYPE_SESSION));
  ++test_sink->begin_count;
  if (test_sink->fail_begin_session_status_code != IREE_STATUS_OK) {
    return iree_make_status(test_sink->fail_begin_session_status_code,
                            "injected profile sink begin_session failure");
  }
  test_sink->session_id = metadata->session_id;
  return iree_ok_status();
}

static iree_status_t CommandBufferProfileSinkWrite(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  CommandBufferProfileSink* test_sink = CommandBufferProfileSinkCast(sink);
  EXPECT_EQ(1, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  if (test_sink->end_count != 0) test_sink->write_after_end = true;
  EXPECT_EQ(test_sink->session_id, metadata->session_id);
  if (test_sink->fail_write_remaining != 0 &&
      iree_string_view_equal(metadata->content_type,
                             test_sink->fail_write_content_type)) {
    --test_sink->fail_write_remaining;
    return iree_make_status(test_sink->fail_write_status_code,
                            "injected profile sink write failure");
  }
  const bool is_truncated =
      iree_any_bit_set(metadata->flags, IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
  if (is_truncated) {
    ++test_sink->truncated_chunk_count;
    test_sink->dropped_record_count += metadata->dropped_record_count;
  }
  if (iovec_count == 0) {
    if (iree_string_view_equal(metadata->content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
      test_sink->queue_event_dropped_record_count +=
          metadata->dropped_record_count;
      ++test_sink->queue_event_count;
    } else if (iree_string_view_equal(
                   metadata->content_type,
                   IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
      test_sink->memory_event_dropped_record_count +=
          metadata->dropped_record_count;
      ++test_sink->memory_event_count;
    }
    return iree_ok_status();
  }
  if (iovec_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected exactly one profile chunk iovec");
  }

  if (iree_string_view_equal(metadata->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
    ++test_sink->device_metadata_count;
  } else if (iree_string_view_equal(metadata->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    ++test_sink->queue_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
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
      EXPECT_NE(0u, records[i].flags &
                        IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH);
      EXPECT_NE(
          0u, records[i].code_object_hash[0] | records[i].code_object_hash[1]);
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
      EXPECT_NE(0u, record.flags &
                        IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH);
      EXPECT_NE(0u, record.pipeline_hash[0] | record.pipeline_hash[1]);
      EXPECT_EQ(record.name_length,
                record.record_length - (uint32_t)sizeof(record));
      test_sink->executable_export_ids.push_back(record.executable_id);
      payload_offset += record.record_length;
    }
    ++test_sink->executable_export_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_command_buffer_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_command_buffer_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length /
        sizeof(iree_hal_profile_command_buffer_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_command_buffer_record_t),
                records[i].record_length);
      EXPECT_NE(0u, records[i].command_buffer_id);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      test_sink->command_buffer_ids.push_back(records[i].command_buffer_id);
    }
    ++test_sink->command_buffer_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_command_operation_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_command_operation_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length /
        sizeof(iree_hal_profile_command_operation_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_command_operation_record_t),
                records[i].record_length);
      EXPECT_NE(IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_NONE, records[i].type);
      EXPECT_NE(UINT32_MAX, records[i].command_index);
      EXPECT_NE(0u, records[i].command_buffer_id);
      EXPECT_TRUE(
          iree_hal_profile_command_operation_has_block_structure(&records[i]));
      EXPECT_NE(UINT32_MAX, records[i].block_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].block_command_ordinal);
      if (records[i].type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH) {
        EXPECT_NE(0u, records[i].executable_id);
        EXPECT_NE(UINT32_MAX, records[i].export_ordinal);
        EXPECT_NE(0u, records[i].binding_count);
        EXPECT_NE(0u, records[i].workgroup_size[0]);
      }
    }
    test_sink->command_operations.insert(test_sink->command_operations.end(),
                                         records, records + record_count);
    ++test_sink->command_operation_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
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
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS)) {
    iree_host_size_t payload_offset = 0;
    while (payload_offset < iovecs[0].data_length) {
      if (iovecs[0].data_length - payload_offset <
          sizeof(iree_hal_profile_counter_set_record_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "truncated counter set profile record");
      }
      iree_hal_profile_counter_set_record_t record;
      memcpy(&record, iovecs[0].data + payload_offset, sizeof(record));
      if (record.record_length < sizeof(record) ||
          record.record_length > iovecs[0].data_length - payload_offset) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid counter set profile record");
      }
      EXPECT_NE(0u, record.counter_set_id);
      EXPECT_GT(record.counter_count, 0u);
      EXPECT_GT(record.sample_value_count, 0u);
      EXPECT_EQ(record.name_length,
                record.record_length - (uint32_t)sizeof(record));
      test_sink->counter_set_records.push_back(record);
      payload_offset += record.record_length;
    }
    ++test_sink->counter_set_metadata_count;
  } else if (iree_string_view_equal(metadata->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS)) {
    iree_host_size_t payload_offset = 0;
    while (payload_offset < iovecs[0].data_length) {
      if (iovecs[0].data_length - payload_offset <
          sizeof(iree_hal_profile_counter_record_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "truncated counter profile record");
      }
      iree_hal_profile_counter_record_t record;
      memcpy(&record, iovecs[0].data + payload_offset, sizeof(record));
      if (record.record_length < sizeof(record) ||
          record.record_length > iovecs[0].data_length - payload_offset) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid counter profile record");
      }
      EXPECT_NE(0u, record.counter_set_id);
      EXPECT_GT(record.sample_value_count, 0u);
      const uint32_t string_length = record.block_name_length +
                                     record.name_length +
                                     record.description_length;
      EXPECT_EQ(string_length, record.record_length - (uint32_t)sizeof(record));
      test_sink->counter_records.push_back(record);
      payload_offset += record.record_length;
    }
    ++test_sink->counter_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
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
    test_sink->dispatch_events.insert(test_sink->dispatch_events.end(), records,
                                      records + record_count);
    test_sink->dispatch_event_physical_device_ordinals.insert(
        test_sink->dispatch_event_physical_device_ordinals.end(), record_count,
        metadata->physical_device_ordinal);
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
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
      EXPECT_NE(0, records[i].host_time_ns);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
    }
    test_sink->queue_events.insert(test_sink->queue_events.end(), records,
                                   records + record_count);
    test_sink->queue_event_dropped_record_count +=
        metadata->dropped_record_count;
    ++test_sink->queue_event_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
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
      EXPECT_NE(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE, records[i].type);
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0u, records[i].submission_id);
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
                 IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    EXPECT_EQ(0u,
              iovecs[0].data_length % sizeof(iree_hal_profile_memory_event_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_memory_event_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_memory_event_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_memory_event_t),
                records[i].record_length);
      EXPECT_NE(IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_NONE, records[i].type);
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0, records[i].host_time_ns);
      EXPECT_NE(0u, records[i].allocation_id);
    }
    test_sink->memory_events.insert(test_sink->memory_events.end(), records,
                                    records + record_count);
    test_sink->memory_event_dropped_record_count +=
        metadata->dropped_record_count;
    ++test_sink->memory_event_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS)) {
    EXPECT_NE(UINT32_MAX, metadata->physical_device_ordinal);
    EXPECT_NE(UINT32_MAX, metadata->queue_ordinal);
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
      EXPECT_NE(IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_NONE, records[i].type);
      EXPECT_NE(0u, records[i].relationship_id);
      EXPECT_NE(IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_NONE,
                records[i].source_type);
      EXPECT_NE(IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_NONE,
                records[i].target_type);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
      EXPECT_NE(0u, records[i].source_id);
      EXPECT_NE(0u, records[i].target_id);
    }
    test_sink->event_relationships.insert(test_sink->event_relationships.end(),
                                          records, records + record_count);
    ++test_sink->relationship_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
    iree_host_size_t payload_offset = 0;
    while (payload_offset < iovecs[0].data_length) {
      if (iovecs[0].data_length - payload_offset <
          sizeof(iree_hal_profile_counter_sample_record_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "truncated counter sample profile record");
      }
      iree_hal_profile_counter_sample_record_t record;
      memcpy(&record, iovecs[0].data + payload_offset, sizeof(record));
      if (record.record_length < sizeof(record) ||
          record.record_length > iovecs[0].data_length - payload_offset) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid counter sample profile record");
      }
      EXPECT_NE(0u, record.sample_id);
      EXPECT_NE(0u, record.counter_set_id);
      EXPECT_NE(0u, record.dispatch_event_id);
      EXPECT_GT(record.sample_value_count, 0u);
      EXPECT_EQ(record.record_length,
                sizeof(record) +
                    record.sample_value_count * (uint32_t)sizeof(uint64_t));
      const auto* values = reinterpret_cast<const uint64_t*>(
          iovecs[0].data + payload_offset + sizeof(record));
      test_sink->counter_sample_values.insert(
          test_sink->counter_sample_values.end(), values,
          values + record.sample_value_count);
      test_sink->counter_samples.push_back(record);
      payload_offset += record.record_length;
    }
    ++test_sink->counter_sample_count;
  }

  return iree_ok_status();
}

static iree_status_t CommandBufferProfileSinkEndSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  CommandBufferProfileSink* test_sink = CommandBufferProfileSinkCast(sink);
  EXPECT_EQ(1, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  EXPECT_TRUE(iree_string_view_equal(metadata->content_type,
                                     IREE_HAL_PROFILE_CONTENT_TYPE_SESSION));
  EXPECT_EQ(test_sink->session_id, metadata->session_id);
  EXPECT_EQ(test_sink->expected_end_session_status_code, session_status_code);
  test_sink->observed_end_session_status_code = session_status_code;
  test_sink->end_count = 1;
  if (test_sink->fail_end_session_status_code != IREE_STATUS_OK) {
    return iree_make_status(test_sink->fail_end_session_status_code,
                            "injected profile sink end_session failure");
  }
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t kCommandBufferProfileSinkVTable = {
    /*.destroy=*/CommandBufferProfileSinkDestroy,
    /*.begin_session=*/CommandBufferProfileSinkBeginSession,
    /*.write=*/CommandBufferProfileSinkWrite,
    /*.end_session=*/CommandBufferProfileSinkEndSession,
};

static void CommandBufferProfileSinkInitialize(CommandBufferProfileSink* sink) {
  iree_hal_resource_initialize(&kCommandBufferProfileSinkVTable,
                               &sink->resource);
}

static iree_hal_profile_sink_t* CommandBufferProfileSinkAsBase(
    CommandBufferProfileSink* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

static void ExpectQueueEventProfilingCanBeginAndEnd(
    TestLogicalDevice* test_device) {
  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device->base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
}

static void ExpectDispatchEventsWithinClockCorrelationRange(
    const CommandBufferProfileSink& sink) {
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

static const iree_hal_profile_command_operation_record_t* FindCommandOperation(
    const CommandBufferProfileSink& sink, uint64_t command_buffer_id,
    uint32_t command_index) {
  for (const auto& operation : sink.command_operations) {
    if (operation.command_buffer_id == command_buffer_id &&
        operation.command_index == command_index) {
      return &operation;
    }
  }
  return nullptr;
}

static const iree_hal_profile_event_relationship_record_t*
FindEventRelationship(const CommandBufferProfileSink& sink,
                      iree_hal_profile_event_relationship_type_t type,
                      iree_hal_profile_event_endpoint_type_t source_type,
                      uint64_t source_id,
                      iree_hal_profile_event_endpoint_type_t target_type,
                      uint64_t target_id) {
  for (const auto& relationship : sink.event_relationships) {
    if (relationship.type == type && relationship.source_type == source_type &&
        relationship.source_id == source_id &&
        relationship.target_type == target_type &&
        relationship.target_id == target_id) {
      return &relationship;
    }
  }
  return nullptr;
}

static const iree_hal_profile_queue_event_t* FindUniqueQueueEvent(
    const CommandBufferProfileSink& sink,
    iree_hal_profile_queue_event_type_t type) {
  const iree_hal_profile_queue_event_t* result = nullptr;
  for (const auto& event : sink.queue_events) {
    if (event.type != type) continue;
    EXPECT_EQ(nullptr, result);
    result = &event;
  }
  return result;
}

static iree_host_size_t CountQueueEvents(
    const CommandBufferProfileSink& sink,
    iree_hal_profile_queue_event_type_t type) {
  iree_host_size_t count = 0;
  for (const auto& event : sink.queue_events) {
    if (event.type == type) ++count;
  }
  return count;
}

static uint32_t SumQueueEventOperationCounts(
    const CommandBufferProfileSink& sink,
    iree_hal_profile_queue_event_type_t type) {
  uint32_t operation_count = 0;
  for (const auto& event : sink.queue_events) {
    if (event.type == type) {
      operation_count += event.operation_count;
    }
  }
  return operation_count;
}

static bool IsProfilingUnsupported(iree_status_t status) {
  return iree_status_is_unimplemented(status) ||
         iree_status_is_invalid_argument(status);
}

static bool IsHardwareCounterProfilingUnavailable(iree_status_t status) {
  return IsProfilingUnsupported(status) || iree_status_is_not_found(status) ||
         iree_status_is_failed_precondition(status);
}

static bool IsQueueDeviceProfilingUnavailable(iree_status_t status) {
  return IsProfilingUnsupported(status) ||
         iree_status_is_failed_precondition(status);
}

static iree_status_t BeginHardwareCounterProfiling(
    DeviceProfilingScope* profiling, CommandBufferProfileSink* sink,
    iree_host_size_t counter_name_count, iree_string_view_t* counter_names) {
  iree_hal_profile_counter_set_selection_t counter_set = {
      /*.flags=*/IREE_HAL_PROFILE_COUNTER_SET_SELECTION_FLAG_NONE,
      /*.name=*/IREE_SV("smoke"),
      /*.counter_name_count=*/counter_name_count,
      /*.counter_names=*/counter_names,
  };
  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES;
  profiling_options.sink = CommandBufferProfileSinkAsBase(sink);
  profiling_options.counter_set_count = 1;
  profiling_options.counter_sets = &counter_set;
  return profiling->Begin(&profiling_options);
}

static iree_status_t BeginSqWavesProfiling(DeviceProfilingScope* profiling,
                                           CommandBufferProfileSink* sink) {
  iree_string_view_t counter_names[] = {
      IREE_SV("SQ_WAVES"),
  };
  return BeginHardwareCounterProfiling(
      profiling, sink, IREE_ARRAYSIZE(counter_names), counter_names);
}

static iree_status_t BeginSqWaveWidthProfiling(DeviceProfilingScope* profiling,
                                               CommandBufferProfileSink* sink) {
  iree_string_view_t counter_names[] = {
      IREE_SV("SQ_WAVES"),
      IREE_SV("SQ_WAVES_32"),
      IREE_SV("SQ_WAVES_64"),
      IREE_SV("SQ_BUSY_CYCLES"),
  };
  return BeginHardwareCounterProfiling(
      profiling, sink, IREE_ARRAYSIZE(counter_names), counter_names);
}

TEST_F(HostQueueCommandBufferTest,
       ExplicitHardwareCounterSelectionEmitsMetadataWhenAvailable) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = BeginSqWavesProfiling(&profiling, &sink);
  if (IsQueueDeviceProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "AMDGPU hardware counter profiling unavailable";
  }
  IREE_ASSERT_OK(profiling_status);
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.counter_set_metadata_count);
  EXPECT_EQ(1, sink.counter_metadata_count);
}

TEST_F(HostQueueCommandBufferTest,
       MultipleHardwareCounterSelectionEmitsLayoutWhenAvailable) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = BeginSqWaveWidthProfiling(&profiling, &sink);
  if (IsHardwareCounterProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "AMDGPU hardware counter profiling unavailable";
  }
  IREE_ASSERT_OK(profiling_status);
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.counter_set_metadata_count);
  EXPECT_EQ(1, sink.counter_metadata_count);
  ASSERT_FALSE(sink.counter_set_records.empty());
  ASSERT_EQ(sink.counter_set_records.size() * 4u, sink.counter_records.size());
  const iree_hal_profile_counter_unit_t expected_units[] = {
      IREE_HAL_PROFILE_COUNTER_UNIT_COUNT,
      IREE_HAL_PROFILE_COUNTER_UNIT_COUNT,
      IREE_HAL_PROFILE_COUNTER_UNIT_COUNT,
      IREE_HAL_PROFILE_COUNTER_UNIT_CYCLES,
  };
  iree_host_size_t counter_record_index = 0;
  for (const auto& counter_set_record : sink.counter_set_records) {
    ASSERT_EQ(4u, counter_set_record.counter_count);
    uint32_t sample_value_count = 0;
    for (uint32_t i = 0; i < counter_set_record.counter_count; ++i) {
      const auto& counter_record = sink.counter_records[counter_record_index++];
      EXPECT_EQ(counter_set_record.counter_set_id,
                counter_record.counter_set_id);
      EXPECT_EQ(sample_value_count, counter_record.sample_value_offset);
      EXPECT_GT(counter_record.sample_value_count, 0u);
      EXPECT_EQ(expected_units[i], counter_record.unit);
      sample_value_count += counter_record.sample_value_count;
    }
    EXPECT_EQ(sample_value_count, counter_set_record.sample_value_count);
  }
}

static iree_status_t AppendConstantsBindingsDispatch(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, iree_hal_buffer_ref_list_t bindings) {
  const uint32_t constant_values[2] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_values, sizeof(constant_values));
  return iree_hal_command_buffer_dispatch(
      command_buffer, executable, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE);
}

struct TwoDispatchCommandBuffer {
  ~TwoDispatchCommandBuffer() {
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
  }

  // Executable cache owning |executable|.
  iree_hal_executable_cache_t* executable_cache = NULL;

  // CTS executable containing the constants+bindings dispatch entry point.
  iree_hal_executable_t* executable = NULL;

  // Host-visible input buffer shared by both dispatches.
  Ref<iree_hal_buffer_t> input_buffer;

  // Host-visible output buffer written by command index 0.
  Ref<iree_hal_buffer_t> output_buffer0;

  // Host-visible output buffer written by command index 1.
  Ref<iree_hal_buffer_t> output_buffer1;

  // Command buffer containing two equivalent dispatch operations.
  Ref<iree_hal_command_buffer_t> command_buffer;
};

static iree_status_t CreateTwoDispatchCommandBuffer(
    TestLogicalDevice* test_device, TwoDispatchCommandBuffer* out_fixture) {
  IREE_RETURN_IF_ERROR(LoadCtsExecutable(
      test_device->base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &out_fixture->executable_cache, &out_fixture->executable));

  IREE_RETURN_IF_ERROR(CreateHostVisibleDispatchBuffer(
      test_device->allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      out_fixture->input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_write(out_fixture->input_buffer, /*target_offset=*/0,
                                input_values, sizeof(input_values)));

  IREE_RETURN_IF_ERROR(CreateHostVisibleDispatchBuffer(
      test_device->allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      out_fixture->output_buffer0.out()));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_zero(
      out_fixture->output_buffer0, /*offset=*/0, IREE_HAL_WHOLE_BUFFER));

  IREE_RETURN_IF_ERROR(CreateHostVisibleDispatchBuffer(
      test_device->allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      out_fixture->output_buffer1.out()));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_zero(
      out_fixture->output_buffer1, /*offset=*/0, IREE_HAL_WHOLE_BUFFER));

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      test_device->base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, out_fixture->command_buffer.out()));
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_begin(out_fixture->command_buffer));
  iree_hal_buffer_ref_t binding_refs0[2] = {
      iree_hal_make_buffer_ref(
          out_fixture->input_buffer, /*offset=*/0,
          iree_hal_buffer_byte_length(out_fixture->input_buffer)),
      iree_hal_make_buffer_ref(
          out_fixture->output_buffer0, /*offset=*/0,
          iree_hal_buffer_byte_length(out_fixture->output_buffer0)),
  };
  const iree_hal_buffer_ref_list_t bindings0 = {
      /*count=*/IREE_ARRAYSIZE(binding_refs0),
      /*values=*/binding_refs0,
  };
  IREE_RETURN_IF_ERROR(AppendConstantsBindingsDispatch(
      out_fixture->command_buffer, out_fixture->executable, bindings0));
  iree_hal_buffer_ref_t binding_refs1[2] = {
      iree_hal_make_buffer_ref(
          out_fixture->input_buffer, /*offset=*/0,
          iree_hal_buffer_byte_length(out_fixture->input_buffer)),
      iree_hal_make_buffer_ref(
          out_fixture->output_buffer1, /*offset=*/0,
          iree_hal_buffer_byte_length(out_fixture->output_buffer1)),
  };
  const iree_hal_buffer_ref_list_t bindings1 = {
      /*count=*/IREE_ARRAYSIZE(binding_refs1),
      /*values=*/binding_refs1,
  };
  IREE_RETURN_IF_ERROR(AppendConstantsBindingsDispatch(
      out_fixture->command_buffer, out_fixture->executable, bindings1));
  return iree_hal_command_buffer_end(out_fixture->command_buffer);
}

static void ExpectTwoDispatchOutputs(const TwoDispatchCommandBuffer& fixture) {
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  uint32_t output_values0[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(fixture.output_buffer0, /*offset=*/0,
                                          output_values0,
                                          sizeof(output_values0)));
  EXPECT_EQ(0,
            memcmp(output_values0, expected_values, sizeof(expected_values)));
  uint32_t output_values1[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(fixture.output_buffer1, /*offset=*/0,
                                          output_values1,
                                          sizeof(output_values1)));
  EXPECT_EQ(0,
            memcmp(output_values1, expected_values, sizeof(expected_values)));
}

#if !defined(NDEBUG)
static bool AqlHeaderHasBarrier(uint16_t header) {
  return ((header >> IREE_HSA_PACKET_HEADER_BARRIER) &
          ((1u << IREE_HSA_PACKET_HEADER_WIDTH_BARRIER) - 1u)) != 0;
}

TEST_F(HostQueueCommandBufferTest,
       PacketSummaryOmitsInteriorBarriersWithoutExecutionBarrier) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  Ref<iree_hal_buffer_t> output_buffer0;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer0.out()));
  Ref<iree_hal_buffer_t> output_buffer1;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer1.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t binding_refs0[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer0, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer0)),
  };
  const iree_hal_buffer_ref_list_t bindings0 = {
      /*count=*/IREE_ARRAYSIZE(binding_refs0),
      /*values=*/binding_refs0,
  };
  IREE_ASSERT_OK(
      AppendConstantsBindingsDispatch(command_buffer, executable, bindings0));
  iree_hal_buffer_ref_t binding_refs1[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer1, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer1)),
  };
  const iree_hal_buffer_ref_list_t bindings1 = {
      /*count=*/IREE_ARRAYSIZE(binding_refs1),
      /*values=*/binding_refs1,
  };
  IREE_ASSERT_OK(
      AppendConstantsBindingsDispatch(command_buffer, executable, bindings1));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);
  ASSERT_EQ(program->first_block->aql_packet_count, 2u);

  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t summary = {0};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_host_queue_summarize_command_buffer_block_packets(
          queue, &resolution, iree_hal_semaphore_list_empty(),
          program->first_block, &summary));
  EXPECT_EQ(summary.packet_count, 2u);
  EXPECT_EQ(summary.barrier_packet_count, 1u);
  EXPECT_FALSE(AqlHeaderHasBarrier(summary.first_packet_header));
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.last_packet_header));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest,
       PacketSummaryBarriersIndirectParameterPatchDispatch) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_multi_workgroup_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));
  Ref<iree_hal_buffer_t> parameter_buffer;
  IREE_ASSERT_OK(CreateHostVisibleIndirectParameterBuffer(
      test_device.allocator(), /*buffer_size=*/3 * sizeof(uint32_t),
      parameter_buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  iree_hal_dispatch_config_t config =
      iree_hal_make_static_dispatch_config(4, 1, 1);
  config.workgroup_count_ref = iree_hal_make_buffer_ref(
      parameter_buffer, /*offset=*/0, 3 * sizeof(uint32_t));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, /*entry_point=*/0, config,
      iree_const_byte_span_empty(), bindings,
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);
  ASSERT_EQ(program->first_block->aql_packet_count, 2u);

  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t summary = {0};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_host_queue_summarize_command_buffer_block_packets(
          queue, &resolution, iree_hal_semaphore_list_empty(),
          program->first_block, &summary));
  EXPECT_EQ(summary.packet_count, 2u);
  EXPECT_EQ(summary.barrier_packet_count, 2u);
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.first_packet_header));
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.last_packet_header));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest,
       PacketSummaryHonorsExplicitExecutionBarrier) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  Ref<iree_hal_buffer_t> output_buffer0;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer0.out()));
  Ref<iree_hal_buffer_t> output_buffer1;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer1.out()));
  Ref<iree_hal_buffer_t> output_buffer2;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer2.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t binding_refs0[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer0, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer0)),
  };
  const iree_hal_buffer_ref_list_t bindings0 = {
      /*count=*/IREE_ARRAYSIZE(binding_refs0),
      /*values=*/binding_refs0,
  };
  IREE_ASSERT_OK(
      AppendConstantsBindingsDispatch(command_buffer, executable, bindings0));
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  iree_hal_buffer_ref_t binding_refs1[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer1, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer1)),
  };
  const iree_hal_buffer_ref_list_t bindings1 = {
      /*count=*/IREE_ARRAYSIZE(binding_refs1),
      /*values=*/binding_refs1,
  };
  IREE_ASSERT_OK(
      AppendConstantsBindingsDispatch(command_buffer, executable, bindings1));
  iree_hal_buffer_ref_t binding_refs2[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer2, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer2)),
  };
  const iree_hal_buffer_ref_list_t bindings2 = {
      /*count=*/IREE_ARRAYSIZE(binding_refs2),
      /*values=*/binding_refs2,
  };
  IREE_ASSERT_OK(
      AppendConstantsBindingsDispatch(command_buffer, executable, bindings2));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);
  ASSERT_EQ(program->first_block->aql_packet_count, 3u);

  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t summary = {0};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_host_queue_summarize_command_buffer_block_packets(
          queue, &resolution, iree_hal_semaphore_list_empty(),
          program->first_block, &summary));
  EXPECT_EQ(summary.packet_count, 3u);
  EXPECT_EQ(summary.barrier_packet_count, 2u);
  EXPECT_FALSE(AqlHeaderHasBarrier(summary.first_packet_header));
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.last_packet_header));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest,
       PacketSummaryAppliesSystemAcquireOnlyToFirstDynamicKernargPacket) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_indirect_buffer_ref(/*binding=*/0, /*offset=*/0,
                                        IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_indirect_buffer_ref(/*binding=*/1, /*offset=*/0,
                                        IREE_HAL_WHOLE_BUFFER),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  IREE_ASSERT_OK(
      AppendConstantsBindingsDispatch(command_buffer, executable, bindings));
  IREE_ASSERT_OK(
      AppendConstantsBindingsDispatch(command_buffer, executable, bindings));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);
  ASSERT_EQ(program->first_block->aql_packet_count, 2u);
  const iree_hal_amdgpu_command_buffer_command_header_t* first_command =
      iree_hal_amdgpu_command_buffer_block_commands_const(program->first_block);
  ASSERT_EQ(first_command->opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command =
      (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)first_command;
  EXPECT_NE(dispatch_command->kernarg_strategy,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED);

  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  resolution.inline_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
  iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t summary = {0};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_host_queue_summarize_command_buffer_block_packets(
          queue, &resolution, iree_hal_semaphore_list_empty(),
          program->first_block, &summary));
  EXPECT_EQ(summary.packet_count, 2u);
  EXPECT_EQ(summary.barrier_packet_count, 1u);
  EXPECT_EQ(summary.system_acquire_packet_count, 1u);
  EXPECT_EQ(summary.system_release_packet_count, 0u);

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest,
       PacketSummaryLargeDispatchProgramOmitsInteriorBarriers) {
  static constexpr uint32_t kDispatchCount = 1000;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  for (uint32_t i = 0; i < kDispatchCount; ++i) {
    IREE_ASSERT_OK(
        AppendConstantsBindingsDispatch(command_buffer, executable, bindings));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);

  uint32_t payload_block_count = 0;
  uint32_t packet_count = 0;
  uint32_t barrier_packet_count = 0;
  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  const iree_hal_amdgpu_command_buffer_block_header_t* block =
      program->first_block;
  while (block) {
    iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t summary = {0};
    IREE_ASSERT_OK(
        iree_hal_amdgpu_host_queue_summarize_command_buffer_block_packets(
            queue, &resolution, iree_hal_semaphore_list_empty(), block,
            &summary));
    if (summary.packet_count > 0) {
      ++payload_block_count;
      packet_count += summary.packet_count;
      barrier_packet_count += summary.barrier_packet_count;
    }
    block = iree_hal_amdgpu_aql_program_block_next(program->block_pool, block);
  }

  EXPECT_EQ(packet_count, kDispatchCount);
  EXPECT_EQ(barrier_packet_count, payload_block_count);
  EXPECT_LT(barrier_packet_count, packet_count);

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}
#endif  // !defined(NDEBUG)

TEST_F(HostQueueCommandBufferTest, DirectDispatchUsesPrepublishedKernargs) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  const uint32_t constant_values[2] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_values, sizeof(constant_values));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);
  EXPECT_EQ(program->max_block_kernarg_length, 0u);
  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(program->first_block);
  ASSERT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command =
      (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)command;
  EXPECT_EQ(dispatch_command->kernarg_strategy,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED);
  const uint32_t kernarg_length =
      (uint32_t)dispatch_command->kernarg_length_qwords * 8u;
  EXPECT_NE(
      iree_hal_amdgpu_aql_command_buffer_prepublished_kernarg(
          command_buffer, dispatch_command->payload_reference, kernarg_length),
      nullptr);

  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  const iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_command_buffer_t> one_shot_command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, one_shot_command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(one_shot_command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      one_shot_command_buffer, executable, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(one_shot_command_buffer));

  const iree_hal_amdgpu_aql_program_t* one_shot_program =
      iree_hal_amdgpu_aql_command_buffer_program(one_shot_command_buffer);
  ASSERT_NE(one_shot_program->first_block, nullptr);
  EXPECT_GT(one_shot_program->max_block_kernarg_length, 0u);
  const iree_hal_amdgpu_command_buffer_command_header_t* one_shot_command =
      iree_hal_amdgpu_command_buffer_block_commands_const(
          one_shot_program->first_block);
  ASSERT_EQ(one_shot_command->opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  const iree_hal_amdgpu_command_buffer_dispatch_command_t*
      one_shot_dispatch_command =
          (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
              one_shot_command;
  EXPECT_EQ(one_shot_dispatch_command->kernarg_strategy,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL);
  EXPECT_EQ(
      iree_hal_amdgpu_aql_command_buffer_prepublished_kernarg(
          one_shot_command_buffer, one_shot_dispatch_command->payload_reference,
          (uint32_t)one_shot_dispatch_command->kernarg_length_qwords * 8u),
      nullptr);

  command_buffer_signal_value = 2;
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      one_shot_command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  memset(output_values, 0, sizeof(output_values));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest, SinklessProfilingBeginFails) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_device_profiling_begin(
                            test_device.base_device(), &profiling_options));
  IREE_EXPECT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_EXPECT_OK(iree_hal_device_profiling_end(test_device.base_device()));
}

TEST_F(HostQueueCommandBufferTest, ProfilingBeginSinkBeginFailureAllowsRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  sink.fail_begin_session_status_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink)));
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(0, sink.end_count);

  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);
}

TEST_F(HostQueueCommandBufferTest,
       ProfilingBeginMetadataWriteFailureEndsSessionAndAllowsRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_DATA_LOSS;
  sink.expected_end_session_status_code = IREE_STATUS_DATA_LOSS;
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink)));
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, sink.observed_end_session_status_code);
  EXPECT_EQ(0, sink.device_metadata_count);

  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);
}

TEST_F(HostQueueCommandBufferTest,
       ProfilingFlushWriteFailurePreservesQueueEventsForRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  IREE_ASSERT_OK(SubmitProfiledQueueFill(&test_device));
  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_device_profiling_flush(test_device.base_device()));
  EXPECT_EQ(0, sink.queue_event_count);
  EXPECT_TRUE(sink.queue_events.empty());

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.queue_event_count);
  ASSERT_EQ(1u, sink.queue_events.size());
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL, sink.queue_events[0].type);
}

TEST_F(HostQueueCommandBufferTest,
       ProfiledQueueEventsReportDroppedRecordsWhenRingFull) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  const iree_host_size_t event_capacity =
      test_device.logical_device()->profiling.queue_event_capacity;
  ASSERT_GT(event_capacity, 0u);
  for (iree_host_size_t i = 0; i <= event_capacity; ++i) {
    iree_hal_profile_queue_event_t event =
        iree_hal_profile_queue_event_default();
    event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL;
    event.physical_device_ordinal = 0;
    event.queue_ordinal = 0;
    event.operation_count = 1;
    iree_hal_amdgpu_logical_device_record_profile_queue_event(
        test_device.base_device(), &event);
  }

  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.queue_event_count);
  EXPECT_EQ(event_capacity, sink.queue_events.size());
  EXPECT_EQ(1u, sink.queue_event_dropped_record_count);
  EXPECT_EQ(1u, sink.dropped_record_count);
  EXPECT_EQ(1, sink.truncated_chunk_count);
}

TEST_F(HostQueueCommandBufferTest,
       ProfiledMemoryEventsReportDroppedRecordsWhenRingFull) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  const iree_host_size_t event_capacity =
      test_device.logical_device()->profiling.memory_event_capacity;
  ASSERT_GT(event_capacity, 0u);
  iree_host_size_t recorded_count = 0;
  for (iree_host_size_t i = 0; i <= event_capacity; ++i) {
    iree_hal_profile_memory_event_t event =
        iree_hal_profile_memory_event_default();
    event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE;
    event.allocation_id = i + 1;
    event.physical_device_ordinal = 0;
    event.queue_ordinal = 0;
    event.length = sizeof(uint32_t);
    if (iree_hal_amdgpu_logical_device_record_profile_memory_event(
            test_device.base_device(), &event)) {
      ++recorded_count;
    }
  }

  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(event_capacity, recorded_count);
  EXPECT_EQ(1, sink.memory_event_count);
  EXPECT_EQ(event_capacity, sink.memory_events.size());
  EXPECT_EQ(1u, sink.memory_event_dropped_record_count);
  EXPECT_EQ(1u, sink.dropped_record_count);
  EXPECT_EQ(1, sink.truncated_chunk_count);
}

TEST_F(HostQueueCommandBufferTest,
       ProfilingFlushMetadataWriteFailurePreservesCursorForRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  const iree_hal_semaphore_list_t signal_list = {
      /*count=*/1,
      /*semaphores=*/&signal_ptr,
      /*payload_values=*/&signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal, signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_device_profiling_flush(test_device.base_device()));
  EXPECT_EQ(0, sink.command_buffer_metadata_count);
  EXPECT_TRUE(sink.command_buffer_ids.empty());

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  ASSERT_EQ(1u, sink.command_buffer_ids.size());
}

TEST_F(HostQueueCommandBufferTest,
       ProfilingEndWriteFailureReportsSessionStatusAndAllowsRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  IREE_ASSERT_OK(SubmitProfiledQueueFill(&test_device));
  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_DATA_LOSS;
  sink.expected_end_session_status_code = IREE_STATUS_DATA_LOSS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, profiling.End());
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, sink.observed_end_session_status_code);
  EXPECT_EQ(0, sink.queue_event_count);

  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);
}

TEST_F(HostQueueCommandBufferTest,
       ProfilingEndSessionFailureClearsStateAndAllowsRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  sink.fail_end_session_status_code = IREE_STATUS_ABORTED;
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, profiling.End());
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(IREE_STATUS_OK, sink.observed_end_session_status_code);

  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);
}

TEST_F(HostQueueCommandBufferTest, CommandBufferDispatchesEmitProfileEvents) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  const iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      fixture.command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_EQ(1, sink.executable_metadata_count);
  EXPECT_EQ(1, sink.executable_export_metadata_count);
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  EXPECT_EQ(1, sink.command_operation_metadata_count);
  EXPECT_GE(sink.clock_correlation_count, 2);
  EXPECT_FALSE(sink.write_after_end);
  ASSERT_EQ(3u, sink.command_operations.size());
  uint32_t dispatch_operation_count = 0;
  uint32_t return_operation_count = 0;
  for (const auto& operation : sink.command_operations) {
    EXPECT_EQ(sink.command_buffer_ids[0], operation.command_buffer_id);
    switch (operation.type) {
      case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH:
        EXPECT_TRUE(
            iree_hal_profile_command_operation_has_block_structure(&operation));
        EXPECT_EQ(dispatch_operation_count, operation.command_index);
        EXPECT_NE(0u, operation.executable_id);
        EXPECT_NE(
            sink.executable_ids.end(),
            std::find(sink.executable_ids.begin(), sink.executable_ids.end(),
                      operation.executable_id));
        EXPECT_EQ(0u, operation.export_ordinal);
        EXPECT_EQ(2u, operation.binding_count);
        EXPECT_EQ(1u, operation.workgroup_count[0]);
        EXPECT_EQ(1u, operation.workgroup_count[1]);
        EXPECT_EQ(1u, operation.workgroup_count[2]);
        EXPECT_NE(0u, operation.workgroup_size[0]);
        EXPECT_NE(0u,
                  operation.flags &
                      IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS);
        ++dispatch_operation_count;
        break;
      case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN:
        EXPECT_TRUE(
            iree_hal_profile_command_operation_has_block_structure(&operation));
        EXPECT_EQ(2u, operation.command_index);
        EXPECT_NE(0u, operation.flags &
                          IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_CONTROL_FLOW);
        ++return_operation_count;
        break;
      default:
        FAIL() << "unexpected command operation type " << operation.type;
    }
  }
  EXPECT_EQ(2u, dispatch_operation_count);
  EXPECT_EQ(1u, return_operation_count);
  ASSERT_EQ(2u, sink.dispatch_events.size());
  for (iree_host_size_t i = 0; i < sink.dispatch_events.size(); ++i) {
    const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[i];
    EXPECT_EQ(sizeof(iree_hal_profile_dispatch_event_t), event.record_length);
    EXPECT_EQ(IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER, event.flags);
    EXPECT_NE(0u, event.event_id);
    EXPECT_NE(0u, event.submission_id);
    EXPECT_NE(0u, event.command_buffer_id);
    EXPECT_NE(
        sink.command_buffer_ids.end(),
        std::find(sink.command_buffer_ids.begin(),
                  sink.command_buffer_ids.end(), event.command_buffer_id));
    EXPECT_NE(0u, event.executable_id);
    EXPECT_NE(sink.executable_ids.end(),
              std::find(sink.executable_ids.begin(), sink.executable_ids.end(),
                        event.executable_id));
    EXPECT_NE(sink.executable_export_ids.end(),
              std::find(sink.executable_export_ids.begin(),
                        sink.executable_export_ids.end(), event.executable_id));
    EXPECT_EQ((uint32_t)i, event.command_index);
    const iree_hal_profile_command_operation_record_t* operation =
        FindCommandOperation(sink, event.command_buffer_id,
                             event.command_index);
    ASSERT_NE(nullptr, operation);
    EXPECT_EQ(IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH,
              operation->type);
    EXPECT_EQ(event.executable_id, operation->executable_id);
    EXPECT_EQ(event.export_ordinal, operation->export_ordinal);
    EXPECT_EQ(0u, event.export_ordinal);
    EXPECT_EQ(1u, event.workgroup_count[0]);
    EXPECT_EQ(1u, event.workgroup_count[1]);
    EXPECT_EQ(1u, event.workgroup_count[2]);
    EXPECT_NE(0u, event.workgroup_size[0]);
    EXPECT_NE(0u, event.start_tick);
    EXPECT_NE(0u, event.end_tick);
    EXPECT_GE(event.end_tick, event.start_tick);
  }
  ExpectDispatchEventsWithinClockCorrelationRange(sink);
}

TEST_F(HostQueueCommandBufferTest,
       CommandBufferExecuteEmitsQueueDeviceSpansAndRelationships) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
                          IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsQueueDeviceProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "queue-device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  const iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      fixture.command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_FALSE(sink.write_after_end);
  EXPECT_EQ(1, sink.queue_event_count);
  EXPECT_EQ(1, sink.queue_device_event_count);
  EXPECT_EQ(1, sink.relationship_count);
  ASSERT_EQ(1u, sink.queue_events.size());
  ASSERT_EQ(1u, sink.queue_device_events.size());
  ASSERT_EQ(2u, sink.dispatch_events.size());
  ASSERT_EQ(3u, sink.event_relationships.size());

  const iree_hal_profile_queue_event_t& queue_event = sink.queue_events[0];
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, queue_event.type);
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE,
            queue_event.dependency_strategy);
  EXPECT_NE(0u, queue_event.submission_id);
  EXPECT_NE(0u, queue_event.command_buffer_id);
  EXPECT_EQ(0u, queue_event.wait_count);
  EXPECT_EQ(1u, queue_event.signal_count);
  EXPECT_EQ(0u, queue_event.barrier_count);
  EXPECT_EQ(3u, queue_event.operation_count);
  EXPECT_NE(
      sink.command_buffer_ids.end(),
      std::find(sink.command_buffer_ids.begin(), sink.command_buffer_ids.end(),
                queue_event.command_buffer_id));

  const iree_hal_profile_queue_device_event_t& device_event =
      sink.queue_device_events[0];
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, device_event.type);
  EXPECT_EQ(queue_event.submission_id, device_event.submission_id);
  EXPECT_EQ(queue_event.command_buffer_id, device_event.command_buffer_id);
  EXPECT_EQ(queue_event.stream_id, device_event.stream_id);
  EXPECT_EQ(queue_event.physical_device_ordinal,
            device_event.physical_device_ordinal);
  EXPECT_EQ(queue_event.queue_ordinal, device_event.queue_ordinal);
  EXPECT_EQ(queue_event.operation_count, device_event.operation_count);

  for (const auto& dispatch_event : sink.dispatch_events) {
    EXPECT_EQ(queue_event.submission_id, dispatch_event.submission_id);
    EXPECT_EQ(queue_event.command_buffer_id, dispatch_event.command_buffer_id);
    EXPECT_NE(
        nullptr,
        FindEventRelationship(
            sink,
            IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_DISPATCH,
            IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_SUBMISSION,
            dispatch_event.submission_id,
            IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_DISPATCH_EVENT,
            dispatch_event.event_id));
  }
  EXPECT_NE(
      nullptr,
      FindEventRelationship(
          sink,
          IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_QUEUE_DEVICE_EVENT,
          IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_SUBMISSION,
          device_event.submission_id,
          IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_DEVICE_EVENT,
          device_event.event_id));
  ExpectDispatchEventsWithinClockCorrelationRange(sink);
}

TEST_F(HostQueueCommandBufferTest,
       CommandBufferDispatchesEmitHardwareCounterSamples) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = BeginSqWavesProfiling(&profiling, &sink);
  if (IsHardwareCounterProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "AMDGPU hardware counter profiling unavailable";
  }
  IREE_ASSERT_OK(profiling_status);

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  const iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      fixture.command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.counter_set_metadata_count);
  EXPECT_EQ(1, sink.counter_metadata_count);
  EXPECT_GE(sink.counter_sample_count, 1);
  ASSERT_EQ(2u, sink.dispatch_events.size());
  ASSERT_EQ(sink.dispatch_events.size(), sink.counter_samples.size());
  iree_host_size_t sample_value_count = 0;
  for (iree_host_size_t i = 0; i < sink.counter_samples.size(); ++i) {
    const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[i];
    const iree_hal_profile_counter_sample_record_t& sample =
        sink.counter_samples[i];
    EXPECT_TRUE(iree_all_bits_set(
        sample.flags,
        IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DISPATCH_EVENT |
            IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_COMMAND_OPERATION));
    EXPECT_EQ(sample.dispatch_event_id, event.event_id);
    EXPECT_EQ(sample.submission_id, event.submission_id);
    EXPECT_EQ(sample.command_buffer_id, event.command_buffer_id);
    EXPECT_EQ(sample.executable_id, event.executable_id);
    EXPECT_EQ(sample.command_index, event.command_index);
    EXPECT_EQ(sample.export_ordinal, event.export_ordinal);
    sample_value_count += sample.sample_value_count;
  }
  ASSERT_EQ(sample_value_count, sink.counter_sample_values.size());
  EXPECT_NE(sink.counter_sample_values.end(),
            std::find_if(sink.counter_sample_values.begin(),
                         sink.counter_sample_values.end(),
                         [](uint64_t value) { return value != 0; }));
}

TEST_F(HostQueueCommandBufferTest,
       ProfilingFlushCounterSampleWriteFailurePreservesSamplesForRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = BeginSqWavesProfiling(&profiling, &sink);
  if (IsHardwareCounterProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "AMDGPU hardware counter profiling unavailable";
  }
  IREE_ASSERT_OK(profiling_status);

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  const iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      fixture.command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_device_profiling_flush(test_device.base_device()));
  EXPECT_EQ(0, sink.counter_sample_count);
  EXPECT_TRUE(sink.counter_samples.empty());

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);
  EXPECT_GE(sink.counter_sample_count, 1);
  ASSERT_EQ(2u, sink.counter_samples.size());
  EXPECT_GE(sink.dispatch_events.size(), sink.counter_samples.size());
}

TEST_F(HostQueueCommandBufferTest,
       CommandBufferDispatchProfileFilterSelectsCommandIndex) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  profiling_options.sink = CommandBufferProfileSinkAsBase(&sink);
  profiling_options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX;
  profiling_options.capture_filter.command_index = 1;
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = profiling.Begin(&profiling_options);
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  const iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      fixture.command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  ASSERT_EQ(1u, sink.dispatch_events.size());
  const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[0];
  EXPECT_EQ(IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER, event.flags);
  EXPECT_NE(0u, event.event_id);
  EXPECT_NE(0u, event.submission_id);
  EXPECT_NE(0u, event.command_buffer_id);
  EXPECT_EQ(1u, event.command_index);
  EXPECT_NE(0u, event.start_tick);
  EXPECT_NE(0u, event.end_tick);
  EXPECT_GE(event.end_tick, event.start_tick);
}

TEST_F(HostQueueCommandBufferTest,
       DispatchProfileFilterCopiesExecutableExportPattern) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  std::string export_pattern = "scale_*";
  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  profiling_options.sink = CommandBufferProfileSinkAsBase(&sink);
  profiling_options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN;
  profiling_options.capture_filter.executable_export_pattern =
      iree_make_string_view(export_pattern.data(), export_pattern.size());
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = profiling.Begin(&profiling_options);
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  export_pattern.assign("nomatch");

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  const uint64_t executable_id =
      iree_hal_amdgpu_executable_profile_id(executable);
  EXPECT_TRUE(iree_hal_amdgpu_logical_device_should_profile_dispatch(
      test_device.logical_device(), executable_id, /*export_ordinal=*/0,
      /*command_buffer_id=*/0, /*command_index=*/0,
      /*physical_device_ordinal=*/0, /*queue_ordinal=*/0));

  IREE_ASSERT_OK(profiling.End());
  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest,
       ProfiledDispatchReservationFailsWhenRingFull) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(nullptr, queue);

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation = {0};
  iree_hal_amdgpu_profile_dispatch_event_reservation_t exhausted_reservation = {
      0};
  const uint32_t dispatch_event_capacity =
      iree_hal_amdgpu_host_queue_profile_dispatch_event_capacity(queue);
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_status_t status =
      iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
          queue, dispatch_event_capacity, &reservation);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
        queue, 1, &exhausted_reservation);
  }
  iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue, reservation);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_EQ(dispatch_event_capacity, reservation.event_count);
  EXPECT_EQ(0u, exhausted_reservation.event_count);

  IREE_ASSERT_OK(profiling.End());
}

TEST_F(HostQueueCommandBufferTest,
       ProfiledQueueDeviceReservationFailsWhenRingFull) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(nullptr, queue);

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsQueueDeviceProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "queue-device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  iree_hal_amdgpu_profile_queue_device_event_reservation_t reservation = {0};
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      exhausted_reservation = {0};
  const uint32_t queue_device_event_capacity =
      queue->profiling.queue_device_event_capacity;
  ASSERT_GT(queue_device_event_capacity, 0u);
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_status_t status =
      iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
          queue, queue_device_event_capacity, &reservation);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
        queue, 1, &exhausted_reservation);
  }
  iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(queue,
                                                                reservation);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_EQ(queue_device_event_capacity, reservation.event_count);
  EXPECT_EQ(0u, exhausted_reservation.event_count);

  IREE_ASSERT_OK(profiling.End());
}

TEST_F(HostQueueCommandBufferTest,
       ProfiledCommandBufferDispatchSignalsSurviveAqlSlotReuse) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kDispatchCount = kAqlCapacity + 32;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.kernarg_capacity = 2 * kAqlCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> input_buffer;
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  for (uint32_t i = 0; i < kDispatchCount; ++i) {
    IREE_ASSERT_OK(
        AppendConstantsBindingsDispatch(command_buffer, executable, bindings));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  const iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  const uint32_t expected_values[4] = {13, 16, 19, 22};
  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_FALSE(sink.write_after_end);
  ASSERT_EQ(kDispatchCount, sink.dispatch_events.size());
  for (iree_host_size_t i = 0; i < sink.dispatch_events.size(); ++i) {
    const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[i];
    EXPECT_EQ(sizeof(iree_hal_profile_dispatch_event_t), event.record_length);
    EXPECT_EQ(IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER, event.flags);
    EXPECT_NE(0u, event.event_id);
    EXPECT_NE(0u, event.submission_id);
    EXPECT_NE(0u, event.start_tick);
    EXPECT_NE(0u, event.end_tick);
    EXPECT_GE(event.end_tick, event.start_tick);
  }

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest,
       SingleBlockCommandBufferParksAndResumesUnderNotificationPressure) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kNotificationCapacity = 1;
  static constexpr uint32_t kKernargCapacity = 2 * kAqlCapacity;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.notification_capacity = kNotificationCapacity;
  options.host_queues.kernarg_capacity = kKernargCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(target_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t expected = 0xBD3A0001u;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*binding=*/0, /*offset=*/0,
                                        sizeof(expected)),
      &expected, sizeof(expected), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_EQ(program->block_count, 1u);
  ASSERT_GT(program->max_block_aql_packet_count, 0u);

  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(queue, blocker_signal));

  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_ptr = pressure_signal.get();
  iree_hal_semaphore_list_t pressure_signal_list = {
      /*count=*/1,
      /*semaphores=*/&pressure_signal_ptr,
      /*payload_values=*/&pressure_signal_value,
  };
  const uint32_t pressure_pattern = 0xABCD1234u;
  iree_status_t status = iree_hal_device_queue_fill(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), pressure_signal_list, pressure_buffer,
      /*target_offset=*/0, sizeof(pressure_pattern), &pressure_pattern,
      sizeof(pressure_pattern), IREE_HAL_FILL_FLAG_NONE);

  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  iree_hal_buffer_binding_t binding = {
      /*buffer=*/target_buffer.get(),
      /*offset=*/0,
      /*length=*/IREE_HAL_WHOLE_BUFFER,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*count=*/1,
      /*bindings=*/&binding,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), command_buffer_signal_list,
        command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  const bool replay_parked =
      iree_status_is_ok(status) && HostQueueHasPostDrainAction(queue);

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(
        command_buffer_signal, command_buffer_signal_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);
  EXPECT_TRUE(replay_parked);

  uint32_t actual = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_buffer, /*offset=*/0, &actual,
                                          sizeof(actual)));
  EXPECT_EQ(actual, expected);
}

TEST_F(HostQueueCommandBufferTest,
       MetadataOnlyCommandBufferParksAndResumesUnderNotificationPressure) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kNotificationCapacity = 1;
  static constexpr uint32_t kKernargCapacity = 2 * kAqlCapacity;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.notification_capacity = kNotificationCapacity;
  options.host_queues.kernarg_capacity = kKernargCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_EQ(program->max_block_aql_packet_count, 0u);

  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(queue, blocker_signal));

  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_ptr = pressure_signal.get();
  iree_hal_semaphore_list_t pressure_signal_list = {
      /*count=*/1,
      /*semaphores=*/&pressure_signal_ptr,
      /*payload_values=*/&pressure_signal_value,
  };
  const uint32_t pressure_pattern = 0xABCD1234u;
  iree_status_t status = iree_hal_device_queue_fill(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), pressure_signal_list, pressure_buffer,
      /*target_offset=*/0, sizeof(pressure_pattern), &pressure_pattern,
      sizeof(pressure_pattern), IREE_HAL_FILL_FLAG_NONE);

  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), command_buffer_signal_list,
        command_buffer, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
  }
  const bool replay_parked =
      iree_status_is_ok(status) && HostQueueHasPostDrainAction(queue);

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(
        command_buffer_signal, command_buffer_signal_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_TRUE(replay_parked);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  EXPECT_EQ(1, sink.queue_event_count);
  ASSERT_EQ(1u, sink.command_buffer_ids.size());
  EXPECT_EQ(1u, CountQueueEvents(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL));
  EXPECT_EQ(1u,
            CountQueueEvents(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE));
  const iree_hal_profile_queue_event_t* execute_event =
      FindUniqueQueueEvent(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE);
  ASSERT_NE(nullptr, execute_event);
  EXPECT_EQ(sink.command_buffer_ids[0], execute_event->command_buffer_id);
  EXPECT_EQ(0u, execute_event->operation_count);
  EXPECT_EQ(1u, execute_event->signal_count);
  EXPECT_LE(sink.command_operations.size(), 1u);
  for (const auto& operation : sink.command_operations) {
    EXPECT_EQ(sink.command_buffer_ids[0], operation.command_buffer_id);
    EXPECT_EQ(IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN, operation.type);
  }
}

TEST_F(HostQueueCommandBufferTest,
       DeferredTransientBindingSurvivesQueuedDealloca) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kNotificationCapacity = 1;
  static constexpr uint32_t kKernargCapacity = 2 * kAqlCapacity;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.notification_capacity = kNotificationCapacity;
  options.host_queues.kernarg_capacity = kKernargCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_semaphore_t> alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca_signal.out()));
  uint64_t alloca_signal_value = 1;
  iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
  iree_hal_semaphore_list_t alloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&alloca_signal_ptr,
      /*payload_values=*/&alloca_signal_value,
  };
  iree_hal_buffer_t* transient_raw = NULL;
  IREE_ASSERT_OK(QueueTransientTransferBuffer(
      test_device.base_device(), alloca_signal_list, sizeof(uint32_t),
      &transient_raw));
  Ref<iree_hal_buffer_t> transient_buffer(transient_raw);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca_signal, alloca_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t expected = 0xBD3A0002u;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*binding=*/0, /*offset=*/0,
                                        sizeof(expected)),
      &expected, sizeof(expected), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*binding=*/0, /*offset=*/0,
                                        sizeof(expected)),
      iree_hal_make_indirect_buffer_ref(/*binding=*/1, /*offset=*/0,
                                        sizeof(expected)),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  Ref<iree_hal_semaphore_t> dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dealloca_signal.out()));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(queue, blocker_signal));

  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_ptr = pressure_signal.get();
  iree_hal_semaphore_list_t pressure_signal_list = {
      /*count=*/1,
      /*semaphores=*/&pressure_signal_ptr,
      /*payload_values=*/&pressure_signal_value,
  };
  const uint32_t pressure_pattern = 0xABCD1234u;
  iree_status_t status = iree_hal_device_queue_fill(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), pressure_signal_list, pressure_buffer,
      /*target_offset=*/0, sizeof(pressure_pattern), &pressure_pattern,
      sizeof(pressure_pattern), IREE_HAL_FILL_FLAG_NONE);

  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  iree_hal_buffer_binding_t bindings[2] = {
      {
          /*buffer=*/transient_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/output_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*count=*/IREE_ARRAYSIZE(bindings),
      /*bindings=*/bindings,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), command_buffer_signal_list,
        command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  const bool replay_parked =
      iree_status_is_ok(status) && HostQueueHasPostDrainAction(queue);

  uint64_t dealloca_signal_value = 1;
  iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
  iree_hal_semaphore_list_t dealloca_wait_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  iree_hal_semaphore_list_t dealloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&dealloca_signal_ptr,
      /*payload_values=*/&dealloca_signal_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        dealloca_wait_list, dealloca_signal_list, transient_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE);
  }

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(dealloca_signal, dealloca_signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);
  EXPECT_TRUE(replay_parked);

  uint32_t actual = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer, /*offset=*/0, &actual,
                                          sizeof(actual)));
  EXPECT_EQ(actual, expected);
}

TEST_F(HostQueueCommandBufferTest,
       LargeCommandBufferParksAndResumesUnderNotificationPressure) {
  static constexpr uint32_t kFillCount = 2048;
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kNotificationCapacity = 1;
  static constexpr uint32_t kKernargCapacity = 2 * kAqlCapacity;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.notification_capacity = kNotificationCapacity;
  options.host_queues.kernarg_capacity = kKernargCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  const iree_device_size_t target_buffer_size = kFillCount * sizeof(uint32_t);
  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), target_buffer_size, target_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(target_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  std::vector<uint32_t> expected(kFillCount);
  for (uint32_t i = 0; i < kFillCount; ++i) {
    expected[i] = 0xBD3A0000u | i;
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer,
        iree_hal_make_indirect_buffer_ref(/*binding=*/0, i * sizeof(uint32_t),
                                          sizeof(uint32_t)),
        &expected[i], sizeof(expected[i]), IREE_HAL_FILL_FLAG_NONE));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_GT(program->block_count, 1u);
  ASSERT_GT(kFillCount, kAqlCapacity);
  ASSERT_LE(program->max_block_aql_packet_count, kAqlCapacity);

  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(queue, blocker_signal));

  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_ptr = pressure_signal.get();
  iree_hal_semaphore_list_t pressure_signal_list = {
      /*count=*/1,
      /*semaphores=*/&pressure_signal_ptr,
      /*payload_values=*/&pressure_signal_value,
  };
  const uint32_t pressure_pattern = 0xABCD1234u;
  iree_status_t status = iree_hal_device_queue_fill(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), pressure_signal_list, pressure_buffer,
      /*target_offset=*/0, sizeof(pressure_pattern), &pressure_pattern,
      sizeof(pressure_pattern), IREE_HAL_FILL_FLAG_NONE);

  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  iree_hal_buffer_binding_t binding = {
      /*buffer=*/target_buffer.get(),
      /*offset=*/0,
      /*length=*/IREE_HAL_WHOLE_BUFFER,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*count=*/1,
      /*bindings=*/&binding,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), command_buffer_signal_list,
        command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  const bool replay_parked =
      iree_status_is_ok(status) && HostQueueHasPostDrainAction(queue);

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(
        command_buffer_signal, command_buffer_signal_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_TRUE(replay_parked);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  EXPECT_EQ(1, sink.queue_event_count);
  ASSERT_EQ(1u, sink.command_buffer_ids.size());
  EXPECT_EQ(1u, CountQueueEvents(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL));
  EXPECT_EQ(program->block_count,
            CountQueueEvents(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE));
  EXPECT_EQ(program->command_count,
            SumQueueEventOperationCounts(
                sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE));
  uint32_t execute_signal_count = 0;
  for (const auto& event : sink.queue_events) {
    if (event.type != IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE) continue;
    EXPECT_EQ(sink.command_buffer_ids[0], event.command_buffer_id);
    execute_signal_count += event.signal_count;
  }
  EXPECT_EQ(1u, execute_signal_count);

  std::vector<uint32_t> actual(kFillCount);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_buffer, /*offset=*/0,
                                          actual.data(), target_buffer_size));
  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace iree::hal::amdgpu
