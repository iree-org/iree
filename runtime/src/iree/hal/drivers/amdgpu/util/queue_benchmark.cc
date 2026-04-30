// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/registration/driver_module.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/benchmark_flags.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/io/file_contents.h"
#include "runtime/src/iree/hal/drivers/amdgpu/cts/testdata_amdgpu.h"
#include "runtime/src/iree/hal/drivers/amdgpu/util/testdata_amdgpu_queue_benchmark.h"

IREE_FLAG(
    string, binding_count_executable_file, "",
    "Optional raw HSACO or AMDGPU executable file used for binding-count "
    "dispatch benchmark rows. When empty, the embedded benchmark executable "
    "is used.");
IREE_FLAG(
    int32_t, binding_count_workgroup_size_x, 0,
    "Optional 1D workgroup-size override for binding-count dispatch benchmark "
    "rows. Use 64 to match common HIP launch benchmark shapes. The default 0 "
    "uses the executable export's reflected workgroup size.");

namespace {

constexpr int64_t kBatchCount = 20;
constexpr uint32_t kFrontierAxisTableCapacity = 256;
constexpr iree_device_size_t kPayloadBufferAlignment = 16;
constexpr iree_device_size_t kPayloadLength = sizeof(uint32_t);
constexpr iree_host_size_t kDispatchBindingBenchmarkMaxCount = 256;
constexpr iree_host_size_t kDispatchBindingBenchmarkVariantCapacity =
    kDispatchBindingBenchmarkMaxCount + 1;
constexpr int64_t kProfileGuardrailBindingCount = 1;
constexpr int64_t kProfileGuardrailIterations = 200;
constexpr int64_t kProfileGuardrailOperationCount = 20;
constexpr iree_hal_queue_affinity_t kQueue0 = ((iree_hal_queue_affinity_t)1ull)
                                              << 0;
constexpr iree_hal_queue_affinity_t kQueue1 = ((iree_hal_queue_affinity_t)1ull)
                                              << 1;

enum class PayloadKind {
  kCopy,
  kDispatch,
  kFill,
  kNoopDispatch,
  kPreResolvedDispatch,
};

enum class ProfileGuardrailMode : int64_t {
  kDisabled = 0,
  kQueueDeviceEvents = 1,
  kDispatchEvents = 2,
  kQueueDeviceAndDispatchEvents = 3,
};

enum class QueueAllocaTlsfGrowthMode : int64_t {
  kWarm = 0,
  kForcedGrowth = 1,
};

constexpr iree_hal_buffer_params_t kQueueAllocaBufferParams = {
    /*usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER |
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
    /*access=*/IREE_HAL_MEMORY_ACCESS_ALL,
    /*type=*/IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
    /*queue_affinity=*/0,
    /*min_alignment=*/0,
};

iree_hal_device_profiling_data_families_t ProfileGuardrailDataFamilies(
    ProfileGuardrailMode mode) {
  switch (mode) {
    case ProfileGuardrailMode::kDisabled:
      return IREE_HAL_DEVICE_PROFILING_DATA_NONE;
    case ProfileGuardrailMode::kQueueDeviceEvents:
      return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
             IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS;
    case ProfileGuardrailMode::kDispatchEvents:
      return IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
    case ProfileGuardrailMode::kQueueDeviceAndDispatchEvents:
      return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
             IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
             IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  }
  return IREE_HAL_DEVICE_PROFILING_DATA_NONE;
}

typedef struct QueueBenchmarkDiscardProfileSink {
  // Resource header for iree_hal_profile_sink_t lifetime management.
  iree_hal_resource_t resource;
  // Host allocator used for sink lifetime.
  iree_allocator_t host_allocator;
  // Number of profile chunks observed by the sink.
  uint64_t write_count;
  // Total bytes observed across all profile chunk iovecs.
  uint64_t payload_byte_count;
} QueueBenchmarkDiscardProfileSink;

static void QueueBenchmarkDiscardProfileSinkDestroy(
    iree_hal_profile_sink_t* base_sink) {
  QueueBenchmarkDiscardProfileSink* sink =
      (QueueBenchmarkDiscardProfileSink*)base_sink;
  iree_allocator_t host_allocator = sink->host_allocator;
  iree_allocator_free(host_allocator, sink);
}

static iree_status_t QueueBenchmarkDiscardProfileSinkBeginSession(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  (void)base_sink;
  (void)metadata;
  return iree_ok_status();
}

static iree_status_t QueueBenchmarkDiscardProfileSinkWrite(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  (void)metadata;
  QueueBenchmarkDiscardProfileSink* sink =
      (QueueBenchmarkDiscardProfileSink*)base_sink;
  ++sink->write_count;
  for (iree_host_size_t i = 0; i < iovec_count; ++i) {
    sink->payload_byte_count += iovecs[i].data_length;
  }
  return iree_ok_status();
}

static iree_status_t QueueBenchmarkDiscardProfileSinkEndSession(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)base_sink;
  (void)metadata;
  (void)session_status_code;
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t
    kQueueBenchmarkDiscardProfileSinkVtable = {
        .destroy = QueueBenchmarkDiscardProfileSinkDestroy,
        .begin_session = QueueBenchmarkDiscardProfileSinkBeginSession,
        .write = QueueBenchmarkDiscardProfileSinkWrite,
        .end_session = QueueBenchmarkDiscardProfileSinkEndSession,
};

iree_status_t QueueBenchmarkDiscardProfileSinkCreate(
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink) {
  *out_sink = nullptr;
  QueueBenchmarkDiscardProfileSink* sink = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*sink), (void**)&sink));
  iree_hal_resource_initialize(&kQueueBenchmarkDiscardProfileSinkVtable,
                               &sink->resource);
  sink->host_allocator = host_allocator;
  sink->write_count = 0;
  sink->payload_byte_count = 0;
  *out_sink = (iree_hal_profile_sink_t*)sink;
  return iree_ok_status();
}

class QueueBenchmark : public benchmark::Fixture {
 public:
  static void InitializeOnce() {
    if (initialized_) return;
    initialized_ = true;
    host_allocator_ = iree_allocator_system();

    iree_status_t status = iree_hal_amdgpu_driver_module_register(
        iree_hal_driver_registry_default());
    if (iree_status_is_already_exists(status)) {
      iree_status_free(status);
      status = iree_ok_status();
    }

    if (iree_status_is_ok(status)) {
      status = iree_hal_driver_registry_try_create(
          iree_hal_driver_registry_default(), iree_make_cstring_view("amdgpu"),
          host_allocator_, &driver_);
    }

    iree_async_proactor_pool_t* proactor_pool = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_async_proactor_pool_create(
          iree_numa_node_count(), /*node_ids=*/nullptr,
          iree_async_proactor_pool_options_default(), host_allocator_,
          &proactor_pool);
    }

    if (iree_status_is_ok(status)) {
      iree_hal_device_create_params_t create_params =
          iree_hal_device_create_params_default();
      create_params.proactor_pool = proactor_pool;
      status = iree_hal_driver_create_default_device(driver_, &create_params,
                                                     host_allocator_, &device_);
    }
    iree_async_proactor_pool_release(proactor_pool);

    iree_async_frontier_tracker_t* frontier_tracker = nullptr;
    if (iree_status_is_ok(status)) {
      iree_async_frontier_tracker_options_t options =
          iree_async_frontier_tracker_options_default();
      options.axis_table_capacity = kFrontierAxisTableCapacity;
      status = iree_async_frontier_tracker_create(options, host_allocator_,
                                                  &frontier_tracker);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          device_, frontier_tracker, host_allocator_, &device_group_);
    }
    iree_async_frontier_tracker_release(frontier_tracker);

    if (iree_status_is_ok(status)) {
      available_ = true;
      return;
    }

    iree_status_fprint(stderr, status);
    iree_status_free(status);
    iree_hal_device_release(device_);
    iree_hal_driver_release(driver_);
    device_ = nullptr;
    driver_ = nullptr;
  }

  static void DeinitializeOnce() {
    if (!initialized_) return;
    iree_hal_executable_release(binding_count_executable_);
    iree_hal_executable_cache_release(binding_count_executable_cache_);
    iree_io_file_contents_free(binding_count_executable_file_contents_);
    iree_hal_executable_release(dispatch_executable_);
    iree_hal_executable_cache_release(dispatch_executable_cache_);
    iree_hal_device_release(device_);
    iree_hal_device_group_release(device_group_);
    iree_hal_driver_release(driver_);
    binding_count_executable_ = nullptr;
    binding_count_executable_cache_ = nullptr;
    binding_count_executable_file_contents_ = nullptr;
    dispatch_executable_ = nullptr;
    dispatch_executable_cache_ = nullptr;
    device_ = nullptr;
    device_group_ = nullptr;
    driver_ = nullptr;
    available_ = false;
  }

  void SetUp(benchmark::State& state) override {
    InitializeOnce();
    if (!available_) {
      state.SkipWithError("AMDGPU HAL device not available");
      return;
    }

    if (!CreatePublicSemaphore(state, &completion_semaphore_) ||
        !CreatePrivateStreamSemaphore(state, &stream_semaphore_) ||
        !CreatePrivateStreamSemaphore(state, &producer_semaphore_)) {
      return;
    }
  }

  void TearDown(benchmark::State& state) override {
    if (profile_session_active_) {
      EndProfileSession(state,
                        "profiling end failed during benchmark teardown");
    }
    iree_hal_profile_sink_release(profile_sink_);
    profile_sink_ = nullptr;
    ReleasePreResolvedDispatch();
    for (iree_host_size_t i = 0; i < kDispatchBindingBenchmarkVariantCapacity;
         ++i) {
      iree_hal_command_buffer_release(binding_count_command_buffers_[i]);
      binding_count_command_buffers_[i] = nullptr;
    }
    for (iree_host_size_t i = 0; i < kDispatchBindingBenchmarkMaxCount; ++i) {
      iree_hal_buffer_release(binding_count_buffers_[i]);
      binding_count_buffers_[i] = nullptr;
    }
    iree_hal_buffer_release(source_buffer_);
    iree_hal_buffer_release(target_buffer_);
    iree_hal_semaphore_release(completion_semaphore_);
    iree_hal_semaphore_release(stream_semaphore_);
    iree_hal_semaphore_release(producer_semaphore_);
    source_buffer_ = nullptr;
    target_buffer_ = nullptr;
    completion_semaphore_ = nullptr;
    stream_semaphore_ = nullptr;
    producer_semaphore_ = nullptr;
    completion_payload_value_ = 0;
    stream_payload_value_ = 0;
    producer_payload_value_ = 0;
  }

 protected:
  struct SubmittedCompletion {
    iree_hal_semaphore_t* semaphore;
    uint64_t payload_value;
  };

  static iree_hal_queue_affinity_t CrossQueuePingPongFinalQueue(
      int64_t handoff_count) {
    return (handoff_count & 1) ? kQueue1 : kQueue0;
  }

  bool EnsureQueueAvailable(benchmark::State& state,
                            iree_hal_queue_affinity_t queue_affinity) {
    if (queue_affinity == kQueue1) {
      // Cross-queue rows are same-physical-agent measurements; skip instead
      // of silently turning them into cross-device/system-scope rows.
      return EnsurePrivateStreamQueuePair(state);
    }
    return HandleStatus(state,
                        iree_hal_device_queue_flush(device_, queue_affinity),
                        "queue affinity not available");
  }

  bool BeginProfileSession(benchmark::State& state, ProfileGuardrailMode mode) {
    const iree_hal_device_profiling_data_families_t data_families =
        ProfileGuardrailDataFamilies(mode);
    if (data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) return true;

    if (!HandleStatus(state,
                      QueueBenchmarkDiscardProfileSinkCreate(host_allocator_,
                                                             &profile_sink_),
                      "failed to create profile benchmark sink")) {
      return false;
    }
    iree_hal_device_profiling_options_t options = {0};
    options.data_families = data_families;
    options.sink = profile_sink_;
    if (!HandleStatus(state, iree_hal_device_profiling_begin(device_, &options),
                      "failed to begin profile benchmark session")) {
      iree_hal_profile_sink_release(profile_sink_);
      profile_sink_ = nullptr;
      return false;
    }
    profile_session_active_ = true;
    return true;
  }

  void EndProfileSession(benchmark::State& state, const char* message) {
    if (!profile_session_active_) return;
    profile_session_active_ = false;
    HandleStatus(state, iree_hal_device_profiling_end(device_), message);
  }

  bool FlushProfileSession(benchmark::State& state, const char* message) {
    if (!profile_session_active_) return true;
    return HandleStatus(state, iree_hal_device_profiling_flush(device_),
                        message);
  }

  bool FlushProfileSessionWithTimingPaused(benchmark::State& state,
                                           const char* message) {
    if (!profile_session_active_) return true;
    state.PauseTiming();
    const bool result = FlushProfileSession(state, message);
    state.ResumeTiming();
    return result;
  }

  bool WaitAndFlushProfileSessionWithTimingPaused(
      benchmark::State& state, const SubmittedCompletion& completion,
      const char* wait_message, const char* flush_message) {
    state.PauseTiming();
    iree_status_t status = Wait(completion.semaphore, completion.payload_value);
    bool result = HandleStatus(state, status, wait_message);
    if (result) {
      result = FlushProfileSession(state, flush_message);
    }
    state.ResumeTiming();
    return result;
  }

  template <typename RunWaitFn>
  void RunProfileGuardrailFinalWaitBenchmark(
      benchmark::State& state, int64_t queue_submissions_per_sync,
      int64_t profiled_operations_per_sync, const char* flush_message,
      const char* end_message, RunWaitFn run_wait) {
    const ProfileGuardrailMode profile_mode =
        static_cast<ProfileGuardrailMode>(state.range(0));
    if (!BeginProfileSession(state, profile_mode)) return;
    for (auto _ : state) {
      if (!run_wait()) break;
      if (!FlushProfileSessionWithTimingPaused(state, flush_message)) break;
    }
    SetProfileGuardrailCounters(state, profile_mode, queue_submissions_per_sync,
                                profiled_operations_per_sync);
    EndProfileSession(state, end_message);
  }

  template <typename SubmitFn>
  void RunProfileGuardrailSubmitOnlyBenchmark(
      benchmark::State& state, int64_t queue_submissions_per_sync,
      int64_t profiled_operations_per_sync, const char* wait_message,
      const char* flush_message, const char* end_message, SubmitFn submit) {
    const ProfileGuardrailMode profile_mode =
        static_cast<ProfileGuardrailMode>(state.range(0));
    if (!BeginProfileSession(state, profile_mode)) return;
    for (auto _ : state) {
      SubmittedCompletion completion;
      if (!submit(&completion)) break;
      if (!WaitAndFlushProfileSessionWithTimingPaused(
              state, completion, wait_message, flush_message)) {
        break;
      }
    }
    SetProfileGuardrailCounters(state, profile_mode, queue_submissions_per_sync,
                                profiled_operations_per_sync);
    EndProfileSession(state, end_message);
  }

  iree_status_t LookupHostQueue(iree_hal_queue_affinity_t queue_affinity,
                                iree_hal_amdgpu_host_queue_t** out_host_queue) {
    *out_host_queue = nullptr;
    auto* logical_device =
        reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(device_);
    const iree_hal_amdgpu_queue_affinity_domain_t domain = {
        .supported_affinity = logical_device->queue_affinity_mask,
        .physical_device_count = logical_device->physical_device_count,
        .queue_count_per_physical_device =
            logical_device->system->topology.gpu_agent_queue_count,
    };
    iree_hal_amdgpu_queue_affinity_resolved_t resolved;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_resolve(
        domain, queue_affinity, &resolved));
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[resolved.physical_device_ordinal];
    if (IREE_UNLIKELY(resolved.physical_queue_ordinal >=
                      physical_device->host_queue_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "queue ordinal has no initialized host queue");
    }

    *out_host_queue =
        &physical_device->host_queues[resolved.physical_queue_ordinal];
    return iree_ok_status();
  }

  iree_status_t LookupHostQueueByAxis(
      iree_async_axis_t axis, iree_hal_amdgpu_host_queue_t** out_host_queue) {
    *out_host_queue = nullptr;
    if (IREE_UNLIKELY(iree_async_axis_domain(axis) !=
                      IREE_ASYNC_CAUSAL_DOMAIN_QUEUE)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "producer axis is not a queue axis");
    }

    auto* logical_device =
        reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(device_);
    const uint8_t device_index = iree_async_axis_device_index(axis);
    if (IREE_UNLIKELY(device_index >= logical_device->physical_device_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "producer axis has no physical device");
    }
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[device_index];
    const uint8_t queue_index = iree_async_axis_queue_index(axis);
    if (IREE_UNLIKELY(queue_index >= physical_device->host_queue_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "producer axis has no initialized host queue");
    }

    *out_host_queue = &physical_device->host_queues[queue_index];
    return iree_ok_status();
  }

  bool EnsurePrivateStreamQueuePair(benchmark::State& state) {
    iree_hal_amdgpu_host_queue_t* queue0 = nullptr;
    iree_hal_amdgpu_host_queue_t* queue1 = nullptr;
    if (!HandleStatus(state, LookupHostQueue(kQueue0, &queue0),
                      "queue 0 is not available")) {
      return false;
    }
    if (!HandleStatus(state, LookupHostQueue(kQueue1, &queue1),
                      "queue 1 is not available")) {
      return false;
    }
    if (queue0->device_ordinal != queue1->device_ordinal) {
      state.SkipWithError(
          "queue 0 and queue 1 are not on the same physical AMDGPU agent");
      return false;
    }
    return true;
  }

  iree_status_t WaitForSubmittedProducerEpoch(
      const SubmittedCompletion& completion) {
    if (IREE_UNLIKELY(!iree_hal_amdgpu_semaphore_isa(completion.semaphore))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "epoch completion floor requires an AMDGPU semaphore");
    }

    iree_hal_amdgpu_last_signal_flags_t signal_flags =
        IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
    iree_async_axis_t producer_axis = 0;
    uint64_t producer_epoch = 0;
    uint64_t producer_value = 0;
    if (IREE_UNLIKELY(!iree_hal_amdgpu_last_signal_load(
            iree_hal_amdgpu_semaphore_last_signal(completion.semaphore),
            &signal_flags, &producer_axis, &producer_epoch, &producer_value))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "completion semaphore has no submitted producer epoch");
    }
    if (IREE_UNLIKELY(producer_value < completion.payload_value)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "completion semaphore producer value %" PRIu64
                              " is below target %" PRIu64,
                              producer_value, completion.payload_value);
    }

    iree_hal_amdgpu_host_queue_t* host_queue = nullptr;
    IREE_RETURN_IF_ERROR(LookupHostQueueByAxis(producer_axis, &host_queue));
    hsa_signal_t epoch_signal = iree_hal_amdgpu_notification_ring_epoch_signal(
        &host_queue->notification_ring);
    const hsa_signal_value_t compare_value =
        (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                             producer_epoch + 1);

    auto* logical_device =
        reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(device_);
    uint64_t wait_timeout_hint =
        logical_device->system->info.timestamp_frequency / 1000;
    if (wait_timeout_hint == 0) wait_timeout_hint = 1;

    for (;;) {
      hsa_signal_value_t signal_value = iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(host_queue->libhsa), epoch_signal,
          HSA_SIGNAL_CONDITION_LT, compare_value, wait_timeout_hint,
          HSA_WAIT_STATE_BLOCKED);
      if (signal_value < compare_value) return iree_ok_status();

      iree_status_t queue_error = (iree_status_t)iree_atomic_load(
          &host_queue->error_status, iree_memory_order_acquire);
      if (IREE_UNLIKELY(!iree_status_is_ok(queue_error))) {
        return iree_status_clone(queue_error);
      }
    }
  }

  static iree_status_t ResolveBufferDevicePointer(iree_hal_buffer_t* buffer,
                                                  uint64_t* out_device_ptr) {
    *out_device_ptr = 0;
    iree_hal_buffer_t* allocated_buffer =
        iree_hal_buffer_allocated_buffer(buffer);
    void* device_ptr = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
    if (IREE_UNLIKELY(!device_ptr)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch benchmark buffer must be backed by an AMDGPU allocation");
    }

    const iree_device_size_t device_offset =
        iree_hal_buffer_byte_offset(buffer);
    if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch benchmark buffer offset exceeds host pointer size");
    }
    *out_device_ptr = (uint64_t)((uintptr_t)device_ptr + device_offset);
    return iree_ok_status();
  }

  iree_status_t SubmitBarrierWithLists(
      iree_hal_queue_affinity_t queue_affinity,
      iree_hal_semaphore_list_t wait_semaphore_list,
      iree_hal_semaphore_list_t signal_semaphore_list) {
    return iree_hal_device_queue_barrier(
        device_, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  iree_status_t SubmitPayloadWithLists(
      PayloadKind payload_kind, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_semaphore_list_t wait_semaphore_list,
      iree_hal_semaphore_list_t signal_semaphore_list) {
    if (payload_kind == PayloadKind::kCopy) {
      return iree_hal_device_queue_copy(
          device_, queue_affinity, wait_semaphore_list, signal_semaphore_list,
          source_buffer_, /*source_offset=*/0, target_buffer_,
          /*target_offset=*/0, kPayloadLength, IREE_HAL_COPY_FLAG_NONE);
    }
    if (payload_kind == PayloadKind::kDispatch) {
      const uint32_t constant_data[] = {3, 10};
      iree_const_byte_span_t constants =
          iree_make_const_byte_span(constant_data, sizeof(constant_data));
      iree_hal_buffer_ref_t binding_refs[2] = {
          iree_hal_make_buffer_ref(source_buffer_, /*offset=*/0,
                                   iree_hal_buffer_byte_length(source_buffer_)),
          iree_hal_make_buffer_ref(target_buffer_, /*offset=*/0,
                                   iree_hal_buffer_byte_length(target_buffer_)),
      };
      iree_hal_buffer_ref_list_t bindings = {
          /*count=*/IREE_ARRAYSIZE(binding_refs),
          /*values=*/binding_refs,
      };
      return iree_hal_device_queue_dispatch(
          device_, queue_affinity, wait_semaphore_list, signal_semaphore_list,
          dispatch_executable_, /*export_ordinal=*/0,
          iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
          IREE_HAL_DISPATCH_FLAG_NONE);
    }
    if (payload_kind == PayloadKind::kNoopDispatch) {
      return iree_hal_device_queue_dispatch(
          device_, queue_affinity, wait_semaphore_list, signal_semaphore_list,
          dispatch_executable_, /*export_ordinal=*/0,
          iree_hal_make_static_dispatch_config(0, 0, 0),
          iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(),
          IREE_HAL_DISPATCH_FLAG_NONE);
    }
    if (payload_kind == PayloadKind::kPreResolvedDispatch) {
      return SubmitPreResolvedDispatchWithLists(
          queue_affinity, wait_semaphore_list, signal_semaphore_list);
    }
    return iree_hal_device_queue_fill(
        device_, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        target_buffer_, /*target_offset=*/0, kPayloadLength, &fill_pattern_,
        sizeof(fill_pattern_), IREE_HAL_FILL_FLAG_NONE);
  }

  iree_status_t SubmitBarrier(iree_hal_queue_affinity_t queue_affinity,
                              iree_hal_semaphore_t* wait_semaphore,
                              uint64_t wait_payload_value,
                              iree_hal_semaphore_t* signal_semaphore,
                              uint64_t signal_payload_value) {
    iree_hal_semaphore_t* wait_semaphore_storage = wait_semaphore;
    iree_hal_semaphore_t* signal_semaphore_storage = signal_semaphore;
    iree_hal_semaphore_list_t wait_semaphore_list =
        iree_hal_semaphore_list_empty();
    iree_hal_semaphore_list_t signal_semaphore_list =
        iree_hal_semaphore_list_empty();
    if (wait_semaphore) {
      wait_semaphore_list = {
          /*count=*/1,
          /*semaphores=*/&wait_semaphore_storage,
          /*payload_values=*/&wait_payload_value,
      };
    }
    if (signal_semaphore) {
      signal_semaphore_list = {
          /*count=*/1,
          /*semaphores=*/&signal_semaphore_storage,
          /*payload_values=*/&signal_payload_value,
      };
    }
    return SubmitBarrierWithLists(queue_affinity, wait_semaphore_list,
                                  signal_semaphore_list);
  }

  iree_status_t SubmitBarrierWithWaitList(
      iree_hal_queue_affinity_t queue_affinity,
      iree_hal_semaphore_list_t wait_semaphore_list,
      iree_hal_semaphore_t* signal_semaphore, uint64_t signal_payload_value) {
    iree_hal_semaphore_t* signal_semaphore_storage = signal_semaphore;
    iree_hal_semaphore_list_t signal_semaphore_list =
        iree_hal_semaphore_list_empty();
    if (signal_semaphore) {
      signal_semaphore_list = {
          /*count=*/1,
          /*semaphores=*/&signal_semaphore_storage,
          /*payload_values=*/&signal_payload_value,
      };
    }
    return SubmitBarrierWithLists(queue_affinity, wait_semaphore_list,
                                  signal_semaphore_list);
  }

  iree_status_t SubmitBarrierWithSingleWaitAndSignalList(
      iree_hal_queue_affinity_t queue_affinity,
      iree_hal_semaphore_t* wait_semaphore, uint64_t wait_payload_value,
      iree_hal_semaphore_list_t signal_semaphore_list) {
    iree_hal_semaphore_t* wait_semaphore_storage = wait_semaphore;
    iree_hal_semaphore_list_t wait_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&wait_semaphore_storage,
        /*payload_values=*/&wait_payload_value,
    };
    return SubmitBarrierWithLists(queue_affinity, wait_semaphore_list,
                                  signal_semaphore_list);
  }

  iree_status_t Wait(iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
    return iree_hal_semaphore_wait(
        semaphore, payload_value, iree_infinite_timeout(),
        iree_hal_amdgpu_benchmark_completion_wait_flags());
  }

  iree_status_t CreateQueueAllocaTlsfPool(iree_device_size_t allocation_size,
                                          iree_hal_pool_t** out_pool) {
    *out_pool = nullptr;
    iree_hal_queue_pool_backend_t backend = {0};
    IREE_RETURN_IF_ERROR(
        iree_hal_device_query_queue_pool_backend(device_, kQueue0, &backend));
    if (!backend.slab_provider || !backend.notification) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "queue pool backend query returned an incomplete backend bundle");
    }

    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = allocation_size;
    options.tlsf_options.alignment = kPayloadBufferAlignment;
    options.tlsf_options.initial_block_capacity = 16;
    options.tlsf_options.frontier_capacity = 2;
    return iree_hal_tlsf_pool_create(
        options, backend.slab_provider, backend.notification,
        iree_hal_pool_epoch_query_null(), host_allocator_, out_pool);
  }

  iree_status_t QueueAllocaSubmit(iree_hal_pool_t* pool,
                                  iree_device_size_t allocation_size,
                                  iree_hal_buffer_t** out_buffer,
                                  SubmittedCompletion* out_completion) {
    *out_buffer = nullptr;
    uint64_t signal_payload_value = ++completion_payload_value_;
    iree_hal_semaphore_t* signal_semaphore = completion_semaphore_;
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&signal_payload_value,
    };
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_alloca(
        device_, kQueue0, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, pool, kQueueAllocaBufferParams, allocation_size,
        IREE_HAL_ALLOCA_FLAG_NONE, out_buffer));
    *out_completion = {completion_semaphore_, signal_payload_value};
    return iree_ok_status();
  }

  iree_status_t QueueAllocaCleanup(iree_hal_buffer_t* buffer,
                                   SubmittedCompletion alloca_completion) {
    IREE_RETURN_IF_ERROR(
        Wait(alloca_completion.semaphore, alloca_completion.payload_value));

    uint64_t signal_payload_value = ++completion_payload_value_;
    iree_hal_semaphore_t* wait_semaphore = alloca_completion.semaphore;
    iree_hal_semaphore_t* signal_semaphore = completion_semaphore_;
    iree_hal_semaphore_list_t wait_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&wait_semaphore,
        /*payload_values=*/&alloca_completion.payload_value,
    };
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&signal_payload_value,
    };
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_dealloca(
        device_, kQueue0, wait_semaphore_list, signal_semaphore_list, buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE));
    IREE_RETURN_IF_ERROR(Wait(completion_semaphore_, signal_payload_value));
    iree_hal_buffer_release(buffer);
    return iree_ok_status();
  }

  iree_status_t QueueAllocaSubmitAndCleanup(
      iree_hal_pool_t* pool, iree_device_size_t allocation_size) {
    iree_hal_buffer_t* buffer = nullptr;
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(
        QueueAllocaSubmit(pool, allocation_size, &buffer, &completion));
    return QueueAllocaCleanup(buffer, completion);
  }

  iree_status_t FillBufferAndWait(iree_hal_buffer_t* target_buffer,
                                  const void* pattern,
                                  iree_host_size_t pattern_length) {
    uint64_t payload_value = ++completion_payload_value_;
    iree_hal_semaphore_t* signal_semaphore = completion_semaphore_;
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&payload_value,
    };
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
        device_, kQueue0, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, target_buffer, /*target_offset=*/0,
        kPayloadBufferAlignment, pattern, pattern_length,
        IREE_HAL_FILL_FLAG_NONE));
    return Wait(completion_semaphore_, payload_value);
  }

  iree_status_t SameQueueBarrierAndWait() {
    uint64_t payload_value = ++completion_payload_value_;
    IREE_RETURN_IF_ERROR(SubmitBarrier(kQueue0, /*wait_semaphore=*/nullptr,
                                       /*wait_payload_value=*/0,
                                       completion_semaphore_, payload_value));
    return Wait(completion_semaphore_, payload_value);
  }

  static iree_status_t NoopHostCall(void* user_data, const uint64_t args[4],
                                    iree_hal_host_call_context_t* context) {
    (void)user_data;
    (void)args;
    (void)context;
    return iree_ok_status();
  }

  iree_status_t HostCallAndWait(iree_hal_host_call_flags_t flags) {
    uint64_t payload_value = ++completion_payload_value_;
    iree_hal_semaphore_t* signal_semaphore = completion_semaphore_;
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&payload_value,
    };
    iree_hal_host_call_t call =
        iree_hal_make_host_call(NoopHostCall, /*user_data=*/nullptr);
    uint64_t args[4] = {0, 0, 0, 0};
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_host_call(
        device_, kQueue0, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, call, args, flags));
    return Wait(completion_semaphore_, payload_value);
  }

  iree_status_t HostCallBatchSubmit(iree_hal_host_call_flags_t flags,
                                    int64_t batch_count,
                                    SubmittedCompletion* out_completion) {
    for (int64_t i = 0; i < batch_count; ++i) {
      const bool is_final_operation = i + 1 == batch_count;
      uint64_t wait_payload_value = stream_payload_value_;
      uint64_t signal_payload_value = stream_payload_value_ + 1;
      iree_hal_semaphore_t* wait_semaphore = stream_semaphore_;
      iree_hal_semaphore_list_t wait_semaphore_list =
          iree_hal_semaphore_list_empty();
      if (i > 0) {
        wait_semaphore_list = iree_hal_semaphore_list_t{
            /*count=*/1,
            /*semaphores=*/&wait_semaphore,
            /*payload_values=*/&wait_payload_value,
        };
      }

      iree_hal_semaphore_t* signal_semaphores[2] = {
          stream_semaphore_,
          completion_semaphore_,
      };
      uint64_t signal_payload_values[2] = {
          signal_payload_value,
          completion_payload_value_ + 1,
      };
      iree_hal_semaphore_list_t signal_semaphore_list = {
          /*count=*/is_final_operation ? 2u : 1u,
          /*semaphores=*/signal_semaphores,
          /*payload_values=*/signal_payload_values,
      };
      iree_hal_host_call_t call =
          iree_hal_make_host_call(NoopHostCall, /*user_data=*/nullptr);
      uint64_t args[4] = {0, 0, 0, 0};
      IREE_RETURN_IF_ERROR(iree_hal_device_queue_host_call(
          device_, kQueue0, wait_semaphore_list, signal_semaphore_list, call,
          args, flags));
      stream_payload_value_ = signal_payload_value;
      if (is_final_operation) {
        ++completion_payload_value_;
      }
    }

    *out_completion = {completion_semaphore_, completion_payload_value_};
    return iree_ok_status();
  }

  iree_status_t HostCallBatchAndWait(iree_hal_host_call_flags_t flags,
                                     int64_t batch_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(HostCallBatchSubmit(flags, batch_count, &completion));
    return Wait(completion.semaphore, completion.payload_value);
  }

  iree_status_t SameQueueBarrierBatchSubmit(
      int64_t batch_count, SubmittedCompletion* out_completion) {
    uint64_t payload_value = completion_payload_value_;
    for (int64_t i = 0; i < batch_count; ++i) {
      const uint64_t wait_payload_value = payload_value;
      const uint64_t signal_payload_value = payload_value + 1;
      IREE_RETURN_IF_ERROR(SubmitBarrier(
          kQueue0, i == 0 ? nullptr : completion_semaphore_, wait_payload_value,
          completion_semaphore_, signal_payload_value));
      payload_value = signal_payload_value;
    }
    completion_payload_value_ = payload_value;
    *out_completion = {completion_semaphore_, payload_value};
    return iree_ok_status();
  }

  iree_status_t SameQueueBarrierBatchAndWait(int64_t batch_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(SameQueueBarrierBatchSubmit(batch_count, &completion));
    return Wait(completion.semaphore, completion.payload_value);
  }

  iree_status_t SameQueueEpochChainSubmit(int64_t batch_count,
                                          SubmittedCompletion* out_completion) {
    for (int64_t i = 0; i < batch_count; ++i) {
      const uint64_t wait_payload_value = stream_payload_value_;
      const uint64_t signal_payload_value = stream_payload_value_ + 1;
      IREE_RETURN_IF_ERROR(SubmitBarrier(
          kQueue0, i == 0 ? nullptr : stream_semaphore_, wait_payload_value,
          stream_semaphore_, signal_payload_value));
      stream_payload_value_ = signal_payload_value;
    }
    *out_completion = {stream_semaphore_, stream_payload_value_};
    return iree_ok_status();
  }

  iree_status_t SameQueueEpochChainAndWait(int64_t batch_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(SameQueueEpochChainSubmit(batch_count, &completion));
    return Wait(completion.semaphore, completion.payload_value);
  }

  iree_status_t CrossQueueAlreadyCompletedWaitAndSignal() {
    const uint64_t completion_payload_value = ++completion_payload_value_;
    IREE_RETURN_IF_ERROR(
        SubmitBarrier(kQueue1, producer_semaphore_, producer_payload_value_,
                      completion_semaphore_, completion_payload_value));
    return Wait(completion_semaphore_, completion_payload_value);
  }

  iree_status_t PrimeProducerSemaphore() {
    producer_payload_value_ = 1;
    IREE_RETURN_IF_ERROR(SubmitBarrier(
        kQueue0, /*wait_semaphore=*/nullptr, /*wait_payload_value=*/0,
        producer_semaphore_, producer_payload_value_));
    return Wait(producer_semaphore_, producer_payload_value_);
  }

  iree_status_t CrossQueueBarrierValueAndWait() {
    const uint64_t producer_payload_value = ++producer_payload_value_;
    const uint64_t completion_payload_value = ++completion_payload_value_;
    IREE_RETURN_IF_ERROR(SubmitBarrier(
        kQueue0, /*wait_semaphore=*/nullptr, /*wait_payload_value=*/0,
        producer_semaphore_, producer_payload_value));
    IREE_RETURN_IF_ERROR(
        SubmitBarrier(kQueue1, producer_semaphore_, producer_payload_value,
                      completion_semaphore_, completion_payload_value));
    return Wait(completion_semaphore_, completion_payload_value);
  }

  iree_status_t CrossQueueBarrierValueBatchSubmit(
      int64_t batch_count, SubmittedCompletion* out_completion) {
    uint64_t completion_payload_value = completion_payload_value_;
    for (int64_t i = 0; i < batch_count; ++i) {
      const uint64_t producer_payload_value = ++producer_payload_value_;
      IREE_RETURN_IF_ERROR(SubmitBarrier(
          kQueue0, /*wait_semaphore=*/nullptr, /*wait_payload_value=*/0,
          producer_semaphore_, producer_payload_value));

      iree_hal_semaphore_t* wait_semaphores[2] = {producer_semaphore_, nullptr};
      uint64_t wait_payload_values[2] = {producer_payload_value, 0};
      iree_host_size_t wait_semaphore_count = 1;
      if (i > 0) {
        wait_semaphores[wait_semaphore_count] = completion_semaphore_;
        wait_payload_values[wait_semaphore_count] = completion_payload_value;
        ++wait_semaphore_count;
      }
      iree_hal_semaphore_list_t wait_semaphore_list = {
          /*count=*/wait_semaphore_count,
          /*semaphores=*/wait_semaphores,
          /*payload_values=*/wait_payload_values,
      };

      const uint64_t signal_completion_payload_value =
          completion_payload_value + 1;
      IREE_RETURN_IF_ERROR(SubmitBarrierWithWaitList(
          kQueue1, wait_semaphore_list, completion_semaphore_,
          signal_completion_payload_value));
      completion_payload_value = signal_completion_payload_value;
    }
    completion_payload_value_ = completion_payload_value;
    *out_completion = {completion_semaphore_, completion_payload_value};
    return iree_ok_status();
  }

  iree_status_t CrossQueueBarrierValueBatchAndWait(int64_t batch_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(
        CrossQueueBarrierValueBatchSubmit(batch_count, &completion));
    return Wait(completion.semaphore, completion.payload_value);
  }

  iree_status_t CrossQueuePingPongChainSubmit(
      int64_t handoff_count, SubmittedCompletion* out_completion) {
    uint64_t producer_payload_value = ++producer_payload_value_;
    IREE_RETURN_IF_ERROR(SubmitBarrier(
        kQueue0, /*wait_semaphore=*/nullptr, /*wait_payload_value=*/0,
        producer_semaphore_, producer_payload_value));

    iree_hal_semaphore_t* final_semaphore = producer_semaphore_;
    uint64_t final_payload_value = producer_payload_value;
    for (int64_t i = 0; i < handoff_count; ++i) {
      if ((i & 1) == 0) {
        const uint64_t stream_payload_value = ++stream_payload_value_;
        IREE_RETURN_IF_ERROR(
            SubmitBarrier(kQueue1, producer_semaphore_, producer_payload_value,
                          stream_semaphore_, stream_payload_value));
        final_semaphore = stream_semaphore_;
        final_payload_value = stream_payload_value;
      } else {
        producer_payload_value = ++producer_payload_value_;
        IREE_RETURN_IF_ERROR(
            SubmitBarrier(kQueue0, stream_semaphore_, stream_payload_value_,
                          producer_semaphore_, producer_payload_value));
        final_semaphore = producer_semaphore_;
        final_payload_value = producer_payload_value;
      }
    }
    *out_completion = {final_semaphore, final_payload_value};
    return iree_ok_status();
  }

  iree_status_t CrossQueuePingPongChainSubmitPublicFinalInline(
      int64_t handoff_count, SubmittedCompletion* out_completion) {
    if (handoff_count == 0) {
      SubmittedCompletion private_completion;
      IREE_RETURN_IF_ERROR(
          CrossQueuePingPongChainSubmit(handoff_count, &private_completion));
      const uint64_t completion_payload_value = ++completion_payload_value_;
      IREE_RETURN_IF_ERROR(SubmitBarrier(
          CrossQueuePingPongFinalQueue(handoff_count),
          private_completion.semaphore, private_completion.payload_value,
          completion_semaphore_, completion_payload_value));
      *out_completion = {completion_semaphore_, completion_payload_value};
      return iree_ok_status();
    }

    uint64_t producer_payload_value = ++producer_payload_value_;
    IREE_RETURN_IF_ERROR(SubmitBarrier(
        kQueue0, /*wait_semaphore=*/nullptr, /*wait_payload_value=*/0,
        producer_semaphore_, producer_payload_value));

    for (int64_t i = 0; i < handoff_count; ++i) {
      const bool is_final_handoff = i + 1 == handoff_count;
      if ((i & 1) == 0) {
        const uint64_t stream_payload_value = ++stream_payload_value_;
        iree_hal_semaphore_t* signal_semaphores[2] = {
            stream_semaphore_,
            completion_semaphore_,
        };
        uint64_t signal_payload_values[2] = {
            stream_payload_value,
            completion_payload_value_ + 1,
        };
        iree_hal_semaphore_list_t signal_semaphore_list = {
            /*count=*/is_final_handoff ? 2u : 1u,
            /*semaphores=*/signal_semaphores,
            /*payload_values=*/signal_payload_values,
        };
        IREE_RETURN_IF_ERROR(SubmitBarrierWithSingleWaitAndSignalList(
            kQueue1, producer_semaphore_, producer_payload_value,
            signal_semaphore_list));
        if (is_final_handoff) {
          ++completion_payload_value_;
        }
      } else {
        producer_payload_value = ++producer_payload_value_;
        iree_hal_semaphore_t* signal_semaphores[2] = {
            producer_semaphore_,
            completion_semaphore_,
        };
        uint64_t signal_payload_values[2] = {
            producer_payload_value,
            completion_payload_value_ + 1,
        };
        iree_hal_semaphore_list_t signal_semaphore_list = {
            /*count=*/is_final_handoff ? 2u : 1u,
            /*semaphores=*/signal_semaphores,
            /*payload_values=*/signal_payload_values,
        };
        IREE_RETURN_IF_ERROR(SubmitBarrierWithSingleWaitAndSignalList(
            kQueue0, stream_semaphore_, stream_payload_value_,
            signal_semaphore_list));
        if (is_final_handoff) {
          ++completion_payload_value_;
        }
      }
    }
    *out_completion = {completion_semaphore_, completion_payload_value_};
    return iree_ok_status();
  }

  iree_status_t CrossQueuePingPongChainSubmitPublicFinalSeparate(
      int64_t handoff_count, SubmittedCompletion* out_completion) {
    SubmittedCompletion private_completion;
    IREE_RETURN_IF_ERROR(
        CrossQueuePingPongChainSubmit(handoff_count, &private_completion));
    const uint64_t completion_payload_value = ++completion_payload_value_;
    IREE_RETURN_IF_ERROR(SubmitBarrier(
        CrossQueuePingPongFinalQueue(handoff_count),
        private_completion.semaphore, private_completion.payload_value,
        completion_semaphore_, completion_payload_value));
    *out_completion = {completion_semaphore_, completion_payload_value};
    return iree_ok_status();
  }

  iree_status_t CrossQueuePingPongPayloadSubmitPublicFinalInline(
      PayloadKind payload_kind, int64_t handoff_count,
      SubmittedCompletion* out_completion) {
    uint64_t producer_payload_value = ++producer_payload_value_;
    IREE_RETURN_IF_ERROR(SubmitBarrier(
        kQueue0, /*wait_semaphore=*/nullptr, /*wait_payload_value=*/0,
        producer_semaphore_, producer_payload_value));

    for (int64_t i = 0; i < handoff_count; ++i) {
      const bool is_final_handoff = i + 1 == handoff_count;
      iree_hal_semaphore_t* signal_semaphores[2] = {
          nullptr,
          completion_semaphore_,
      };
      uint64_t signal_payload_values[2] = {
          0,
          completion_payload_value_ + 1,
      };
      iree_hal_semaphore_list_t signal_semaphore_list = {
          /*count=*/is_final_handoff ? 2u : 1u,
          /*semaphores=*/signal_semaphores,
          /*payload_values=*/signal_payload_values,
      };

      if ((i & 1) == 0) {
        const uint64_t stream_payload_value = ++stream_payload_value_;
        iree_hal_semaphore_t* wait_semaphore = producer_semaphore_;
        iree_hal_semaphore_list_t wait_semaphore_list = {
            /*count=*/1,
            /*semaphores=*/&wait_semaphore,
            /*payload_values=*/&producer_payload_value,
        };
        signal_semaphores[0] = stream_semaphore_;
        signal_payload_values[0] = stream_payload_value;
        IREE_RETURN_IF_ERROR(SubmitPayloadWithLists(
            payload_kind, kQueue1, wait_semaphore_list, signal_semaphore_list));
      } else {
        producer_payload_value = ++producer_payload_value_;
        iree_hal_semaphore_t* wait_semaphore = stream_semaphore_;
        iree_hal_semaphore_list_t wait_semaphore_list = {
            /*count=*/1,
            /*semaphores=*/&wait_semaphore,
            /*payload_values=*/&stream_payload_value_,
        };
        signal_semaphores[0] = producer_semaphore_;
        signal_payload_values[0] = producer_payload_value;
        IREE_RETURN_IF_ERROR(SubmitPayloadWithLists(
            payload_kind, kQueue0, wait_semaphore_list, signal_semaphore_list));
      }
      if (is_final_handoff) {
        ++completion_payload_value_;
      }
    }

    *out_completion = {completion_semaphore_, completion_payload_value_};
    return iree_ok_status();
  }

  iree_status_t CrossQueuePingPongPayloadPublicFinalInlineAndWait(
      PayloadKind payload_kind, int64_t handoff_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(CrossQueuePingPongPayloadSubmitPublicFinalInline(
        payload_kind, handoff_count, &completion));
    return Wait(completion.semaphore, completion.payload_value);
  }

  iree_status_t SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
      PayloadKind payload_kind, int64_t operation_count,
      SubmittedCompletion* out_completion) {
    for (int64_t i = 0; i < operation_count; ++i) {
      const bool is_final_operation = i + 1 == operation_count;
      uint64_t wait_payload_value = stream_payload_value_;
      uint64_t signal_payload_value = stream_payload_value_ + 1;
      iree_hal_semaphore_t* wait_semaphore = stream_semaphore_;
      iree_hal_semaphore_list_t wait_semaphore_list =
          iree_hal_semaphore_list_empty();
      if (i > 0) {
        wait_semaphore_list = iree_hal_semaphore_list_t{
            /*count=*/1,
            /*semaphores=*/&wait_semaphore,
            /*payload_values=*/&wait_payload_value,
        };
      }
      iree_hal_semaphore_t* signal_semaphores[2] = {
          stream_semaphore_,
          completion_semaphore_,
      };
      uint64_t signal_payload_values[2] = {
          signal_payload_value,
          completion_payload_value_ + 1,
      };
      iree_hal_semaphore_list_t signal_semaphore_list = {
          /*count=*/is_final_operation ? 2u : 1u,
          /*semaphores=*/signal_semaphores,
          /*payload_values=*/signal_payload_values,
      };
      IREE_RETURN_IF_ERROR(SubmitPayloadWithLists(
          payload_kind, kQueue0, wait_semaphore_list, signal_semaphore_list));
      stream_payload_value_ = signal_payload_value;
      if (is_final_operation) {
        ++completion_payload_value_;
      }
    }

    *out_completion = {completion_semaphore_, completion_payload_value_};
    return iree_ok_status();
  }

  iree_status_t SameQueuePrivateStreamPayloadPublicFinalInlineAndWait(
      PayloadKind payload_kind, int64_t operation_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
        payload_kind, operation_count, &completion));
    return Wait(completion.semaphore, completion.payload_value);
  }

  iree_status_t CrossQueuePingPongChainAndWait(int64_t handoff_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(
        CrossQueuePingPongChainSubmit(handoff_count, &completion));
    return Wait(completion.semaphore, completion.payload_value);
  }

  iree_status_t CrossQueuePingPongChainPublicFinalInlineAndWait(
      int64_t handoff_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(CrossQueuePingPongChainSubmitPublicFinalInline(
        handoff_count, &completion));
    return Wait(completion.semaphore, completion.payload_value);
  }

  iree_status_t CrossQueuePingPongChainPublicFinalSeparateAndWait(
      int64_t handoff_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(CrossQueuePingPongChainSubmitPublicFinalSeparate(
        handoff_count, &completion));
    return Wait(completion.semaphore, completion.payload_value);
  }

  iree_status_t WaitBeforeSignalChainAndWait() {
    const uint64_t producer_payload_value = ++producer_payload_value_;
    const uint64_t completion_payload_value = ++completion_payload_value_;
    IREE_RETURN_IF_ERROR(
        SubmitBarrier(kQueue1, producer_semaphore_, producer_payload_value,
                      completion_semaphore_, completion_payload_value));
    IREE_RETURN_IF_ERROR(SubmitBarrier(
        kQueue0, /*wait_semaphore=*/nullptr, /*wait_payload_value=*/0,
        producer_semaphore_, producer_payload_value));
    return Wait(completion_semaphore_, completion_payload_value);
  }

  bool HandleStatus(benchmark::State& state, iree_status_t status,
                    const char* message) {
    if (iree_status_is_ok(status)) return true;
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    state.SkipWithError(message);
    return false;
  }

  void SetQueueSubmissionsProcessed(benchmark::State& state,
                                    int64_t queue_submissions_per_sync) {
    state.counters["queue_submissions_per_sync"] =
        static_cast<double>(queue_submissions_per_sync);
    iree_hal_amdgpu_benchmark_set_completion_wait_counters(state);
    state.SetItemsProcessed(state.iterations() * queue_submissions_per_sync);
  }

  void SetCrossQueuePingPongCounters(
      benchmark::State& state, int64_t handoff_count,
      int64_t queue_submissions_per_sync,
      int64_t public_completion_signals_per_sync) {
    state.counters["cross_queue_handoffs_per_sync"] =
        static_cast<double>(handoff_count);
    state.counters["hip_equivalent_round_trips_per_sync"] =
        static_cast<double>(handoff_count) / 2.0;
    state.counters["public_completion_signals_per_sync"] =
        static_cast<double>(public_completion_signals_per_sync);
    SetQueueSubmissionsProcessed(state, queue_submissions_per_sync);
  }

  void SetPayloadPingPongCounters(benchmark::State& state,
                                  int64_t handoff_count,
                                  int64_t queue_submissions_per_sync,
                                  int64_t public_completion_signals_per_sync) {
    SetCrossQueuePingPongCounters(state, handoff_count,
                                  queue_submissions_per_sync,
                                  public_completion_signals_per_sync);
    state.counters["payload_operations_per_sync"] =
        static_cast<double>(handoff_count);
  }

  void SetSingleStreamPayloadCounters(benchmark::State& state,
                                      int64_t operation_count) {
    state.counters["operations_per_sync"] =
        static_cast<double>(operation_count);
    state.counters["public_completion_signals_per_sync"] = 1.0;
    SetQueueSubmissionsProcessed(state, operation_count);
  }

  void SetQueueAllocaCounters(benchmark::State& state,
                              iree_device_size_t allocation_size,
                              QueueAllocaTlsfGrowthMode growth_mode) {
    state.counters["queue_allocas_per_sync"] = 1.0;
    state.counters["allocation_bytes_per_sync"] =
        static_cast<double>(allocation_size);
    state.counters["forced_tlsf_growth"] =
        growth_mode == QueueAllocaTlsfGrowthMode::kForcedGrowth ? 1.0 : 0.0;
    SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/1);
  }

  void SetProfileGuardrailCounters(benchmark::State& state,
                                   ProfileGuardrailMode mode,
                                   int64_t queue_submissions_per_sync,
                                   int64_t profiled_operations_per_sync) {
    const iree_hal_device_profiling_data_families_t data_families =
        ProfileGuardrailDataFamilies(mode);
    state.counters["profile_data_families"] =
        static_cast<double>(data_families);
    state.counters["profile_queue_events"] =
        iree_any_bit_set(data_families,
                         IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)
            ? 1.0
            : 0.0;
    state.counters["profile_device_queue_events"] =
        iree_any_bit_set(data_families,
                         IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS)
            ? 1.0
            : 0.0;
    state.counters["profile_dispatch_events"] =
        iree_any_bit_set(data_families,
                         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS)
            ? 1.0
            : 0.0;
    state.counters["profiled_operations_per_sync"] =
        static_cast<double>(profiled_operations_per_sync);
    SetQueueSubmissionsProcessed(state, queue_submissions_per_sync);
  }

  bool WaitWithTimingPaused(benchmark::State& state,
                            const SubmittedCompletion& completion,
                            const char* message) {
    state.PauseTiming();
    iree_status_t status = Wait(completion.semaphore, completion.payload_value);
    state.ResumeTiming();
    return HandleStatus(state, status, message);
  }

  bool EnsurePayloadBuffers(benchmark::State& state) {
    if (source_buffer_ && target_buffer_) return true;
    return AllocatePayloadBuffers(state);
  }

  iree_status_t LoadExecutableFromData(
      iree_const_byte_span_t executable_data,
      iree_hal_executable_cache_t** out_executable_cache,
      iree_hal_executable_t** out_executable) {
    *out_executable_cache = nullptr;
    *out_executable = nullptr;

    iree_hal_executable_cache_t* executable_cache = nullptr;
    iree_hal_executable_t* executable = nullptr;
    iree_status_t status = iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache);

    char executable_format[128] = {0};
    iree_host_size_t inferred_size = 0;
    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_cache_infer_format(
          executable_cache,
          IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA, executable_data,
          IREE_ARRAYSIZE(executable_format), executable_format, &inferred_size);
    }

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

  bool EnsureDispatchExecutable(benchmark::State& state) {
    if (dispatch_executable_) return true;

    iree_const_byte_span_t executable_data = iree_const_byte_span_empty();
    iree_status_t status = iree_ok_status();
    executable_data = FindCtsExecutableData(iree_make_cstring_view(
        "command_buffer_dispatch_constants_bindings_test.bin"));
    if (executable_data.data_length == 0) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "AMDGPU CTS dispatch executable not found");
    }

    if (iree_status_is_ok(status)) {
      status = LoadExecutableFromData(
          executable_data, &dispatch_executable_cache_, &dispatch_executable_);
    }
    return HandleStatus(state, status, "failed to load dispatch executable");
  }

  bool EnsureBindingCountExecutable(benchmark::State& state) {
    if (binding_count_executable_) return true;

    const iree_string_view_t executable_file =
        iree_make_cstring_view(FLAG_binding_count_executable_file);
    iree_io_file_contents_t* executable_file_contents = nullptr;
    iree_const_byte_span_t executable_data = iree_const_byte_span_empty();
    iree_status_t status = iree_ok_status();
    if (!iree_string_view_is_empty(executable_file)) {
      status = iree_io_file_contents_read(executable_file, host_allocator_,
                                          &executable_file_contents);
      if (iree_status_is_ok(status)) {
        executable_data = executable_file_contents->const_buffer;
      }
    } else {
      executable_data = FindQueueBenchmarkExecutableData(
          iree_make_cstring_view("queue_benchmark_testdata.bin"));
      if (executable_data.data_length == 0) {
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "AMDGPU queue benchmark dispatch executable not found");
      }
    }
    if (iree_status_is_ok(status)) {
      status = LoadExecutableFromData(executable_data,
                                      &binding_count_executable_cache_,
                                      &binding_count_executable_);
    }
    if (iree_status_is_ok(status)) {
      binding_count_executable_file_contents_ = executable_file_contents;
    } else {
      iree_io_file_contents_free(executable_file_contents);
    }
    return HandleStatus(state, status,
                        "failed to load binding-count dispatch executable");
  }

  bool EnsureBindingCountBuffers(benchmark::State& state,
                                 int64_t binding_count) {
    if (binding_count < 0 ||
        binding_count > (int64_t)kDispatchBindingBenchmarkMaxCount) {
      return HandleStatus(
          state,
          iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "unsupported dispatch benchmark binding count %" PRId64,
              binding_count),
          "invalid dispatch benchmark binding count");
    }

    iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
    iree_hal_buffer_params_t params = {0};
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.min_alignment = kPayloadBufferAlignment;
    for (iree_host_size_t i = 0; i < (iree_host_size_t)binding_count; ++i) {
      if (binding_count_buffers_[i]) continue;
      iree_status_t status = iree_hal_allocator_allocate_buffer(
          allocator, params, kPayloadBufferAlignment,
          &binding_count_buffers_[i]);
      if (!HandleStatus(state, status,
                        "failed to allocate dispatch benchmark binding")) {
        return false;
      }
    }
    return true;
  }

  bool EnsurePreResolvedDispatch(benchmark::State& state) {
    if (pre_resolved_dispatch_kernargs_) return true;
    if (!EnsurePayloadBuffers(state) || !EnsureDispatchExecutable(state)) {
      return false;
    }
    return HandleStatus(state, PreparePreResolvedDispatch(),
                        "failed to prepare pre-resolved dispatch");
  }

  iree_status_t PreparePreResolvedDispatch() {
    iree_hal_amdgpu_host_queue_t* host_queue = nullptr;
    IREE_RETURN_IF_ERROR(LookupHostQueue(kQueue0, &host_queue));

    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor =
        nullptr;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
            dispatch_executable_, /*export_ordinal=*/0,
            host_queue->device_ordinal, &descriptor));
    if (IREE_UNLIKELY(!descriptor)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "dispatch executable has no descriptor for device ordinal "
          "%" PRIhsz,
          host_queue->device_ordinal);
    }
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
        &descriptor->kernel_args;

    uint64_t binding_ptrs[2] = {0, 0};
    IREE_RETURN_IF_ERROR(
        ResolveBufferDevicePointer(source_buffer_, &binding_ptrs[0]));
    IREE_RETURN_IF_ERROR(
        ResolveBufferDevicePointer(target_buffer_, &binding_ptrs[1]));

    const uint32_t workgroup_count[3] = {1, 1, 1};
    const uint32_t dynamic_workgroup_local_memory = 0;
    const uint32_t kernarg_block_count = descriptor->hal_kernarg_block_count;

    iree_host_size_t kernarg_length = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            kernarg_block_count, sizeof(iree_hal_amdgpu_kernarg_block_t),
            &kernarg_length))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "pre-resolved dispatch kernarg storage "
                              "overflows host size");
    }

    uint8_t* kernargs = nullptr;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator_, kernarg_length,
                                               (void**)&kernargs));
    std::memset(kernargs, 0, kernarg_length);

    const uint32_t constant_data[] = {3, 10};
    iree_hal_amdgpu_device_dispatch_emplace_hal_kernargs(
        kernel_args, workgroup_count, dynamic_workgroup_local_memory,
        &descriptor->hal_kernarg_layout, binding_ptrs, constant_data, kernargs);
    std::memset(&pre_resolved_dispatch_packet_template_, 0,
                sizeof(pre_resolved_dispatch_packet_template_));
    iree_hal_amdgpu_device_dispatch_emplace_packet(
        kernel_args, workgroup_count, dynamic_workgroup_local_memory,
        &pre_resolved_dispatch_packet_template_, /*kernarg_ptr=*/nullptr);

    pre_resolved_dispatch_kernargs_ = kernargs;
    pre_resolved_dispatch_kernarg_length_ = kernarg_length;
    pre_resolved_dispatch_kernarg_block_count_ = (uint32_t)kernarg_block_count;
    return iree_ok_status();
  }

  void ReleasePreResolvedDispatch() {
    iree_allocator_free(host_allocator_, pre_resolved_dispatch_kernargs_);
    pre_resolved_dispatch_kernargs_ = nullptr;
    pre_resolved_dispatch_kernarg_length_ = 0;
    pre_resolved_dispatch_kernarg_block_count_ = 0;
    std::memset(&pre_resolved_dispatch_packet_template_, 0,
                sizeof(pre_resolved_dispatch_packet_template_));
  }

  iree_status_t SubmitPreResolvedDispatchWithLists(
      iree_hal_queue_affinity_t queue_affinity,
      iree_hal_semaphore_list_t wait_semaphore_list,
      iree_hal_semaphore_list_t signal_semaphore_list) {
    iree_hal_amdgpu_host_queue_t* host_queue = nullptr;
    IREE_RETURN_IF_ERROR(LookupHostQueue(queue_affinity, &host_queue));

    iree_slim_mutex_lock(&host_queue->locks.submission_mutex);
    iree_hal_amdgpu_wait_resolution_t resolution;
    iree_hal_amdgpu_host_queue_resolve_waits(host_queue, wait_semaphore_list,
                                             &resolution);
    iree_status_t status = iree_ok_status();
    if (IREE_UNLIKELY(resolution.needs_deferral)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "pre-resolved dispatch benchmark path cannot defer waits");
    }

    if (iree_status_is_ok(status)) {
      iree_hal_resource_t* operation_resources[3] = {
          (iree_hal_resource_t*)dispatch_executable_,
          (iree_hal_resource_t*)source_buffer_,
          (iree_hal_resource_t*)target_buffer_,
      };
      iree_hal_amdgpu_host_queue_dispatch_submission_t submission;
      bool ready = false;
      status = iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
          host_queue, &resolution, signal_semaphore_list,
          IREE_ARRAYSIZE(operation_resources),
          pre_resolved_dispatch_kernarg_block_count_,
          iree_hal_amdgpu_profile_dispatch_event_reservation_t{0},
          /*profile_queue_event_info=*/nullptr, &ready, &submission);
      if (iree_status_is_ok(status) && !ready) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "pre-resolved dispatch benchmark path hit temporary queue "
            "capacity");
      }
      if (iree_status_is_ok(status)) {
        std::memcpy(submission.kernel.kernargs.blocks->data,
                    pre_resolved_dispatch_kernargs_,
                    pre_resolved_dispatch_kernarg_length_);
        submission.dispatch_setup =
            iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
                &submission.dispatch_slot->dispatch,
                &pre_resolved_dispatch_packet_template_,
                submission.kernel.kernargs.blocks->data,
                submission.dispatch_completion_signal);
        iree_hal_amdgpu_host_queue_finish_dispatch_submission(
            host_queue, &resolution, signal_semaphore_list, operation_resources,
            IREE_ARRAYSIZE(operation_resources),
            /*profile_queue_event_info=*/nullptr,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
            &submission);
      }
    }
    iree_slim_mutex_unlock(&host_queue->locks.submission_mutex);
    return status;
  }

  iree_status_t ValidateDispatchOnce(
      iree_hal_amdgpu_host_queue_t* host_queue,
      iree_host_size_t* out_operation_resource_count) {
    const uint32_t constant_data[] = {3, 10};
    iree_const_byte_span_t constants =
        iree_make_const_byte_span(constant_data, sizeof(constant_data));
    iree_hal_buffer_ref_t binding_refs[2] = {
        iree_hal_make_buffer_ref(source_buffer_, /*offset=*/0,
                                 iree_hal_buffer_byte_length(source_buffer_)),
        iree_hal_make_buffer_ref(target_buffer_, /*offset=*/0,
                                 iree_hal_buffer_byte_length(target_buffer_)),
    };
    iree_hal_buffer_ref_list_t bindings = {
        /*count=*/IREE_ARRAYSIZE(binding_refs),
        /*values=*/binding_refs,
    };
    return iree_hal_amdgpu_host_queue_validate_dispatch(
        host_queue, dispatch_executable_, /*export_ordinal=*/0,
        iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
        IREE_HAL_DISPATCH_FLAG_NONE, out_operation_resource_count);
  }

  static iree_status_t BindingCountDispatchConfig(
      iree_hal_dispatch_config_t* out_config) {
    *out_config = iree_hal_make_static_dispatch_config(1, 1, 1);
    if (FLAG_binding_count_workgroup_size_x == 0) {
      return iree_ok_status();
    }
    if (FLAG_binding_count_workgroup_size_x < 0 ||
        FLAG_binding_count_workgroup_size_x > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "binding_count_workgroup_size_x must be in [0, %" PRIu16 "]",
          UINT16_MAX);
    }
    out_config->workgroup_size[0] =
        (uint32_t)FLAG_binding_count_workgroup_size_x;
    out_config->workgroup_size[1] = 1;
    out_config->workgroup_size[2] = 1;
    return iree_ok_status();
  }

  static iree_status_t BindingCountExportName(int64_t binding_count,
                                              char* buffer,
                                              iree_host_size_t capacity,
                                              iree_string_view_t* out_name) {
    *out_name = iree_string_view_empty();
    if (binding_count < 0 ||
        binding_count > (int64_t)kDispatchBindingBenchmarkMaxCount) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "unsupported dispatch benchmark binding count "
                              "%" PRId64,
                              binding_count);
    }
    int result =
        snprintf(buffer, capacity, "binding_count_%" PRId64, binding_count);
    if (result < 0 || (iree_host_size_t)result >= capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "binding-count export name is too long");
    }
    *out_name = iree_make_string_view(buffer, (iree_host_size_t)result);
    return iree_ok_status();
  }

  iree_status_t BindingCountExportOrdinal(
      int64_t binding_count,
      iree_hal_executable_export_ordinal_t* out_export_ordinal) {
    *out_export_ordinal = 0;
    char export_name_buffer[32] = {0};
    iree_string_view_t export_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(BindingCountExportName(
        binding_count, export_name_buffer, IREE_ARRAYSIZE(export_name_buffer),
        &export_name));
    return iree_hal_executable_lookup_export_by_name(
        binding_count_executable_, export_name, out_export_ordinal);
  }

  iree_hal_buffer_ref_list_t BindingCountDispatchBindings(
      int64_t binding_count) {
    for (iree_host_size_t i = 0; i < (iree_host_size_t)binding_count; ++i) {
      binding_count_binding_ref_scratch_[i] = iree_hal_make_buffer_ref(
          binding_count_buffers_[i], /*offset=*/0,
          iree_hal_buffer_byte_length(binding_count_buffers_[i]));
    }
    return (iree_hal_buffer_ref_list_t){
        /*count=*/(iree_host_size_t)binding_count,
        /*values=*/binding_count_binding_ref_scratch_,
    };
  }

  iree_status_t ValidateBindingCountDispatchOnce(
      iree_hal_amdgpu_host_queue_t* host_queue, int64_t binding_count,
      iree_host_size_t* out_operation_resource_count) {
    iree_hal_executable_export_ordinal_t export_ordinal = 0;
    IREE_RETURN_IF_ERROR(
        BindingCountExportOrdinal(binding_count, &export_ordinal));
    iree_hal_dispatch_config_t dispatch_config;
    IREE_RETURN_IF_ERROR(BindingCountDispatchConfig(&dispatch_config));
    return iree_hal_amdgpu_host_queue_validate_dispatch(
        host_queue, binding_count_executable_, export_ordinal, dispatch_config,
        iree_const_byte_span_empty(),
        BindingCountDispatchBindings(binding_count),
        IREE_HAL_DISPATCH_FLAG_NONE, out_operation_resource_count);
  }

  iree_status_t SubmitBindingCountDispatchWithLists(
      iree_hal_queue_affinity_t queue_affinity,
      iree_hal_semaphore_list_t wait_semaphore_list,
      iree_hal_semaphore_list_t signal_semaphore_list, int64_t binding_count) {
    iree_hal_executable_export_ordinal_t export_ordinal = 0;
    IREE_RETURN_IF_ERROR(
        BindingCountExportOrdinal(binding_count, &export_ordinal));
    iree_hal_dispatch_config_t dispatch_config;
    IREE_RETURN_IF_ERROR(BindingCountDispatchConfig(&dispatch_config));
    return iree_hal_device_queue_dispatch(
        device_, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        binding_count_executable_, export_ordinal, dispatch_config,
        iree_const_byte_span_empty(),
        BindingCountDispatchBindings(binding_count),
        IREE_HAL_DISPATCH_FLAG_NONE);
  }

  iree_status_t RecordBindingCountCommandBuffer(
      int64_t binding_count, iree_hal_command_buffer_t** out_command_buffer) {
    *out_command_buffer = nullptr;
    iree_hal_executable_export_ordinal_t export_ordinal = 0;
    IREE_RETURN_IF_ERROR(
        BindingCountExportOrdinal(binding_count, &export_ordinal));
    iree_hal_dispatch_config_t dispatch_config;
    IREE_RETURN_IF_ERROR(BindingCountDispatchConfig(&dispatch_config));

    iree_hal_command_buffer_t* command_buffer = nullptr;
    iree_status_t status = iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, &command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_begin(command_buffer);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_dispatch(
          command_buffer, binding_count_executable_, export_ordinal,
          dispatch_config, iree_const_byte_span_empty(),
          BindingCountDispatchBindings(binding_count),
          IREE_HAL_DISPATCH_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_end(command_buffer);
    }
    if (iree_status_is_ok(status)) {
      *out_command_buffer = command_buffer;
    } else {
      iree_hal_command_buffer_release(command_buffer);
    }
    return status;
  }

  bool EnsureBindingCountCommandBuffer(benchmark::State& state,
                                       int64_t binding_count) {
    if (!EnsureBindingCountExecutable(state) ||
        !EnsureBindingCountBuffers(state, binding_count)) {
      return false;
    }
    iree_hal_command_buffer_t** command_buffer_slot =
        &binding_count_command_buffers_[binding_count];
    if (*command_buffer_slot) return true;
    iree_status_t status =
        RecordBindingCountCommandBuffer(binding_count, command_buffer_slot);
    return HandleStatus(state, status,
                        "failed to record binding-count command buffer");
  }

  iree_status_t RecordBindingCountDispatchChainCommandBuffer(
      int64_t operation_count, int64_t binding_count,
      iree_hal_command_buffer_t** out_command_buffer) {
    *out_command_buffer = nullptr;
    if (operation_count <= 0) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "operation count must be positive");
    }

    iree_hal_executable_export_ordinal_t export_ordinal = 0;
    IREE_RETURN_IF_ERROR(
        BindingCountExportOrdinal(binding_count, &export_ordinal));
    iree_hal_dispatch_config_t dispatch_config;
    IREE_RETURN_IF_ERROR(BindingCountDispatchConfig(&dispatch_config));

    iree_hal_command_buffer_t* command_buffer = nullptr;
    iree_status_t status = iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, &command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_begin(command_buffer);
    }
    for (int64_t i = 0; i < operation_count && iree_status_is_ok(status); ++i) {
      status = iree_hal_command_buffer_dispatch(
          command_buffer, binding_count_executable_, export_ordinal,
          dispatch_config, iree_const_byte_span_empty(),
          BindingCountDispatchBindings(binding_count),
          IREE_HAL_DISPATCH_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_end(command_buffer);
    }
    if (iree_status_is_ok(status)) {
      *out_command_buffer = command_buffer;
    } else {
      iree_hal_command_buffer_release(command_buffer);
    }
    return status;
  }

  void SetBindingCountCounters(benchmark::State& state, int64_t binding_count) {
    state.counters["binding_count"] = static_cast<double>(binding_count);
    state.counters["binding_count_external_executable"] =
        iree_string_view_is_empty(
            iree_make_cstring_view(FLAG_binding_count_executable_file))
            ? 0.0
            : 1.0;
    state.counters["binding_count_workgroup_size_x"] =
        static_cast<double>(FLAG_binding_count_workgroup_size_x);
  }

  void SetCommandBufferProgramCounters(
      benchmark::State& state, iree_hal_command_buffer_t* command_buffer,
      int64_t operation_count, int64_t binding_count) {
    const iree_hal_amdgpu_aql_program_t* program =
        iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
    uint64_t payload_block_count = 0;
    uint64_t total_aql_packet_count = 0;
    uint64_t total_block_bytes = 0;
    uint64_t total_used_bytes = 0;
    struct {
      uint64_t dispatch_count = 0;
      uint64_t payload_bytes = 0;
      uint64_t storage_span_bytes = 0;
    } prepublished_kernarg;
    struct {
      uint64_t dispatch_count = 0;
      uint64_t payload_bytes = 0;
      uint64_t reserved_bytes = 0;
    } queue_kernarg;
    for (const iree_hal_amdgpu_command_buffer_block_header_t* block =
             program->first_block;
         block; block = iree_hal_amdgpu_aql_program_block_next(
                    program->block_pool, block)) {
      const uint64_t binding_source_length =
          (uint64_t)block->binding_source_count *
          sizeof(iree_hal_amdgpu_command_buffer_binding_source_t);
      const uint64_t used_bytes = block->header_length + block->command_length +
                                  binding_source_length + block->rodata_length;
      if (block->aql_packet_count > 0) ++payload_block_count;
      total_aql_packet_count += block->aql_packet_count;
      total_block_bytes += block->block_length;
      total_used_bytes += used_bytes;

      const uint8_t* command_end =
          (const uint8_t*)block + block->command_offset + block->command_length;
      for (const iree_hal_amdgpu_command_buffer_command_header_t* command =
               iree_hal_amdgpu_command_buffer_block_commands_const(block);
           (const uint8_t*)command < command_end;
           command =
               iree_hal_amdgpu_command_buffer_command_next_const(command)) {
        if (command->opcode != IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH) {
          continue;
        }
        const iree_hal_amdgpu_command_buffer_dispatch_command_t*
            dispatch_command =
                (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
                    command;
        const uint64_t kernarg_bytes =
            (uint64_t)dispatch_command->kernarg_length_qwords * 8u;
        if (dispatch_command->kernarg_strategy ==
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED) {
          ++prepublished_kernarg.dispatch_count;
          prepublished_kernarg.payload_bytes += kernarg_bytes;
          const uint64_t storage_end =
              (uint64_t)dispatch_command->payload_reference + kernarg_bytes;
          prepublished_kernarg.storage_span_bytes =
              std::max(prepublished_kernarg.storage_span_bytes, storage_end);
        } else {
          ++queue_kernarg.dispatch_count;
          queue_kernarg.payload_bytes += kernarg_bytes;
          uint64_t kernarg_block_count = std::max<uint64_t>(
              1, (kernarg_bytes + sizeof(iree_hal_amdgpu_kernarg_block_t) - 1) /
                     sizeof(iree_hal_amdgpu_kernarg_block_t));
          if (iree_any_bit_set(
                  dispatch_command->dispatch_flags,
                  IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS)) {
            ++kernarg_block_count;
          }
          queue_kernarg.reserved_bytes +=
              kernarg_block_count * sizeof(iree_hal_amdgpu_kernarg_block_t);
        }
      }
    }

    state.counters["operation_count"] = static_cast<double>(operation_count);
    SetBindingCountCounters(state, binding_count);
    state.counters["command_buffer_blocks"] =
        static_cast<double>(program->block_count);
    state.counters["command_buffer_payload_blocks"] =
        static_cast<double>(payload_block_count);
    state.counters["command_buffer_block_bytes"] =
        static_cast<double>(program->first_block->block_length);
    state.counters["command_buffer_occupancy_pct"] =
        total_block_bytes
            ? 100.0 * (double)total_used_bytes / (double)total_block_bytes
            : 0.0;
    state.counters["aql_packets_per_sync"] =
        static_cast<double>(total_aql_packet_count);
    state.counters["max_block_aql_packets"] =
        static_cast<double>(program->max_block_aql_packet_count);
    state.counters["max_block_kernarg_bytes"] =
        static_cast<double>(program->max_block_kernarg_length);
    state.counters["prepublished_dispatches_per_sync"] =
        static_cast<double>(prepublished_kernarg.dispatch_count);
    state.counters["prepublished_kernarg_payload_bytes_per_sync"] =
        static_cast<double>(prepublished_kernarg.payload_bytes);
    state.counters["prepublished_storage_span_bytes"] =
        static_cast<double>(prepublished_kernarg.storage_span_bytes);
    state.counters["queue_kernarg_dispatches_per_sync"] =
        static_cast<double>(queue_kernarg.dispatch_count);
    state.counters["queue_kernarg_payload_bytes_per_sync"] =
        static_cast<double>(queue_kernarg.payload_bytes);
    state.counters["queue_kernarg_reserved_bytes_per_sync"] =
        static_cast<double>(queue_kernarg.reserved_bytes);
    state.counters["queue_submissions_per_sync"] = 1.0;
    iree_hal_amdgpu_benchmark_set_completion_wait_counters(state);
    state.SetItemsProcessed(state.iterations() * operation_count);
  }

  iree_status_t BindingCountDispatchSubmitPublicFinalInline(
      int64_t binding_count, SubmittedCompletion* out_completion) {
    uint64_t completion_payload_value = ++completion_payload_value_;
    iree_hal_semaphore_t* signal_semaphore = completion_semaphore_;
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&completion_payload_value,
    };
    IREE_RETURN_IF_ERROR(SubmitBindingCountDispatchWithLists(
        kQueue0, iree_hal_semaphore_list_empty(), signal_semaphore_list,
        binding_count));
    *out_completion = {completion_semaphore_, completion_payload_value};
    return iree_ok_status();
  }

  iree_status_t BindingCountCommandBufferSubmitPublicFinalInline(
      int64_t binding_count, SubmittedCompletion* out_completion) {
    uint64_t completion_payload_value = ++completion_payload_value_;
    iree_hal_semaphore_t* signal_semaphore = completion_semaphore_;
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&completion_payload_value,
    };
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
        device_, kQueue0, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, binding_count_command_buffers_[binding_count],
        iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
    *out_completion = {completion_semaphore_, completion_payload_value};
    return iree_ok_status();
  }

  iree_status_t BindingCountDispatchChainCommandBufferSubmitPublicFinalInline(
      iree_hal_command_buffer_t* command_buffer,
      SubmittedCompletion* out_completion) {
    uint64_t completion_payload_value = ++completion_payload_value_;
    iree_hal_semaphore_t* signal_semaphore = completion_semaphore_;
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&completion_payload_value,
    };
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
        device_, kQueue0, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, command_buffer,
        iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
    *out_completion = {completion_semaphore_, completion_payload_value};
    return iree_ok_status();
  }

 private:
  static iree_const_byte_span_t FindExecutableData(
      const iree_file_toc_t* toc, iree_string_view_t file_name) {
    for (iree_host_size_t i = 0; toc[i].name != nullptr; ++i) {
      if (iree_string_view_equal(file_name,
                                 iree_make_cstring_view(toc[i].name))) {
        return iree_make_const_byte_span(
            reinterpret_cast<const uint8_t*>(toc[i].data), toc[i].size);
      }
    }
    return iree_const_byte_span_empty();
  }

  static iree_const_byte_span_t FindCtsExecutableData(
      iree_string_view_t file_name) {
    return FindExecutableData(iree_cts_testdata_amdgpu_create(), file_name);
  }

  static iree_const_byte_span_t FindQueueBenchmarkExecutableData(
      iree_string_view_t file_name) {
    return FindExecutableData(iree_queue_benchmark_testdata_amdgpu_create(),
                              file_name);
  }

  bool AllocatePayloadBuffers(benchmark::State& state) {
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
    iree_hal_buffer_params_t params = {0};
    params.usage =
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.min_alignment = kPayloadBufferAlignment;

    if (!HandleStatus(
            state,
            iree_hal_allocator_allocate_buffer(
                allocator, params, kPayloadBufferAlignment, &source_buffer_),
            "failed to allocate source payload buffer")) {
      return false;
    }
    if (!HandleStatus(
            state,
            iree_hal_allocator_allocate_buffer(
                allocator, params, kPayloadBufferAlignment, &target_buffer_),
            "failed to allocate target payload buffer")) {
      return false;
    }

    uint8_t source_pattern = 0x5A;
    if (!HandleStatus(state,
                      FillBufferAndWait(source_buffer_, &source_pattern,
                                        sizeof(source_pattern)),
                      "failed to initialize source payload buffer")) {
      return false;
    }
    uint8_t target_pattern = 0x00;
    return HandleStatus(state,
                        FillBufferAndWait(target_buffer_, &target_pattern,
                                          sizeof(target_pattern)),
                        "failed to initialize target payload buffer");
  }

  bool CreatePublicSemaphore(benchmark::State& state,
                             iree_hal_semaphore_t** out_semaphore) {
    return HandleStatus(state,
                        iree_hal_semaphore_create(
                            device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                            /*initial_value=*/0,
                            IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_semaphore),
                        "failed to create semaphore");
  }

  bool CreatePrivateStreamSemaphore(benchmark::State& state,
                                    iree_hal_semaphore_t** out_semaphore) {
    return HandleStatus(
        state,
        iree_hal_semaphore_create(device_, kQueue0 | kQueue1,
                                  /*initial_value=*/0,
                                  IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
                                      IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER,
                                  out_semaphore),
        "failed to create private stream semaphore");
  }

  static bool initialized_;
  static bool available_;
  static iree_allocator_t host_allocator_;
  static iree_hal_driver_t* driver_;
  static iree_hal_device_group_t* device_group_;
  static iree_hal_device_t* device_;
  // Executable cache used for the CTS-derived tiny dispatch benchmark payload.
  static iree_hal_executable_cache_t* dispatch_executable_cache_;
  // CTS-derived tiny dispatch executable shared by dispatch benchmark rows.
  static iree_hal_executable_t* dispatch_executable_;
  // Executable cache used for binding-count dispatch benchmark exports.
  static iree_hal_executable_cache_t* binding_count_executable_cache_;
  // Empty-kernel executable with exports that vary only in ABI binding count.
  static iree_hal_executable_t* binding_count_executable_;
  // Optional file contents backing an externally loaded binding-count HSACO.
  static iree_io_file_contents_t* binding_count_executable_file_contents_;

  // Precomputed dispatch packet body used by direct-substrate attribution rows.
  iree_hsa_kernel_dispatch_packet_t pre_resolved_dispatch_packet_template_ = {};
  // Precomputed kernarg bytes used by direct-substrate attribution rows.
  uint8_t* pre_resolved_dispatch_kernargs_ = nullptr;
  // Byte length of |pre_resolved_dispatch_kernargs_|.
  iree_host_size_t pre_resolved_dispatch_kernarg_length_ = 0;
  // Queue kernarg-ring block count required by the precomputed kernarg bytes.
  uint32_t pre_resolved_dispatch_kernarg_block_count_ = 0;
  // Small source buffer used by queue copy payload benchmark rows.
  iree_hal_buffer_t* source_buffer_ = nullptr;
  // Small target buffer used by queue copy and fill payload benchmark rows.
  iree_hal_buffer_t* target_buffer_ = nullptr;
  // Unique device buffers passed to binding-count dispatch benchmark rows.
  iree_hal_buffer_t* binding_count_buffers_[kDispatchBindingBenchmarkMaxCount] =
      {};
  // Queue_dispatch binding refs assembled from |binding_count_buffers_|.
  iree_hal_buffer_ref_t
      binding_count_binding_ref_scratch_[kDispatchBindingBenchmarkMaxCount] =
          {};
  // Static direct command buffers for binding-count dispatch benchmark rows.
  iree_hal_command_buffer_t*
      binding_count_command_buffers_[kDispatchBindingBenchmarkVariantCapacity] =
          {};
  // Public semaphore used for final host-observable completion.
  iree_hal_semaphore_t* completion_semaphore_ = nullptr;
  // Private single-producer stream semaphore used by queue 1.
  iree_hal_semaphore_t* stream_semaphore_ = nullptr;
  // Private single-producer stream semaphore used by queue 0.
  iree_hal_semaphore_t* producer_semaphore_ = nullptr;
  // Discard sink used by profiling overhead guardrail rows.
  iree_hal_profile_sink_t* profile_sink_ = nullptr;
  // True while the fixture owns an active HAL profiling session.
  bool profile_session_active_ = false;
  // Next public completion payload value.
  uint64_t completion_payload_value_ = 0;
  // Next private queue 1 stream payload value.
  uint64_t stream_payload_value_ = 0;
  // Next private queue 0 stream payload value.
  uint64_t producer_payload_value_ = 0;
  // Dword fill pattern used by fill payload benchmark rows.
  uint32_t fill_pattern_ = 0xDEADBEEFu;
};

bool QueueBenchmark::initialized_ = false;
bool QueueBenchmark::available_ = false;
iree_allocator_t QueueBenchmark::host_allocator_;
iree_hal_driver_t* QueueBenchmark::driver_ = nullptr;
iree_hal_device_group_t* QueueBenchmark::device_group_ = nullptr;
iree_hal_device_t* QueueBenchmark::device_ = nullptr;
iree_hal_executable_cache_t* QueueBenchmark::dispatch_executable_cache_ =
    nullptr;
iree_hal_executable_t* QueueBenchmark::dispatch_executable_ = nullptr;
iree_hal_executable_cache_t* QueueBenchmark::binding_count_executable_cache_ =
    nullptr;
iree_hal_executable_t* QueueBenchmark::binding_count_executable_ = nullptr;
iree_io_file_contents_t*
    QueueBenchmark::binding_count_executable_file_contents_ = nullptr;

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueueBarrierWait)(benchmark::State& state) {
  for (auto _ : state) {
    if (!HandleStatus(state, SameQueueBarrierAndWait(),
                      "same-queue barrier failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   QueueAllocaWarmTlsfSubmitOnly)(benchmark::State& state) {
  const iree_device_size_t allocation_size =
      static_cast<iree_device_size_t>(state.range(0));
  iree_hal_pool_t* pool = nullptr;
  if (!HandleStatus(state, CreateQueueAllocaTlsfPool(allocation_size, &pool),
                    "failed to create queue_alloca TLSF pool")) {
    return;
  }
  if (!HandleStatus(state, QueueAllocaSubmitAndCleanup(pool, allocation_size),
                    "failed to prewarm queue_alloca TLSF pool")) {
    iree_hal_pool_release(pool);
    return;
  }

  for (auto _ : state) {
    iree_hal_buffer_t* buffer = nullptr;
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            QueueAllocaSubmit(pool, allocation_size, &buffer, &completion),
            "warm queue_alloca submit failed")) {
      break;
    }
    state.PauseTiming();
    const bool cleanup_ok =
        HandleStatus(state, QueueAllocaCleanup(buffer, completion),
                     "warm queue_alloca cleanup failed");
    state.ResumeTiming();
    if (!cleanup_ok) break;
  }

  iree_hal_pool_release(pool);
  SetQueueAllocaCounters(state, allocation_size,
                         QueueAllocaTlsfGrowthMode::kWarm);
}

BENCHMARK_DEFINE_F(QueueBenchmark, QueueAllocaForcedTlsfGrowthSubmitOnly)(
    benchmark::State& state) {
  const iree_device_size_t allocation_size =
      static_cast<iree_device_size_t>(state.range(0));

  for (auto _ : state) {
    state.PauseTiming();
    iree_hal_pool_t* pool = nullptr;
    iree_hal_buffer_t* held_buffer = nullptr;
    SubmittedCompletion held_completion = {};
    iree_status_t status = CreateQueueAllocaTlsfPool(allocation_size, &pool);
    if (iree_status_is_ok(status)) {
      status = QueueAllocaSubmit(pool, allocation_size, &held_buffer,
                                 &held_completion);
    }
    if (iree_status_is_ok(status)) {
      status = Wait(held_completion.semaphore, held_completion.payload_value);
    }
    if (!HandleStatus(state, status,
                      "failed to seed forced-growth queue_alloca pool")) {
      iree_hal_buffer_release(held_buffer);
      iree_hal_pool_release(pool);
      state.ResumeTiming();
      break;
    }

    state.ResumeTiming();
    iree_hal_buffer_t* grown_buffer = nullptr;
    SubmittedCompletion grown_completion;
    status = QueueAllocaSubmit(pool, allocation_size, &grown_buffer,
                               &grown_completion);
    state.PauseTiming();

    bool ok =
        HandleStatus(state, status, "forced-growth queue_alloca submit failed");
    if (ok) {
      ok = HandleStatus(state,
                        QueueAllocaCleanup(grown_buffer, grown_completion),
                        "forced-growth queue_alloca cleanup failed");
    } else {
      iree_hal_buffer_release(grown_buffer);
    }
    ok = HandleStatus(state, QueueAllocaCleanup(held_buffer, held_completion),
                      "forced-growth held alloca cleanup failed") &&
         ok;
    iree_hal_pool_release(pool);
    state.ResumeTiming();
    if (!ok) break;
  }

  SetQueueAllocaCounters(state, allocation_size,
                         QueueAllocaTlsfGrowthMode::kForcedGrowth);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   HostCallBlockingWait)(benchmark::State& state) {
  for (auto _ : state) {
    if (!HandleStatus(state, HostCallAndWait(IREE_HAL_HOST_CALL_FLAG_NONE),
                      "blocking host call failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   HostCallBlockingRelaxedWait)(benchmark::State& state) {
  for (auto _ : state) {
    if (!HandleStatus(state, HostCallAndWait(IREE_HAL_HOST_CALL_FLAG_RELAXED),
                      "relaxed blocking host call failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   HostCallNonBlockingWait)(benchmark::State& state) {
  for (auto _ : state) {
    if (!HandleStatus(state,
                      HostCallAndWait(IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING |
                                      IREE_HAL_HOST_CALL_FLAG_RELAXED),
                      "relaxed nonblocking host call failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   HostCallBlockingBatch20FinalWait)(benchmark::State& state) {
  for (auto _ : state) {
    if (!HandleStatus(
            state,
            HostCallBatchAndWait(IREE_HAL_HOST_CALL_FLAG_NONE, kBatchCount),
            "blocking host call batch failed")) {
      break;
    }
  }
  state.counters["host_functions_per_sync"] = static_cast<double>(kBatchCount);
  SetQueueSubmissionsProcessed(state, kBatchCount);
}

BENCHMARK_DEFINE_F(QueueBenchmark, HostCallBlockingRelaxedBatch20FinalWait)(
    benchmark::State& state) {
  for (auto _ : state) {
    if (!HandleStatus(
            state,
            HostCallBatchAndWait(IREE_HAL_HOST_CALL_FLAG_RELAXED, kBatchCount),
            "relaxed blocking host call batch failed")) {
      break;
    }
  }
  state.counters["host_functions_per_sync"] = static_cast<double>(kBatchCount);
  SetQueueSubmissionsProcessed(state, kBatchCount);
}

BENCHMARK_DEFINE_F(QueueBenchmark, HostCallNonBlockingBatch20FinalWait)(
    benchmark::State& state) {
  for (auto _ : state) {
    if (!HandleStatus(
            state,
            HostCallBatchAndWait(IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING |
                                     IREE_HAL_HOST_CALL_FLAG_RELAXED,
                                 kBatchCount),
            "relaxed nonblocking host call batch failed")) {
      break;
    }
  }
  state.counters["host_functions_per_sync"] = static_cast<double>(kBatchCount);
  SetQueueSubmissionsProcessed(state, kBatchCount);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueueBarrierBatch20FinalWait)(benchmark::State& state) {
  for (auto _ : state) {
    if (!HandleStatus(state, SameQueueBarrierBatchAndWait(kBatchCount),
                      "same-queue barrier batch failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, kBatchCount);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueueBarrierBatchFinalWait)(benchmark::State& state) {
  const int64_t batch_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state, SameQueueBarrierBatchAndWait(batch_count),
                      "same-queue barrier batch failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, batch_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueueBarrierBatchSubmitOnly)(benchmark::State& state) {
  const int64_t batch_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      SameQueueBarrierBatchSubmit(batch_count, &completion),
                      "same-queue barrier batch submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(state, completion,
                              "same-queue barrier batch wait failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, batch_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueueEpochChain20)(benchmark::State& state) {
  for (auto _ : state) {
    if (!HandleStatus(state, SameQueueEpochChainAndWait(kBatchCount),
                      "same-queue epoch chain failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, kBatchCount);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueueEpochChain)(benchmark::State& state) {
  const int64_t batch_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state, SameQueueEpochChainAndWait(batch_count),
                      "same-queue epoch chain failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, batch_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueueEpochChainSubmitOnly)(benchmark::State& state) {
  const int64_t batch_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      SameQueueEpochChainSubmit(batch_count, &completion),
                      "same-queue epoch chain submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(state, completion,
                              "same-queue epoch chain wait failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, batch_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueueAlreadyCompletedWait)(benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!HandleStatus(state, PrimeProducerSemaphore(),
                    "failed to prime producer semaphore")) {
    return;
  }

  for (auto _ : state) {
    if (!HandleStatus(state, CrossQueueAlreadyCompletedWaitAndSignal(),
                      "cross-queue completed wait failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueueBarrierValue)(benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, CrossQueueBarrierValueAndWait(),
                      "cross-queue barrier-value wait failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/2);
}

BENCHMARK_DEFINE_F(QueueBenchmark, CrossQueueBarrierValueBatch20FinalWait)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, CrossQueueBarrierValueBatchAndWait(kBatchCount),
                      "cross-queue barrier-value batch failed")) {
      break;
    }
  }
  state.counters["cross_queue_handoffs_per_sync"] =
      static_cast<double>(kBatchCount);
  SetQueueSubmissionsProcessed(state,
                               /*queue_submissions_per_sync=*/2 * kBatchCount);
}

BENCHMARK_DEFINE_F(QueueBenchmark, CrossQueueBarrierValueBatchFinalWait)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  const int64_t batch_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state, CrossQueueBarrierValueBatchAndWait(batch_count),
                      "cross-queue barrier-value batch failed")) {
      break;
    }
  }
  state.counters["cross_queue_handoffs_per_sync"] =
      static_cast<double>(batch_count);
  SetQueueSubmissionsProcessed(state,
                               /*queue_submissions_per_sync=*/2 * batch_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark, CrossQueueBarrierValueBatchSubmitOnly)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  const int64_t batch_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state, CrossQueueBarrierValueBatchSubmit(batch_count, &completion),
            "cross-queue barrier-value batch submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(state, completion,
                              "cross-queue barrier-value batch wait failed")) {
      break;
    }
  }
  state.counters["cross_queue_handoffs_per_sync"] =
      static_cast<double>(batch_count);
  SetQueueSubmissionsProcessed(state,
                               /*queue_submissions_per_sync=*/2 * batch_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongChain20)(benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, CrossQueuePingPongChainAndWait(kBatchCount),
                      "cross-queue ping-pong chain failed")) {
      break;
    }
  }
  SetCrossQueuePingPongCounters(state, kBatchCount,
                                /*queue_submissions_per_sync=*/1 + kBatchCount,
                                /*public_completion_signals_per_sync=*/0);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongChain)(benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state, CrossQueuePingPongChainAndWait(handoff_count),
                      "cross-queue ping-pong chain failed")) {
      break;
    }
  }
  SetCrossQueuePingPongCounters(
      state, handoff_count, /*queue_submissions_per_sync=*/1 + handoff_count,
      /*public_completion_signals_per_sync=*/0);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongChainSubmitOnly)(benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      CrossQueuePingPongChainSubmit(handoff_count, &completion),
                      "cross-queue ping-pong chain submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(state, completion,
                              "cross-queue ping-pong chain wait failed")) {
      break;
    }
  }
  SetCrossQueuePingPongCounters(
      state, handoff_count, /*queue_submissions_per_sync=*/1 + handoff_count,
      /*public_completion_signals_per_sync=*/0);
}

BENCHMARK_DEFINE_F(QueueBenchmark, CrossQueuePingPongPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(
            state,
            CrossQueuePingPongChainPublicFinalInlineAndWait(handoff_count),
            "cross-queue ping-pong public-final inline chain failed")) {
      break;
    }
  }
  SetCrossQueuePingPongCounters(
      state, handoff_count, /*queue_submissions_per_sync=*/1 + handoff_count,
      /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            CrossQueuePingPongChainSubmitPublicFinalInline(handoff_count,
                                                           &completion),
            "cross-queue ping-pong public-final inline submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "cross-queue ping-pong public-final inline wait failed")) {
      break;
    }
  }
  SetCrossQueuePingPongCounters(
      state, handoff_count, /*queue_submissions_per_sync=*/1 + handoff_count,
      /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark, CrossQueuePingPongPublicFinalSeparate)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(
            state,
            CrossQueuePingPongChainPublicFinalSeparateAndWait(handoff_count),
            "cross-queue ping-pong public-final separate chain failed")) {
      break;
    }
  }
  SetCrossQueuePingPongCounters(
      state, handoff_count, /*queue_submissions_per_sync=*/2 + handoff_count,
      /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongPublicFinalSeparateSubmitOnly)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            CrossQueuePingPongChainSubmitPublicFinalSeparate(handoff_count,
                                                             &completion),
            "cross-queue ping-pong public-final separate submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "cross-queue ping-pong public-final separate wait failed")) {
      break;
    }
  }
  SetCrossQueuePingPongCounters(
      state, handoff_count, /*queue_submissions_per_sync=*/2 + handoff_count,
      /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark, CrossQueuePingPongCopyPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsurePayloadBuffers(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(
            state,
            CrossQueuePingPongPayloadPublicFinalInlineAndWait(
                PayloadKind::kCopy, handoff_count),
            "cross-queue ping-pong copy public-final inline chain failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongCopyPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsurePayloadBuffers(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            CrossQueuePingPongPayloadSubmitPublicFinalInline(
                PayloadKind::kCopy, handoff_count, &completion),
            "cross-queue ping-pong copy public-final inline submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "cross-queue ping-pong copy public-final inline wait failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark, CrossQueuePingPongFillPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsurePayloadBuffers(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(
            state,
            CrossQueuePingPongPayloadPublicFinalInlineAndWait(
                PayloadKind::kFill, handoff_count),
            "cross-queue ping-pong fill public-final inline chain failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongFillPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsurePayloadBuffers(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            CrossQueuePingPongPayloadSubmitPublicFinalInline(
                PayloadKind::kFill, handoff_count, &completion),
            "cross-queue ping-pong fill public-final inline submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "cross-queue ping-pong fill public-final inline wait failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark, CrossQueuePingPongDispatchPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsurePayloadBuffers(state)) return;
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state,
                      CrossQueuePingPongPayloadPublicFinalInlineAndWait(
                          PayloadKind::kDispatch, handoff_count),
                      "cross-queue ping-pong dispatch public-final inline "
                      "chain failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(
    QueueBenchmark,
    CrossQueuePingPongDispatchPublicFinalInlineEpochCompletionFloor)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsurePayloadBuffers(state)) return;
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      CrossQueuePingPongPayloadSubmitPublicFinalInline(
                          PayloadKind::kDispatch, handoff_count, &completion),
                      "cross-queue ping-pong dispatch public-final inline "
                      "submit failed")) {
      break;
    }
    if (!HandleStatus(state, WaitForSubmittedProducerEpoch(completion),
                      "cross-queue ping-pong dispatch producer epoch wait "
                      "failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "cross-queue ping-pong dispatch public-final inline wait failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongDispatchPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsurePayloadBuffers(state)) return;
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      CrossQueuePingPongPayloadSubmitPublicFinalInline(
                          PayloadKind::kDispatch, handoff_count, &completion),
                      "cross-queue ping-pong dispatch public-final inline "
                      "submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "cross-queue ping-pong dispatch public-final inline wait failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongNoopDispatchPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state,
                      CrossQueuePingPongPayloadPublicFinalInlineAndWait(
                          PayloadKind::kNoopDispatch, handoff_count),
                      "cross-queue ping-pong noop dispatch public-final inline "
                      "chain failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongNoopDispatchPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            CrossQueuePingPongPayloadSubmitPublicFinalInline(
                PayloadKind::kNoopDispatch, handoff_count, &completion),
            "cross-queue ping-pong noop dispatch public-final inline "
            "submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "cross-queue ping-pong noop dispatch public-final inline wait "
            "failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   CrossQueuePingPongPreResolvedDispatchPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsurePreResolvedDispatch(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state,
                      CrossQueuePingPongPayloadPublicFinalInlineAndWait(
                          PayloadKind::kPreResolvedDispatch, handoff_count),
                      "cross-queue ping-pong pre-resolved dispatch "
                      "public-final inline chain failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(
    QueueBenchmark,
    CrossQueuePingPongPreResolvedDispatchPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  if (!EnsurePreResolvedDispatch(state)) return;
  const int64_t handoff_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            CrossQueuePingPongPayloadSubmitPublicFinalInline(
                PayloadKind::kPreResolvedDispatch, handoff_count, &completion),
            "cross-queue ping-pong pre-resolved dispatch "
            "public-final inline submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "cross-queue ping-pong pre-resolved dispatch public-final inline "
            "wait failed")) {
      break;
    }
  }
  SetPayloadPingPongCounters(state, handoff_count,
                             /*queue_submissions_per_sync=*/1 + handoff_count,
                             /*public_completion_signals_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueuePrivateStreamCopyChainPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadPublicFinalInlineAndWait(
                          PayloadKind::kCopy, operation_count),
                      "same-queue private-stream copy chain failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueuePrivateStreamCopyChainPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
                          PayloadKind::kCopy, operation_count, &completion),
                      "same-queue private-stream copy submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(state, completion,
                              "same-queue private-stream copy wait failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueuePrivateStreamDispatchChainPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadPublicFinalInlineAndWait(
                          PayloadKind::kDispatch, operation_count),
                      "same-queue private-stream dispatch chain failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(
    QueueBenchmark,
    SameQueuePrivateStreamDispatchChainPublicFinalInlineEpochCompletionFloor)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
                          PayloadKind::kDispatch, operation_count, &completion),
                      "same-queue private-stream dispatch submit failed")) {
      break;
    }
    if (!HandleStatus(state, WaitForSubmittedProducerEpoch(completion),
                      "same-queue private-stream dispatch producer epoch wait "
                      "failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "same-queue private-stream dispatch wait failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(
    QueueBenchmark,
    SameQueuePrivateStreamDispatchChainPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
                          PayloadKind::kDispatch, operation_count, &completion),
                      "same-queue private-stream dispatch submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "same-queue private-stream dispatch wait failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueuePrivateStreamNoopDispatchChainPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadPublicFinalInlineAndWait(
                          PayloadKind::kNoopDispatch, operation_count),
                      "same-queue private-stream noop dispatch chain failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(
    QueueBenchmark,
    SameQueuePrivateStreamNoopDispatchChainPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsureDispatchExecutable(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
                PayloadKind::kNoopDispatch, operation_count, &completion),
            "same-queue private-stream noop dispatch submit "
            "failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "same-queue private-stream noop dispatch wait failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(
    QueueBenchmark,
    SameQueuePrivateStreamPreResolvedDispatchChainPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsurePreResolvedDispatch(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadPublicFinalInlineAndWait(
                          PayloadKind::kPreResolvedDispatch, operation_count),
                      "same-queue private-stream pre-resolved dispatch chain "
                      "failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(
    QueueBenchmark,
    SameQueuePrivateStreamPreResolvedDispatchChainPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsurePreResolvedDispatch(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
                          PayloadKind::kPreResolvedDispatch, operation_count,
                          &completion),
                      "same-queue private-stream pre-resolved dispatch submit "
                      "failed")) {
      break;
    }
    if (!WaitWithTimingPaused(
            state, completion,
            "same-queue private-stream pre-resolved dispatch wait failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueuePrivateStreamFillChainPublicFinalInline)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadPublicFinalInlineAndWait(
                          PayloadKind::kFill, operation_count),
                      "same-queue private-stream fill chain failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueuePrivateStreamFillChainPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  const int64_t operation_count = state.range(0);
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
                          PayloadKind::kFill, operation_count, &completion),
                      "same-queue private-stream fill submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(state, completion,
                              "same-queue private-stream fill wait failed")) {
      break;
    }
  }
  SetSingleStreamPayloadCounters(state, operation_count);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   DispatchValidateOnly)(benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  if (!EnsureDispatchExecutable(state)) return;
  iree_hal_amdgpu_host_queue_t* host_queue = nullptr;
  if (!HandleStatus(state, LookupHostQueue(kQueue0, &host_queue),
                    "failed to find queue 0")) {
    return;
  }

  for (auto _ : state) {
    iree_host_size_t operation_resource_count = 0;
    iree_status_t status =
        ValidateDispatchOnce(host_queue, &operation_resource_count);
    if (!HandleStatus(state, status, "dispatch validation failed")) {
      break;
    }
    benchmark::DoNotOptimize(operation_resource_count);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   DispatchBindingCountValidateOnly)(benchmark::State& state) {
  const int64_t binding_count = state.range(0);
  if (!EnsureBindingCountExecutable(state)) return;
  if (!EnsureBindingCountBuffers(state, binding_count)) return;
  iree_hal_amdgpu_host_queue_t* host_queue = nullptr;
  if (!HandleStatus(state, LookupHostQueue(kQueue0, &host_queue),
                    "failed to find queue 0")) {
    return;
  }

  for (auto _ : state) {
    iree_host_size_t operation_resource_count = 0;
    iree_status_t status = ValidateBindingCountDispatchOnce(
        host_queue, binding_count, &operation_resource_count);
    if (!HandleStatus(state, status,
                      "binding-count dispatch validation failed")) {
      break;
    }
    benchmark::DoNotOptimize(operation_resource_count);
  }
  SetBindingCountCounters(state, binding_count);
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueueDispatchBindingCountPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  const int64_t binding_count = state.range(0);
  if (!EnsureBindingCountExecutable(state)) return;
  if (!EnsureBindingCountBuffers(state, binding_count)) return;
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      BindingCountDispatchSubmitPublicFinalInline(binding_count,
                                                                  &completion),
                      "binding-count dispatch submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(state, completion,
                              "binding-count dispatch wait failed")) {
      break;
    }
  }
  SetBindingCountCounters(state, binding_count);
  SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/1);
}

BENCHMARK_DEFINE_F(
    QueueBenchmark,
    SameQueueCommandBufferBindingCountStaticPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  const int64_t binding_count = state.range(0);
  if (!EnsureBindingCountCommandBuffer(state, binding_count)) return;
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(state,
                      BindingCountCommandBufferSubmitPublicFinalInline(
                          binding_count, &completion),
                      "binding-count command-buffer submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(state, completion,
                              "binding-count command-buffer wait failed")) {
      break;
    }
  }
  SetBindingCountCounters(state, binding_count);
  SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/1);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   SameQueueCommandBufferDispatchChainStaticPublicFinalInline)(
    benchmark::State& state) {
  const int64_t operation_count = state.range(0);
  const int64_t binding_count = state.range(1);
  if (!EnsureBindingCountExecutable(state)) return;
  if (!EnsureBindingCountBuffers(state, binding_count)) return;

  iree_hal_command_buffer_t* command_buffer = nullptr;
  if (!HandleStatus(state,
                    RecordBindingCountDispatchChainCommandBuffer(
                        operation_count, binding_count, &command_buffer),
                    "failed to record command-buffer dispatch chain")) {
    return;
  }
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            BindingCountDispatchChainCommandBufferSubmitPublicFinalInline(
                command_buffer, &completion),
            "command-buffer dispatch chain submit failed")) {
      break;
    }
    if (!HandleStatus(state,
                      Wait(completion.semaphore, completion.payload_value),
                      "command-buffer dispatch chain wait failed")) {
      break;
    }
  }
  SetCommandBufferProgramCounters(state, command_buffer, operation_count,
                                  binding_count);
  iree_hal_command_buffer_release(command_buffer);
}

BENCHMARK_DEFINE_F(QueueBenchmark, ProfileGuardrailBarrierBatch20FinalWait)(
    benchmark::State& state) {
  RunProfileGuardrailFinalWaitBenchmark(
      state, kProfileGuardrailOperationCount, kProfileGuardrailOperationCount,
      "profile guardrail barrier batch flush failed",
      "profile guardrail barrier batch end failed", [&]() {
        return HandleStatus(
            state,
            SameQueueBarrierBatchAndWait(kProfileGuardrailOperationCount),
            "profile guardrail barrier batch failed");
      });
}

BENCHMARK_DEFINE_F(QueueBenchmark, ProfileGuardrailBarrierBatch20SubmitOnly)(
    benchmark::State& state) {
  RunProfileGuardrailSubmitOnlyBenchmark(
      state, kProfileGuardrailOperationCount, kProfileGuardrailOperationCount,
      "profile guardrail barrier batch wait failed",
      "profile guardrail barrier batch flush failed",
      "profile guardrail barrier batch end failed",
      [&](SubmittedCompletion* completion) {
        return HandleStatus(state,
                            SameQueueBarrierBatchSubmit(
                                kProfileGuardrailOperationCount, completion),
                            "profile guardrail barrier batch submit failed");
      });
}

BENCHMARK_DEFINE_F(QueueBenchmark, ProfileGuardrailCopyBatch20FinalWait)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  RunProfileGuardrailFinalWaitBenchmark(
      state, kProfileGuardrailOperationCount, kProfileGuardrailOperationCount,
      "profile guardrail copy batch flush failed",
      "profile guardrail copy batch end failed", [&]() {
        return HandleStatus(
            state,
            SameQueuePrivateStreamPayloadPublicFinalInlineAndWait(
                PayloadKind::kCopy, kProfileGuardrailOperationCount),
            "profile guardrail copy batch failed");
      });
}

BENCHMARK_DEFINE_F(QueueBenchmark, ProfileGuardrailCopyBatch20SubmitOnly)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  RunProfileGuardrailSubmitOnlyBenchmark(
      state, kProfileGuardrailOperationCount, kProfileGuardrailOperationCount,
      "profile guardrail copy batch wait failed",
      "profile guardrail copy batch flush failed",
      "profile guardrail copy batch end failed",
      [&](SubmittedCompletion* completion) {
        return HandleStatus(
            state,
            SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
                PayloadKind::kCopy, kProfileGuardrailOperationCount,
                completion),
            "profile guardrail copy batch submit failed");
      });
}

BENCHMARK_DEFINE_F(QueueBenchmark, ProfileGuardrailFillBatch20FinalWait)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  RunProfileGuardrailFinalWaitBenchmark(
      state, kProfileGuardrailOperationCount, kProfileGuardrailOperationCount,
      "profile guardrail fill batch flush failed",
      "profile guardrail fill batch end failed", [&]() {
        return HandleStatus(
            state,
            SameQueuePrivateStreamPayloadPublicFinalInlineAndWait(
                PayloadKind::kFill, kProfileGuardrailOperationCount),
            "profile guardrail fill batch failed");
      });
}

BENCHMARK_DEFINE_F(QueueBenchmark, ProfileGuardrailFillBatch20SubmitOnly)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  RunProfileGuardrailSubmitOnlyBenchmark(
      state, kProfileGuardrailOperationCount, kProfileGuardrailOperationCount,
      "profile guardrail fill batch wait failed",
      "profile guardrail fill batch flush failed",
      "profile guardrail fill batch end failed",
      [&](SubmittedCompletion* completion) {
        return HandleStatus(
            state,
            SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
                PayloadKind::kFill, kProfileGuardrailOperationCount,
                completion),
            "profile guardrail fill batch submit failed");
      });
}

BENCHMARK_DEFINE_F(QueueBenchmark, ProfileGuardrailDispatchBatch20FinalWait)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  if (!EnsureDispatchExecutable(state)) return;
  RunProfileGuardrailFinalWaitBenchmark(
      state, kProfileGuardrailOperationCount, kProfileGuardrailOperationCount,
      "profile guardrail dispatch batch flush failed",
      "profile guardrail dispatch batch end failed", [&]() {
        return HandleStatus(
            state,
            SameQueuePrivateStreamPayloadPublicFinalInlineAndWait(
                PayloadKind::kDispatch, kProfileGuardrailOperationCount),
            "profile guardrail dispatch batch failed");
      });
}

BENCHMARK_DEFINE_F(QueueBenchmark, ProfileGuardrailDispatchBatch20SubmitOnly)(
    benchmark::State& state) {
  if (!EnsurePayloadBuffers(state)) return;
  if (!EnsureDispatchExecutable(state)) return;
  RunProfileGuardrailSubmitOnlyBenchmark(
      state, kProfileGuardrailOperationCount, kProfileGuardrailOperationCount,
      "profile guardrail dispatch batch wait failed",
      "profile guardrail dispatch batch flush failed",
      "profile guardrail dispatch batch end failed",
      [&](SubmittedCompletion* completion) {
        return HandleStatus(
            state,
            SameQueuePrivateStreamPayloadSubmitPublicFinalInline(
                PayloadKind::kDispatch, kProfileGuardrailOperationCount,
                completion),
            "profile guardrail dispatch batch submit failed");
      });
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   ProfileGuardrailCommandBufferDispatchChain20FinalWait)(
    benchmark::State& state) {
  if (!EnsureBindingCountExecutable(state)) return;
  if (!EnsureBindingCountBuffers(state, kProfileGuardrailBindingCount)) return;

  iree_hal_command_buffer_t* command_buffer = nullptr;
  if (!HandleStatus(state,
                    RecordBindingCountDispatchChainCommandBuffer(
                        kProfileGuardrailOperationCount,
                        kProfileGuardrailBindingCount, &command_buffer),
                    "failed to record profile guardrail command buffer")) {
    return;
  }
  RunProfileGuardrailFinalWaitBenchmark(
      state, /*queue_submissions_per_sync=*/1, kProfileGuardrailOperationCount,
      "profile guardrail command-buffer flush failed",
      "profile guardrail command-buffer end failed", [&]() {
        SubmittedCompletion completion;
        if (!HandleStatus(
                state,
                BindingCountDispatchChainCommandBufferSubmitPublicFinalInline(
                    command_buffer, &completion),
                "profile guardrail command-buffer submit failed")) {
          return false;
        }
        return HandleStatus(
            state, Wait(completion.semaphore, completion.payload_value),
            "profile guardrail command-buffer wait failed");
      });
  iree_hal_command_buffer_release(command_buffer);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   ProfileGuardrailCommandBufferDispatchChain20SubmitOnly)(
    benchmark::State& state) {
  if (!EnsureBindingCountExecutable(state)) return;
  if (!EnsureBindingCountBuffers(state, kProfileGuardrailBindingCount)) return;

  iree_hal_command_buffer_t* command_buffer = nullptr;
  if (!HandleStatus(state,
                    RecordBindingCountDispatchChainCommandBuffer(
                        kProfileGuardrailOperationCount,
                        kProfileGuardrailBindingCount, &command_buffer),
                    "failed to record profile guardrail command buffer")) {
    return;
  }
  RunProfileGuardrailSubmitOnlyBenchmark(
      state, /*queue_submissions_per_sync=*/1, kProfileGuardrailOperationCount,
      "profile guardrail command-buffer wait failed",
      "profile guardrail command-buffer flush failed",
      "profile guardrail command-buffer end failed",
      [&](SubmittedCompletion* completion) {
        return HandleStatus(
            state,
            BindingCountDispatchChainCommandBufferSubmitPublicFinalInline(
                command_buffer, completion),
            "profile guardrail command-buffer submit failed");
      });
  iree_hal_command_buffer_release(command_buffer);
}

BENCHMARK_DEFINE_F(
    QueueBenchmark,
    SameQueueCommandBufferDispatchChainStaticPublicFinalInlineSubmitOnly)(
    benchmark::State& state) {
  const int64_t operation_count = state.range(0);
  const int64_t binding_count = state.range(1);
  if (!EnsureBindingCountExecutable(state)) return;
  if (!EnsureBindingCountBuffers(state, binding_count)) return;

  iree_hal_command_buffer_t* command_buffer = nullptr;
  if (!HandleStatus(state,
                    RecordBindingCountDispatchChainCommandBuffer(
                        operation_count, binding_count, &command_buffer),
                    "failed to record command-buffer dispatch chain")) {
    return;
  }
  for (auto _ : state) {
    SubmittedCompletion completion;
    if (!HandleStatus(
            state,
            BindingCountDispatchChainCommandBufferSubmitPublicFinalInline(
                command_buffer, &completion),
            "command-buffer dispatch chain submit failed")) {
      break;
    }
    if (!WaitWithTimingPaused(state, completion,
                              "command-buffer dispatch chain wait failed")) {
      break;
    }
  }
  SetCommandBufferProgramCounters(state, command_buffer, operation_count,
                                  binding_count);
  iree_hal_command_buffer_release(command_buffer);
}

BENCHMARK_DEFINE_F(QueueBenchmark,
                   WaitBeforeSignalChain)(benchmark::State& state) {
  if (!EnsureQueueAvailable(state, kQueue1)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, WaitBeforeSignalChainAndWait(),
                      "wait-before-signal chain failed")) {
      break;
    }
  }
  SetQueueSubmissionsProcessed(state, /*queue_submissions_per_sync=*/2);
}

void ApplyProfileGuardrailModes(benchmark::Benchmark* benchmark) {
  benchmark->ArgName("profile_mode");
  benchmark->Arg((int64_t)ProfileGuardrailMode::kDisabled);
  benchmark->Arg((int64_t)ProfileGuardrailMode::kQueueDeviceEvents);
  benchmark->Arg((int64_t)ProfileGuardrailMode::kDispatchEvents);
  benchmark->Arg((int64_t)ProfileGuardrailMode::kQueueDeviceAndDispatchEvents);
  benchmark->Iterations(kProfileGuardrailIterations);
}

BENCHMARK_REGISTER_F(QueueBenchmark, SameQueueBarrierWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, QueueAllocaWarmTlsfSubmitOnly)
    ->Arg(4096)
    ->Arg(65536)
    ->ArgName("allocation_size")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, QueueAllocaForcedTlsfGrowthSubmitOnly)
    ->Arg(4096)
    ->Arg(65536)
    ->ArgName("allocation_size")
    ->Iterations(100)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, HostCallBlockingWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, HostCallBlockingRelaxedWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, HostCallNonBlockingWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, HostCallBlockingBatch20FinalWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, HostCallBlockingRelaxedBatch20FinalWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, HostCallNonBlockingBatch20FinalWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, SameQueueBarrierBatch20FinalWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, SameQueueBarrierBatchFinalWait)
    ->Arg(20)
    ->Arg(100)
    ->Arg(1000)
    ->ArgName("batch_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, SameQueueBarrierBatchSubmitOnly)
    ->Arg(20)
    ->Arg(100)
    ->Arg(1000)
    ->ArgName("batch_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, SameQueueEpochChain20)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, SameQueueEpochChain)
    ->Arg(20)
    ->Arg(100)
    ->Arg(1000)
    ->ArgName("batch_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, SameQueueEpochChainSubmitOnly)
    ->Arg(20)
    ->Arg(100)
    ->Arg(1000)
    ->ArgName("batch_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueueAlreadyCompletedWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueueBarrierValue)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueueBarrierValueBatch20FinalWait)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueueBarrierValueBatchFinalWait)
    ->Arg(20)
    ->Arg(100)
    ->Arg(1000)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueueBarrierValueBatchSubmitOnly)
    ->Arg(20)
    ->Arg(100)
    ->Arg(1000)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueuePingPongChain20)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueuePingPongChain)
    ->Arg(20)
    ->Arg(100)
    ->Arg(1000)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueuePingPongChainSubmitOnly)
    ->Arg(20)
    ->Arg(100)
    ->Arg(1000)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueuePingPongPublicFinalInline)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     CrossQueuePingPongPublicFinalInlineSubmitOnly)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueuePingPongPublicFinalSeparate)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     CrossQueuePingPongPublicFinalSeparateSubmitOnly)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueuePingPongCopyPublicFinalInline)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     CrossQueuePingPongCopyPublicFinalInlineSubmitOnly)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, CrossQueuePingPongFillPublicFinalInline)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     CrossQueuePingPongFillPublicFinalInlineSubmitOnly)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     CrossQueuePingPongDispatchPublicFinalInline)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(
    QueueBenchmark,
    CrossQueuePingPongDispatchPublicFinalInlineEpochCompletionFloor)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     CrossQueuePingPongDispatchPublicFinalInlineSubmitOnly)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     CrossQueuePingPongNoopDispatchPublicFinalInline)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     CrossQueuePingPongNoopDispatchPublicFinalInlineSubmitOnly)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     CrossQueuePingPongPreResolvedDispatchPublicFinalInline)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(
    QueueBenchmark,
    CrossQueuePingPongPreResolvedDispatchPublicFinalInlineSubmitOnly)
    ->Arg(2)
    ->Arg(32)
    ->Arg(512)
    ->Arg(2048)
    ->ArgName("handoff_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     SameQueuePrivateStreamCopyChainPublicFinalInline)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     SameQueuePrivateStreamCopyChainPublicFinalInlineSubmitOnly)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     SameQueuePrivateStreamDispatchChainPublicFinalInline)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(
    QueueBenchmark,
    SameQueuePrivateStreamDispatchChainPublicFinalInlineEpochCompletionFloor)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(
    QueueBenchmark,
    SameQueuePrivateStreamDispatchChainPublicFinalInlineSubmitOnly)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     SameQueuePrivateStreamNoopDispatchChainPublicFinalInline)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(
    QueueBenchmark,
    SameQueuePrivateStreamNoopDispatchChainPublicFinalInlineSubmitOnly)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(
    QueueBenchmark,
    SameQueuePrivateStreamPreResolvedDispatchChainPublicFinalInline)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(
    QueueBenchmark,
    SameQueuePrivateStreamPreResolvedDispatchChainPublicFinalInlineSubmitOnly)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     SameQueuePrivateStreamFillChainPublicFinalInline)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     SameQueuePrivateStreamFillChainPublicFinalInlineSubmitOnly)
    ->Arg(1)
    ->Arg(20)
    ->Arg(1000)
    ->ArgName("operation_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, DispatchValidateOnly)
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, DispatchBindingCountValidateOnly)
    ->Arg(0)
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(9)
    ->Arg(16)
    ->Arg(17)
    ->Arg(24)
    ->Arg(25)
    ->Arg(256)
    ->ArgName("binding_count")
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     SameQueueDispatchBindingCountPublicFinalInlineSubmitOnly)
    ->Arg(0)
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(9)
    ->Arg(16)
    ->Arg(17)
    ->Arg(24)
    ->Arg(25)
    ->Arg(256)
    ->ArgName("binding_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(
    QueueBenchmark,
    SameQueueCommandBufferBindingCountStaticPublicFinalInlineSubmitOnly)
    ->Arg(0)
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(9)
    ->Arg(16)
    ->Arg(17)
    ->Arg(24)
    ->Arg(25)
    ->Arg(256)
    ->ArgName("binding_count")
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     SameQueueCommandBufferDispatchChainStaticPublicFinalInline)
    ->ArgsProduct({{1, 10, 100, 1000, 5000}, {0, 1, 4, 8, 16}})
    ->ArgNames({"operation_count", "binding_count"})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, ProfileGuardrailBarrierBatch20FinalWait)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, ProfileGuardrailBarrierBatch20SubmitOnly)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, ProfileGuardrailCopyBatch20FinalWait)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, ProfileGuardrailCopyBatch20SubmitOnly)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, ProfileGuardrailFillBatch20FinalWait)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, ProfileGuardrailFillBatch20SubmitOnly)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, ProfileGuardrailDispatchBatch20FinalWait)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, ProfileGuardrailDispatchBatch20SubmitOnly)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     ProfileGuardrailCommandBufferDispatchChain20FinalWait)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark,
                     ProfileGuardrailCommandBufferDispatchChain20SubmitOnly)
    ->Apply(ApplyProfileGuardrailModes)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(
    QueueBenchmark,
    SameQueueCommandBufferDispatchChainStaticPublicFinalInlineSubmitOnly)
    ->ArgsProduct({{1, 10, 100, 1000, 5000}, {0, 1, 4, 8, 16}})
    ->ArgNames({"operation_count", "binding_count"})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(QueueBenchmark, WaitBeforeSignalChain)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  QueueBenchmark::DeinitializeOnce();
  return 0;
}
