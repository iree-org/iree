// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <cstdint>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/registration/driver_module.h"

namespace {

constexpr int64_t kBatchCount = 20;
constexpr uint32_t kFrontierAxisTableCapacity = 256;
constexpr iree_hal_queue_affinity_t kQueue0 = ((iree_hal_queue_affinity_t)1ull)
                                              << 0;
constexpr iree_hal_queue_affinity_t kQueue1 = ((iree_hal_queue_affinity_t)1ull)
                                              << 1;

class QueueBenchmark : public benchmark::Fixture {
 public:
  static void InitializeOnce() {
    if (initialized_) return;
    initialized_ = true;
    host_allocator_ = iree_allocator_system();

    iree_status_t status = iree_hal_amdgpu_driver_module_register(
        iree_hal_driver_registry_default());
    if (iree_status_is_already_exists(status)) {
      iree_status_ignore(status);
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
    iree_status_ignore(status);
    iree_hal_device_release(device_);
    iree_hal_driver_release(driver_);
    device_ = nullptr;
    driver_ = nullptr;
  }

  static void DeinitializeOnce() {
    if (!initialized_) return;
    iree_hal_device_release(device_);
    iree_hal_device_group_release(device_group_);
    iree_hal_driver_release(driver_);
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

    if (!CreateSemaphore(state, &completion_semaphore_) ||
        !CreateSemaphore(state, &stream_semaphore_) ||
        !CreateSemaphore(state, &producer_semaphore_)) {
      return;
    }
  }

  void TearDown(benchmark::State& state) override {
    iree_hal_semaphore_release(completion_semaphore_);
    iree_hal_semaphore_release(stream_semaphore_);
    iree_hal_semaphore_release(producer_semaphore_);
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

  bool EnsureQueueAvailable(benchmark::State& state,
                            iree_hal_queue_affinity_t queue_affinity) {
    return HandleStatus(state,
                        iree_hal_device_queue_flush(device_, queue_affinity),
                        "queue affinity not available");
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
    return iree_hal_device_queue_barrier(
        device_, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  iree_status_t Wait(iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
    return iree_hal_semaphore_wait(semaphore, payload_value,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_status_t SameQueueBarrierAndWait() {
    uint64_t payload_value = ++completion_payload_value_;
    IREE_RETURN_IF_ERROR(SubmitBarrier(kQueue0, /*wait_semaphore=*/nullptr,
                                       /*wait_payload_value=*/0,
                                       completion_semaphore_, payload_value));
    return Wait(completion_semaphore_, payload_value);
  }

  iree_status_t SameQueueBarrierBatchSubmit(
      int64_t batch_count, SubmittedCompletion* out_completion) {
    uint64_t payload_value = ++completion_payload_value_;
    for (int64_t i = 0; i < batch_count; ++i) {
      IREE_RETURN_IF_ERROR(SubmitBarrier(
          kQueue0, /*wait_semaphore=*/nullptr, /*wait_payload_value=*/0,
          i + 1 == batch_count ? completion_semaphore_ : nullptr,
          payload_value));
    }
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
    const uint64_t final_completion_payload_value = ++completion_payload_value_;
    for (int64_t i = 0; i < batch_count; ++i) {
      const uint64_t producer_payload_value = ++producer_payload_value_;
      IREE_RETURN_IF_ERROR(SubmitBarrier(
          kQueue0, /*wait_semaphore=*/nullptr, /*wait_payload_value=*/0,
          producer_semaphore_, producer_payload_value));
      const bool signal_completion = i + 1 == batch_count;
      IREE_RETURN_IF_ERROR(
          SubmitBarrier(kQueue1, producer_semaphore_, producer_payload_value,
                        signal_completion ? completion_semaphore_ : nullptr,
                        final_completion_payload_value));
    }
    *out_completion = {completion_semaphore_, final_completion_payload_value};
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

  iree_status_t CrossQueuePingPongChainAndWait(int64_t handoff_count) {
    SubmittedCompletion completion;
    IREE_RETURN_IF_ERROR(
        CrossQueuePingPongChainSubmit(handoff_count, &completion));
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
    iree_status_ignore(status);
    state.SkipWithError(message);
    return false;
  }

  void SetQueueSubmissionsProcessed(benchmark::State& state,
                                    int64_t queue_submissions_per_sync) {
    state.counters["queue_submissions_per_sync"] =
        static_cast<double>(queue_submissions_per_sync);
    state.SetItemsProcessed(state.iterations() * queue_submissions_per_sync);
  }

  bool WaitWithTimingPaused(benchmark::State& state,
                            const SubmittedCompletion& completion,
                            const char* message) {
    state.PauseTiming();
    iree_status_t status = Wait(completion.semaphore, completion.payload_value);
    state.ResumeTiming();
    return HandleStatus(state, status, message);
  }

 private:
  bool CreateSemaphore(benchmark::State& state,
                       iree_hal_semaphore_t** out_semaphore) {
    return HandleStatus(
        state,
        iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                                  /*initial_value=*/0,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT |
                                      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL,
                                  out_semaphore),
        "failed to create semaphore");
  }

  static bool initialized_;
  static bool available_;
  static iree_allocator_t host_allocator_;
  static iree_hal_driver_t* driver_;
  static iree_hal_device_group_t* device_group_;
  static iree_hal_device_t* device_;

  iree_hal_semaphore_t* completion_semaphore_ = nullptr;
  iree_hal_semaphore_t* stream_semaphore_ = nullptr;
  iree_hal_semaphore_t* producer_semaphore_ = nullptr;
  uint64_t completion_payload_value_ = 0;
  uint64_t stream_payload_value_ = 0;
  uint64_t producer_payload_value_ = 0;
};

bool QueueBenchmark::initialized_ = false;
bool QueueBenchmark::available_ = false;
iree_allocator_t QueueBenchmark::host_allocator_;
iree_hal_driver_t* QueueBenchmark::driver_ = nullptr;
iree_hal_device_group_t* QueueBenchmark::device_group_ = nullptr;
iree_hal_device_t* QueueBenchmark::device_ = nullptr;

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
  state.counters["cross_queue_handoffs_per_sync"] =
      static_cast<double>(kBatchCount);
  SetQueueSubmissionsProcessed(state,
                               /*queue_submissions_per_sync=*/1 + kBatchCount);
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
  state.counters["cross_queue_handoffs_per_sync"] =
      static_cast<double>(handoff_count);
  SetQueueSubmissionsProcessed(
      state, /*queue_submissions_per_sync=*/1 + handoff_count);
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
  state.counters["cross_queue_handoffs_per_sync"] =
      static_cast<double>(handoff_count);
  SetQueueSubmissionsProcessed(
      state, /*queue_submissions_per_sync=*/1 + handoff_count);
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

BENCHMARK_REGISTER_F(QueueBenchmark, SameQueueBarrierWait)
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
BENCHMARK_REGISTER_F(QueueBenchmark, WaitBeforeSignalChain)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  QueueBenchmark::DeinitializeOnce();
  return 0;
}
