// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO(benvanik): move this to the CTS (add a benchmark path ala iree/async/).

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/registration/driver_module.h"
#include "iree/hal/drivers/amdgpu/util/benchmark_flags.h"

namespace {

constexpr iree_device_size_t kBenchmarkBufferAlignment = 16;
constexpr int64_t kBatchCount = 20;
constexpr int64_t kSubmitOnlyIterations = 200;
constexpr uint32_t kFrontierAxisTableCapacity = 256;

class BlitBenchmark : public benchmark::Fixture {
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
    }
  }

  void TearDown(benchmark::State& state) override { ReleaseBuffers(); }

 protected:
  bool PrepareCopy(benchmark::State& state) {
    if (!available_) return false;
    length_ = static_cast<iree_device_size_t>(state.range(0));
    source_offset_ = static_cast<iree_device_size_t>(state.range(1));
    target_offset_ = static_cast<iree_device_size_t>(state.range(2));
    batch_count_ = 1;

    iree_device_size_t required_size = source_offset_ + length_;
    if (target_offset_ + length_ > required_size) {
      required_size = target_offset_ + length_;
    }
    allocation_size_ = iree_device_align(
        required_size + kBenchmarkBufferAlignment, kBenchmarkBufferAlignment);
    return AllocateBenchmarkBuffers(state, /*needs_source=*/true);
  }

  bool PrepareCopyBatch(benchmark::State& state, int64_t batch_count) {
    if (!PrepareCopy(state)) return false;
    batch_count_ = batch_count;
    return true;
  }

  bool PrepareFill(benchmark::State& state) {
    if (!available_) return false;
    length_ = static_cast<iree_device_size_t>(state.range(0));
    target_offset_ = static_cast<iree_device_size_t>(state.range(1));
    pattern_length_ = static_cast<iree_host_size_t>(state.range(2));
    batch_count_ = 1;
    if ((target_offset_ % pattern_length_) != 0 ||
        (length_ % pattern_length_) != 0) {
      state.SkipWithError("fill offset/length are not pattern-aligned");
      return false;
    }

    allocation_size_ =
        iree_device_align(target_offset_ + length_ + kBenchmarkBufferAlignment,
                          kBenchmarkBufferAlignment);
    return AllocateBenchmarkBuffers(state, /*needs_source=*/false);
  }

  bool PrepareFillBatch(benchmark::State& state, int64_t batch_count) {
    if (!PrepareFill(state)) return false;
    batch_count_ = batch_count;
    return true;
  }

  bool PrepareUpdate(benchmark::State& state) {
    if (!available_) return false;
    length_ = static_cast<iree_device_size_t>(state.range(0));
    source_offset_ = static_cast<iree_device_size_t>(state.range(1));
    target_offset_ = static_cast<iree_device_size_t>(state.range(2));
    batch_count_ = 1;

    allocation_size_ =
        iree_device_align(target_offset_ + length_ + kBenchmarkBufferAlignment,
                          kBenchmarkBufferAlignment);
    update_source_.resize(source_offset_ + length_ + kBenchmarkBufferAlignment);
    for (size_t i = 0; i < update_source_.size(); ++i) {
      update_source_[i] = static_cast<uint8_t>(0xA0u + (i & 0x3Fu));
    }
    return AllocateBenchmarkBuffers(state, /*needs_source=*/false);
  }

  bool PrepareUpdateBatch(benchmark::State& state, int64_t batch_count) {
    if (!PrepareUpdate(state)) return false;
    batch_count_ = batch_count;
    return true;
  }

  iree_status_t QueueCopyAndWait() {
    iree_hal_semaphore_t* semaphore = completion_semaphore_;
    uint64_t payload_value = ++completion_payload_value_;
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&semaphore,
        /*payload_values=*/&payload_value,
    };
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_copy(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, source_buffer_, source_offset_, target_buffer_,
        target_offset_, length_, IREE_HAL_COPY_FLAG_NONE));
    return WaitForCompletion(payload_value);
  }

  iree_status_t QueueCopyBatchAndWait() {
    uint64_t payload_value = 0;
    IREE_RETURN_IF_ERROR(QueueCopyBatchSubmit(&payload_value));
    return WaitForCompletion(payload_value);
  }

  iree_status_t QueueCopyBatchSubmit(uint64_t* out_payload_value) {
    iree_hal_semaphore_t* semaphore = completion_semaphore_;
    uint64_t payload_value = completion_payload_value_;
    for (int64_t i = 0; i < batch_count_; ++i) {
      uint64_t wait_payload_value = payload_value;
      uint64_t signal_payload_value = payload_value + 1;
      iree_hal_semaphore_list_t wait_semaphore_list =
          iree_hal_semaphore_list_empty();
      if (i > 0) {
        wait_semaphore_list = {
            /*count=*/1,
            /*semaphores=*/&semaphore,
            /*payload_values=*/&wait_payload_value,
        };
      }
      iree_hal_semaphore_list_t signal_semaphore_list = {
          /*count=*/1,
          /*semaphores=*/&semaphore,
          /*payload_values=*/&signal_payload_value,
      };
      IREE_RETURN_IF_ERROR(iree_hal_device_queue_copy(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
          signal_semaphore_list, source_buffer_, source_offset_, target_buffer_,
          target_offset_, length_, IREE_HAL_COPY_FLAG_NONE));
      payload_value = signal_payload_value;
    }
    completion_payload_value_ = payload_value;
    *out_payload_value = payload_value;
    return iree_ok_status();
  }

  iree_status_t QueueFillAndWait(iree_hal_buffer_t* target_buffer,
                                 iree_device_size_t target_offset,
                                 iree_device_size_t length, const void* pattern,
                                 iree_host_size_t pattern_length) {
    iree_hal_semaphore_t* semaphore = completion_semaphore_;
    uint64_t payload_value = ++completion_payload_value_;
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&semaphore,
        /*payload_values=*/&payload_value,
    };
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, target_buffer, target_offset, length, pattern,
        pattern_length, IREE_HAL_FILL_FLAG_NONE));
    return WaitForCompletion(payload_value);
  }

  iree_status_t QueueBenchmarkFillAndWait() {
    return QueueFillAndWait(target_buffer_, target_offset_, length_,
                            &fill_pattern_, pattern_length_);
  }

  iree_status_t QueueFillBatchAndWait() {
    uint64_t payload_value = 0;
    IREE_RETURN_IF_ERROR(QueueFillBatchSubmit(&payload_value));
    return WaitForCompletion(payload_value);
  }

  iree_status_t QueueFillBatchSubmit(uint64_t* out_payload_value) {
    iree_hal_semaphore_t* semaphore = completion_semaphore_;
    uint64_t payload_value = completion_payload_value_;
    for (int64_t i = 0; i < batch_count_; ++i) {
      uint64_t wait_payload_value = payload_value;
      uint64_t signal_payload_value = payload_value + 1;
      iree_hal_semaphore_list_t wait_semaphore_list =
          iree_hal_semaphore_list_empty();
      if (i > 0) {
        wait_semaphore_list = {
            /*count=*/1,
            /*semaphores=*/&semaphore,
            /*payload_values=*/&wait_payload_value,
        };
      }
      iree_hal_semaphore_list_t signal_semaphore_list = {
          /*count=*/1,
          /*semaphores=*/&semaphore,
          /*payload_values=*/&signal_payload_value,
      };
      IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
          signal_semaphore_list, target_buffer_, target_offset_, length_,
          &fill_pattern_, pattern_length_, IREE_HAL_FILL_FLAG_NONE));
      payload_value = signal_payload_value;
    }
    completion_payload_value_ = payload_value;
    *out_payload_value = payload_value;
    return iree_ok_status();
  }

  iree_status_t QueueUpdateAndWait() {
    iree_hal_semaphore_t* semaphore = completion_semaphore_;
    uint64_t payload_value = ++completion_payload_value_;
    iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&semaphore,
        /*payload_values=*/&payload_value,
    };
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_update(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, update_source_.data(),
        (iree_host_size_t)source_offset_, target_buffer_, target_offset_,
        length_, IREE_HAL_UPDATE_FLAG_NONE));
    return WaitForCompletion(payload_value);
  }

  iree_status_t QueueUpdateBatchAndWait() {
    uint64_t payload_value = 0;
    IREE_RETURN_IF_ERROR(QueueUpdateBatchSubmit(&payload_value));
    return WaitForCompletion(payload_value);
  }

  iree_status_t QueueUpdateBatchSubmit(uint64_t* out_payload_value) {
    iree_hal_semaphore_t* semaphore = completion_semaphore_;
    uint64_t payload_value = completion_payload_value_;
    for (int64_t i = 0; i < batch_count_; ++i) {
      uint64_t wait_payload_value = payload_value;
      uint64_t signal_payload_value = payload_value + 1;
      iree_hal_semaphore_list_t wait_semaphore_list =
          iree_hal_semaphore_list_empty();
      if (i > 0) {
        wait_semaphore_list = {
            /*count=*/1,
            /*semaphores=*/&semaphore,
            /*payload_values=*/&wait_payload_value,
        };
      }
      iree_hal_semaphore_list_t signal_semaphore_list = {
          /*count=*/1,
          /*semaphores=*/&semaphore,
          /*payload_values=*/&signal_payload_value,
      };
      IREE_RETURN_IF_ERROR(iree_hal_device_queue_update(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
          signal_semaphore_list, update_source_.data(),
          (iree_host_size_t)source_offset_, target_buffer_, target_offset_,
          length_, IREE_HAL_UPDATE_FLAG_NONE));
      payload_value = signal_payload_value;
    }
    completion_payload_value_ = payload_value;
    *out_payload_value = payload_value;
    return iree_ok_status();
  }

  iree_status_t WaitForCompletion(uint64_t payload_value) {
    return iree_hal_semaphore_wait(
        completion_semaphore_, payload_value, iree_infinite_timeout(),
        iree_hal_amdgpu_benchmark_completion_wait_flags());
  }

  bool HandleStatus(benchmark::State& state, iree_status_t status,
                    const char* message) {
    if (iree_status_is_ok(status)) return true;
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    state.SkipWithError(message);
    return false;
  }

  void SetBytesProcessed(benchmark::State& state) {
    iree_hal_amdgpu_benchmark_set_completion_wait_counters(state);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            batch_count_ * static_cast<int64_t>(length_));
  }

 private:
  bool AllocateBenchmarkBuffers(benchmark::State& state, bool needs_source) {
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
    iree_hal_buffer_params_t params = {0};
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.min_alignment = kBenchmarkBufferAlignment;

    if (!HandleStatus(state,
                      iree_hal_semaphore_create(
                          device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                          /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                          &completion_semaphore_),
                      "failed to create completion semaphore")) {
      return false;
    }

    if (needs_source) {
      if (!HandleStatus(
              state,
              iree_hal_allocator_allocate_buffer(
                  allocator, params, allocation_size_, &source_buffer_),
              "failed to allocate source buffer")) {
        return false;
      }
    }
    if (!HandleStatus(state,
                      iree_hal_allocator_allocate_buffer(
                          allocator, params, allocation_size_, &target_buffer_),
                      "failed to allocate target buffer")) {
      return false;
    }

    uint8_t source_pattern = 0x5A;
    if (needs_source &&
        !HandleStatus(state,
                      QueueFillAndWait(source_buffer_, /*target_offset=*/0,
                                       allocation_size_, &source_pattern,
                                       sizeof(source_pattern)),
                      "failed to pre-initialize source buffer")) {
      return false;
    }
    uint8_t target_pattern = 0x00;
    return HandleStatus(
        state,
        QueueFillAndWait(target_buffer_, /*target_offset=*/0, allocation_size_,
                         &target_pattern, sizeof(target_pattern)),
        "failed to pre-initialize target buffer");
  }

  void ReleaseBuffers() {
    iree_hal_buffer_release(source_buffer_);
    iree_hal_buffer_release(target_buffer_);
    iree_hal_semaphore_release(completion_semaphore_);
    source_buffer_ = nullptr;
    target_buffer_ = nullptr;
    completion_semaphore_ = nullptr;
    completion_payload_value_ = 0;
    update_source_.clear();
  }

  static bool initialized_;
  static bool available_;
  static iree_allocator_t host_allocator_;
  static iree_hal_driver_t* driver_;
  static iree_hal_device_group_t* device_group_;
  static iree_hal_device_t* device_;

  iree_hal_buffer_t* source_buffer_ = nullptr;
  iree_hal_buffer_t* target_buffer_ = nullptr;
  iree_hal_semaphore_t* completion_semaphore_ = nullptr;
  uint64_t completion_payload_value_ = 0;
  iree_device_size_t allocation_size_ = 0;
  iree_device_size_t length_ = 0;
  iree_device_size_t source_offset_ = 0;
  iree_device_size_t target_offset_ = 0;
  iree_host_size_t pattern_length_ = 1;
  int64_t batch_count_ = 1;
  uint32_t fill_pattern_ = 0xDEADBEEFu;
  std::vector<uint8_t> update_source_;
};

bool BlitBenchmark::initialized_ = false;
bool BlitBenchmark::available_ = false;
iree_allocator_t BlitBenchmark::host_allocator_;
iree_hal_driver_t* BlitBenchmark::driver_ = nullptr;
iree_hal_device_group_t* BlitBenchmark::device_group_ = nullptr;
iree_hal_device_t* BlitBenchmark::device_ = nullptr;

BENCHMARK_DEFINE_F(BlitBenchmark, QueueCopy)(benchmark::State& state) {
  if (!PrepareCopy(state)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, QueueCopyAndWait(), "queue_copy failed")) break;
  }
  SetBytesProcessed(state);
}

BENCHMARK_DEFINE_F(BlitBenchmark, QueueFill)(benchmark::State& state) {
  if (!PrepareFill(state)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, QueueBenchmarkFillAndWait(),
                      "queue_fill failed")) {
      break;
    }
  }
  SetBytesProcessed(state);
}

BENCHMARK_DEFINE_F(BlitBenchmark, QueueCopyBatch20)(benchmark::State& state) {
  if (!PrepareCopyBatch(state, kBatchCount)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, QueueCopyBatchAndWait(),
                      "queue_copy batch failed")) {
      break;
    }
  }
  SetBytesProcessed(state);
}

BENCHMARK_DEFINE_F(BlitBenchmark,
                   QueueCopyBatch20SubmitOnly)(benchmark::State& state) {
  if (!PrepareCopyBatch(state, kBatchCount)) return;
  for (auto _ : state) {
    uint64_t payload_value = 0;
    if (!HandleStatus(state, QueueCopyBatchSubmit(&payload_value),
                      "queue_copy batch submit failed")) {
      break;
    }
    state.PauseTiming();
    iree_status_t status = WaitForCompletion(payload_value);
    state.ResumeTiming();
    if (!HandleStatus(state, status, "queue_copy batch wait failed")) break;
  }
  SetBytesProcessed(state);
}

BENCHMARK_DEFINE_F(BlitBenchmark, QueueFillBatch20)(benchmark::State& state) {
  if (!PrepareFillBatch(state, kBatchCount)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, QueueFillBatchAndWait(),
                      "queue_fill batch failed")) {
      break;
    }
  }
  SetBytesProcessed(state);
}

BENCHMARK_DEFINE_F(BlitBenchmark,
                   QueueFillBatch20SubmitOnly)(benchmark::State& state) {
  if (!PrepareFillBatch(state, kBatchCount)) return;
  for (auto _ : state) {
    uint64_t payload_value = 0;
    if (!HandleStatus(state, QueueFillBatchSubmit(&payload_value),
                      "queue_fill batch submit failed")) {
      break;
    }
    state.PauseTiming();
    iree_status_t status = WaitForCompletion(payload_value);
    state.ResumeTiming();
    if (!HandleStatus(state, status, "queue_fill batch wait failed")) break;
  }
  SetBytesProcessed(state);
}

BENCHMARK_DEFINE_F(BlitBenchmark, QueueUpdate)(benchmark::State& state) {
  if (!PrepareUpdate(state)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, QueueUpdateAndWait(), "queue_update failed")) {
      break;
    }
  }
  SetBytesProcessed(state);
}

BENCHMARK_DEFINE_F(BlitBenchmark, QueueUpdateBatch20)(benchmark::State& state) {
  if (!PrepareUpdateBatch(state, kBatchCount)) return;
  for (auto _ : state) {
    if (!HandleStatus(state, QueueUpdateBatchAndWait(),
                      "queue_update batch failed")) {
      break;
    }
  }
  SetBytesProcessed(state);
}

BENCHMARK_DEFINE_F(BlitBenchmark,
                   QueueUpdateBatch20SubmitOnly)(benchmark::State& state) {
  if (!PrepareUpdateBatch(state, kBatchCount)) return;
  for (auto _ : state) {
    uint64_t payload_value = 0;
    if (!HandleStatus(state, QueueUpdateBatchSubmit(&payload_value),
                      "queue_update batch submit failed")) {
      break;
    }
    state.PauseTiming();
    iree_status_t status = WaitForCompletion(payload_value);
    state.ResumeTiming();
    if (!HandleStatus(state, status, "queue_update batch wait failed")) break;
  }
  SetBytesProcessed(state);
}

void ApplyCopyArguments(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"length", "source_offset", "target_offset"});
  const int64_t common_sizes[] = {
      4,   8,   16,   31,       32,        33,        64,
      128, 256, 1024, 4 * 1024, 16 * 1024, 64 * 1024, 2 * 1024 * 1024,
  };
  const int64_t alignment_cases[][2] = {
      {0, 0},
      {8, 8},
      {4, 4},
      {1, 2},
  };
  for (int64_t length : common_sizes) {
    for (const auto& alignment_case : alignment_cases) {
      benchmark->Args({length, alignment_case[0], alignment_case[1]});
    }
  }
  benchmark->Args({500ll * 1024 * 1024, 0, 0});
  benchmark->Args({1024ll * 1024 * 1024, 0, 0});
}

void ApplyFillArguments(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"length", "target_offset", "pattern_length"});
  const int64_t common_sizes[] = {
      4,  8,  16,  28,  30,   31,       32,        33,        34,
      36, 64, 128, 256, 1024, 4 * 1024, 16 * 1024, 64 * 1024, 2 * 1024 * 1024,
  };
  const int64_t pattern_lengths[] = {1, 2, 4};
  for (int64_t pattern_length : pattern_lengths) {
    for (int64_t length : common_sizes) {
      if ((length % pattern_length) != 0) continue;
      const int64_t offsets[] = {0, pattern_length};
      for (int64_t offset : offsets) {
        benchmark->Args({length, offset, pattern_length});
      }
    }
  }
  benchmark->Args({500ll * 1024 * 1024, 0, 4});
  benchmark->Args({1024ll * 1024 * 1024, 0, 4});
}

void ApplyCopySubmitOnlyArguments(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"length", "source_offset", "target_offset"});
  benchmark->Args({4, 0, 0});
  benchmark->Args({8, 0, 0});
  benchmark->Args({64, 0, 0});
  benchmark->Args({4096, 0, 0});
}

void ApplyFillSubmitOnlyArguments(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"length", "target_offset", "pattern_length"});
  benchmark->Args({4, 0, 4});
  benchmark->Args({64, 0, 4});
  benchmark->Args({4096, 0, 4});
}

void ApplyUpdateArguments(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"length", "source_offset", "target_offset"});
  const int64_t common_sizes[] = {
      4, 8, 16, 31, 32, 33, 64, 128, 256, 1024, 4 * 1024, 16 * 1024, 64 * 1024,
  };
  const int64_t alignment_cases[][2] = {
      {0, 0},
      {3, 0},
      {5, 4},
      {1, 1},
  };
  for (int64_t length : common_sizes) {
    for (const auto& alignment_case : alignment_cases) {
      benchmark->Args({length, alignment_case[0], alignment_case[1]});
    }
  }
}

void ApplyUpdateSubmitOnlyArguments(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"length", "source_offset", "target_offset"});
  benchmark->Args({4, 0, 0});
  benchmark->Args({8, 3, 0});
  benchmark->Args({64, 0, 0});
  benchmark->Args({4096, 0, 0});
}

BENCHMARK_REGISTER_F(BlitBenchmark, QueueCopy)
    ->Apply(ApplyCopyArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BlitBenchmark, QueueFill)
    ->Apply(ApplyFillArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BlitBenchmark, QueueCopyBatch20)
    ->Apply(ApplyCopyArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BlitBenchmark, QueueCopyBatch20SubmitOnly)
    ->Apply(ApplyCopySubmitOnlyArguments)
    ->Iterations(kSubmitOnlyIterations)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BlitBenchmark, QueueFillBatch20)
    ->Apply(ApplyFillArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BlitBenchmark, QueueFillBatch20SubmitOnly)
    ->Apply(ApplyFillSubmitOnlyArguments)
    ->Iterations(kSubmitOnlyIterations)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BlitBenchmark, QueueUpdate)
    ->Apply(ApplyUpdateArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BlitBenchmark, QueueUpdateBatch20)
    ->Apply(ApplyUpdateArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BlitBenchmark, QueueUpdateBatch20SubmitOnly)
    ->Apply(ApplyUpdateSubmitOnlyArguments)
    ->Iterations(kSubmitOnlyIterations)
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
  BlitBenchmark::DeinitializeOnce();
  return 0;
}
