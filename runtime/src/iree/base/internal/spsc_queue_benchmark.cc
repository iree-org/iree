// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/spsc_queue.h"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Manages aligned memory for a queue.
class QueueMemory {
 public:
  explicit QueueMemory(uint32_t capacity)
      : capacity_(capacity), size_(iree_spsc_queue_required_size(capacity)) {
    // Allocate with extra space for cache-line alignment.
    raw_.resize(size_ + 64);
    uintptr_t base = (uintptr_t)raw_.data();
    aligned_ = (void*)((base + 63) & ~(uintptr_t)63);
  }
  void* data() { return aligned_; }
  iree_host_size_t size() const { return size_; }
  uint32_t capacity() const { return capacity_; }

 private:
  uint32_t capacity_;
  iree_host_size_t size_;
  std::vector<uint8_t> raw_;
  void* aligned_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Single-threaded benchmarks (baseline)
//===----------------------------------------------------------------------===//

// Alternating write/read of a single entry — measures per-operation overhead.
static void BM_WriteReadCycle(benchmark::State& state) {
  const size_t entry_size = static_cast<size_t>(state.range(0));
  QueueMemory mem(65536);
  iree_spsc_queue_t queue;
  iree_status_ignore(iree_spsc_queue_initialize(mem.data(), mem.size(),
                                                mem.capacity(), &queue));

  std::vector<uint8_t> write_buffer(entry_size, 0xAB);
  std::vector<uint8_t> read_buffer(entry_size);

  for (auto _ : state) {
    iree_spsc_queue_write(&queue, write_buffer.data(), entry_size);
    iree_host_size_t length = 0;
    iree_spsc_queue_read(&queue, read_buffer.data(), read_buffer.size(),
                         &length);
    benchmark::DoNotOptimize(length);
  }

  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(entry_size));
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WriteReadCycle)->Arg(8)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// Write entries until full — measures sustained write throughput.
static void BM_WriteOnly(benchmark::State& state) {
  const size_t entry_size = static_cast<size_t>(state.range(0));
  const uint32_t capacity = static_cast<uint32_t>(state.range(1));
  QueueMemory mem(capacity);
  iree_spsc_queue_t queue;

  std::vector<uint8_t> write_buffer(entry_size, 0xAB);
  int64_t total_bytes = 0;
  int64_t total_entries = 0;

  for (auto _ : state) {
    state.PauseTiming();
    iree_status_ignore(iree_spsc_queue_initialize(mem.data(), mem.size(),
                                                  mem.capacity(), &queue));
    state.ResumeTiming();

    while (iree_spsc_queue_write(&queue, write_buffer.data(), entry_size)) {
      total_bytes += static_cast<int64_t>(entry_size);
      ++total_entries;
    }
  }

  state.counters["bytes/s"] = benchmark::Counter(
      static_cast<double>(total_bytes), benchmark::Counter::kIsRate);
  state.counters["entries/s"] = benchmark::Counter(
      static_cast<double>(total_entries), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_WriteOnly)
    ->Args({64, 65536})
    ->Args({256, 65536})
    ->Args({1024, 65536})
    ->Args({64, 1048576})
    ->Args({1024, 1048576});

// Read entries from a pre-filled queue — measures sustained read throughput.
static void BM_ReadOnly(benchmark::State& state) {
  const size_t entry_size = static_cast<size_t>(state.range(0));
  const uint32_t capacity = static_cast<uint32_t>(state.range(1));
  QueueMemory mem(capacity);
  iree_spsc_queue_t queue;

  std::vector<uint8_t> write_buffer(entry_size, 0xAB);
  std::vector<uint8_t> read_buffer(entry_size);
  int64_t total_bytes = 0;
  int64_t total_entries = 0;

  for (auto _ : state) {
    state.PauseTiming();
    iree_status_ignore(iree_spsc_queue_initialize(mem.data(), mem.size(),
                                                  mem.capacity(), &queue));
    // Fill the queue.
    while (iree_spsc_queue_write(&queue, write_buffer.data(), entry_size)) {
    }
    state.ResumeTiming();

    iree_host_size_t length = 0;
    while (iree_spsc_queue_read(&queue, read_buffer.data(), read_buffer.size(),
                                &length)) {
      total_bytes += static_cast<int64_t>(length);
      ++total_entries;
      benchmark::DoNotOptimize(length);
    }
  }

  state.counters["bytes/s"] = benchmark::Counter(
      static_cast<double>(total_bytes), benchmark::Counter::kIsRate);
  state.counters["entries/s"] = benchmark::Counter(
      static_cast<double>(total_entries), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_ReadOnly)
    ->Args({64, 65536})
    ->Args({256, 65536})
    ->Args({1024, 65536})
    ->Args({64, 1048576})
    ->Args({1024, 1048576});

//===----------------------------------------------------------------------===//
// Two-threaded benchmarks (producer-consumer)
//===----------------------------------------------------------------------===//

// Sustained throughput: one producer, one consumer, measure bytes/second.
// Args: [0] = entry_size, [1] = ring capacity
static void BM_Throughput(benchmark::State& state) {
  const size_t entry_size = static_cast<size_t>(state.range(0));
  const uint32_t capacity = static_cast<uint32_t>(state.range(1));
  QueueMemory mem(capacity);
  iree_spsc_queue_t queue;
  iree_status_ignore(iree_spsc_queue_initialize(mem.data(), mem.size(),
                                                mem.capacity(), &queue));

  std::vector<uint8_t> write_buffer(entry_size, 0xCD);
  std::vector<uint8_t> read_buffer(entry_size);

  // The benchmark framework calls the loop body from a single thread.
  // We use an inner loop with a fixed message count per iteration to allow
  // the consumer to run on a separate thread.
  const int64_t messages_per_iteration = 10000;
  std::atomic<bool> stop{false};
  std::atomic<int64_t> consumed_bytes{0};
  std::atomic<int64_t> consumed_entries{0};

  // Consumer thread.
  std::thread consumer([&]() {
    iree_host_size_t length = 0;
    while (!stop.load(std::memory_order_acquire)) {
      if (iree_spsc_queue_read(&queue, read_buffer.data(), read_buffer.size(),
                               &length)) {
        consumed_bytes.fetch_add(static_cast<int64_t>(length),
                                 std::memory_order_relaxed);
        consumed_entries.fetch_add(1, std::memory_order_relaxed);
      }
    }
    // Drain remaining.
    while (iree_spsc_queue_read(&queue, read_buffer.data(), read_buffer.size(),
                                &length)) {
      consumed_bytes.fetch_add(static_cast<int64_t>(length),
                               std::memory_order_relaxed);
      consumed_entries.fetch_add(1, std::memory_order_relaxed);
    }
  });

  for (auto _ : state) {
    for (int64_t i = 0; i < messages_per_iteration; ++i) {
      while (!iree_spsc_queue_write(&queue, write_buffer.data(), entry_size)) {
        // Spin until space.
      }
    }
  }

  stop.store(true, std::memory_order_release);
  consumer.join();

  state.SetBytesProcessed(consumed_bytes.load(std::memory_order_relaxed));
  state.SetItemsProcessed(consumed_entries.load(std::memory_order_relaxed));
}
BENCHMARK(BM_Throughput)
    ->UseRealTime()
    ->Args({64, 65536})
    ->Args({256, 65536})
    ->Args({1024, 65536})
    ->Args({4096, 65536})
    ->Args({64, 262144})
    ->Args({256, 262144})
    ->Args({1024, 262144})
    ->Args({64, 1048576})
    ->Args({1024, 1048576});

// Zero-copy write throughput: begin_write + commit_write.
static void BM_ZeroCopyWriteReadCycle(benchmark::State& state) {
  const size_t entry_size = static_cast<size_t>(state.range(0));
  QueueMemory mem(65536);
  iree_spsc_queue_t queue;
  iree_status_ignore(iree_spsc_queue_initialize(mem.data(), mem.size(),
                                                mem.capacity(), &queue));

  for (auto _ : state) {
    void* payload = iree_spsc_queue_begin_write(&queue, entry_size);
    if (payload) {
      memset(payload, 0xAB, entry_size);
      iree_spsc_queue_commit_write(&queue, entry_size);

      iree_host_size_t length = 0;
      const void* peek_data = iree_spsc_queue_peek(&queue, &length);
      benchmark::DoNotOptimize(peek_data);
      iree_spsc_queue_consume(&queue);
    }
  }

  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(entry_size));
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ZeroCopyWriteReadCycle)
    ->Arg(8)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096);

}  // namespace

BENCHMARK_MAIN();
