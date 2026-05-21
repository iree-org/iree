// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks for iree_atomic_slist_t characterizing single-threaded operation
// costs and multi-threaded contention behavior.
//
// Single-threaded baselines measure the raw cost of each operation in
// isolation. Multi-threaded benchmarks use Google Benchmark's ->Threads(N)
// to measure how operations scale under contention with increasing thread
// counts on the same list.
//
// Key scenarios:
//   - Push/pop/flush round-trip cost (uncontended)
//   - Fast-path empty check cost (pop/flush on empty list)
//   - Batch push then pop/flush throughput
//   - Write-write contention (N threads pushing)
//   - Mixed contention (N threads doing push+pop)
//   - Producer/consumer (N-1 pushers, 1 popper or flusher)
//   - Empty fast-path under contention (N threads popping/flushing empty)
//
// Run with:
//   iree-bazel-run //runtime/src/iree/base/internal:atomic_slist_benchmark
// Filter:
//   --benchmark_filter='Contention'

#include <algorithm>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/atomic_slist.h"

namespace {

//===----------------------------------------------------------------------===//
// Test entry type
//===----------------------------------------------------------------------===//

struct bench_entry_t {
  int value;
  iree_atomic_slist_intrusive_ptr_t slist_next;
};
IREE_TYPED_ATOMIC_SLIST_WRAPPER(bench, bench_entry_t,
                                offsetof(bench_entry_t, slist_next));

// Returns the number of hardware threads, clamped to at least 1.
static int HardwareConcurrency() {
  int count = static_cast<int>(std::thread::hardware_concurrency());
  return count > 0 ? count : 1;
}

// Skips benchmarks that request more threads than the machine has cores.
// Running N threads on M << N cores produces meaningless contention noise.
static bool ShouldSkipThreadCount(benchmark::State& state) {
  if (state.threads() > HardwareConcurrency()) {
    state.SkipWithMessage("thread count exceeds available cores");
    return true;
  }
  return false;
}

// Thread counts to sweep for contention benchmarks. Powers of two up to 256,
// filtered at runtime by ShouldSkipThreadCount.
static void ThreadRange(::benchmark::Benchmark* benchmark) {
  for (int threads = 1; threads <= 256; threads *= 2) {
    benchmark->Threads(threads);
  }
}

// Fixed iteration count for intrusive-entry contention benchmarks. Each push
// needs a unique entry until the shared list is drained at teardown; otherwise
// repeatedly pushing the same intrusive node would corrupt the list.
static constexpr benchmark::IterationCount kContentionIterations = 10000;

//===----------------------------------------------------------------------===//
// Single-threaded baselines
//===----------------------------------------------------------------------===//

// Push then pop a single entry. Measures the uncontended round-trip cost
// through the mutex (lock, store head, unlock) x2.
void BM_PushPop(benchmark::State& state) {
  bench_slist_t list;
  bench_slist_initialize(&list);
  bench_entry_t entry = {42, nullptr};

  for (auto _ : state) {
    bench_slist_push(&list, &entry);
    bench_entry_t* popped = bench_slist_pop(&list);
    benchmark::DoNotOptimize(popped);
  }

  bench_slist_deinitialize(&list);
}
BENCHMARK(BM_PushPop);

// Pop from an empty list. Measures the fast-path relaxed load cost: should
// return without ever touching the mutex.
void BM_PopEmpty(benchmark::State& state) {
  bench_slist_t list;
  bench_slist_initialize(&list);

  for (auto _ : state) {
    bench_entry_t* popped = bench_slist_pop(&list);
    benchmark::DoNotOptimize(popped);
  }

  bench_slist_deinitialize(&list);
}
BENCHMARK(BM_PopEmpty);

// Flush from an empty list. Same fast-path as pop: relaxed load of NULL head
// returns immediately without locking.
void BM_FlushEmpty(benchmark::State& state) {
  bench_slist_t list;
  bench_slist_initialize(&list);

  for (auto _ : state) {
    bench_entry_t* head = nullptr;
    bool flushed = bench_slist_flush(
        &list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &head, nullptr);
    benchmark::DoNotOptimize(flushed);
  }

  bench_slist_deinitialize(&list);
}
BENCHMARK(BM_FlushEmpty);

// Push N entries then pop them all one at a time. Measures throughput of
// sequential pop draining a populated list (N mutex acquisitions).
// The final pop hits the fast-path empty check.
void BM_PushNPopN(benchmark::State& state) {
  const int count = static_cast<int>(state.range(0));
  std::vector<bench_entry_t> entries(count);
  for (int i = 0; i < count; ++i) entries[i].value = i;

  bench_slist_t list;
  bench_slist_initialize(&list);

  for (auto _ : state) {
    for (int i = 0; i < count; ++i) {
      bench_slist_push(&list, &entries[i]);
    }
    for (int i = 0; i < count; ++i) {
      bench_entry_t* popped = bench_slist_pop(&list);
      benchmark::DoNotOptimize(popped);
    }
  }
  state.SetItemsProcessed(state.iterations() * count * 2);

  bench_slist_deinitialize(&list);
}
BENCHMARK(BM_PushNPopN)->Arg(1)->Arg(4)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

// Push N entries then flush them all at once (LIFO order). Measures the
// amortized cost: one mutex acquisition drains the entire list. The flush
// walks the list to find the tail only when out_tail is requested.
void BM_PushNFlushLIFO(benchmark::State& state) {
  const int count = static_cast<int>(state.range(0));
  std::vector<bench_entry_t> entries(count);
  for (int i = 0; i < count; ++i) entries[i].value = i;

  bench_slist_t list;
  bench_slist_initialize(&list);

  for (auto _ : state) {
    for (int i = 0; i < count; ++i) {
      bench_slist_push(&list, &entries[i]);
    }
    bench_entry_t* head = nullptr;
    bench_slist_flush(&list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                      &head, nullptr);
    benchmark::DoNotOptimize(head);
  }
  state.SetItemsProcessed(state.iterations() * count);

  bench_slist_deinitialize(&list);
}
BENCHMARK(BM_PushNFlushLIFO)
    ->Arg(1)
    ->Arg(4)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024);

// Push N entries then flush them all at once (FIFO order). FIFO flush
// reverses the list in-place, touching every entry's next pointer. This
// shows the cost delta between LIFO (O(1) without tail) and FIFO (O(N)
// reversal).
void BM_PushNFlushFIFO(benchmark::State& state) {
  const int count = static_cast<int>(state.range(0));
  std::vector<bench_entry_t> entries(count);
  for (int i = 0; i < count; ++i) entries[i].value = i;

  bench_slist_t list;
  bench_slist_initialize(&list);

  for (auto _ : state) {
    for (int i = 0; i < count; ++i) {
      bench_slist_push(&list, &entries[i]);
    }
    bench_entry_t* head = nullptr;
    bench_slist_flush(&list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                      &head, nullptr);
    benchmark::DoNotOptimize(head);
  }
  state.SetItemsProcessed(state.iterations() * count);

  bench_slist_deinitialize(&list);
}
BENCHMARK(BM_PushNFlushFIFO)
    ->Arg(1)
    ->Arg(4)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024);

// Concat a chain of N entries at once. Measures the cost of a single mutex
// acquisition to prepend an entire chain (O(1) regardless of chain length).
void BM_Concat(benchmark::State& state) {
  const int count = static_cast<int>(state.range(0));
  std::vector<bench_entry_t> entries(count);
  for (int i = 0; i < count; ++i) entries[i].value = i;

  bench_slist_t list;
  bench_slist_initialize(&list);

  for (auto _ : state) {
    // Build chain: link entries[0] -> entries[1] -> ... -> entries[N-1].
    for (int i = 0; i < count - 1; ++i) {
      bench_slist_set_next(&entries[i], &entries[i + 1]);
    }
    bench_slist_set_next(&entries[count - 1], nullptr);

    bench_slist_concat(&list, &entries[0], &entries[count - 1]);

    // Drain so the list is empty for the next iteration.
    bench_entry_t* head = nullptr;
    bench_slist_flush(&list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                      &head, nullptr);
    benchmark::DoNotOptimize(head);
  }
  state.SetItemsProcessed(state.iterations() * count);

  bench_slist_deinitialize(&list);
}
BENCHMARK(BM_Concat)->Arg(1)->Arg(4)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

//===----------------------------------------------------------------------===//
// Multi-threaded contention: write-write
//===----------------------------------------------------------------------===//

// N threads all pushing to the same list. Measures mutex contention on the
// write path. Each thread pushes its own entry (no data sharing beyond the
// list head). After each iteration, entries accumulate in the list; a barrier
// and flush between iterations would add noise, so we accept the growing list
// and report per-operation throughput.
void BM_ContentionPush(benchmark::State& state) {
  if (ShouldSkipThreadCount(state)) return;

  // Shared list, initialized once by thread 0.
  static bench_slist_t list;
  if (state.thread_index() == 0) {
    bench_slist_initialize(&list);
  }

  // Each push uses a unique intrusive entry. The benchmark loop has a start
  // barrier, so thread 0 setup is complete before other threads push.
  std::vector<bench_entry_t> entries(static_cast<size_t>(state.max_iterations));
  size_t entry_index = 0;

  for (auto _ : state) {
    (void)_;
    entries[entry_index].value = static_cast<int>(entry_index);
    bench_slist_push(&list, &entries[entry_index]);
    ++entry_index;
  }

  // The benchmark loop has a stop barrier before teardown, so thread 0 drains
  // only after all pushes have completed.
  if (state.thread_index() == 0) {
    bench_entry_t* head = nullptr;
    bench_slist_flush(&list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                      &head, nullptr);
    bench_slist_deinitialize(&list);
  }
}
BENCHMARK(BM_ContentionPush)
    ->Apply(ThreadRange)
    ->Iterations(kContentionIterations)
    ->UseRealTime();

// N threads each doing push-then-pop cycles on the same list. Measures
// mixed read-write contention. Each thread pushes its own entry then
// immediately pops (which may get its own entry back, or another thread's —
// that's fine, we're measuring contention cost not correctness).
void BM_ContentionPushPop(benchmark::State& state) {
  if (ShouldSkipThreadCount(state)) return;

  static bench_slist_t list;
  if (state.thread_index() == 0) {
    bench_slist_initialize(&list);
  }

  std::vector<bench_entry_t> entries(static_cast<size_t>(state.max_iterations));
  size_t entry_index = 0;

  for (auto _ : state) {
    (void)_;
    entries[entry_index].value = static_cast<int>(entry_index);
    bench_slist_push(&list, &entries[entry_index]);
    ++entry_index;
    bench_entry_t* popped = bench_slist_pop(&list);
    benchmark::DoNotOptimize(popped);
  }

  if (state.thread_index() == 0) {
    bench_entry_t* head = nullptr;
    bench_slist_flush(&list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                      &head, nullptr);
    bench_slist_deinitialize(&list);
  }
}
BENCHMARK(BM_ContentionPushPop)
    ->Apply(ThreadRange)
    ->Iterations(kContentionIterations)
    ->UseRealTime();

//===----------------------------------------------------------------------===//
// Multi-threaded contention: empty fast-path
//===----------------------------------------------------------------------===//

// N threads all popping from an empty list. This is the critical benchmark
// for the fast-path empty check: with the relaxed load, threads should never
// touch the mutex and should scale perfectly. Without it, every pop would
// contend on the mutex even though the list is always empty.
void BM_PopEmptyContention(benchmark::State& state) {
  if (ShouldSkipThreadCount(state)) return;

  static bench_slist_t list;
  if (state.thread_index() == 0) {
    bench_slist_initialize(&list);
  }

  for (auto _ : state) {
    bench_entry_t* popped = bench_slist_pop(&list);
    benchmark::DoNotOptimize(popped);
  }

  if (state.thread_index() == 0) {
    bench_slist_deinitialize(&list);
  }
}
BENCHMARK(BM_PopEmptyContention)->Apply(ThreadRange)->UseRealTime();

// N threads all flushing an empty list. Same fast-path test as PopEmpty
// but for the flush path.
void BM_FlushEmptyContention(benchmark::State& state) {
  if (ShouldSkipThreadCount(state)) return;

  static bench_slist_t list;
  if (state.thread_index() == 0) {
    bench_slist_initialize(&list);
  }

  for (auto _ : state) {
    bench_entry_t* head = nullptr;
    bool flushed = bench_slist_flush(
        &list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &head, nullptr);
    benchmark::DoNotOptimize(flushed);
  }

  if (state.thread_index() == 0) {
    bench_slist_deinitialize(&list);
  }
}
BENCHMARK(BM_FlushEmptyContention)->Apply(ThreadRange)->UseRealTime();

//===----------------------------------------------------------------------===//
// Multi-threaded contention: producer/consumer
//===----------------------------------------------------------------------===//

// (N-1) producers pushing, 1 consumer popping. Simulates the typical
// pattern where worker threads produce results into a shared collection
// list and a coordinator drains them one at a time. The consumer (thread 0)
// may frequently see an empty list, exercising the fast-path.
//
// For the single-thread case, the one thread does push+pop (no point in
// having zero producers or zero consumers).
void BM_ProducerConsumerPop(benchmark::State& state) {
  if (ShouldSkipThreadCount(state)) return;

  static bench_slist_t list;
  if (state.thread_index() == 0) {
    bench_slist_initialize(&list);
  }

  std::vector<bench_entry_t> entries(static_cast<size_t>(state.max_iterations));
  size_t entry_index = 0;
  const bool is_consumer = (state.thread_index() == 0);

  for (auto _ : state) {
    (void)_;
    if (state.threads() == 1 || !is_consumer) {
      entries[entry_index].value = static_cast<int>(entry_index);
      bench_slist_push(&list, &entries[entry_index]);
      ++entry_index;
    }
    if (state.threads() == 1 || is_consumer) {
      bench_entry_t* popped = bench_slist_pop(&list);
      benchmark::DoNotOptimize(popped);
    }
  }

  if (state.thread_index() == 0) {
    bench_entry_t* head = nullptr;
    bench_slist_flush(&list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                      &head, nullptr);
    bench_slist_deinitialize(&list);
  }
}
BENCHMARK(BM_ProducerConsumerPop)
    ->Apply(ThreadRange)
    ->Iterations(kContentionIterations)
    ->UseRealTime();

// (N-1) producers pushing, 1 consumer flushing. Simulates the pattern where
// a coordinator periodically drains the entire list in one shot. This is the
// typical pattern for work-stealing schedulers and batch processing: the
// flush amortizes mutex cost across all accumulated entries.
void BM_ProducerConsumerFlush(benchmark::State& state) {
  if (ShouldSkipThreadCount(state)) return;

  static bench_slist_t list;
  if (state.thread_index() == 0) {
    bench_slist_initialize(&list);
  }

  std::vector<bench_entry_t> entries(static_cast<size_t>(state.max_iterations));
  size_t entry_index = 0;
  const bool is_consumer = (state.thread_index() == 0);

  for (auto _ : state) {
    (void)_;
    if (state.threads() == 1 || !is_consumer) {
      entries[entry_index].value = static_cast<int>(entry_index);
      bench_slist_push(&list, &entries[entry_index]);
      ++entry_index;
    }
    if (state.threads() == 1 || is_consumer) {
      bench_entry_t* head = nullptr;
      bool flushed = bench_slist_flush(
          &list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &head,
          nullptr);
      benchmark::DoNotOptimize(flushed);
    }
  }

  if (state.thread_index() == 0) {
    bench_entry_t* head = nullptr;
    bench_slist_flush(&list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                      &head, nullptr);
    bench_slist_deinitialize(&list);
  }
}
BENCHMARK(BM_ProducerConsumerFlush)
    ->Apply(ThreadRange)
    ->Iterations(kContentionIterations)
    ->UseRealTime();

}  // namespace
