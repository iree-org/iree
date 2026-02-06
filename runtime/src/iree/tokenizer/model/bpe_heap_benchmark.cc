// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/tokenizer/model/bpe_heap.h"

// BPE heap benchmarks for measuring min-heap performance.
// This establishes baseline performance for potential d-ary heap experiments.
//
// The BPE algorithm uses a min-heap to track merge candidates ordered by rank.
// During encoding, the heap experiences a mix of pushes (when new token pairs
// form) and pops (when checking/applying merges). Most pops encounter stale
// entries that are discarded, so pop efficiency is critical.

namespace {

//===----------------------------------------------------------------------===//
// Heap Benchmark Fixture
//===----------------------------------------------------------------------===//

class HeapBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    capacity_ = state.range(0);
    entries_.resize(capacity_);
    iree_tokenizer_bpe_heap_initialize(&heap_, entries_.data(), capacity_);
  }

  void TearDown(benchmark::State&) override {
    entries_.clear();
    entries_.shrink_to_fit();
  }

 protected:
  iree_tokenizer_bpe_heap_t heap_ = {};
  std::vector<iree_tokenizer_bpe_heap_entry_t> entries_;
  iree_host_size_t capacity_ = 0;
};

//===----------------------------------------------------------------------===//
// Fill and Drain Benchmark
//===----------------------------------------------------------------------===//

// Fills the heap to capacity, then drains it completely.
// This measures raw push/pop throughput without the interleaving that occurs
// in actual BPE encoding.
BENCHMARK_DEFINE_F(HeapBenchmark, FillAndDrain)(benchmark::State& state) {
  for (auto _ : state) {
    iree_tokenizer_bpe_heap_reset(&heap_);

    // Fill with pseudo-random ranks to avoid best/worst case behavior.
    for (uint32_t i = 0; i < capacity_; ++i) {
      iree_tokenizer_bpe_heap_entry_t entry = {
          .rank = (i * 7919) % (uint32_t)capacity_,
          .left_start_byte = i,
      };
      iree_tokenizer_bpe_heap_push(&heap_, entry);
    }

    // Drain completely, consuming all entries.
    while (!iree_tokenizer_bpe_heap_is_empty(&heap_)) {
      iree_tokenizer_bpe_heap_entry_t entry =
          iree_tokenizer_bpe_heap_pop(&heap_);
      benchmark::DoNotOptimize(entry);
    }
  }
  // Each iteration does capacity pushes + capacity pops.
  state.SetItemsProcessed(state.iterations() * capacity_ * 2);
}

BENCHMARK_REGISTER_F(HeapBenchmark, FillAndDrain)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond);

//===----------------------------------------------------------------------===//
// BPE Pattern Benchmark
//===----------------------------------------------------------------------===//

// Simulates BPE merge pattern: push 2 new candidates, pop 1 to check.
// This models the interleaved push/pop behavior during actual encoding where
// each merge creates up to 2 new merge candidates (with left and right
// neighbors) and the algorithm pops to find the next best merge.
BENCHMARK_DEFINE_F(HeapBenchmark, BPEPattern)(benchmark::State& state) {
  // Calculate expected ops outside the timed loop for accurate reporting.
  // Pattern: capacity/2 iterations of (push 2, pop 1) = 3 ops each.
  // Then drain remaining ~capacity/2 entries.
  // Total: (capacity/2) * 3 + capacity/2 = 2 * capacity ops.
  iree_host_size_t expected_ops = capacity_ * 2;

  for (auto _ : state) {
    iree_tokenizer_bpe_heap_reset(&heap_);

    // Simulate a segment with capacity/2 initial tokens being merged.
    iree_host_size_t iterations = capacity_ / 2;
    for (uint32_t i = 0; i < iterations; ++i) {
      // Each merge creates 2 new candidates (left pair + right pair).
      iree_tokenizer_bpe_heap_entry_t entry1 = {
          .rank = (i * 7919) % 1000,
          .left_start_byte = i * 2,
      };
      iree_tokenizer_bpe_heap_entry_t entry2 = {
          .rank = (i * 7927) % 1000,
          .left_start_byte = i * 2 + 1,
      };
      iree_tokenizer_bpe_heap_push(&heap_, entry1);
      iree_tokenizer_bpe_heap_push(&heap_, entry2);

      // Pop to get the next merge candidate.
      if (!iree_tokenizer_bpe_heap_is_empty(&heap_)) {
        iree_tokenizer_bpe_heap_entry_t entry =
            iree_tokenizer_bpe_heap_pop(&heap_);
        benchmark::DoNotOptimize(entry);
      }
    }

    // Drain remaining entries (end of segment).
    while (!iree_tokenizer_bpe_heap_is_empty(&heap_)) {
      iree_tokenizer_bpe_heap_entry_t entry =
          iree_tokenizer_bpe_heap_pop(&heap_);
      benchmark::DoNotOptimize(entry);
    }
  }
  state.SetItemsProcessed(state.iterations() * expected_ops);
}

BENCHMARK_REGISTER_F(HeapBenchmark, BPEPattern)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond);

//===----------------------------------------------------------------------===//
// Push-Heavy Benchmark
//===----------------------------------------------------------------------===//

// Measures push performance when heap is already partially filled.
// In BPE, pushes happen after each merge when new token pairs form.
BENCHMARK_DEFINE_F(HeapBenchmark, PushHeavy)(benchmark::State& state) {
  // Pre-fill heap to 50% capacity.
  iree_host_size_t prefill = capacity_ / 2;
  for (uint32_t i = 0; i < prefill; ++i) {
    iree_tokenizer_bpe_heap_entry_t entry = {
        .rank = (i * 7919) % (uint32_t)capacity_,
        .left_start_byte = i,
    };
    iree_tokenizer_bpe_heap_push(&heap_, entry);
  }

  uint32_t counter = (uint32_t)prefill;
  for (auto _ : state) {
    // Push one entry, pop one to keep size stable.
    iree_tokenizer_bpe_heap_entry_t entry = {
        .rank = (counter * 7919) % (uint32_t)capacity_,
        .left_start_byte = counter,
    };
    iree_tokenizer_bpe_heap_push(&heap_, entry);
    iree_tokenizer_bpe_heap_entry_t popped =
        iree_tokenizer_bpe_heap_pop(&heap_);
    benchmark::DoNotOptimize(popped);
    counter++;
  }

  state.SetItemsProcessed(state.iterations() * 2);  // 1 push + 1 pop
}

BENCHMARK_REGISTER_F(HeapBenchmark, PushHeavy)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kNanosecond);

}  // namespace
