// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TSAN stress test for proactor submit→complete ordering.
//
// Proactor backends may use kernel-mediated mechanisms (shared memory rings,
// completion ports) where the submit→complete ordering is invisible to TSAN's
// userspace instrumentation. The iree_async_operation_t::tsan_bridge atomic
// provides a release/acquire pair that makes this ordering visible.
//
// This test exercises the pattern that exposes missing bridges: a fixed pool
// of operation structs is reused across many submit/complete cycles from
// multiple threads. Each cycle writes base struct fields (type, completion_fn,
// user_data) before submit and reads them during completion dispatch. Without
// proper ordering, TSAN reports a data race on these fields.
//
// Proactor-only — no carriers, no sockets, no network code. NOP operations
// exercise the full submit→complete pipeline with the same TSAN bridge path
// as all other operation types.

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/internal/math.h"

namespace iree::async::cts {

class TsanBridgeTest : public CtsTestBase<> {};

//===----------------------------------------------------------------------===//
// Operation pool with bitmap-based free tracking
//===----------------------------------------------------------------------===//

// Fixed-size pool of NOP operations with atomic bitmap free tracking.
// Mimics the carrier send slot pattern: claim via CAS, release via OR.
struct OperationPool {
  static constexpr int kSlotCount = 8;

  iree_async_nop_operation_t slots[kSlotCount];

  // Bitmap: bit i set = slot i is free. All slots start free.
  std::atomic<uint32_t> free_bitmap{(1u << kSlotCount) - 1};

  // Total completions observed (for test termination).
  std::atomic<int> completions{0};

  // Claims a free slot, returns index or -1 if none available.
  int Claim() {
    uint32_t bitmap = free_bitmap.load(std::memory_order_acquire);
    while (bitmap != 0) {
      int index = iree_math_count_trailing_zeros_u32(bitmap);
      uint32_t cleared = bitmap & ~(1u << index);
      if (free_bitmap.compare_exchange_weak(bitmap, cleared,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return index;
      }
    }
    return -1;
  }

  // Releases a slot back to the pool.
  void Release(int index) {
    free_bitmap.fetch_or(1u << index, std::memory_order_release);
  }

  // Completion callback: reads operation fields (the TSAN-sensitive part),
  // then releases the slot back to the pool for reuse.
  static void CompletionCallback(void* user_data,
                                 iree_async_operation_t* operation,
                                 iree_status_t status,
                                 iree_async_completion_flags_t flags) {
    auto* pool = static_cast<OperationPool*>(user_data);
    iree_status_ignore(status);

    // Read fields that were written during initialization. These are the
    // accesses that race without proper submit→complete ordering visible
    // to TSAN.
    IREE_ASSERT_EQ(operation->type, IREE_ASYNC_OPERATION_TYPE_NOP);
    IREE_ASSERT_EQ(operation->completion_fn, CompletionCallback);
    IREE_ASSERT_EQ(operation->user_data, pool);

    // Find which slot this is by pointer arithmetic.
    auto* nop = reinterpret_cast<iree_async_nop_operation_t*>(operation);
    int index = static_cast<int>(nop - pool->slots);
    IREE_ASSERT(index >= 0 && index < kSlotCount);

    pool->completions.fetch_add(1, std::memory_order_relaxed);
    pool->Release(index);
  }
};

// 2-slot pool for maximum reuse frequency in high-contention tests.
// Defined at namespace scope because C++ disallows static constexpr members
// in local structs.
struct TinyPool {
  static constexpr int kSlotCount = 2;
  iree_async_nop_operation_t slots[kSlotCount];
  std::atomic<uint32_t> free_bitmap{(1u << kSlotCount) - 1};
  std::atomic<int> completions{0};

  int Claim() {
    uint32_t bitmap = free_bitmap.load(std::memory_order_acquire);
    while (bitmap != 0) {
      int index = iree_math_count_trailing_zeros_u32(bitmap);
      uint32_t cleared = bitmap & ~(1u << index);
      if (free_bitmap.compare_exchange_weak(bitmap, cleared,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return index;
      }
    }
    return -1;
  }

  void Release(int index) {
    free_bitmap.fetch_or(1u << index, std::memory_order_release);
  }

  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    auto* pool = static_cast<TinyPool*>(user_data);
    iree_status_ignore(status);
    IREE_ASSERT_EQ(operation->type, IREE_ASYNC_OPERATION_TYPE_NOP);
    auto* nop = reinterpret_cast<iree_async_nop_operation_t*>(operation);
    int index = static_cast<int>(nop - pool->slots);
    pool->completions.fetch_add(1, std::memory_order_relaxed);
    pool->Release(index);
  }
};

//===----------------------------------------------------------------------===//
// Single-threaded reuse stress
//===----------------------------------------------------------------------===//

// Rapidly reuses operation structs from a single thread. Validates that the
// proactor handles same-thread submit→complete→reinitialize→resubmit without
// TSAN false positives.
TEST_P(TsanBridgeTest, SingleThreadReuse) {
  constexpr int kIterations = 200;

  OperationPool pool;

  for (int i = 0; i < kIterations; ++i) {
    int index = pool.Claim();
    ASSERT_GE(index, 0) << "Pool exhausted at iteration " << i;

    iree_async_nop_operation_t* nop = &pool.slots[index];
    iree_async_operation_zero(&nop->base, sizeof(*nop));
    iree_async_operation_initialize(&nop->base, IREE_ASYNC_OPERATION_TYPE_NOP,
                                    IREE_ASYNC_OPERATION_FLAG_NONE,
                                    OperationPool::CompletionCallback, &pool);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &nop->base));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(500));
  }

  EXPECT_EQ(pool.completions.load(), kIterations);
}

//===----------------------------------------------------------------------===//
// Multi-threaded reuse stress
//===----------------------------------------------------------------------===//

// Sender thread function: claim slot, initialize, submit, repeat.
// Uses a simple index+bitmap protocol compatible with both pool types.
struct SenderArgs {
  iree_async_nop_operation_t* slots;
  std::atomic<uint32_t>* free_bitmap;
  std::atomic<int>* completions;
  iree_async_proactor_t* proactor;
  iree_async_completion_fn_t callback;
  void* callback_user_data;
  int sends_per_thread;
  std::atomic<bool>* stop;
};

static void SenderThreadFn(SenderArgs args) {
  for (int j = 0; j < args.sends_per_thread; ++j) {
    if (args.stop->load(std::memory_order_relaxed)) return;

    // Spin until a slot is available or stop is requested.
    int index;
    for (;;) {
      if (args.stop->load(std::memory_order_relaxed)) return;
      uint32_t bitmap = args.free_bitmap->load(std::memory_order_acquire);
      while (bitmap != 0) {
        index = iree_math_count_trailing_zeros_u32(bitmap);
        uint32_t cleared = bitmap & ~(1u << index);
        if (args.free_bitmap->compare_exchange_weak(
                bitmap, cleared, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          goto claimed;
        }
      }
      std::this_thread::yield();
    }
  claimed:

    iree_async_nop_operation_t* nop = &args.slots[index];
    iree_async_operation_zero(&nop->base, sizeof(*nop));
    iree_async_operation_initialize(&nop->base, IREE_ASYNC_OPERATION_TYPE_NOP,
                                    IREE_ASYNC_OPERATION_FLAG_NONE,
                                    args.callback, args.callback_user_data);

    iree_status_t status =
        iree_async_proactor_submit_one(args.proactor, &nop->base);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      args.free_bitmap->fetch_or(1u << index, std::memory_order_release);
      --j;
      std::this_thread::yield();
    }
  }
}

// Polls until the expected number of completions arrive or timeout.
static void PollUntilComplete(iree_async_proactor_t* proactor,
                              std::atomic<int>& completions,
                              int expected_total) {
  iree_time_t deadline_ns = iree_time_now() + iree_make_duration_ms(30000);
  while (completions.load(std::memory_order_relaxed) < expected_total) {
    if (iree_time_now() >= deadline_ns) break;
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor, iree_make_timeout_ms(10), &completed);
    if (!iree_status_is_deadline_exceeded(status)) {
      IREE_ASSERT_OK(status);
    } else {
      iree_status_ignore(status);
    }
  }
}

// Multiple sender threads contend for a shared operation pool, submitting NOP
// operations that the main thread polls. Different threads write the same
// operation fields (type, completion_fn) across reuse cycles, with only the
// proactor's internal mechanism mediating the ordering.
TEST_P(TsanBridgeTest, MultiThreadReuse) {
  constexpr int kSenderCount = 4;
  constexpr int kSendsPerThread = 10;
  constexpr int kTotalSends = kSenderCount * kSendsPerThread;

  OperationPool pool;
  std::atomic<bool> stop{false};

  SenderArgs args = {pool.slots,
                     &pool.free_bitmap,
                     &pool.completions,
                     proactor_,
                     OperationPool::CompletionCallback,
                     &pool,
                     kSendsPerThread,
                     &stop};

  std::vector<std::thread> senders;
  for (int i = 0; i < kSenderCount; ++i) {
    senders.emplace_back(SenderThreadFn, args);
  }

  PollUntilComplete(proactor_, pool.completions, kTotalSends);

  stop.store(true, std::memory_order_relaxed);
  for (auto& sender : senders) {
    sender.join();
  }
  DrainPending();

  EXPECT_EQ(pool.completions.load(), kTotalSends);
}

//===----------------------------------------------------------------------===//
// High-frequency reuse from two threads targeting same slots
//===----------------------------------------------------------------------===//

// Two sender threads aggressively contend for a tiny pool (2 slots), forcing
// maximum reuse frequency. This maximizes the chance of TSAN false positives
// if the proactor's submit→complete ordering bridge is insufficient.
TEST_P(TsanBridgeTest, TwoThreadsTwoSlots) {
  constexpr int kSendsPerThread = 20;
  constexpr int kTotalSends = 2 * kSendsPerThread;

  TinyPool pool;
  std::atomic<bool> stop{false};

  SenderArgs args = {pool.slots,         &pool.free_bitmap,
                     &pool.completions,  proactor_,
                     TinyPool::Callback, &pool,
                     kSendsPerThread,    &stop};

  std::thread sender_a(SenderThreadFn, args);
  std::thread sender_b(SenderThreadFn, args);

  PollUntilComplete(proactor_, pool.completions, kTotalSends);

  stop.store(true, std::memory_order_relaxed);
  sender_a.join();
  sender_b.join();
  DrainPending();

  EXPECT_EQ(pool.completions.load(), kTotalSends);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(TsanBridgeTest);

}  // namespace iree::async::cts
