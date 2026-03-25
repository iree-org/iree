// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TSAN bridge validation for proactor submit→complete ordering.
//
// Proactor backends may use kernel-mediated mechanisms (shared memory rings,
// completion ports) where the submit→complete ordering is invisible to TSAN's
// userspace instrumentation. The iree_async_operation_t::tsan_bridge atomic
// provides a release/acquire pair that makes this ordering visible.
//
// This test validates the bridge works correctly by reusing operation structs
// across multiple submit/complete cycles. Each cycle writes base struct fields
// (type, completion_fn, user_data) before submit and reads them during
// completion dispatch. Without proper ordering, TSAN reports a data race.

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
template <int kSlotCount>
struct Pool {
  static_assert(kSlotCount <= 32, "kSlotCount must be <= 32");

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
    auto* pool = static_cast<Pool*>(user_data);
    iree_status_ignore(status);

    // Read fields that were written during initialization. These are the
    // accesses that race without proper submit→complete ordering.
    IREE_ASSERT_EQ(operation->type, IREE_ASYNC_OPERATION_TYPE_NOP);
    IREE_ASSERT_EQ(operation->completion_fn, Callback);
    IREE_ASSERT_EQ(operation->user_data, pool);

    auto* nop = reinterpret_cast<iree_async_nop_operation_t*>(operation);
    int index = static_cast<int>(nop - pool->slots);
    IREE_ASSERT(index >= 0 && index < kSlotCount);

    pool->completions.fetch_add(1, std::memory_order_relaxed);
    pool->Release(index);
  }
};

//===----------------------------------------------------------------------===//
// Single-threaded reuse
//===----------------------------------------------------------------------===//

// Reuses operation structs from a single thread. Validates that the proactor
// handles same-thread submit→complete→reinitialize→resubmit without TSAN
// false positives.
TEST_P(TsanBridgeTest, SingleThreadReuse) {
  constexpr int kIterations = 200;
  Pool<8> pool;

  for (int i = 0; i < kIterations; ++i) {
    int index = pool.Claim();
    ASSERT_GE(index, 0) << "Pool exhausted at iteration " << i;

    iree_async_nop_operation_t* nop = &pool.slots[index];
    iree_async_operation_zero(&nop->base, sizeof(*nop));
    iree_async_operation_initialize(&nop->base, IREE_ASYNC_OPERATION_TYPE_NOP,
                                    IREE_ASYNC_OPERATION_FLAG_NONE,
                                    Pool<8>::Callback, &pool);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &nop->base));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(500));
  }

  EXPECT_EQ(pool.completions.load(), kIterations);
}

//===----------------------------------------------------------------------===//
// Multi-threaded reuse
//===----------------------------------------------------------------------===//

// Multiple sender threads submit NOPs to a shared pool while the main thread
// polls. The pool has enough slots to avoid contention (goal is to test the
// TSAN bridge, not stress contention). Different threads write the same
// operation fields across reuse cycles.
TEST_P(TsanBridgeTest, MultiThreadReuse) {
  constexpr int kSenderCount = 4;
  constexpr int kSendsPerThread = 5;
  constexpr int kSlotCount = kSenderCount * 2;  // 2x to reduce contention
  constexpr int kTotalSends = kSenderCount * kSendsPerThread;

  Pool<kSlotCount> pool;
  std::atomic<bool> stop{false};

  auto sender_fn = [&]() {
    for (int j = 0; j < kSendsPerThread; ++j) {
      if (stop.load(std::memory_order_relaxed)) return;

      int index = pool.Claim();
      if (index < 0) {
        // Should not happen with 2x slots, but handle gracefully.
        std::this_thread::yield();
        --j;
        continue;
      }

      iree_async_nop_operation_t* nop = &pool.slots[index];
      iree_async_operation_zero(&nop->base, sizeof(*nop));
      iree_async_operation_initialize(&nop->base, IREE_ASYNC_OPERATION_TYPE_NOP,
                                      IREE_ASYNC_OPERATION_FLAG_NONE,
                                      Pool<kSlotCount>::Callback, &pool);

      iree_status_t status =
          iree_async_proactor_submit_one(proactor_, &nop->base);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        pool.Release(index);
        --j;
        std::this_thread::yield();
      }
    }
  };

  std::vector<std::thread> senders;
  for (int i = 0; i < kSenderCount; ++i) {
    senders.emplace_back(sender_fn);
  }

  // Poll with short timeout until all completions arrive.
  iree_time_t deadline_ns = iree_time_now() + iree_make_duration_ms(5000);
  while (pool.completions.load(std::memory_order_relaxed) < kTotalSends) {
    if (iree_time_now() >= deadline_ns) break;
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_timeout_ms(10), &completed);
    if (!iree_status_is_deadline_exceeded(status)) {
      IREE_ASSERT_OK(status);
    } else {
      iree_status_ignore(status);
    }
  }

  stop.store(true, std::memory_order_relaxed);
  for (auto& sender : senders) {
    sender.join();
  }
  DrainPending();

  EXPECT_EQ(pool.completions.load(), kTotalSends);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(TsanBridgeTest);

}  // namespace iree::async::cts
