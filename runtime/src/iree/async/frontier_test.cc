// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Helper to allocate a frontier on the stack with a given capacity.
// The returned pointer is valid for the lifetime of the enclosing scope.
#define FRONTIER_ALLOC(name, capacity)                                  \
  alignas(16) uint8_t                                                   \
      name##_storage[sizeof(iree_async_frontier_t) +                    \
                     (capacity) * sizeof(iree_async_frontier_entry_t)]; \
  iree_async_frontier_t* name =                                         \
      reinterpret_cast<iree_async_frontier_t*>(name##_storage);         \
  memset(name##_storage, 0, sizeof(name##_storage))

// Helper to build a frontier from a list of (axis, epoch) pairs.
// Assumes entries are provided in sorted axis order.
static iree_async_frontier_t* BuildFrontier(
    uint8_t* storage, iree_host_size_t storage_size,
    std::initializer_list<iree_async_frontier_entry_t> entries) {
  iree_async_frontier_t* frontier =
      reinterpret_cast<iree_async_frontier_t*>(storage);
  iree_async_frontier_initialize(frontier,
                                 static_cast<uint8_t>(entries.size()));
  uint8_t i = 0;
  for (const auto& entry : entries) {
    frontier->entries[i++] = entry;
  }
  return frontier;
}

#define MAKE_FRONTIER(name, capacity, ...)                              \
  alignas(16) uint8_t                                                   \
      name##_storage[sizeof(iree_async_frontier_t) +                    \
                     (capacity) * sizeof(iree_async_frontier_entry_t)]; \
  iree_async_frontier_t* name =                                         \
      BuildFrontier(name##_storage, sizeof(name##_storage), {__VA_ARGS__})

// Shorthand for creating frontier entries.
static iree_async_frontier_entry_t E(iree_async_axis_t axis, uint64_t epoch) {
  return {axis, epoch};
}

// Axes for testing: use simple values that are easy to reason about.
// Session=1, Machine=0, Domain=QUEUE, Device=0, Queue=0..7.
static iree_async_axis_t TestQueueAxis(uint8_t queue_index) {
  return iree_async_axis_make_queue(1, 0, 0, queue_index);
}

// Session=1, Machine=1, Domain=QUEUE, Device=0, Queue=0..7.
static iree_async_axis_t RemoteQueueAxis(uint8_t queue_index) {
  return iree_async_axis_make_queue(1, 1, 0, queue_index);
}

// Session=1, Machine=0, Domain=COLLECTIVE, Channel=N.
static iree_async_axis_t CollectiveAxis(uint64_t channel_id) {
  return iree_async_axis_make(1, 0, IREE_ASYNC_CAUSAL_DOMAIN_COLLECTIVE,
                              channel_id);
}

//===----------------------------------------------------------------------===//
// Axis construction/extraction round-trips
//===----------------------------------------------------------------------===//

TEST(AxisTest, QueueAxisRoundTrip) {
  iree_async_axis_t axis = iree_async_axis_make_queue(3, 7, 2, 5);
  EXPECT_EQ(iree_async_axis_session(axis), 3);
  EXPECT_EQ(iree_async_axis_machine(axis), 7);
  EXPECT_EQ(iree_async_axis_domain(axis), IREE_ASYNC_CAUSAL_DOMAIN_QUEUE);
  EXPECT_EQ(iree_async_axis_device_index(axis), 2);
  EXPECT_EQ(iree_async_axis_queue_index(axis), 5);
}

TEST(AxisTest, QueueAxisMaxValues) {
  iree_async_axis_t axis = iree_async_axis_make_queue(255, 255, 255, 255);
  EXPECT_EQ(iree_async_axis_session(axis), 255);
  EXPECT_EQ(iree_async_axis_machine(axis), 255);
  EXPECT_EQ(iree_async_axis_domain(axis), IREE_ASYNC_CAUSAL_DOMAIN_QUEUE);
  EXPECT_EQ(iree_async_axis_device_index(axis), 255);
  EXPECT_EQ(iree_async_axis_queue_index(axis), 255);
}

TEST(AxisTest, CollectiveAxisRoundTrip) {
  iree_async_axis_t axis =
      iree_async_axis_make(2, 5, IREE_ASYNC_CAUSAL_DOMAIN_COLLECTIVE, 42);
  EXPECT_EQ(iree_async_axis_session(axis), 2);
  EXPECT_EQ(iree_async_axis_machine(axis), 5);
  EXPECT_EQ(iree_async_axis_domain(axis), IREE_ASYNC_CAUSAL_DOMAIN_COLLECTIVE);
  EXPECT_EQ(iree_async_axis_ordinal(axis), 42u);
}

TEST(AxisTest, HostAxisRoundTrip) {
  iree_async_axis_t axis =
      iree_async_axis_make(1, 0, IREE_ASYNC_CAUSAL_DOMAIN_HOST, 7);
  EXPECT_EQ(iree_async_axis_session(axis), 1);
  EXPECT_EQ(iree_async_axis_machine(axis), 0);
  EXPECT_EQ(iree_async_axis_domain(axis), IREE_ASYNC_CAUSAL_DOMAIN_HOST);
  EXPECT_EQ(iree_async_axis_ordinal(axis), 7u);
}

TEST(AxisTest, EpilogueAxisRoundTrip) {
  iree_async_axis_t axis =
      iree_async_axis_make(1, 3, IREE_ASYNC_CAUSAL_DOMAIN_EPILOGUE, 99);
  EXPECT_EQ(iree_async_axis_session(axis), 1);
  EXPECT_EQ(iree_async_axis_machine(axis), 3);
  EXPECT_EQ(iree_async_axis_domain(axis), IREE_ASYNC_CAUSAL_DOMAIN_EPILOGUE);
  EXPECT_EQ(iree_async_axis_ordinal(axis), 99u);
}

TEST(AxisTest, OrdinalMasking) {
  // Ordinal is 40 bits — upper bits must not leak.
  uint64_t large_ordinal = 0xFFFFFFFFFFull;  // 40-bit max.
  iree_async_axis_t axis =
      iree_async_axis_make(1, 0, IREE_ASYNC_CAUSAL_DOMAIN_HOST, large_ordinal);
  EXPECT_EQ(iree_async_axis_ordinal(axis), large_ordinal);

  // If we pass a value wider than 40 bits, it gets masked.
  uint64_t too_large = 0x1FFFFFFFFFFull;  // 41 bits.
  axis = iree_async_axis_make(1, 0, IREE_ASYNC_CAUSAL_DOMAIN_HOST, too_large);
  EXPECT_EQ(iree_async_axis_ordinal(axis), 0xFFFFFFFFFFull);
}

TEST(AxisTest, SortingOrder) {
  // Axes with lower session/machine/domain/ordinal sort first.
  iree_async_axis_t a = iree_async_axis_make_queue(1, 0, 0, 0);
  iree_async_axis_t b = iree_async_axis_make_queue(1, 0, 0, 1);
  iree_async_axis_t c = iree_async_axis_make_queue(1, 0, 1, 0);
  iree_async_axis_t d = iree_async_axis_make_queue(1, 1, 0, 0);
  EXPECT_LT(a, b);
  EXPECT_LT(b, c);
  EXPECT_LT(c, d);
}

//===----------------------------------------------------------------------===//
// Validate
//===----------------------------------------------------------------------===//

TEST(ValidateTest, EmptyFrontierIsValid) {
  FRONTIER_ALLOC(f, 0);
  iree_async_frontier_initialize(f, 0);
  IREE_EXPECT_OK(iree_async_frontier_validate(f));
}

TEST(ValidateTest, SingleEntryValid) {
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 5));
  IREE_EXPECT_OK(iree_async_frontier_validate(f));
}

TEST(ValidateTest, TwoEntriesSorted) {
  MAKE_FRONTIER(f, 2, E(TestQueueAxis(0), 3), E(TestQueueAxis(1), 7));
  IREE_EXPECT_OK(iree_async_frontier_validate(f));
}

TEST(ValidateTest, TwoEntriesUnsorted) {
  MAKE_FRONTIER(f, 2, E(TestQueueAxis(1), 3), E(TestQueueAxis(0), 7));
  iree_status_t status = iree_async_frontier_validate(f);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
}

TEST(ValidateTest, DuplicateAxes) {
  MAKE_FRONTIER(f, 2, E(TestQueueAxis(0), 3), E(TestQueueAxis(0), 7));
  iree_status_t status = iree_async_frontier_validate(f);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
}

TEST(ValidateTest, ZeroEpoch) {
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 0));
  iree_status_t status = iree_async_frontier_validate(f);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
}

TEST(ValidateTest, TenEntriesAllValid) {
  FRONTIER_ALLOC(f, 10);
  iree_async_frontier_initialize(f, 10);
  for (uint8_t i = 0; i < 10; ++i) {
    f->entries[i] = E(TestQueueAxis(i), i + 1);
  }
  IREE_EXPECT_OK(iree_async_frontier_validate(f));
}

TEST(ValidateTest, TenEntriesUnsortedMiddle) {
  FRONTIER_ALLOC(f, 10);
  iree_async_frontier_initialize(f, 10);
  for (uint8_t i = 0; i < 10; ++i) {
    f->entries[i] = E(TestQueueAxis(i), i + 1);
  }
  // Swap entries 4 and 5 to create unsorted pair in the middle.
  iree_async_frontier_entry_t tmp = f->entries[4];
  f->entries[4] = f->entries[5];
  f->entries[5] = tmp;
  iree_status_t status = iree_async_frontier_validate(f);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
}

//===----------------------------------------------------------------------===//
// Compare — slow path (different axis sets)
//===----------------------------------------------------------------------===//

TEST(CompareTest, BothEmpty) {
  FRONTIER_ALLOC(a, 0);
  FRONTIER_ALLOC(b, 0);
  iree_async_frontier_initialize(a, 0);
  iree_async_frontier_initialize(b, 0);
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_EQUAL);
}

TEST(CompareTest, OneEmptyOneNonEmpty) {
  FRONTIER_ALLOC(a, 0);
  iree_async_frontier_initialize(a, 0);
  MAKE_FRONTIER(b, 1, E(TestQueueAxis(0), 5));
  // Empty frontier is BEFORE any non-empty frontier.
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_BEFORE);
  EXPECT_EQ(iree_async_frontier_compare(b, a), IREE_ASYNC_FRONTIER_AFTER);
}

TEST(CompareTest, DisjointAxes) {
  MAKE_FRONTIER(a, 1, E(TestQueueAxis(0), 5));
  MAKE_FRONTIER(b, 1, E(TestQueueAxis(1), 3));
  // Each has an axis the other doesn't — concurrent.
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_CONCURRENT);
}

TEST(CompareTest, SubsetAxesBefore) {
  MAKE_FRONTIER(a, 1, E(TestQueueAxis(0), 5));
  MAKE_FRONTIER(b, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 3));
  // a is missing axis 1 (implicit epoch 0), b has it at 3 → b is ahead.
  // Both have axis 0 at 5 (equal). So a is BEFORE b.
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_BEFORE);
}

TEST(CompareTest, SupersetAxesAfter) {
  MAKE_FRONTIER(a, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 3));
  MAKE_FRONTIER(b, 1, E(TestQueueAxis(0), 5));
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_AFTER);
}

TEST(CompareTest, PartialOverlapConcurrent) {
  // a has axes 0,2 — b has axes 1,2.
  // Shared axis 2: a is ahead (epoch 10 vs 5).
  // a has axis 0 (b doesn't) → a ahead. b has axis 1 (a doesn't) → b ahead.
  // Result: concurrent (both ahead on some axis).
  MAKE_FRONTIER(a, 2, E(TestQueueAxis(0), 3), E(TestQueueAxis(2), 10));
  MAKE_FRONTIER(b, 2, E(TestQueueAxis(1), 7), E(TestQueueAxis(2), 5));
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_CONCURRENT);
}

//===----------------------------------------------------------------------===//
// Compare — fast path (identical axis sets)
//===----------------------------------------------------------------------===//

TEST(CompareTest, FastPathEqual) {
  MAKE_FRONTIER(a, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  MAKE_FRONTIER(b, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_EQUAL);
}

TEST(CompareTest, FastPathBefore) {
  MAKE_FRONTIER(a, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  MAKE_FRONTIER(b, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 12));
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_BEFORE);
}

TEST(CompareTest, FastPathAfter) {
  MAKE_FRONTIER(a, 2, E(TestQueueAxis(0), 7), E(TestQueueAxis(1), 10));
  MAKE_FRONTIER(b, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_AFTER);
}

TEST(CompareTest, FastPathConcurrent) {
  MAKE_FRONTIER(a, 2, E(TestQueueAxis(0), 7), E(TestQueueAxis(1), 3));
  MAKE_FRONTIER(b, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_CONCURRENT);
}

TEST(CompareTest, FastPathEightEntries) {
  FRONTIER_ALLOC(a, 8);
  FRONTIER_ALLOC(b, 8);
  iree_async_frontier_initialize(a, 8);
  iree_async_frontier_initialize(b, 8);
  for (uint8_t i = 0; i < 8; ++i) {
    a->entries[i] = E(TestQueueAxis(i), 100);
    b->entries[i] = E(TestQueueAxis(i), 100);
  }
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_EQUAL);

  // One axis differs → BEFORE.
  b->entries[3].epoch = 101;
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_BEFORE);

  // Two axes differ in opposite directions → CONCURRENT.
  a->entries[5].epoch = 200;
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_CONCURRENT);
}

TEST(CompareTest, FastPathDetectionBoundary) {
  // Same entry count but different axes — must fall through to slow path.
  MAKE_FRONTIER(a, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  MAKE_FRONTIER(b, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(2), 10));
  // a has axis 1, b has axis 2 — concurrent.
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_CONCURRENT);
}

//===----------------------------------------------------------------------===//
// Compare — property tests
//===----------------------------------------------------------------------===//

TEST(CompareTest, Reflexivity) {
  MAKE_FRONTIER(f, 3, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10),
                E(TestQueueAxis(2), 15));
  EXPECT_EQ(iree_async_frontier_compare(f, f), IREE_ASYNC_FRONTIER_EQUAL);
}

TEST(CompareTest, Antisymmetry) {
  MAKE_FRONTIER(a, 2, E(TestQueueAxis(0), 3), E(TestQueueAxis(1), 5));
  MAKE_FRONTIER(b, 2, E(TestQueueAxis(0), 3), E(TestQueueAxis(1), 7));
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_BEFORE);
  EXPECT_EQ(iree_async_frontier_compare(b, a), IREE_ASYNC_FRONTIER_AFTER);
}

TEST(CompareTest, Transitivity) {
  MAKE_FRONTIER(a, 2, E(TestQueueAxis(0), 1), E(TestQueueAxis(1), 1));
  MAKE_FRONTIER(b, 2, E(TestQueueAxis(0), 2), E(TestQueueAxis(1), 2));
  MAKE_FRONTIER(c, 2, E(TestQueueAxis(0), 3), E(TestQueueAxis(1), 3));
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_BEFORE);
  EXPECT_EQ(iree_async_frontier_compare(b, c), IREE_ASYNC_FRONTIER_BEFORE);
  EXPECT_EQ(iree_async_frontier_compare(a, c), IREE_ASYNC_FRONTIER_BEFORE);
}

//===----------------------------------------------------------------------===//
// Merge — path 1 (source empty)
//===----------------------------------------------------------------------===//

TEST(MergeTest, SourceEmptyTargetEmpty) {
  FRONTIER_ALLOC(target, 4);
  iree_async_frontier_initialize(target, 0);
  FRONTIER_ALLOC(source, 0);
  iree_async_frontier_initialize(source, 0);
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 0);
}

TEST(MergeTest, SourceEmptyTargetNonEmpty) {
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  FRONTIER_ALLOC(source, 0);
  iree_async_frontier_initialize(source, 0);
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 2);
  EXPECT_EQ(target->entries[0].epoch, 5u);
  EXPECT_EQ(target->entries[1].epoch, 10u);
}

//===----------------------------------------------------------------------===//
// Merge — path 2 (same axis set, epoch-max only)
//===----------------------------------------------------------------------===//

TEST(MergeTest, SameAxesSourceHigher) {
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(0), 8), E(TestQueueAxis(1), 15));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 2);
  EXPECT_EQ(target->entries[0].epoch, 8u);
  EXPECT_EQ(target->entries[1].epoch, 15u);
}

TEST(MergeTest, SameAxesSourceLower) {
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(0), 10), E(TestQueueAxis(1), 20));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(0), 3), E(TestQueueAxis(1), 5));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 2);
  EXPECT_EQ(target->entries[0].epoch, 10u);
  EXPECT_EQ(target->entries[1].epoch, 20u);
}

TEST(MergeTest, SameAxesEqual) {
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 2);
  EXPECT_EQ(target->entries[0].epoch, 5u);
  EXPECT_EQ(target->entries[1].epoch, 10u);
}

TEST(MergeTest, SameAxesMixed) {
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 20),
                E(TestQueueAxis(2), 8));
  MAKE_FRONTIER(source, 3, E(TestQueueAxis(0), 10), E(TestQueueAxis(1), 3),
                E(TestQueueAxis(2), 8));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 3);
  EXPECT_EQ(target->entries[0].epoch, 10u);  // Source higher.
  EXPECT_EQ(target->entries[1].epoch, 20u);  // Target higher.
  EXPECT_EQ(target->entries[2].epoch, 8u);   // Equal.
}

TEST(MergeTest, SameAxesEightEntries) {
  FRONTIER_ALLOC(target, 8);
  FRONTIER_ALLOC(source, 8);
  iree_async_frontier_initialize(target, 8);
  iree_async_frontier_initialize(source, 8);
  for (uint8_t i = 0; i < 8; ++i) {
    target->entries[i] = E(TestQueueAxis(i), 100 + i);
    source->entries[i] = E(TestQueueAxis(i), 100 + (7 - i));
  }
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 8, source));
  EXPECT_EQ(target->entry_count, 8);
  // Each entry should have max(100+i, 100+(7-i)).
  for (uint8_t i = 0; i < 8; ++i) {
    uint64_t expected = 100 + (i > (7 - i) ? i : (7 - i));
    EXPECT_EQ(target->entries[i].epoch, expected) << "at index " << (int)i;
  }
}

TEST(MergeTest, SameAxesEntryCountUnchanged) {
  MAKE_FRONTIER(target, 8, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10),
                E(TestQueueAxis(2), 15));
  MAKE_FRONTIER(source, 3, E(TestQueueAxis(0), 99), E(TestQueueAxis(1), 99),
                E(TestQueueAxis(2), 99));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 8, source));
  EXPECT_EQ(target->entry_count, 3);
}

//===----------------------------------------------------------------------===//
// Merge — path 3 (different axis sets, right-to-left)
//===----------------------------------------------------------------------===//

TEST(MergeTest, TargetEmptySourceNonEmpty) {
  FRONTIER_ALLOC(target, 4);
  iree_async_frontier_initialize(target, 0);
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 2);
  EXPECT_EQ(target->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(target->entries[0].epoch, 5u);
  EXPECT_EQ(target->entries[1].axis, TestQueueAxis(1));
  EXPECT_EQ(target->entries[1].epoch, 10u);
}

TEST(MergeTest, PartialOverlap) {
  // Target: axes 0, 2. Source: axes 1, 2.
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(0), 5), E(TestQueueAxis(2), 10));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(1), 7), E(TestQueueAxis(2), 15));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 3);
  EXPECT_EQ(target->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(target->entries[0].epoch, 5u);
  EXPECT_EQ(target->entries[1].axis, TestQueueAxis(1));
  EXPECT_EQ(target->entries[1].epoch, 7u);
  EXPECT_EQ(target->entries[2].axis, TestQueueAxis(2));
  EXPECT_EQ(target->entries[2].epoch, 15u);  // max(10, 15)
}

TEST(MergeTest, DisjointAxes) {
  // Target: axes 0, 2. Source: axes 1, 3.
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(0), 5), E(TestQueueAxis(2), 10));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(1), 7), E(TestQueueAxis(3), 15));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 4);
  EXPECT_EQ(target->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(target->entries[1].axis, TestQueueAxis(1));
  EXPECT_EQ(target->entries[2].axis, TestQueueAxis(2));
  EXPECT_EQ(target->entries[3].axis, TestQueueAxis(3));
}

TEST(MergeTest, SourceEntirelyBeforeTarget) {
  // Source axes are all smaller than target axes.
  MAKE_FRONTIER(target, 6, E(TestQueueAxis(4), 10), E(TestQueueAxis(5), 20));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(0), 1), E(TestQueueAxis(1), 2));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 6, source));
  EXPECT_EQ(target->entry_count, 4);
  EXPECT_EQ(target->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(target->entries[1].axis, TestQueueAxis(1));
  EXPECT_EQ(target->entries[2].axis, TestQueueAxis(4));
  EXPECT_EQ(target->entries[3].axis, TestQueueAxis(5));
}

TEST(MergeTest, SourceEntirelyAfterTarget) {
  // Source axes are all larger than target axes.
  MAKE_FRONTIER(target, 6, E(TestQueueAxis(0), 1), E(TestQueueAxis(1), 2));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(4), 10), E(TestQueueAxis(5), 20));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 6, source));
  EXPECT_EQ(target->entry_count, 4);
  EXPECT_EQ(target->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(target->entries[1].axis, TestQueueAxis(1));
  EXPECT_EQ(target->entries[2].axis, TestQueueAxis(4));
  EXPECT_EQ(target->entries[3].axis, TestQueueAxis(5));
}

TEST(MergeTest, InterleaveSourceBetweenTarget) {
  // Target: 0, 2, 4. Source: 1, 3.
  MAKE_FRONTIER(target, 8, E(TestQueueAxis(0), 10), E(TestQueueAxis(2), 20),
                E(TestQueueAxis(4), 30));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(1), 15), E(TestQueueAxis(3), 25));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 8, source));
  EXPECT_EQ(target->entry_count, 5);
  EXPECT_EQ(target->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(target->entries[1].axis, TestQueueAxis(1));
  EXPECT_EQ(target->entries[2].axis, TestQueueAxis(2));
  EXPECT_EQ(target->entries[3].axis, TestQueueAxis(3));
  EXPECT_EQ(target->entries[4].axis, TestQueueAxis(4));
}

TEST(MergeTest, CapacityInsufficient) {
  MAKE_FRONTIER(target, 8, E(TestQueueAxis(0), 5), E(TestQueueAxis(2), 10));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(1), 7), E(TestQueueAxis(3), 15));
  // Merged would be 4 entries, but capacity is 3.
  iree_status_t status = iree_async_frontier_merge(target, 3, source);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);
  // Target must be unchanged.
  EXPECT_EQ(target->entry_count, 2);
  EXPECT_EQ(target->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(target->entries[0].epoch, 5u);
  EXPECT_EQ(target->entries[1].axis, TestQueueAxis(2));
  EXPECT_EQ(target->entries[1].epoch, 10u);
}

TEST(MergeTest, CapacityOverflowLargeDisjointSets) {
  // Regression test: merged_count must use a type wider than uint8_t.
  // With 200 target entries and 100 disjoint source entries, merged_count =
  // 300. If merged_count were uint8_t, 300 wraps to 44, passing the capacity
  // check and causing a buffer overwrite.
  FRONTIER_ALLOC(target, 200);
  iree_async_frontier_initialize(target, 200);
  for (int i = 0; i < 200; ++i) {
    // Machine=0, 200 device axes — all less than any machine=1 axis.
    target->entries[i].axis = iree_async_axis_make_queue(1, 0, (uint8_t)i, 0);
    target->entries[i].epoch = 1;
  }

  FRONTIER_ALLOC(source, 100);
  iree_async_frontier_initialize(source, 100);
  for (int i = 0; i < 100; ++i) {
    // Machine=1, 100 device axes — all greater than any machine=0 axis.
    source->entries[i].axis = iree_async_axis_make_queue(1, 1, (uint8_t)i, 0);
    source->entries[i].epoch = 1;
  }

  // Capacity 255 is the max a uint8_t can hold, but 300 entries are needed.
  iree_status_t status = iree_async_frontier_merge(target, 255, source);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);
  // Target must be unchanged.
  EXPECT_EQ(target->entry_count, 200);
}

TEST(MergeTest, CapacityExactlyRight) {
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(0), 5), E(TestQueueAxis(2), 10));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(1), 7), E(TestQueueAxis(3), 15));
  // Merged is exactly 4 entries, capacity is 4.
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 4);
}

// Right-to-left merge correctness: verify entries shift correctly.
TEST(MergeTest, RightToLeftSingleShift) {
  // Target has 1 entry at position 0. After merge, it moves to position 2.
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(2), 99));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(0), 1), E(TestQueueAxis(1), 2));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 3);
  EXPECT_EQ(target->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(target->entries[1].axis, TestQueueAxis(1));
  EXPECT_EQ(target->entries[2].axis, TestQueueAxis(2));
  EXPECT_EQ(target->entries[2].epoch, 99u);  // Original value preserved.
}

TEST(MergeTest, RightToLeftInsertBetween) {
  // Target {A, C}, Source {B}: result is {A, B, C}.
  iree_async_axis_t axis_a = TestQueueAxis(0);
  iree_async_axis_t axis_b = TestQueueAxis(1);
  iree_async_axis_t axis_c = TestQueueAxis(2);
  MAKE_FRONTIER(target, 4, E(axis_a, 1), E(axis_c, 3));
  MAKE_FRONTIER(source, 1, E(axis_b, 2));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 3);
  EXPECT_EQ(target->entries[0].axis, axis_a);
  EXPECT_EQ(target->entries[0].epoch, 1u);
  EXPECT_EQ(target->entries[1].axis, axis_b);
  EXPECT_EQ(target->entries[1].epoch, 2u);
  EXPECT_EQ(target->entries[2].axis, axis_c);
  EXPECT_EQ(target->entries[2].epoch, 3u);
}

TEST(MergeTest, RightToLeftLargeShift) {
  // Target has 2 entries, source has 6 new entries.
  // Target entries shift from positions 0-1 to positions 6-7.
  FRONTIER_ALLOC(target, 8);
  iree_async_frontier_initialize(target, 2);
  target->entries[0] = E(TestQueueAxis(6), 70);
  target->entries[1] = E(TestQueueAxis(7), 80);

  FRONTIER_ALLOC(source, 6);
  iree_async_frontier_initialize(source, 6);
  for (uint8_t i = 0; i < 6; ++i) {
    source->entries[i] = E(TestQueueAxis(i), (uint64_t)(i + 1) * 10);
  }

  IREE_EXPECT_OK(iree_async_frontier_merge(target, 8, source));
  EXPECT_EQ(target->entry_count, 8);
  // All entries sorted, original values preserved after shift.
  for (uint8_t i = 0; i < 8; ++i) {
    EXPECT_EQ(target->entries[i].axis, TestQueueAxis(i));
    EXPECT_EQ(target->entries[i].epoch, (uint64_t)(i + 1) * 10);
  }
}

//===----------------------------------------------------------------------===//
// Merge — property tests
//===----------------------------------------------------------------------===//

TEST(MergeTest, Idempotent) {
  MAKE_FRONTIER(target, 4, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, source));
  EXPECT_EQ(target->entry_count, 2);
  EXPECT_EQ(target->entries[0].epoch, 5u);
  EXPECT_EQ(target->entries[1].epoch, 10u);
}

TEST(MergeTest, ResultDominatesInputs) {
  MAKE_FRONTIER(target, 8, E(TestQueueAxis(0), 3), E(TestQueueAxis(2), 10));
  MAKE_FRONTIER(source, 2, E(TestQueueAxis(1), 7), E(TestQueueAxis(2), 5));

  // Save original target for comparison.
  MAKE_FRONTIER(original_target, 2, E(TestQueueAxis(0), 3),
                E(TestQueueAxis(2), 10));

  IREE_EXPECT_OK(iree_async_frontier_merge(target, 8, source));

  // Merged result must not be BEFORE either input.
  EXPECT_NE(iree_async_frontier_compare(target, original_target),
            IREE_ASYNC_FRONTIER_BEFORE);
  EXPECT_NE(iree_async_frontier_compare(target, source),
            IREE_ASYNC_FRONTIER_BEFORE);
}

TEST(MergeTest, ResultIsSorted) {
  MAKE_FRONTIER(target, 8, E(TestQueueAxis(0), 5), E(TestQueueAxis(4), 10));
  MAKE_FRONTIER(source, 3, E(TestQueueAxis(1), 7), E(TestQueueAxis(3), 3),
                E(TestQueueAxis(5), 20));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 8, source));
  IREE_EXPECT_OK(iree_async_frontier_validate(target));
}

TEST(MergeTest, Commutativity) {
  // merge(a, b) then merge(result, c) should equal merge(a, c) then
  // merge(result, b).
  iree_async_axis_t ax0 = TestQueueAxis(0);
  iree_async_axis_t ax1 = TestQueueAxis(1);
  iree_async_axis_t ax2 = TestQueueAxis(2);

  // Path: merge a+b, then merge result+c.
  MAKE_FRONTIER(ab, 8, E(ax0, 3), E(ax1, 5));
  MAKE_FRONTIER(b, 2, E(ax0, 7), E(ax2, 2));
  MAKE_FRONTIER(c, 1, E(ax1, 9));
  IREE_EXPECT_OK(iree_async_frontier_merge(ab, 8, b));
  IREE_EXPECT_OK(iree_async_frontier_merge(ab, 8, c));

  // Path: merge a+c, then merge result+b.
  MAKE_FRONTIER(ac, 8, E(ax0, 3), E(ax1, 5));
  MAKE_FRONTIER(b2, 2, E(ax0, 7), E(ax2, 2));
  MAKE_FRONTIER(c2, 1, E(ax1, 9));
  IREE_EXPECT_OK(iree_async_frontier_merge(ac, 8, c2));
  IREE_EXPECT_OK(iree_async_frontier_merge(ac, 8, b2));

  // Both paths should produce the same result.
  EXPECT_EQ(ab->entry_count, ac->entry_count);
  for (uint8_t i = 0; i < ab->entry_count; ++i) {
    EXPECT_EQ(ab->entries[i].axis, ac->entries[i].axis);
    EXPECT_EQ(ab->entries[i].epoch, ac->entries[i].epoch);
  }
}

//===----------------------------------------------------------------------===//
// is_satisfied
//===----------------------------------------------------------------------===//

TEST(IsSatisfiedTest, EmptyFrontierAlwaysSatisfied) {
  FRONTIER_ALLOC(f, 0);
  iree_async_frontier_initialize(f, 0);
  EXPECT_TRUE(iree_async_frontier_is_satisfied(f, nullptr, 0));

  iree_async_frontier_entry_t current[] = {E(TestQueueAxis(0), 100)};
  EXPECT_TRUE(iree_async_frontier_is_satisfied(f, current, 1));
}

TEST(IsSatisfiedTest, SingleEntryHigherCurrent) {
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 5));
  iree_async_frontier_entry_t current[] = {E(TestQueueAxis(0), 10)};
  EXPECT_TRUE(iree_async_frontier_is_satisfied(f, current, 1));
}

TEST(IsSatisfiedTest, SingleEntryExactEpoch) {
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 5));
  iree_async_frontier_entry_t current[] = {E(TestQueueAxis(0), 5)};
  EXPECT_TRUE(iree_async_frontier_is_satisfied(f, current, 1));
}

TEST(IsSatisfiedTest, SingleEntryLowerCurrent) {
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 5));
  iree_async_frontier_entry_t current[] = {E(TestQueueAxis(0), 3)};
  EXPECT_FALSE(iree_async_frontier_is_satisfied(f, current, 1));
}

TEST(IsSatisfiedTest, AxisNotInCurrent) {
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 5));
  iree_async_frontier_entry_t current[] = {E(TestQueueAxis(1), 100)};
  EXPECT_FALSE(iree_async_frontier_is_satisfied(f, current, 1));
}

TEST(IsSatisfiedTest, MultipleEntriesAllSatisfied) {
  MAKE_FRONTIER(f, 3, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10),
                E(TestQueueAxis(2), 15));
  iree_async_frontier_entry_t current[] = {
      E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10), E(TestQueueAxis(2), 20)};
  EXPECT_TRUE(iree_async_frontier_is_satisfied(f, current, 3));
}

TEST(IsSatisfiedTest, MultipleEntriesOneUnsatisfied) {
  MAKE_FRONTIER(f, 3, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10),
                E(TestQueueAxis(2), 15));
  iree_async_frontier_entry_t current[] = {
      E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 8), E(TestQueueAxis(2), 20)};
  EXPECT_FALSE(iree_async_frontier_is_satisfied(f, current, 3));
}

TEST(IsSatisfiedTest, CurrentHasExtraAxes) {
  MAKE_FRONTIER(f, 2, E(TestQueueAxis(1), 10), E(TestQueueAxis(3), 20));
  iree_async_frontier_entry_t current[] = {
      E(TestQueueAxis(0), 99), E(TestQueueAxis(1), 10), E(TestQueueAxis(2), 99),
      E(TestQueueAxis(3), 25), E(TestQueueAxis(4), 99)};
  EXPECT_TRUE(iree_async_frontier_is_satisfied(f, current, 5));
}

TEST(IsSatisfiedTest, EmptyCurrentNonEmptyFrontier) {
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 5));
  EXPECT_FALSE(iree_async_frontier_is_satisfied(f, nullptr, 0));
}

TEST(IsSatisfiedTest, Monotonicity) {
  // If satisfied at epoch E, still satisfied at E+1.
  MAKE_FRONTIER(f, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  iree_async_frontier_entry_t current_low[] = {E(TestQueueAxis(0), 5),
                                               E(TestQueueAxis(1), 10)};
  iree_async_frontier_entry_t current_high[] = {E(TestQueueAxis(0), 100),
                                                E(TestQueueAxis(1), 200)};
  EXPECT_TRUE(iree_async_frontier_is_satisfied(f, current_low, 2));
  EXPECT_TRUE(iree_async_frontier_is_satisfied(f, current_high, 2));
}

//===----------------------------------------------------------------------===//
// Scenario-based tests
//===----------------------------------------------------------------------===//

TEST(ScenarioTest, GpuPipelineSequential) {
  iree_async_axis_t gpu0_q0 = TestQueueAxis(0);

  // Op1 signals {(GPU0/q0, 1)}.
  // Op2 waits on {(GPU0/q0, 1)}, signals {(GPU0/q0, 2)}.
  MAKE_FRONTIER(wait_for_op1, 1, E(gpu0_q0, 1));
  MAKE_FRONTIER(wait_for_op2, 1, E(gpu0_q0, 2));

  // At epoch 1: op1's wait is satisfied, op2's is not.
  iree_async_frontier_entry_t at_epoch1[] = {E(gpu0_q0, 1)};
  EXPECT_TRUE(iree_async_frontier_is_satisfied(wait_for_op1, at_epoch1, 1));
  EXPECT_FALSE(iree_async_frontier_is_satisfied(wait_for_op2, at_epoch1, 1));

  // At epoch 2: both satisfied.
  iree_async_frontier_entry_t at_epoch2[] = {E(gpu0_q0, 2)};
  EXPECT_TRUE(iree_async_frontier_is_satisfied(wait_for_op1, at_epoch2, 1));
  EXPECT_TRUE(iree_async_frontier_is_satisfied(wait_for_op2, at_epoch2, 1));
}

TEST(ScenarioTest, MultiQueueFanOutFanIn) {
  iree_async_axis_t gpu0_q0 = TestQueueAxis(0);
  iree_async_axis_t gpu1_q0 = RemoteQueueAxis(0);

  // GPU0 produces → signal {(GPU0/q0, 10)}.
  // GPU1 processes → wait {(GPU0/q0, 10)}, signal {(GPU1/q0, 5)}.
  // GPU0 consumes → wait = merge of both signals.
  MAKE_FRONTIER(target, 4, E(gpu0_q0, 10));
  MAKE_FRONTIER(gpu1_signal, 1, E(gpu1_q0, 5));
  IREE_EXPECT_OK(iree_async_frontier_merge(target, 4, gpu1_signal));

  EXPECT_EQ(target->entry_count, 2);
  EXPECT_EQ(target->entries[0].axis, gpu0_q0);
  EXPECT_EQ(target->entries[0].epoch, 10u);
  EXPECT_EQ(target->entries[1].axis, gpu1_q0);
  EXPECT_EQ(target->entries[1].epoch, 5u);

  // The two signals are concurrent (different axes).
  MAKE_FRONTIER(gpu0_signal, 1, E(gpu0_q0, 10));
  EXPECT_EQ(iree_async_frontier_compare(gpu0_signal, gpu1_signal),
            IREE_ASYNC_FRONTIER_CONCURRENT);
}

TEST(ScenarioTest, CollectiveCompression) {
  iree_async_axis_t collective_42 = CollectiveAxis(42);

  MAKE_FRONTIER(wait, 1, E(collective_42, 7));

  // Not yet at epoch 7.
  iree_async_frontier_entry_t at_6[] = {E(collective_42, 6)};
  EXPECT_FALSE(iree_async_frontier_is_satisfied(wait, at_6, 1));

  // At epoch 7: satisfied.
  iree_async_frontier_entry_t at_7[] = {E(collective_42, 7)};
  EXPECT_TRUE(iree_async_frontier_is_satisfied(wait, at_7, 1));
}

TEST(ScenarioTest, NetworkFrontierPropagation) {
  iree_async_axis_t local_q0 = TestQueueAxis(0);
  iree_async_axis_t remote_q0 = RemoteQueueAxis(0);

  MAKE_FRONTIER(local, 4, E(local_q0, 50));
  MAKE_FRONTIER(remote, 1, E(remote_q0, 100));

  // Save local before merge.
  MAKE_FRONTIER(old_local, 1, E(local_q0, 50));

  IREE_EXPECT_OK(iree_async_frontier_merge(local, 4, remote));

  // Merged includes both axes.
  EXPECT_EQ(local->entry_count, 2);
  // Merged is strictly after old_local.
  EXPECT_EQ(iree_async_frontier_compare(old_local, local),
            IREE_ASYNC_FRONTIER_BEFORE);
}

TEST(ScenarioTest, MergeOverflowDetection) {
  // 4 entries in target, 3 new axes from source, but capacity is only 5.
  FRONTIER_ALLOC(target, 8);
  iree_async_frontier_initialize(target, 4);
  for (uint8_t i = 0; i < 4; ++i) {
    target->entries[i] = E(TestQueueAxis(i * 2), (uint64_t)(i + 1) * 10);
  }

  FRONTIER_ALLOC(source, 3);
  iree_async_frontier_initialize(source, 3);
  source->entries[0] = E(TestQueueAxis(1), 5);
  source->entries[1] = E(TestQueueAxis(3), 15);
  source->entries[2] = E(TestQueueAxis(5), 25);

  // Merged would be 7 entries (4 + 3, no overlap), capacity is 5.
  iree_status_t status = iree_async_frontier_merge(target, 5, source);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);

  // Target unchanged.
  EXPECT_EQ(target->entry_count, 4);
  EXPECT_EQ(target->entries[0].epoch, 10u);
}

TEST(ScenarioTest, CausalityChain) {
  iree_async_axis_t axis_x = TestQueueAxis(0);
  iree_async_axis_t axis_y = TestQueueAxis(1);

  MAKE_FRONTIER(a, 1, E(axis_x, 1));
  MAKE_FRONTIER(b, 4, E(axis_x, 2), E(axis_y, 1));
  MAKE_FRONTIER(c, 1, E(axis_y, 2));

  // A happens-before B (X advanced from 1→2).
  EXPECT_EQ(iree_async_frontier_compare(a, b), IREE_ASYNC_FRONTIER_BEFORE);

  // B and C are concurrent (B ahead on X, C ahead on Y).
  EXPECT_EQ(iree_async_frontier_compare(b, c), IREE_ASYNC_FRONTIER_CONCURRENT);

  // Merge B and C → LUB.
  MAKE_FRONTIER(merged, 4, E(axis_x, 2), E(axis_y, 1));
  IREE_EXPECT_OK(iree_async_frontier_merge(merged, 4, c));
  EXPECT_EQ(merged->entry_count, 2);
  EXPECT_EQ(merged->entries[0].epoch, 2u);  // X: max(2, 0) = 2.
  EXPECT_EQ(merged->entries[1].epoch, 2u);  // Y: max(1, 2) = 2.

  // A is transitively before the merged result.
  EXPECT_EQ(iree_async_frontier_compare(a, merged), IREE_ASYNC_FRONTIER_BEFORE);
}

}  // namespace
