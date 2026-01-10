// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/allocator_stats.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace {

TEST(AllocatorStatsTest, CheckThatWrappingHostAllocatorTracksAllocations) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_allocator_with_stats_t stats_allocator;
  host_allocator = iree_allocator_stats_init(&stats_allocator, host_allocator);

  void* ptr1 = nullptr;
  IREE_CHECK_OK(iree_allocator_malloc(host_allocator, 128, &ptr1));
  EXPECT_NE(ptr1, nullptr);
  EXPECT_EQ(stats_allocator.statistics.bytes_allocated, 128);
  EXPECT_EQ(stats_allocator.statistics.bytes_freed, 0);
  EXPECT_EQ(stats_allocator.statistics.bytes_peak, 128);

  void* ptr2 = nullptr;
  IREE_CHECK_OK(iree_allocator_malloc(host_allocator, 256, &ptr2));
  EXPECT_NE(ptr2, nullptr);
  EXPECT_EQ(stats_allocator.statistics.bytes_allocated, 384);
  EXPECT_EQ(stats_allocator.statistics.bytes_freed, 0);
  EXPECT_EQ(stats_allocator.statistics.bytes_peak, 384);

  iree_allocator_free(host_allocator, ptr1);
  EXPECT_EQ(stats_allocator.statistics.bytes_allocated, 384);
  EXPECT_EQ(stats_allocator.statistics.bytes_freed, 128);
  EXPECT_EQ(stats_allocator.statistics.bytes_peak, 384);

  iree_allocator_free(host_allocator, ptr2);
  iree_allocator_free(host_allocator, nullptr);  // should be no-op
  EXPECT_EQ(stats_allocator.statistics.bytes_allocated, 384);
  EXPECT_EQ(stats_allocator.statistics.bytes_freed, 384);
  EXPECT_EQ(stats_allocator.statistics.bytes_peak, 384);

  iree_allocator_stats_deinit(&stats_allocator);
}

TEST(AllocatorStatsTest, CheckReallocStatisticsAreCorrectlyTracked) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_allocator_with_stats_t stats_allocator;
  host_allocator = iree_allocator_stats_init(&stats_allocator, host_allocator);

  void* ptr = nullptr;
  IREE_CHECK_OK(iree_allocator_malloc(host_allocator, 128, &ptr));
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(stats_allocator.statistics.bytes_allocated, 128);
  EXPECT_EQ(stats_allocator.statistics.bytes_freed, 0);
  EXPECT_EQ(stats_allocator.statistics.bytes_peak, 128);

  // Grow the allocation, this may be an alloc+free or a resize in place
  // counting as alloc
  IREE_CHECK_OK(iree_allocator_realloc(host_allocator, 256, &ptr));
  EXPECT_NE(ptr, nullptr);
  EXPECT_GE(stats_allocator.statistics.bytes_allocated, 256);
  EXPECT_LE(stats_allocator.statistics.bytes_allocated, 384);
  EXPECT_GE(stats_allocator.statistics.bytes_freed, 0);
  EXPECT_LE(stats_allocator.statistics.bytes_freed, 128);
  EXPECT_EQ(stats_allocator.statistics.bytes_peak, 256);

  // Shrink the allocation, this may be an alloc+free or a resize in place
  // counting as free
  IREE_CHECK_OK(iree_allocator_realloc(host_allocator, 64, &ptr));
  EXPECT_NE(ptr, nullptr);
  EXPECT_GE(stats_allocator.statistics.bytes_allocated, 256);
  EXPECT_LE(stats_allocator.statistics.bytes_allocated, 448);
  EXPECT_GE(stats_allocator.statistics.bytes_freed, 192);
  EXPECT_LE(stats_allocator.statistics.bytes_freed, 384);
  EXPECT_EQ(stats_allocator.statistics.bytes_peak, 256);

  iree_allocator_free(host_allocator, ptr);
  EXPECT_GE(stats_allocator.statistics.bytes_allocated, 256);
  EXPECT_LE(stats_allocator.statistics.bytes_allocated, 448);
  EXPECT_GE(stats_allocator.statistics.bytes_freed, 256);
  EXPECT_LE(stats_allocator.statistics.bytes_freed, 448);
  EXPECT_EQ(stats_allocator.statistics.bytes_freed,
            stats_allocator.statistics.bytes_allocated);
  EXPECT_EQ(stats_allocator.statistics.bytes_peak, 256);

  iree_allocator_stats_deinit(&stats_allocator);
}

TEST(AllocatorStatsTest, CheckAllocatorRespectsHostAlignment) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_allocator_with_stats_t stats_allocator;
  host_allocator = iree_allocator_stats_init(&stats_allocator, host_allocator);

  void* ptr = nullptr;
  IREE_CHECK_OK(iree_allocator_malloc(host_allocator, 128, &ptr));

  EXPECT_TRUE(
      iree_host_size_has_alignment((iree_host_size_t)ptr, iree_max_align_t));

  iree_allocator_free(host_allocator, ptr);
  iree_allocator_stats_deinit(&stats_allocator);
}

TEST(AllocatorStatsTest, PrintStats) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_allocator_with_stats_t stats_allocator;
  host_allocator = iree_allocator_stats_init(&stats_allocator, host_allocator);

  void* ptr = nullptr;
  IREE_CHECK_OK(iree_allocator_malloc(host_allocator, 128, &ptr));
  iree_allocator_free(host_allocator, ptr);

  ::testing::internal::CaptureStdout();
  IREE_CHECK_OK(iree_allocator_statistics_fprint(stdout, &stats_allocator));

  std::string output = ::testing::internal::GetCapturedStdout();
  EXPECT_EQ(output,
            "[[ iree_allocator_t memory statistics ]]\n"
            "  HOST_ALLOC:          128B peak /          128B allocated /      "
            "    128B freed /            0B live\n");

  iree_allocator_stats_deinit(&stats_allocator);
}

}  // namespace
}  // namespace iree
