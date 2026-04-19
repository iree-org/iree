// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/statistics_internal.h"

#include "iree/testing/gtest.h"

namespace iree::hal {
namespace {

typedef struct statistics_macro_smoke_t {
  uint32_t always_present;
  IREE_HAL_STATISTICS_FIELD(uint32_t stats_only;)
} statistics_macro_smoke_t;

static uint32_t StatisticsMacroSmoke(uint32_t enabled_bits) {
  statistics_macro_smoke_t state = {};
  state.always_present = 1;
  IREE_HAL_STATISTICS({
    state.stats_only = 2;
    state.always_present += state.stats_only;
  });
  IREE_HAL_STATISTICS_IF_ENABLED(enabled_bits, { state.always_present += 4; });
  return state.always_present;
}

TEST(StatisticsInternalTest, MacrosCompileWithStatisticsFields) {
  EXPECT_NE(0u, StatisticsMacroSmoke(/*enabled_bits=*/1));
  IREE_HAL_STATISTICS({
    EXPECT_EQ(3u, StatisticsMacroSmoke(/*enabled_bits=*/0));
    EXPECT_EQ(7u, StatisticsMacroSmoke(/*enabled_bits=*/1));
  });
}

TEST(StatisticsInternalTest, OperationCountsMerge) {
  iree_hal_statistics_operation_counts_t counts = {};
  counts.submitted_count = 4;
  counts.completed_count = 3;

  iree_hal_statistics_operation_counts_t delta = {};
  delta.submitted_count = 2;
  delta.failed_count = 1;
  delta.cancelled_count = 1;

  iree_hal_statistics_operation_counts_merge(&counts, &delta);

  EXPECT_EQ(6ull, counts.submitted_count);
  EXPECT_EQ(3ull, counts.completed_count);
  EXPECT_EQ(1ull, counts.failed_count);
  EXPECT_EQ(1ull, counts.cancelled_count);
}

TEST(StatisticsInternalTest, TimingRecordsAndMergesSamples) {
  iree_hal_statistics_timing_ns_t timing = {};
  iree_hal_statistics_timing_ns_record(&timing, 10);
  iree_hal_statistics_timing_ns_record(&timing, 4);
  iree_hal_statistics_timing_ns_record(&timing, 16);

  EXPECT_EQ(3ull, timing.sample_count);
  EXPECT_EQ(30ull, timing.total_duration_ns);
  EXPECT_EQ(4ull, timing.minimum_duration_ns);
  EXPECT_EQ(16ull, timing.maximum_duration_ns);
  EXPECT_EQ(16ull, timing.last_duration_ns);

  iree_hal_statistics_timing_ns_t delta = {};
  iree_hal_statistics_timing_ns_record(&delta, 7);
  iree_hal_statistics_timing_ns_merge(&timing, &delta);

  EXPECT_EQ(4ull, timing.sample_count);
  EXPECT_EQ(37ull, timing.total_duration_ns);
  EXPECT_EQ(4ull, timing.minimum_duration_ns);
  EXPECT_EQ(16ull, timing.maximum_duration_ns);
  EXPECT_EQ(7ull, timing.last_duration_ns);
}

}  // namespace
}  // namespace iree::hal
