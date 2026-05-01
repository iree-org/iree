// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/sysfs.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_EMSCRIPTEN)

using namespace iree::testing::status;

struct CpuRanges {
  uint32_t start_cpus[4];
  uint32_t end_cpus[4];
  iree_host_size_t count;
};

static bool AppendCpuRange(uint32_t start_cpu, uint32_t end_cpu,
                           void* user_data) {
  CpuRanges* ranges = (CpuRanges*)user_data;
  if (ranges->count >= IREE_ARRAYSIZE(ranges->start_cpus)) return false;
  ranges->start_cpus[ranges->count] = start_cpu;
  ranges->end_cpus[ranges->count] = end_cpu;
  ++ranges->count;
  return true;
}

TEST(SysfsTest, ParseCpuList) {
  CpuRanges ranges = {};
  IREE_ASSERT_OK(
      iree_sysfs_parse_cpu_list(IREE_SV("0,2-3, 7"), AppendCpuRange, &ranges));

  EXPECT_EQ(3, ranges.count);
  EXPECT_EQ(0, ranges.start_cpus[0]);
  EXPECT_EQ(1, ranges.end_cpus[0]);
  EXPECT_EQ(2, ranges.start_cpus[1]);
  EXPECT_EQ(4, ranges.end_cpus[1]);
  EXPECT_EQ(7, ranges.start_cpus[2]);
  EXPECT_EQ(8, ranges.end_cpus[2]);
}

TEST(SysfsTest, TryParseCpuListRejectsMalformedInput) {
  CpuRanges ranges = {};
  EXPECT_FALSE(iree_sysfs_try_parse_cpu_list(IREE_SV("not-a-cpu-list"),
                                             AppendCpuRange, &ranges));
  EXPECT_FALSE(
      iree_sysfs_try_parse_cpu_list(IREE_SV("3-2"), AppendCpuRange, &ranges));
  EXPECT_FALSE(iree_sysfs_try_parse_cpu_list(IREE_SV("4294967295"),
                                             AppendCpuRange, &ranges));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_sysfs_parse_cpu_list(IREE_SV("3-2"), AppendCpuRange, &ranges));
}

TEST(SysfsTest, ParseSizeString) {
  uint64_t size = 0;
  IREE_ASSERT_OK(iree_sysfs_parse_size_string(IREE_SV("32K"), &size));
  EXPECT_EQ(32768u, size);
  IREE_ASSERT_OK(iree_sysfs_parse_size_string(IREE_SV("512"), &size));
  EXPECT_EQ(512u, size);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_sysfs_parse_size_string(IREE_SV(""), &size));
}

#else

TEST(SysfsTest, PlatformDisabled) {}

#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_EMSCRIPTEN

}  // namespace
