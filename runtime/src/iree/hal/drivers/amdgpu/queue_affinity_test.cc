// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/queue_affinity.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static iree_hal_amdgpu_queue_affinity_domain_t TwoDeviceDomain() {
  return (iree_hal_amdgpu_queue_affinity_domain_t){
      .supported_affinity = 0xFull,
      .physical_device_count = 2,
      .queue_count_per_physical_device = 2,
  };
}

TEST(QueueAffinityTest, NormalizeAnyExpandsToSupportedAffinity) {
  iree_hal_queue_affinity_t normalized_affinity = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_affinity_normalize(
      0xFull, IREE_HAL_QUEUE_AFFINITY_ANY, &normalized_affinity));
  EXPECT_EQ(normalized_affinity, 0xFull);
}

TEST(QueueAffinityTest, NormalizeIntersectsExplicitAffinity) {
  iree_hal_queue_affinity_t normalized_affinity = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_affinity_normalize(
      0x3ull, 0x5ull, &normalized_affinity));
  EXPECT_EQ(normalized_affinity, 0x1ull);
}

TEST(QueueAffinityTest, NormalizeRejectsEmptyIntersection) {
  iree_hal_queue_affinity_t normalized_affinity = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_queue_affinity_normalize(
                            0x3ull, 0x4ull, &normalized_affinity));
}

TEST(QueueAffinityTest, ResolveSelectsFirstQueue) {
  iree_hal_amdgpu_queue_affinity_resolved_t resolved;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_affinity_resolve(TwoDeviceDomain(),
                                                        0xAull, &resolved));
  EXPECT_EQ(resolved.queue_affinity, 0xAull);
  EXPECT_EQ(resolved.queue_ordinal, 1);
  EXPECT_EQ(resolved.physical_device_ordinal, 0);
  EXPECT_EQ(resolved.physical_queue_ordinal, 1);
}

TEST(QueueAffinityTest, ResolveOrdinalMapsPhysicalDeviceAndQueue) {
  iree_hal_amdgpu_queue_affinity_resolved_t resolved;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_affinity_resolve_ordinal(
      TwoDeviceDomain(), 3, &resolved));
  EXPECT_EQ(resolved.queue_affinity, 0x8ull);
  EXPECT_EQ(resolved.queue_ordinal, 3);
  EXPECT_EQ(resolved.physical_device_ordinal, 1);
  EXPECT_EQ(resolved.physical_queue_ordinal, 1);
}

TEST(QueueAffinityTest, ResolveOrdinalRejectsOutOfRangeDevice) {
  iree_hal_amdgpu_queue_affinity_resolved_t resolved;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_queue_affinity_resolve_ordinal(
                            TwoDeviceDomain(), 4, &resolved));
}

TEST(QueueAffinityTest, TryResolveReturnsFalseForInvalidAffinity) {
  iree_hal_amdgpu_queue_affinity_resolved_t resolved;
  EXPECT_FALSE(iree_hal_amdgpu_queue_affinity_try_resolve(TwoDeviceDomain(),
                                                          0x10ull, &resolved));
}

TEST(QueueAffinityTest, PhysicalDeviceAffinityBuildsQueueRange) {
  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_affinity_for_physical_device(
      TwoDeviceDomain(), 1, &queue_affinity));
  EXPECT_EQ(queue_affinity, 0xCull);
}

TEST(QueueAffinityTest, NormalizeAnyForPhysicalDeviceSelectsFirstDevice) {
  iree_hal_queue_affinity_t queue_affinity = 0;
  iree_host_size_t physical_device_ordinal = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_affinity_normalize_for_physical_device(
      TwoDeviceDomain(), IREE_HAL_QUEUE_AFFINITY_ANY, &queue_affinity,
      &physical_device_ordinal));
  EXPECT_EQ(queue_affinity, 0x3ull);
  EXPECT_EQ(physical_device_ordinal, 0);
}

TEST(QueueAffinityTest, NormalizeExplicitForPhysicalDeviceKeepsSelectedBits) {
  iree_hal_queue_affinity_t queue_affinity = 0;
  iree_host_size_t physical_device_ordinal = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_affinity_normalize_for_physical_device(
      TwoDeviceDomain(), 0xCull, &queue_affinity, &physical_device_ordinal));
  EXPECT_EQ(queue_affinity, 0xCull);
  EXPECT_EQ(physical_device_ordinal, 1);
}

TEST(QueueAffinityTest, NormalizeForPhysicalDeviceRejectsCrossDeviceMask) {
  iree_hal_queue_affinity_t queue_affinity = 0;
  iree_host_size_t physical_device_ordinal = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amdgpu_queue_affinity_normalize_for_physical_device(
          TwoDeviceDomain(), 0x5ull, &queue_affinity,
          &physical_device_ordinal));
}

TEST(QueueAffinityTest, DeviceLocalAffinity) {
  EXPECT_TRUE(iree_hal_amdgpu_queue_affinity_is_physical_device_local(
      TwoDeviceDomain(), 0xCull, 1));
  EXPECT_FALSE(iree_hal_amdgpu_queue_affinity_is_physical_device_local(
      TwoDeviceDomain(), 0x5ull, 1));
  EXPECT_FALSE(iree_hal_amdgpu_queue_affinity_is_physical_device_local(
      TwoDeviceDomain(), IREE_HAL_QUEUE_AFFINITY_ANY, 1));
}

}  // namespace
}  // namespace iree::hal::amdgpu
