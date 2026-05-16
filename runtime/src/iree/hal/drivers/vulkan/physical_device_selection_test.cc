// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/physical_device_selection.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

class PhysicalDeviceSnapshotBuilder {
 public:
  PhysicalDeviceSnapshotBuilder() {
    std::memset(&snapshot_, 0, sizeof(snapshot_));
    snapshot_.properties2.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    snapshot_.properties2.properties.apiVersion = VK_API_VERSION_1_3;
    snapshot_.features12.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    snapshot_.features12.timelineSemaphore = VK_TRUE;
    snapshot_.features12.scalarBlockLayout = VK_TRUE;
    snapshot_.features13.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    snapshot_.features13.synchronization2 = VK_TRUE;
  }

  void SetOrdinal(uint32_t ordinal) { snapshot_.ordinal = ordinal; }

  void SetApiVersion(uint32_t api_version) {
    snapshot_.properties2.properties.apiVersion = api_version;
  }

  void SetUuid(const uint8_t uuid[VK_UUID_SIZE]) {
    std::memcpy(snapshot_.id_properties.deviceUUID, uuid, VK_UUID_SIZE);
  }

  void AddQueueFamily(VkQueueFlags flags, uint32_t queue_count) {
    VkQueueFamilyProperties2 queue_family;
    std::memset(&queue_family, 0, sizeof(queue_family));
    queue_family.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    queue_family.queueFamilyProperties.queueFlags = flags;
    queue_family.queueFamilyProperties.queueCount = queue_count;
    queue_families_.push_back(queue_family);
    snapshot_.queue_family_count =
        static_cast<uint32_t>(queue_families_.size());
    snapshot_.queue_families = queue_families_.data();
  }

  const iree_hal_vulkan_physical_device_snapshot_t* snapshot() const {
    return &snapshot_;
  }

 private:
  // Snapshot whose pointer fields reference this builder's storage.
  iree_hal_vulkan_physical_device_snapshot_t snapshot_;

  // Backing storage for snapshot_.queue_families.
  std::vector<VkQueueFamilyProperties2> queue_families_;
};

TEST(PhysicalDeviceSelectionTest, DefaultMatchesBaselineDevice) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  const iree_hal_vulkan_physical_device_selector_t selector = {
      IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT,
  };

  bool matches = false;
  IREE_ASSERT_OK(iree_hal_vulkan_physical_device_selector_match(
      &selector, builder.snapshot(), &matches));

  EXPECT_TRUE(matches);
}

TEST(PhysicalDeviceSelectionTest, DefaultSkipsBelowBaselineDevice) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.SetApiVersion(VK_API_VERSION_1_2);
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  const iree_hal_vulkan_physical_device_selector_t selector = {
      IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT,
  };

  bool matches = true;
  IREE_ASSERT_OK(iree_hal_vulkan_physical_device_selector_match(
      &selector, builder.snapshot(), &matches));

  EXPECT_FALSE(matches);
}

TEST(PhysicalDeviceSelectionTest, IdMatchesOrdinalPlusOne) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.SetOrdinal(2);
  const iree_hal_vulkan_physical_device_selector_t selector = {
      IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID,
      3,
  };

  bool matches = false;
  IREE_ASSERT_OK(iree_hal_vulkan_physical_device_selector_match(
      &selector, builder.snapshot(), &matches));

  EXPECT_TRUE(matches);
}

TEST(PhysicalDeviceSelectionTest, IdRejectsDifferentOrdinal) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.SetOrdinal(2);
  const iree_hal_vulkan_physical_device_selector_t selector = {
      IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID,
      4,
  };

  bool matches = true;
  IREE_ASSERT_OK(iree_hal_vulkan_physical_device_selector_match(
      &selector, builder.snapshot(), &matches));

  EXPECT_FALSE(matches);
}

TEST(PhysicalDeviceSelectionTest, PathMatchesDeviceUuid) {
  PhysicalDeviceSnapshotBuilder builder;
  const uint8_t uuid[VK_UUID_SIZE] = {
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
  };
  builder.SetUuid(uuid);
  const std::string path = "GPU-000102030405060708090a0b0c0d0e0f";
  const iree_hal_vulkan_physical_device_selector_t selector = {
      IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH,
      IREE_HAL_DEVICE_ID_DEFAULT,
      iree_make_string_view(path.data(), path.size()),
  };

  bool matches = false;
  IREE_ASSERT_OK(iree_hal_vulkan_physical_device_selector_match(
      &selector, builder.snapshot(), &matches));

  EXPECT_TRUE(matches);
}

TEST(PhysicalDeviceSelectionTest, InvalidSelectorModeFails) {
  PhysicalDeviceSnapshotBuilder builder;
  const iree_hal_vulkan_physical_device_selector_t selector = {
      static_cast<iree_hal_vulkan_physical_device_selector_mode_t>(0x7F),
  };

  bool matches = true;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_physical_device_selector_match(
                            &selector, builder.snapshot(), &matches));
  EXPECT_FALSE(matches);
}

}  // namespace
}  // namespace iree::hal::vulkan
