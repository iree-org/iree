// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/physical_device_selection.h"

#include <inttypes.h>
#include <string.h>

iree_status_t iree_hal_vulkan_physical_device_selector_match(
    const iree_hal_vulkan_physical_device_selector_t* selector,
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    bool* out_matches) {
  IREE_ASSERT_ARGUMENT(selector);
  IREE_ASSERT_ARGUMENT(snapshot);
  IREE_ASSERT_ARGUMENT(out_matches);
  *out_matches = false;

  switch (selector->mode) {
    case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT:
      *out_matches =
          iree_hal_vulkan_physical_device_supports_baseline(snapshot);
      return iree_ok_status();
    case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID:
      *out_matches =
          selector->device_id == (iree_hal_device_id_t)(snapshot->ordinal + 1);
      return iree_ok_status();
    case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH: {
      char path_storage[64] = {0};
      iree_string_builder_t builder;
      iree_string_builder_initialize_with_storage(
          path_storage, sizeof(path_storage), &builder);
      iree_string_view_t candidate_path = iree_string_view_empty();
      iree_status_t status = iree_hal_vulkan_append_device_path(
          snapshot, &builder, &candidate_path);
      if (iree_status_is_ok(status)) {
        *out_matches =
            iree_string_view_equal(candidate_path, selector->device_path);
      }
      iree_string_builder_deinitialize(&builder);
      return status;
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid Vulkan selector mode %u",
                          (uint32_t)selector->mode);
}

iree_status_t iree_hal_vulkan_physical_device_select(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_physical_device_selector_t* selector,
    iree_allocator_t host_allocator, iree_hal_vulkan_instance_t* out_instance,
    iree_hal_vulkan_physical_device_snapshot_t* out_snapshot) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(driver_options);
  IREE_ASSERT_ARGUMENT(selector);
  IREE_ASSERT_ARGUMENT(out_instance);
  IREE_ASSERT_ARGUMENT(out_snapshot);
  memset(out_instance, 0, sizeof(*out_instance));
  memset(out_snapshot, 0, sizeof(*out_snapshot));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_vulkan_instance_initialize(
      libvulkan, driver_options, host_allocator, out_instance);

  uint32_t physical_device_count = 0;
  VkPhysicalDevice* physical_devices = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_enumerate_physical_device_handles(
        out_instance, host_allocator, &physical_device_count,
        &physical_devices);
  }

  if (iree_status_is_ok(status) && !physical_device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "Vulkan driver has no physical devices");
  }

  if (iree_status_is_ok(status) &&
      selector->mode == IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID) {
    const uint32_t ordinal = (uint32_t)(selector->device_id - 1);
    if (selector->device_id == IREE_HAL_DEVICE_ID_DEFAULT ||
        ordinal >= physical_device_count) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan device id %" PRIu64 " out of range; driver has %u devices",
          (uint64_t)selector->device_id, physical_device_count);
    }
  }

  bool selected = false;
  for (uint32_t i = 0;
       i < physical_device_count && iree_status_is_ok(status) && !selected;
       ++i) {
    iree_hal_vulkan_physical_device_snapshot_t snapshot;
    status = iree_hal_vulkan_physical_device_snapshot_initialize(
        out_instance, physical_devices[i], i, host_allocator, &snapshot);
    if (!iree_status_is_ok(status)) break;

    bool matches = false;
    status = iree_hal_vulkan_physical_device_selector_match(selector, &snapshot,
                                                            &matches);
    if (iree_status_is_ok(status) && matches) {
      if (iree_hal_vulkan_physical_device_supports_baseline(&snapshot)) {
        *out_snapshot = snapshot;
        memset(&snapshot, 0, sizeof(snapshot));
        selected = true;
      } else {
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "Vulkan physical device %u does not satisfy the Vulkan 1.3 "
            "baseline",
            i);
      }
    }
    iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                          &snapshot);
  }

  if (iree_status_is_ok(status) && !selected) {
    switch (selector->mode) {
      case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT:
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "no Vulkan physical device satisfies the Vulkan 1.3 baseline");
        break;
      case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID:
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "Vulkan device id %" PRIu64 " not found",
                                  (uint64_t)selector->device_id);
        break;
      case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH:
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND, "Vulkan device path '%.*s' not found",
            (int)selector->device_path.size, selector->device_path.data);
        break;
    }
  }

  iree_allocator_free(host_allocator, physical_devices);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                          out_snapshot);
    iree_hal_vulkan_instance_deinitialize(out_instance);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
