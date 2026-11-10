// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/descriptor_range.h"

iree_status_t iree_hal_vulkan_resolve_descriptor_range(
    iree_device_size_t view_length, iree_device_size_t owned_length,
    iree_device_size_t offset, iree_device_size_t length,
    uint32_t minimum_access_bytes, iree_device_size_t max_range,
    iree_device_size_t* out_length) {
  if (out_length) *out_length = 0;
  if (offset > view_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "descriptor offset %" PRIu64 " is past the %" PRIu64
                            "-byte view",
                            (uint64_t)offset, (uint64_t)view_length);
  }
  if (length == IREE_HAL_WHOLE_BUFFER) length = view_length - offset;
  iree_device_size_t logical_end = 0;
  if (!iree_device_size_checked_add(offset, length, &logical_end) ||
      logical_end > view_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "descriptor range of %" PRIu64 " bytes at offset %" PRIu64
        " exceeds the %" PRIu64 "-byte view",
        (uint64_t)length, (uint64_t)offset, (uint64_t)view_length);
  }
  if (length == 0) return iree_ok_status();

  iree_device_size_t unit = minimum_access_bytes ? minimum_access_bytes : 1;
  if (offset % unit != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "descriptor offset %" PRIu64
                            " is not a multiple of the %" PRIu64
                            "-byte access unit",
                            (uint64_t)offset, (uint64_t)unit);
  }
  iree_device_size_t physical_end = 0;
  if (!iree_device_size_checked_add(logical_end, unit - 1, &physical_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "descriptor range end overflows");
  }
  physical_end -= physical_end % unit;
  // A narrowed range owns its declared bytes; one reaching the view's end also
  // owns the view's tail.
  const iree_device_size_t owned_end =
      logical_end == view_length ? owned_length : logical_end;
  if (physical_end > owned_end) {
    if (logical_end != view_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "binding at offset %" PRIu64 " needs %" PRIu64 " bytes of %" PRIu64
          "-byte access units but owns %" PRIu64
          " bytes; declare the padding it reserves in the binding range",
          (uint64_t)offset, (uint64_t)(physical_end - offset), (uint64_t)unit,
          (uint64_t)(owned_end > offset ? owned_end - offset : 0));
    }
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "binding at offset %" PRIu64 " needs %" PRIu64 " bytes of %" PRIu64
        "-byte access units but its buffer provides %" PRIu64
        " bytes; bind a whole allocation or a view covering the access unit",
        (uint64_t)offset, (uint64_t)(physical_end - offset), (uint64_t)unit,
        (uint64_t)(owned_end > offset ? owned_end - offset : 0));
  }
  iree_device_size_t physical_length = physical_end - offset;
  if (physical_length > max_range) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "descriptor range of %" PRIu64
                            " bytes exceeds the device limit of %" PRIu64,
                            (uint64_t)physical_length, (uint64_t)max_range);
  }
  if (out_length) *out_length = physical_length;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_resolve_bda_range(
    iree_device_size_t view_length, iree_device_size_t owned_length,
    iree_device_size_t offset, iree_device_size_t length, uint64_t base_address,
    uint32_t minimum_access_bytes, uint32_t minimum_alignment,
    uint64_t minimum_length, uint64_t* out_device_address) {
  IREE_ASSERT_ARGUMENT(out_device_address);
  *out_device_address = 0;
  iree_device_size_t logical_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_resolve_descriptor_range(
      view_length, owned_length, offset, length, /*minimum_access_bytes=*/1,
      UINT64_MAX, &logical_length));
  if (logical_length < minimum_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "BDA binding has length %" PRIdsz
                            " but pipeline requires at least %" PRIu64 " bytes",
                            logical_length, minimum_length);
  }
  if (logical_length == 0) return iree_ok_status();
  if (base_address == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "BDA binding buffer has no device address");
  }
  if (offset > UINT64_MAX - base_address) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "BDA binding device address overflows");
  }
  const uint64_t device_address = base_address + offset;
  // BDA aligns the final address, not the offset within an arbitrary view.
  // Once aligned, the physical tail follows the same ownership rules as a
  // descriptor starting at that address.
  const iree_device_size_t owned_range_length =
      offset + logical_length == view_length ? owned_length - offset
                                             : logical_length;
  iree_device_size_t physical_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_resolve_descriptor_range(
      logical_length, owned_range_length, /*offset=*/0, logical_length,
      minimum_access_bytes, UINT64_MAX, &physical_length));
  if (physical_length > UINT64_MAX - device_address) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "BDA binding physical device range overflows");
  }
  const uint32_t alignment = iree_max(minimum_alignment, minimum_access_bytes);
  if (alignment > 1 && (device_address & (alignment - 1)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BDA binding device address 0x%" PRIx64
                            " does not satisfy alignment %u",
                            device_address, alignment);
  }
  *out_device_address = device_address;
  return iree_ok_status();
}
