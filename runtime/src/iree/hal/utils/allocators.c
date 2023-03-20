// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/allocators.h"

#include "iree/base/tracing.h"
#include "iree/hal/utils/caching_allocator.h"
#include "iree/hal/utils/debug_allocator.h"

iree_status_t iree_hal_configure_allocator_from_spec(
    iree_string_view_t spec, iree_hal_device_t* device,
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_t** out_wrapped_allocator) {
  iree_string_view_t allocator_name = iree_string_view_empty();
  iree_string_view_t config_pairs = iree_string_view_empty();
  iree_string_view_split(spec, ':', &allocator_name, &config_pairs);
  iree_status_t status = iree_ok_status();
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(base_allocator);
  if (iree_string_view_equal(allocator_name, IREE_SV("caching"))) {
    status = iree_hal_caching_allocator_create_from_spec(
        config_pairs, base_allocator, host_allocator, out_wrapped_allocator);
  } else if (iree_string_view_equal(allocator_name, IREE_SV("debug"))) {
    status = iree_hal_debug_allocator_create(
        device, base_allocator, host_allocator, out_wrapped_allocator);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized allocator '%.*s'",
                            (int)allocator_name.size, allocator_name.data);
  }
  if (iree_status_is_ok(status)) {
    // New wrapping allocator has taken ownership of the base allocator.
    iree_hal_allocator_release(base_allocator);
  }
  return status;
}

iree_status_t iree_hal_configure_allocator_from_specs(
    iree_host_size_t spec_count, const iree_string_view_t* specs,
    iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // The current device allocator should be the base one registered or created
  // with the device. If no allocator specs were provided this may be no-op and
  // we'll just pass it right back in.
  iree_hal_allocator_t* device_allocator = iree_hal_device_allocator(device);
  iree_hal_allocator_retain(device_allocator);

  // Walk the specs provided and wrap in order from base to last specified.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < spec_count; ++i) {
    status = iree_hal_configure_allocator_from_spec(
        specs[i], device, device_allocator, &device_allocator);
    if (!iree_status_is_ok(status)) break;
  }

  // Swap the allocator on the device - this is only safe because we know no
  // allocations have been made yet.
  if (iree_status_is_ok(status)) {
    iree_hal_device_replace_allocator(device, device_allocator);
  }
  iree_hal_allocator_release(device_allocator);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
