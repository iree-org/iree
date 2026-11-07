// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DESCRIPTOR_RANGE_H_
#define IREE_HAL_DRIVERS_VULKAN_DESCRIPTOR_RANGE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Resolves the physical length of a storage descriptor for the logical range
// [|offset|, |offset| + |length|) of a bound view of |view_length| bytes.
//
// A shader that accesses the binding in units of |minimum_access_bytes| reads
// and writes whole units, so the physical length rounds up to that unit. The
// rounded tail must lie within the bytes the binding owns: its declared range,
// or |owned_length| (the view's bytes, or the allocation's when the view is a
// whole allocation) when the range reaches the end of the view. |length| may be
// IREE_HAL_WHOLE_BUFFER. An empty logical range resolves to 0, which the
// caller binds to a dummy buffer without requiring offset alignment.
// |out_length| may be NULL for validation only.
// A zero |minimum_access_bytes| is treated as one.
// Fails with OUT_OF_RANGE when the range does
// not fit the view, the owned bytes or |max_range|, and with INVALID_ARGUMENT
// when |offset| is not a multiple of |minimum_access_bytes|.
iree_status_t iree_hal_vulkan_resolve_descriptor_range(
    iree_device_size_t view_length, iree_device_size_t owned_length,
    iree_device_size_t offset, iree_device_size_t length,
    uint32_t minimum_access_bytes, iree_device_size_t max_range,
    iree_device_size_t* out_length);

// Resolves a BDA binding within the same logical and owned ranges as above.
// Non-empty bindings require a non-zero |base_address|. The resulting address
// must satisfy both |minimum_alignment| and |minimum_access_bytes|, and the
// logical range must contain at least |minimum_length| bytes. Checks address
// overflow through the physical end of the last access unit. Returns address
// zero for an empty range that meets |minimum_length|; the caller substitutes
// its dummy binding address.
// Units and alignment must be positive powers of two.
iree_status_t iree_hal_vulkan_resolve_bda_range(
    iree_device_size_t view_length, iree_device_size_t owned_length,
    iree_device_size_t offset, iree_device_size_t length, uint64_t base_address,
    uint32_t minimum_access_bytes, uint32_t minimum_alignment,
    uint64_t minimum_length, uint64_t* out_device_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_DESCRIPTOR_RANGE_H_
