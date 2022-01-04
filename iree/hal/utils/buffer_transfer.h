// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_BUFFER_TRANSFER_H_
#define IREE_HAL_UTILS_BUFFER_TRANSFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_device_transfer_range implementations
//===----------------------------------------------------------------------===//

// Performs a full transfer operation on a device transfer queue.
// This creates a transfer command buffer, submits it against the device, and
// waits for it to complete synchronously. Implementations that can do this
// cheaper are encouraged to do so.
//
// Precondition: source and target do not overlap.
IREE_API_EXPORT iree_status_t iree_hal_device_submit_transfer_range_and_wait(
    iree_hal_device_t* device, iree_hal_transfer_buffer_t source,
    iree_device_size_t source_offset, iree_hal_transfer_buffer_t target,
    iree_device_size_t target_offset, iree_device_size_t data_length,
    iree_hal_transfer_buffer_flags_t flags, iree_timeout_t timeout);

// Generic implementation of iree_hal_device_transfer_range for when the buffers
// are mappable. In certain implementations even if buffers are mappable it's
// often cheaper to still use the full queue transfers: instead of wasting CPU
// cycles copying the memory (and possible PCIe round-trips) letting the device
// do it is effectively free.
//
// Precondition: source and target do not overlap.
IREE_API_EXPORT iree_status_t iree_hal_device_transfer_mappable_range(
    iree_hal_device_t* device, iree_hal_transfer_buffer_t source,
    iree_device_size_t source_offset, iree_hal_transfer_buffer_t target,
    iree_device_size_t target_offset, iree_device_size_t data_length,
    iree_hal_transfer_buffer_flags_t flags, iree_timeout_t timeout);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_map_range implementations
//===----------------------------------------------------------------------===//

// Generic implementation of iree_hal_buffer_map_range and unmap_range for when
// the buffer is not mappable and a full device transfer is required. This will
// allocate additional host-local buffers and submit copy commands.
// Implementations able to do this more efficiently should do so.
IREE_API_EXPORT iree_status_t iree_hal_buffer_emulated_map_range(
    iree_hal_device_t* device, iree_hal_buffer_t* buffer,
    iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping);
IREE_API_EXPORT iree_status_t iree_hal_buffer_emulated_unmap_range(
    iree_hal_device_t* device, iree_hal_buffer_t* buffer,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_BUFFER_TRANSFER_H_
