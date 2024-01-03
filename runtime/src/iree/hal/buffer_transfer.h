// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_TRANSFER_H_
#define IREE_HAL_BUFFER_TRANSFER_H_

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Human-friendly/performance-hostile transfer APIs
//===----------------------------------------------------------------------===//

// A transfer source or destination.
typedef struct iree_hal_transfer_buffer_t {
  // A host-allocated void* buffer.
  iree_byte_span_t host_buffer;
  // A device-allocated buffer (may be of any memory type).
  iree_hal_buffer_t* device_buffer;
} iree_hal_transfer_buffer_t;

static inline iree_hal_transfer_buffer_t iree_hal_make_host_transfer_buffer(
    iree_byte_span_t host_buffer) {
  iree_hal_transfer_buffer_t transfer_buffer = {
      host_buffer,
      NULL,
  };
  return transfer_buffer;
}

static inline iree_hal_transfer_buffer_t
iree_hal_make_host_transfer_buffer_span(void* ptr, iree_host_size_t length) {
  iree_hal_transfer_buffer_t transfer_buffer = {
      iree_make_byte_span(ptr, length),
      NULL,
  };
  return transfer_buffer;
}

static inline iree_hal_transfer_buffer_t iree_hal_make_device_transfer_buffer(
    iree_hal_buffer_t* device_buffer) {
  iree_hal_transfer_buffer_t transfer_buffer = {
      iree_byte_span_empty(),
      device_buffer,
  };
  return transfer_buffer;
}

// Synchronously copies data from |source| into |target|.
//
// Supports host->device, device->host, and device->device transfer,
// including across devices. This method will never fail based on device
// capabilities but may incur some extreme transient allocations and copies in
// order to perform the transfer.
//
// The ordering of the transfer is undefined with respect to queue execution on
// the source or target device; some may require full device flushes in order to
// perform this operation while others may immediately perform it while there is
// still work outstanding.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_copy_buffer.
IREE_API_EXPORT iree_status_t iree_hal_device_transfer_range(
    iree_hal_device_t* device, iree_hal_transfer_buffer_t source,
    iree_device_size_t source_offset, iree_hal_transfer_buffer_t target,
    iree_device_size_t target_offset, iree_device_size_t data_length,
    iree_hal_transfer_buffer_flags_t flags, iree_timeout_t timeout);

// Synchronously copies data from host |source| into device |target|.
// Convience wrapper around iree_hal_device_transfer_range.
IREE_API_EXPORT iree_status_t iree_hal_device_transfer_h2d(
    iree_hal_device_t* device, const void* source, iree_hal_buffer_t* target,
    iree_device_size_t target_offset, iree_device_size_t data_length,
    iree_hal_transfer_buffer_flags_t flags, iree_timeout_t timeout);

// Synchronously copies data from device |source| into host |target|.
// Convience wrapper around iree_hal_device_transfer_range.
IREE_API_EXPORT iree_status_t iree_hal_device_transfer_d2h(
    iree_hal_device_t* device, iree_hal_buffer_t* source,
    iree_device_size_t source_offset, void* target,
    iree_device_size_t data_length, iree_hal_transfer_buffer_flags_t flags,
    iree_timeout_t timeout);

// Synchronously copies data from device |source| into device |target|.
// Convience wrapper around iree_hal_device_transfer_range.
IREE_API_EXPORT iree_status_t iree_hal_device_transfer_d2d(
    iree_hal_device_t* device, iree_hal_buffer_t* source,
    iree_device_size_t source_offset, iree_hal_buffer_t* target,
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

#endif  // IREE_HAL_BUFFER_TRANSFER_H_
