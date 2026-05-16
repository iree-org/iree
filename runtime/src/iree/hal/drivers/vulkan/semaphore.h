// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_VULKAN_SEMAPHORE_H_

#include <stdint.h>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_logical_device_t
    iree_hal_vulkan_logical_device_t;

typedef uint8_t iree_hal_vulkan_last_signal_flags_t;
enum iree_hal_vulkan_last_signal_flag_bits_e {
  IREE_HAL_VULKAN_LAST_SIGNAL_FLAG_NONE = 0u,

  // The cache contains a producer queue axis/epoch/value snapshot.
  IREE_HAL_VULKAN_LAST_SIGNAL_FLAG_VALID = 1u << 0,

  // Waiting on the producer queue epoch exactly covers the signal frontier.
  IREE_HAL_VULKAN_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT = 1u << 1,
};

// Seqlock-protected cache of the most recent queue signal on a semaphore.
typedef struct iree_hal_vulkan_last_signal_t {
  // Seqlock sequence counter; odd means a writer is updating payload fields.
  iree_atomic_int32_t sequence;

  // Cached signal validity and producer-frontier precision flags.
  iree_hal_vulkan_last_signal_flags_t flags;

  // Reserved bytes kept zero so the payload stays naturally aligned.
  uint8_t reserved[3];

  // Producer queue axis that submitted the last cached signal.
  iree_async_axis_t producer_axis;

  // Producer queue epoch associated with the last cached signal.
  uint64_t epoch;

  // Semaphore payload value signaled at producer_axis/epoch.
  uint64_t value;
} iree_hal_vulkan_last_signal_t;

// Stores a new last-signal snapshot. Thread-safe.
void iree_hal_vulkan_last_signal_store(
    iree_hal_vulkan_last_signal_t* cache,
    iree_hal_vulkan_last_signal_flags_t flags, iree_async_axis_t producer_axis,
    uint64_t epoch, uint64_t value);

// Loads a last-signal snapshot. Returns false when no valid signal is cached.
bool iree_hal_vulkan_last_signal_load(
    const iree_hal_vulkan_last_signal_t* cache,
    iree_hal_vulkan_last_signal_flags_t* out_flags,
    iree_async_axis_t* out_producer_axis, uint64_t* out_epoch,
    uint64_t* out_value);

// Creates a Vulkan HAL semaphore backed by a native timeline VkSemaphore.
iree_status_t iree_hal_vulkan_semaphore_create(
    iree_hal_vulkan_logical_device_t* device,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is a Vulkan timeline semaphore.
bool iree_hal_vulkan_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns true if |semaphore| belongs to |device|.
bool iree_hal_vulkan_semaphore_is_local(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_vulkan_logical_device_t* device);

// Returns the Vulkan semaphore creation flags.
iree_hal_semaphore_flags_t iree_hal_vulkan_semaphore_flags(
    iree_hal_semaphore_t* semaphore);

// Returns the Vulkan semaphore creation queue affinity.
iree_hal_queue_affinity_t iree_hal_vulkan_semaphore_queue_affinity(
    iree_hal_semaphore_t* semaphore);

// Returns the native Vulkan timeline semaphore handle.
iree_status_t iree_hal_vulkan_semaphore_handle(iree_hal_semaphore_t* semaphore,
                                               VkSemaphore* out_handle);

// Returns a pointer to the queue last-signal cache.
iree_hal_vulkan_last_signal_t* iree_hal_vulkan_semaphore_last_signal(
    iree_hal_semaphore_t* semaphore);

// Publishes submission-time frontier metadata for a future queue signal.
bool iree_hal_vulkan_semaphore_publish_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier, uint64_t producer_epoch,
    uint64_t producer_value);

// Advances the HAL/async timeline after a native queue signal has retired.
iree_status_t iree_hal_vulkan_semaphore_retire_signal(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SEMAPHORE_H_
