// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_QUEUE_AFFINITY_H_
#define IREE_HAL_DRIVERS_AMDGPU_QUEUE_AFFINITY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Describes the AMDGPU logical-device queue affinity domain.
typedef struct iree_hal_amdgpu_queue_affinity_domain_t {
  // Queue bits supported by the logical device.
  iree_hal_queue_affinity_t supported_affinity;

  // Number of physical GPU devices in the logical device.
  iree_host_size_t physical_device_count;

  // Logical queues assigned to each physical GPU device.
  iree_host_size_t queue_count_per_physical_device;
} iree_hal_amdgpu_queue_affinity_domain_t;

// Resolved queue affinity selection in the flattened HAL queue space.
typedef struct iree_hal_amdgpu_queue_affinity_resolved_t {
  // Queue bits remaining after applying the supported affinity mask.
  iree_hal_queue_affinity_t queue_affinity;

  // Flattened logical queue ordinal selected from |queue_affinity|.
  iree_host_size_t queue_ordinal;

  // Physical GPU device ordinal owning |queue_ordinal|.
  iree_host_size_t physical_device_ordinal;

  // Queue ordinal relative to |physical_device_ordinal|.
  iree_host_size_t physical_queue_ordinal;
} iree_hal_amdgpu_queue_affinity_resolved_t;

// Normalizes |requested_affinity| against |supported_affinity|.
//
// IREE_HAL_QUEUE_AFFINITY_ANY expands to |supported_affinity|. Explicit masks
// are intersected with |supported_affinity|, matching HAL queue submission
// behavior where a multi-bit mask means "any of these queues that exist".
iree_status_t iree_hal_amdgpu_queue_affinity_normalize(
    iree_hal_queue_affinity_t supported_affinity,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_queue_affinity_t* out_normalized_affinity);

// Resolves a flattened logical queue ordinal within |domain|.
iree_status_t iree_hal_amdgpu_queue_affinity_resolve_ordinal(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_host_size_t queue_ordinal,
    iree_hal_amdgpu_queue_affinity_resolved_t* out_resolved);

// Resolves |requested_affinity| to the deterministic first selected queue.
iree_status_t iree_hal_amdgpu_queue_affinity_resolve(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_amdgpu_queue_affinity_resolved_t* out_resolved);

// Attempts to resolve |requested_affinity| without constructing a status.
//
// This is for compatibility queries and other predicate-style cold paths where
// invalid input is expected to return false instead of producing diagnostics.
bool iree_hal_amdgpu_queue_affinity_try_resolve(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_amdgpu_queue_affinity_resolved_t* out_resolved);

// Builds the queue affinity mask for one physical device in |domain|.
iree_status_t iree_hal_amdgpu_queue_affinity_for_physical_device(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_host_size_t physical_device_ordinal,
    iree_hal_queue_affinity_t* out_queue_affinity);

// Normalizes |requested_affinity| to queues owned by a single physical device.
//
// IREE_HAL_QUEUE_AFFINITY_ANY selects all supported queues for the first
// supported physical device in the domain. Explicit masks must not span
// physical devices after intersecting with |domain.supported_affinity|.
iree_status_t iree_hal_amdgpu_queue_affinity_normalize_for_physical_device(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_queue_affinity_t* out_queue_affinity,
    iree_host_size_t* out_physical_device_ordinal);

// Returns true if |requested_affinity| selects only queues on one device.
bool iree_hal_amdgpu_queue_affinity_is_physical_device_local(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_hal_queue_affinity_t requested_affinity,
    iree_host_size_t physical_device_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_QUEUE_AFFINITY_H_
