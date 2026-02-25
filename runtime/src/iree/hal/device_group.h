// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DEVICE_GROUP_H_
#define IREE_HAL_DEVICE_GROUP_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/device.h"
#include "iree/hal/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_device_group_t
//===----------------------------------------------------------------------===//

// A group of HAL devices with an immutable topology matrix describing their
// relationships.
//
// The device group is the runtime representation of a set of devices
// cooperating on a workload. It owns (retains) all devices added to it and
// builds an immutable topology matrix during creation that encodes directional
// device relationships: synchronization modes, buffer sharing capabilities,
// and relative costs.
//
// The topology is computed once during group creation from device capabilities
// and optional driver-specific refinement. After creation both the device set
// and topology are immutable. Devices cache a pointer to the group's topology
// for lock-free concurrent edge queries.
//
// Lifetime: the group must outlive all its devices. Whoever holds the devices
// long-term (e.g., the HAL module) must also retain the group. The group
// retains all its devices, so devices cannot be destroyed while the group
// exists.
//
// Thread safety: the group is immutable after creation and can be queried
// concurrently from any thread without synchronization.
typedef struct iree_hal_device_group_t iree_hal_device_group_t;

// Creates a single-device group.
// Equivalent to initializing a builder, adding the device, and finalizing.
IREE_API_EXPORT iree_status_t iree_hal_device_group_create_from_device(
    iree_hal_device_t* device, iree_allocator_t host_allocator,
    iree_hal_device_group_t** out_group);

// Retains the given |group| for the caller.
IREE_API_EXPORT void iree_hal_device_group_retain(
    iree_hal_device_group_t* group);

// Releases the given |group| from the caller.
IREE_API_EXPORT void iree_hal_device_group_release(
    iree_hal_device_group_t* group);

// Returns the number of devices in the group.
IREE_API_EXPORT iree_host_size_t
iree_hal_device_group_device_count(const iree_hal_device_group_t* group);

// Returns the device at |index| in the group.
// The returned pointer is valid for the lifetime of the group. Callers must
// retain the device if they need it to outlive the group.
IREE_API_EXPORT iree_hal_device_t* iree_hal_device_group_device_at(
    const iree_hal_device_group_t* group, iree_host_size_t index);

// Returns a pointer to the immutable topology matrix.
// Valid for the lifetime of the group.
IREE_API_EXPORT const iree_hal_topology_t* iree_hal_device_group_topology(
    const iree_hal_device_group_t* group);

//===----------------------------------------------------------------------===//
// iree_hal_device_group_builder_t
//===----------------------------------------------------------------------===//

// Builder for constructing device groups.
//
// The builder collects devices and is consumed by finalize, which builds the
// topology matrix and creates the group. The builder is stack-allocated and
// does not retain devices (it is ephemeral). After finalize the builder is
// invalidated (zeroed) and must not be reused.
//
// Usage:
//   iree_hal_device_group_builder_t builder;
//   iree_hal_device_group_builder_initialize(&builder);
//   iree_hal_device_group_builder_add_device(&builder, device_a);
//   iree_hal_device_group_builder_add_device(&builder, device_b);
//   iree_hal_device_group_t* group = NULL;
//   iree_hal_device_group_builder_finalize(&builder, allocator, &group);
//   // builder is now zeroed — use group.
typedef struct iree_hal_device_group_builder_t {
  iree_host_size_t count;
  iree_hal_device_t* devices[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
} iree_hal_device_group_builder_t;

// Initializes a device group builder. The builder should be stack-allocated.
IREE_API_EXPORT void iree_hal_device_group_builder_initialize(
    iree_hal_device_group_builder_t* builder);

// Deinitializes a device group builder. Safe to call on an already-finalized
// (zeroed) builder.
IREE_API_EXPORT void iree_hal_device_group_builder_deinitialize(
    iree_hal_device_group_builder_t* builder);

// Adds a |device| to the builder. Does NOT retain the device (the builder
// is ephemeral). Insertion order determines topology index: the first device
// added is index 0.
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the builder is at max capacity.
IREE_API_EXPORT iree_status_t iree_hal_device_group_builder_add_device(
    iree_hal_device_group_builder_t* builder, iree_hal_device_t* device);

// Builds the topology, creates the device group, and invalidates the builder.
//
// This queries capabilities from all devices, computes base edges via
// iree_hal_topology_edge_from_capabilities(), calls
// iree_hal_device_refine_topology_edge() for same-driver pairs, finalizes
// the topology, and pushes topology info into each device via
// iree_hal_device_assign_topology_info().
//
// The group retains all devices. The builder is zeroed after this call
// (whether it succeeds or fails) and must not be reused.
IREE_API_EXPORT iree_status_t iree_hal_device_group_builder_finalize(
    iree_hal_device_group_builder_t* builder, iree_allocator_t host_allocator,
    iree_hal_device_group_t** out_group);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DEVICE_GROUP_H_
