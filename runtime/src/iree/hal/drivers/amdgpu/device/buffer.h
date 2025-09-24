// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_BUFFER_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"

typedef struct iree_hal_amdgpu_device_allocator_pool_t
    iree_hal_amdgpu_device_allocator_pool_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_allocation_handle_t
//===----------------------------------------------------------------------===//

// Fat allocation pool identifier used to allow both the host and the device to
// route to their respective pool implementations without lookups.
typedef struct iree_hal_amdgpu_device_allocation_pool_id_t {
  // Device-side pool in the memory space of the device that owns the
  // allocation. Note that this may not be the local device.
  iree_hal_amdgpu_device_allocator_pool_t* device_pool;
  // Opaque host-side pool token.
  uint64_t host_pool;
} iree_hal_amdgpu_device_allocation_pool_id_t;

// A handle for a dynamically device-allocated pointer.
// The owner of the handle is responsible for storing it in device-visible
// memory and consistently passing it in buffer references with the
// IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE type. The device will dereference
// the handle to get the actual pointer before using it. Device-side allocs and
// frees will update the pointer in queue-order. The handle contents are only
// valid on the device between an alloca/dealloca pair and we assume the client
// code is not going to do something invalid (free and then try to use the
// handle).
//
// Though the on-device allocator is usually responsible for manipulating the
// handle there are cases where the host or a remote device may need to. For
// example if the user has the last iree_hal_buffer_t reference and drops it
// we'll need to enqueue a device-side deallocation to handle the cleanup. To
// avoid extra round-trips we also optimize for host-side pool growth by
// allowing the host to initialize the handle after it has grown a pool without
// needing to requeue the device allocation.
typedef struct iree_hal_amdgpu_device_allocation_handle_t {
  // Allocated pointer, if any assigned.
  void* ptr;
  // Pool identifier the pointer resides in.
  iree_hal_amdgpu_device_allocation_pool_id_t pool_id;
  // Opaque data used by the allocator.
  struct {
    // TODO(benvanik): block the allocation resides in and other information
    // the allocator needs to avoid lookups when deallocating.
    int reserved;
  } metadata;
} iree_hal_amdgpu_device_allocation_handle_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_ref_t
//===----------------------------------------------------------------------===//

// Identifies the type of a buffer reference and how it should be resolved.
typedef uint8_t iree_hal_amdgpu_device_buffer_type_t;
enum iree_hal_amdgpu_device_buffer_type_e {
  // Reference is to an absolute device pointer that can be directly accessed.
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_PTR = 0u,
  // Reference is to a queue-ordered allocation handle that is only valid at
  // the time the buffer is committed. The handle will be valid for the lifetime
  // of the logical buffer and any resources referencing it but the pointer must
  // only be resolved between a corresponding alloca/dealloca.
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE,
  // Reference is to a slot in the binding table provided during execution.
  // Only one indirection is allowed (table slots cannot reference other slots
  // - yet).
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT,
};

// The ordinal of a slot in the binding table.
typedef uint32_t iree_hal_amdgpu_device_buffer_ordinal_t;

// Describes a subrange of a buffer that can be bound to a binding slot.
typedef struct iree_hal_amdgpu_device_buffer_ref_t {
  // Offset, in bytes, into the buffer that the binding starts at.
  // This will be added to the offset specified on each usage of the slot.
  uint64_t offset;
  // Type of the buffer reference used to resolve the device pointer.
  uint64_t type : 2;
  // Length, in bytes, of the buffer that is available to the executable.
  uint64_t length : 62;
  union {
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_PTR: device pointer.
    void* ptr;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE: queue-ordered allocation
    // handle.
    iree_hal_amdgpu_device_allocation_handle_t* handle;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT: binding table slot.
    iree_hal_amdgpu_device_buffer_ordinal_t slot;
    // Used for setting the value.
    uint64_t bits;
  } value;
} iree_hal_amdgpu_device_buffer_ref_t;
static_assert(sizeof(iree_hal_amdgpu_device_buffer_ref_t) == 24,
              "binding table entries should be 8 byte aligned");

// Describes a buffer binding that contains a uint32_t[3] XYZ workgroup count.
// This is a size-optimized version of iree_hal_amdgpu_device_buffer_ref_t so
// that it will fit in our tiny packets. We know the length is a constant 12 and
// only need the offset, type, and value.
typedef struct iree_hal_amdgpu_device_workgroup_count_buffer_ref_t {
  // Type of the buffer reference used to resolve the device pointer.
  uint64_t type : 2;  // iree_hal_amdgpu_device_buffer_type_t
  // Offset, in bytes, into the buffer that the binding starts at.
  // This will be added to the offset specified on each usage of the slot.
  uint64_t offset : 62;
  union {
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_PTR: raw device pointer.
    void* ptr;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE: queue-ordered allocation
    // handle.
    iree_hal_amdgpu_device_allocation_handle_t* handle;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT: binding table slot.
    iree_hal_amdgpu_device_buffer_ordinal_t slot;
    // Used for setting the value.
    uint64_t bits;
  } value;
} iree_hal_amdgpu_device_workgroup_count_buffer_ref_t;
static_assert(sizeof(iree_hal_amdgpu_device_workgroup_count_buffer_ref_t) == 16,
              "binding table entries should be 8 byte aligned and tiny");

#define iree_hal_amdgpu_device_workgroup_count_buffer_ref_length(buffer_ref) \
  (sizeof(uint32_t) * 3)

// Describes a buffer binding that contains a single uint64_t value.
// This is a size-optimized version of iree_hal_amdgpu_device_buffer_ref_t so
// that it will fit in our tiny packets. We know the length is a constant 8 and
// only need the offset, type, and value.
typedef struct iree_hal_amdgpu_device_uint64_buffer_ref_t {
  // Type of the buffer reference used to resolve the device pointer.
  uint64_t type : 2;  // iree_hal_amdgpu_device_buffer_type_t
  // Offset, in bytes, into the buffer that the binding starts at.
  // This will be added to the offset specified on each usage of the slot.
  uint64_t offset : 62;
  union {
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_PTR: raw device pointer.
    void* ptr;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE: queue-ordered allocation
    // handle.
    iree_hal_amdgpu_device_allocation_handle_t* handle;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT: binding table slot.
    iree_hal_amdgpu_device_buffer_ordinal_t slot;
    // Used for setting the value.
    uint64_t bits;
  } value;
} iree_hal_amdgpu_device_uint64_buffer_ref_t;
static_assert(sizeof(iree_hal_amdgpu_device_uint64_buffer_ref_t) == 16,
              "binding table entries should be 8 byte aligned and tiny");

#define iree_hal_amdgpu_device_uint64_buffer_ref_length(buffer_ref) \
  sizeof(uint64_t)

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Resolves a buffer reference to an absolute device pointer.
// Expects that the binding table is provided if needed and has sufficient
// capacity for any slot that may be referenced. All queue-ordered allocations
// that may be provided via allocation handles must be committed prior to
// attempting to resolve them and must remain committed until all commands using
// the returned device pointer have completed.
void* iree_hal_amdgpu_device_buffer_ref_resolve(
    iree_hal_amdgpu_device_buffer_ref_t buffer_ref,
    IREE_AMDGPU_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_AMDGPU_RESTRICT
            binding_table);

// Resolves a workgroup count buffer reference to an absolute device pointer.
// This is equivalent to iree_hal_amdgpu_device_buffer_ref_resolve but for a
// fixed-size uint32_t[3] value. The returned pointer should have 4-byte
// alignment.
void* iree_hal_amdgpu_device_workgroup_count_buffer_ref_resolve(
    iree_hal_amdgpu_device_workgroup_count_buffer_ref_t buffer_ref,
    IREE_AMDGPU_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_AMDGPU_RESTRICT
            binding_table);

// Resolves a scalar uint64_t buffer reference to an absolute device pointer.
// This is equivalent to iree_hal_amdgpu_device_buffer_ref_resolve but for a
// fixed-size uint64_t value. The returned pointer should have 8-byte
// alignment.
void* iree_hal_amdgpu_device_uint64_buffer_ref_resolve(
    iree_hal_amdgpu_device_uint64_buffer_ref_t buffer_ref,
    IREE_AMDGPU_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_AMDGPU_RESTRICT
            binding_table);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_BUFFER_H_
