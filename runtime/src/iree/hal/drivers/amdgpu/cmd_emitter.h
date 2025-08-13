// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_CMD_EMITTER_H_
#define IREE_HAL_DRIVERS_AMDGPU_CMD_EMITTER_H_

#include <stdint.h>

#include "iree/base/api.h"

#define IREE_AMDGPU_RESTRICT IREE_RESTRICT

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

typedef struct iree_hal_amdgpu_cmd_emitter_t iree_hal_amdgpu_cmd_emitter_t;

typedef uint64_t iree_hal_amdgpu_va_addr_t;
typedef uint64_t iree_hal_amdgpu_va_size_t;

typedef uint32_t iree_hal_amdgpu_ib_offset_t;

typedef uint64_t iree_hal_amdgpu_cmd_temporal_hint_t;
enum iree_hal_amdgpu_cmd_temporal_hint_e {
  // Regular temporal (reuse expected).
  IREE_HAL_AMDGPU_CMD_TEMPORAL_RT = 0,
  // Non-temporal (reuse not expected).
  IREE_HAL_AMDGPU_CMD_TEMPORAL_NT = 1,
  // High-priority temporal (precedence over RT).
  IREE_HAL_AMDGPU_CMD_TEMPORAL_HT = 2,
  // Last-use (non-temporal AND discard dirty if it hits).
  IREE_HAL_AMDGPU_CMD_TEMPORAL_LU = 3,
};

typedef uint64_t iree_hal_amdgpu_cmd_scope_t;
enum iree_hal_amdgpu_cmd_scope_e {
  IREE_HAL_AMDGPU_CMD_SCOPE_QUEUE = 0,
  IREE_HAL_AMDGPU_CMD_SCOPE_XCD,
  IREE_HAL_AMDGPU_CMD_SCOPE_DEVICE,
  IREE_HAL_AMDGPU_CMD_SCOPE_SYSTEM,
};

typedef uint64_t iree_hal_amdgpu_cmd_element_size_t;
enum iree_hal_amdgpu_cmd_element_size_e {
  IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_32BITS = 0,
  IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_64BITS = 1,
};

typedef uint64_t iree_hal_amdgpu_cmd_compare_func_t;
enum iree_hal_amdgpu_cmd_compare_func_e {
  // Always pass (unconditional).
  IREE_HAL_AMDGPU_CMD_COMPARE_FUNC_ALWAYS = 0,
  // Pass if value < reference.
  IREE_HAL_AMDGPU_CMD_COMPARE_FUNC_LESS = 1,
  // Pass if value <= reference.
  IREE_HAL_AMDGPU_CMD_COMPARE_FUNC_LESS_EQUAL = 2,
  // Pass if value == reference.
  IREE_HAL_AMDGPU_CMD_COMPARE_FUNC_EQUAL = 3,
  // Pass if value != reference.
  IREE_HAL_AMDGPU_CMD_COMPARE_FUNC_NOT_EQUAL = 4,
  // Pass if value >= reference.
  IREE_HAL_AMDGPU_CMD_COMPARE_FUNC_GREATER_EQUAL = 5,
  // Pass if value > reference.
  IREE_HAL_AMDGPU_CMD_COMPARE_FUNC_GREATER = 6,
  // Never pass (unconditional, no-op).
  IREE_HAL_AMDGPU_CMD_COMPARE_FUNC_NEVER = 7,
};

typedef uint64_t iree_hal_amdgpu_cmd_type_t;
enum iree_hal_amdgpu_cmd_type_e {
  // iree_hal_amdgpu_cmd_intrinsic_t
  IREE_HAL_AMDGPU_CMD_TYPE_INTRINSIC = 0,
  // iree_hal_amdgpu_cmd_memory_ctl_t
  IREE_HAL_AMDGPU_CMD_TYPE_MEMORY_CTL,
  // iree_hal_amdgpu_cmd_atomic_t
  IREE_HAL_AMDGPU_CMD_TYPE_ATOMIC,
  // iree_hal_amdgpu_cmd_alu_t
  IREE_HAL_AMDGPU_CMD_TYPE_ALU,
  // iree_hal_amdgpu_cmd_cond_alu_t
  IREE_HAL_AMDGPU_CMD_TYPE_COND_ALU,
  // iree_hal_amdgpu_cmd_fill_buffer_t
  IREE_HAL_AMDGPU_CMD_TYPE_FILL_BUFFER,
  // iree_hal_amdgpu_cmd_copy_buffer_t
  IREE_HAL_AMDGPU_CMD_TYPE_COPY_BUFFER,
  // iree_hal_amdgpu_cmd_write_buffer_t
  IREE_HAL_AMDGPU_CMD_TYPE_WRITE_BUFFER,
  // iree_hal_amdgpu_cmd_dispatch_direct_t
  IREE_HAL_AMDGPU_CMD_TYPE_DISPATCH_DIRECT,
  // iree_hal_amdgpu_cmd_dispatch_indirect_t
  IREE_HAL_AMDGPU_CMD_TYPE_DISPATCH_INDIRECT,
  // iree_hal_amdgpu_cmd_cond_exec_t
  IREE_HAL_AMDGPU_CMD_TYPE_COND_EXEC,
  // iree_hal_amdgpu_cmd_branch_t
  IREE_HAL_AMDGPU_CMD_TYPE_BRANCH,
  // iree_hal_amdgpu_cmd_cond_branch_t
  IREE_HAL_AMDGPU_CMD_TYPE_COND_BRANCH,
  // iree_hal_amdgpu_cmd_call_t
  IREE_HAL_AMDGPU_CMD_TYPE_CALL,
  // iree_hal_amdgpu_cmd_signal_t
  IREE_HAL_AMDGPU_CMD_TYPE_SIGNAL,
  // iree_hal_amdgpu_cmd_enqueue_ib_t
  IREE_HAL_AMDGPU_CMD_TYPE_ENQUEUE_IB,
  // iree_hal_amdgpu_cmd_enqueue_scheduler_t
  IREE_HAL_AMDGPU_CMD_TYPE_ENQUEUE_SCHEDULER,
};

//===----------------------------------------------------------------------===//
// Patching
//===----------------------------------------------------------------------===//

// DO NOT SUBMIT evaluate if we even want patching like this
// We may need to vtable the patch queries as their exact locations may vary
// per implementation. This currently assumes that we're patching our own
// packets in memory, which is (ideally) not the case in PM4.

typedef uint32_t iree_hal_amdgpu_cmd_patch_type_t;
enum iree_hal_amdgpu_cmd_patch_type_e {
  // Patch a 64-bit constant value loaded from the constant table.
  // src_ordinal must be 8-byte aligned.
  IREE_HAL_AMDGPU_CMD_PATCH_TYPE_CONSTANT_64 = 0,
  // Patch a buffer reference from the binding table via assignment.
  // The existing value being patched will be overwritten.
  IREE_HAL_AMDGPU_CMD_PATCH_TYPE_BINDING_PTR,
  // Patch a buffer reference from the binding table via addition.
  // The existing value being patched will be added to the binding pointer.
  // This allows commands to pre-bake relative addresses from a fixed root
  // address within themselves.
  IREE_HAL_AMDGPU_CMD_PATCH_TYPE_BINDING_PTR_RELATIVE,
};

typedef uint32_t iree_hal_amdgpu_constant_ordinal_t;
typedef uint32_t iree_hal_amdgpu_binding_ordinal_t;

typedef union iree_hal_amdgpu_patch_value_t {
  iree_hal_amdgpu_constant_ordinal_t constant_ordinal;
  iree_hal_amdgpu_binding_ordinal_t binding_ordinal;
} iree_hal_amdgpu_patch_value_t;

typedef struct iree_hal_amdgpu_cmd_patch_t {
  // Type of patch to perform.
  iree_hal_amdgpu_cmd_patch_type_t type : 3;
  // Destination indirect buffer offset in bytes.
  // Must follow the type-specific alignment.
  iree_hal_amdgpu_ib_offset_t dst_offset : 29;
  // Source ordinal in the type-specific table.
  iree_hal_amdgpu_patch_value_t src_value;
} iree_hal_amdgpu_cmd_patch_t;

typedef struct iree_hal_amdgpu_cmd_patch_block_t {
  iree_hal_amdgpu_cmd_patch_t patches[512];
} iree_hal_amdgpu_cmd_patch_block_t;

typedef struct iree_hal_amdgpu_cmd_patch_batch_t {
  uint32_t block_count;
  const iree_hal_amdgpu_cmd_patch_block_t* blocks;
} iree_hal_amdgpu_cmd_patch_batch_t;

// Appends a patch operation to the patch table.
// The last command emitted will be referenced.
// Invalid if no commands have been emitted in the current block.
iree_status_t iree_hal_amdgpu_cmd_emitter_patch_cmd(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_patch_type_t type, size_t cmd_offset,
    iree_hal_amdgpu_patch_value_t src_value);

// Appends a binding table pointer patch.
// Prior to execution the iree_hal_amdgpu_va_addr_t inside of the last command
// at |cmd_offset| will have the resolved pointer from the binding table at
// |binding_ordinal| added to it.
static inline iree_status_t iree_hal_amdgpu_cmd_emitter_patch_cmd_binding_ptr(
    iree_hal_amdgpu_cmd_emitter_t* emitter, size_t cmd_offset,
    iree_hal_amdgpu_binding_ordinal_t binding_ordinal) {
  const iree_hal_amdgpu_patch_value_t src_value = {
      .binding_ordinal = binding_ordinal,
  };
  return iree_hal_amdgpu_cmd_emitter_patch_cmd(
      emitter, IREE_HAL_AMDGPU_CMD_PATCH_TYPE_BINDING_PTR, cmd_offset,
      src_value);
}

// Declares a function to patch a named binding field in a command.
//
// Example:
//   IREE_HAL_AMDGPU_CMD_PATCH_BINDING_PTR(my_cmd, some_binding_ptr);
//   ...
//   status = my_cmd_patch_some_binding_ptr(emitter, binding_ordinal);
#define IREE_HAL_AMDGPU_CMD_PATCH_BINDING_PTR(cmd_type, field)    \
  static inline iree_status_t cmd_type##_patch_value_ptr(         \
      iree_hal_amdgpu_cmd_emitter_t* emitter,                     \
      iree_hal_amdgpu_binding_ordinal_t binding_ordinal) {        \
    return iree_hal_amdgpu_cmd_emitter_patch_cmd_binding_ptr(     \
        emitter, offsetof(cmd_type##_t, field), binding_ordinal); \
  }

//===----------------------------------------------------------------------===//
// Intrinsic Escape
//===----------------------------------------------------------------------===//

// TODO(benvanik): intrinsics.
// These are tiny kernels that do very specific actions that other commands
// can rely on. e.g. enqueuing an IB in an SDMA queue.
iree_status_t iree_hal_amdgpu_cmd_intrinsic(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Memory Control (Acquire/Release)
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_cmd_memory_ctl_t {
  //
} iree_hal_amdgpu_cmd_memory_ctl_t;

iree_status_t iree_hal_amdgpu_cmd_memory_ctl(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Atomic Memory Operation
//===----------------------------------------------------------------------===//

typedef uint64_t iree_hal_amdgpu_cmd_atomic_op_t;
enum iree_hal_amdgpu_cmd_atomic_op_e {
  // *previous_ptr = atomic_fetch_add(object_ptr, operand)
  IREE_HAL_AMDGPU_CMD_ATOMIC_OP_ADD = 0,
  // *previous_ptr = atomic_exchange(object_ptr, operand)
  IREE_HAL_AMDGPU_CMD_ATOMIC_OP_EXCHANGE = 1,
  // *previous_ptr = atomic_compare_exchange(object_ptr, expected, desired)
  IREE_HAL_AMDGPU_CMD_ATOMIC_OP_COMPARE_EXCHANGE = 2,
};

typedef uint64_t iree_hal_amdgpu_cmd_atomic_wait_mode_t;
enum iree_hal_amdgpu_cmd_atomic_wait_mode_e {
  // Single operation, do not loop if comparison fails.
  // If comparison fails the object is left unchanged.
  IREE_HAL_AMDGPU_CMD_ATOMIC_WAIT_MODE_TRY = 0,
  // Loop until the comparison succeeds and then exchange.
  // Hints that the latency of the change is expected to be low (~1-10us) and
  // the queue should try not to yield its timeslice.
  IREE_HAL_AMDGPU_CMD_ATOMIC_WAIT_MODE_LOOP_LATENCY_LOW,
  // Loop until the comparison succeeds and then exchange.
  // Hints that the latency is very high (>100us) and the queue can yield its
  // timeslice.
  IREE_HAL_AMDGPU_CMD_ATOMIC_WAIT_MODE_LOOP_LATENCY_HIGH,
};

typedef struct iree_hal_amdgpu_cmd_atomic_t {
  // IREE_HAL_AMDGPU_CMD_TYPE_ATOMIC.
  iree_hal_amdgpu_cmd_type_t type : 7;
  iree_hal_amdgpu_cmd_scope_t acquire_scope : 3;
  iree_hal_amdgpu_cmd_scope_t release_scope : 3;
  // Atomic operation.
  iree_hal_amdgpu_cmd_atomic_op_t op : 3;
  // Whether a comparison should wait or pass-through.
  iree_hal_amdgpu_cmd_atomic_wait_mode_t wait_mode : 3;
  // Size of the element being operated on.
  iree_hal_amdgpu_cmd_element_size_t element_size : 1;
  // Atomic object holding the value to manipulate.
  // Aligned to the element_size.
  iree_hal_amdgpu_va_addr_t object_ptr;
  // Operand to add/exchange.
  uint64_t operand;
  // Expected value when performing a COMPARE_EXCHANGE operation.
  // The compare succeeds only when *object_ptr exactly matches this value.
  uint64_t expected;
  // Optional address to write the previous value.
  // Aligned to element_size.
  // Passing NULL may allow optimizations.
  iree_hal_amdgpu_va_addr_t previous_ptr;
} iree_hal_amdgpu_cmd_atomic_t;

iree_status_t iree_hal_amdgpu_cmd_atomic(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_cmd_atomic_op_t op,
    iree_hal_amdgpu_cmd_atomic_wait_mode_t wait_mode,
    iree_hal_amdgpu_cmd_element_size_t element_size,
    iree_hal_amdgpu_va_addr_t object_ptr, uint64_t operand, uint64_t expected,
    iree_hal_amdgpu_va_addr_t previous_ptr);

static inline iree_status_t iree_hal_amdgpu_cmd_atomic_fetch_add_32(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_va_addr_t object_ptr, uint32_t operand,
    iree_hal_amdgpu_va_addr_t previous_ptr) {
  return iree_hal_amdgpu_cmd_atomic(
      emitter, acquire_scope, release_scope, IREE_HAL_AMDGPU_CMD_ATOMIC_OP_ADD,
      IREE_HAL_AMDGPU_CMD_ATOMIC_WAIT_MODE_TRY,
      IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_32BITS, object_ptr & ~0x3ull,
      (uint64_t)operand, 0, previous_ptr);
}

static inline iree_status_t iree_hal_amdgpu_cmd_atomic_fetch_add_64(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_va_addr_t object_ptr, uint64_t operand,
    iree_hal_amdgpu_va_addr_t previous_ptr) {
  return iree_hal_amdgpu_cmd_atomic(
      emitter, acquire_scope, release_scope, IREE_HAL_AMDGPU_CMD_ATOMIC_OP_ADD,
      IREE_HAL_AMDGPU_CMD_ATOMIC_WAIT_MODE_TRY,
      IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_64BITS, object_ptr & ~0x7ull, operand, 0,
      previous_ptr);
}

static inline iree_status_t iree_hal_amdgpu_cmd_atomic_exchange_32(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_va_addr_t object_ptr, uint32_t operand,
    iree_hal_amdgpu_va_addr_t previous_ptr) {
  return iree_hal_amdgpu_cmd_atomic(emitter, acquire_scope, release_scope,
                                    IREE_HAL_AMDGPU_CMD_ATOMIC_OP_EXCHANGE,
                                    IREE_HAL_AMDGPU_CMD_ATOMIC_WAIT_MODE_TRY,
                                    IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_32BITS,
                                    object_ptr & ~0x3ull, (uint64_t)operand, 0,
                                    previous_ptr);
}

static inline iree_status_t iree_hal_amdgpu_cmd_atomic_exchange_64(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_va_addr_t object_ptr, uint64_t operand,
    iree_hal_amdgpu_va_addr_t previous_ptr) {
  return iree_hal_amdgpu_cmd_atomic(emitter, acquire_scope, release_scope,
                                    IREE_HAL_AMDGPU_CMD_ATOMIC_OP_EXCHANGE,
                                    IREE_HAL_AMDGPU_CMD_ATOMIC_WAIT_MODE_TRY,
                                    IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_64BITS,
                                    object_ptr & ~0x7ull, operand, 0,
                                    previous_ptr);
}

static inline iree_status_t iree_hal_amdgpu_cmd_atomic_compare_exchange_32(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_cmd_atomic_wait_mode_t wait_mode,
    iree_hal_amdgpu_va_addr_t object_ptr, uint32_t expected, uint32_t desired,
    iree_hal_amdgpu_va_addr_t previous_ptr) {
  return iree_hal_amdgpu_cmd_atomic(
      emitter, acquire_scope, release_scope,
      IREE_HAL_AMDGPU_CMD_ATOMIC_OP_COMPARE_EXCHANGE, wait_mode,
      IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_32BITS, object_ptr & ~0x3ull,
      (uint64_t)desired, (uint64_t)expected, previous_ptr);
}

static inline iree_status_t iree_hal_amdgpu_cmd_atomic_compare_exchange_64(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_cmd_atomic_wait_mode_t wait_mode,
    iree_hal_amdgpu_va_addr_t object_ptr, uint64_t expected, uint64_t desired,
    iree_hal_amdgpu_va_addr_t previous_ptr) {
  return iree_hal_amdgpu_cmd_atomic(
      emitter, acquire_scope, release_scope,
      IREE_HAL_AMDGPU_CMD_ATOMIC_OP_COMPARE_EXCHANGE, wait_mode,
      IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_64BITS, object_ptr & ~0x7ull, desired,
      expected, previous_ptr);
}

//===----------------------------------------------------------------------===//
// Memory Read-Modify-Write (RMW) ALU Op
//===----------------------------------------------------------------------===//

typedef uint64_t iree_hal_amdgpu_cmd_alu_op_t;
enum iree_hal_amdgpu_cmd_alu_op_e {
  // *value_ptr = (*value_ptr & ~mask) | (operand & mask);
  IREE_HAL_AMDGPU_CMD_ALU_OP_IMMEDIATE = 0,
  // *value_ptr = (*value_ptr & ~mask) | ((*value_ptr + operand) & mask);
  IREE_HAL_AMDGPU_CMD_ALU_OP_ADD,
  // *value_ptr = (*value_ptr & ~mask) | ((*value_ptr - operand) & mask);
  IREE_HAL_AMDGPU_CMD_ALU_OP_SUB,
  // *value_ptr = (*value_ptr & ~mask) | ((*value_ptr | operand) & mask);
  IREE_HAL_AMDGPU_CMD_ALU_OP_OR,
  // *value_ptr = (*value_ptr & ~mask) | ((*value_ptr ^ operand) & mask);
  IREE_HAL_AMDGPU_CMD_ALU_OP_XOR,
  // *value_ptr = (*value_ptr & ~mask) | ((*value_ptr << operand) & mask);
  IREE_HAL_AMDGPU_CMD_ALU_OP_LSHL,
  // *value_ptr = (*value_ptr & ~mask) | ((*value_ptr >> operand) & mask);
  IREE_HAL_AMDGPU_CMD_ALU_OP_LSHR,
  // *value_ptr = (*value_ptr & ~mask) | ((*value_ptr a>> operand) & mask);
  IREE_HAL_AMDGPU_CMD_ALU_OP_ASHR,
};

typedef uint64_t iree_hal_amdgpu_cmd_alu_operand_select_t;
enum iree_hal_amdgpu_cmd_alu_operand_select_e {
  // Use the immediate field in iree_hal_amdgpu_cmd_alu_operand_t.
  IREE_HAL_AMDGPU_CMD_ALU_OPERAND_SELECT_IMMEDIATE = 0,
  // Use the pointer field in iree_hal_amdgpu_cmd_alu_operand_t.
  IREE_HAL_AMDGPU_CMD_ALU_OPERAND_SELECT_PTR = 1,
};

// Operand to the ALU operation (RHS, shift amount, etc).
typedef union iree_hal_amdgpu_cmd_alu_operand_t {
  // operand_select = IREE_HAL_AMDGPU_CMD_ALU_OPERAND_SELECT_IMMEDIATE.
  uint64_t immediate;
  // operand_select = IREE_HAL_AMDGPU_CMD_ALU_OPERAND_SELECT_PTR.
  // Aligned to element_size.
  iree_hal_amdgpu_va_addr_t ptr;
} iree_hal_amdgpu_cmd_alu_operand_t;

typedef struct iree_hal_amdgpu_cmd_alu_t {
  // IREE_HAL_AMDGPU_CMD_TYPE_ALU.
  iree_hal_amdgpu_cmd_type_t type : 7;
  iree_hal_amdgpu_cmd_scope_t acquire_scope : 3;
  iree_hal_amdgpu_cmd_scope_t release_scope : 3;
  // ALU operation to perform.
  iree_hal_amdgpu_cmd_alu_op_t op : 4;
  // Size of the element being operated on.
  iree_hal_amdgpu_cmd_element_size_t element_size : 1;
  // Selects how the operand is sourced.
  iree_hal_amdgpu_cmd_alu_operand_select_t operand_select : 1;
  // Pointer to the value to RMW.
  // Aligned to element_size.
  iree_hal_amdgpu_va_addr_t value_ptr;
  // Operand to the ALU operation (RHS, shift amount, etc).
  iree_hal_amdgpu_cmd_alu_operand_t operand;
  // Mask used to only update certain bits of the value_ptr.
  // Any bit set to 0 will not be modified.
  uint64_t mask;
} iree_hal_amdgpu_cmd_alu_t;

iree_status_t iree_hal_amdgpu_cmd_alu(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope, iree_hal_amdgpu_cmd_alu_op_t op,
    iree_hal_amdgpu_cmd_element_size_t element_size,
    iree_hal_amdgpu_va_addr_t value_ptr,
    iree_hal_amdgpu_cmd_alu_operand_select_t operand_select,
    iree_hal_amdgpu_cmd_alu_operand_t operand, uint64_t mask);

static inline iree_status_t iree_hal_amdgpu_cmd_immediate_32(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_va_addr_t value_ptr, uint32_t operand) {
  const iree_hal_amdgpu_cmd_alu_operand_t operand_value = {
      .immediate = (uint64_t)operand,
  };
  return iree_hal_amdgpu_cmd_alu(
      emitter, acquire_scope, release_scope,
      IREE_HAL_AMDGPU_CMD_ALU_OP_IMMEDIATE,
      IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_32BITS, value_ptr,
      IREE_HAL_AMDGPU_CMD_ALU_OPERAND_SELECT_IMMEDIATE, operand_value, ~0ull);
}

static inline iree_status_t iree_hal_amdgpu_cmd_immediate_64(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_va_addr_t value_ptr, uint64_t operand) {
  const iree_hal_amdgpu_cmd_alu_operand_t operand_value = {
      .immediate = operand,
  };
  return iree_hal_amdgpu_cmd_alu(
      emitter, acquire_scope, release_scope,
      IREE_HAL_AMDGPU_CMD_ALU_OP_IMMEDIATE,
      IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_64BITS, value_ptr,
      IREE_HAL_AMDGPU_CMD_ALU_OPERAND_SELECT_IMMEDIATE, operand_value, ~0ull);
}

//===----------------------------------------------------------------------===//
// Conditional Memory Read-Modify-Write (RMW) ALU Op
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_cmd_cond_alu_t {
  // IREE_HAL_AMDGPU_CMD_TYPE_COND_ALU
  iree_hal_amdgpu_cmd_type_t type : 7;
  iree_hal_amdgpu_cmd_scope_t acquire_scope : 3;
  iree_hal_amdgpu_cmd_scope_t release_scope : 3;
  // ALU operation to perform.
  iree_hal_amdgpu_cmd_alu_op_t op : 4;
  // Comparison function used to evaluate the condition.
  iree_hal_amdgpu_cmd_compare_func_t compare_func : 3;
  // Size of the element being operated on.
  iree_hal_amdgpu_cmd_element_size_t element_size : 1;
  // Selects how the operand is sourced.
  iree_hal_amdgpu_cmd_alu_operand_select_t operand_select : 1;
  // Pointer to the value to RMW.
  // Aligned to element_size.
  iree_hal_amdgpu_va_addr_t value_ptr;
  // Reference value used for comparison.
  uint64_t reference;
  // Operand to the ALU operation (RHS, shift amount, etc).
  iree_hal_amdgpu_cmd_alu_operand_t operand;
  // Mask used to only update certain bits of the value_ptr.
  // Any bit set to 0 will not be modified.
  uint64_t mask;
} iree_hal_amdgpu_cmd_cond_alu_t;

IREE_HAL_AMDGPU_CMD_PATCH_BINDING_PTR(iree_hal_amdgpu_cmd_cond_alu, value_ptr);

iree_status_t iree_hal_amdgpu_cmd_emit(iree_hal_amdgpu_cmd_emitter_t* emitter,
                                       iree_hal_amdgpu_cmd_type_t type,
                                       const void* IREE_AMDGPU_RESTRICT cmd);

iree_status_t iree_hal_amdgpu_cmd_cond_alu(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope, iree_hal_amdgpu_cmd_alu_op_t op,
    iree_hal_amdgpu_cmd_element_size_t element_size,
    iree_hal_amdgpu_va_addr_t value_ptr,
    iree_hal_amdgpu_cmd_compare_func_t compare_func, uint64_t reference,
    iree_hal_amdgpu_cmd_alu_operand_select_t operand_select,
    iree_hal_amdgpu_cmd_alu_operand_t operand, uint64_t mask) {
  const iree_hal_amdgpu_cmd_cond_alu_t cmd = {
      .type = IREE_HAL_AMDGPU_CMD_TYPE_COND_ALU,
      .acquire_scope = acquire_scope,
      .release_scope = release_scope,
      .op = op,
      .compare_func = compare_func,
      .element_size = element_size,
      .operand_select = operand_select,
      .value_ptr = value_ptr,
      .reference = reference,
      .operand = operand,
      .mask = mask,
  };
  return iree_hal_amdgpu_cmd_emit(emitter, IREE_HAL_AMDGPU_CMD_TYPE_COND_ALU,
                                  &cmd);
}

static inline iree_status_t iree_hal_amdgpu_cmd_cond_immediate_32(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_va_addr_t value_ptr,
    iree_hal_amdgpu_cmd_compare_func_t compare_func, uint32_t reference,
    uint32_t operand) {
  const iree_hal_amdgpu_cmd_alu_operand_t operand_value = {
      .immediate = (uint32_t)operand,
  };
  const iree_hal_amdgpu_cmd_cond_alu_t cmd = {
      .type = IREE_HAL_AMDGPU_CMD_TYPE_COND_ALU,
      .acquire_scope = acquire_scope,
      .release_scope = release_scope,
      .op = IREE_HAL_AMDGPU_CMD_ALU_OP_IMMEDIATE,
      .compare_func = compare_func,
      .element_size = IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_32BITS,
      .operand_select = IREE_HAL_AMDGPU_CMD_ALU_OPERAND_SELECT_IMMEDIATE,
      .value_ptr = value_ptr,
      .reference = (uint64_t)reference,
      .operand = operand_value,
      .mask = ~0ull,
  };
  return iree_hal_amdgpu_cmd_emit(emitter, IREE_HAL_AMDGPU_CMD_TYPE_COND_ALU,
                                  &cmd);
}

static inline iree_status_t iree_hal_amdgpu_cmd_cond_immediate_64(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope,
    iree_hal_amdgpu_va_addr_t value_ptr,
    iree_hal_amdgpu_cmd_compare_func_t compare_func, uint64_t reference,
    uint64_t operand) {
  const iree_hal_amdgpu_cmd_alu_operand_t operand_value = {
      .immediate = operand,
  };
  const iree_hal_amdgpu_cmd_cond_alu_t cmd = {
      .type = IREE_HAL_AMDGPU_CMD_TYPE_COND_ALU,
      .acquire_scope = acquire_scope,
      .release_scope = release_scope,
      .op = IREE_HAL_AMDGPU_CMD_ALU_OP_IMMEDIATE,
      .compare_func = compare_func,
      .element_size = IREE_HAL_AMDGPU_CMD_ELEMENT_SIZE_64BITS,
      .operand_select = IREE_HAL_AMDGPU_CMD_ALU_OPERAND_SELECT_IMMEDIATE,
      .value_ptr = value_ptr,
      .reference = (uint64_t)reference,
      .operand = operand_value,
      .mask = ~0ull,
  };
  return iree_hal_amdgpu_cmd_emit(emitter, IREE_HAL_AMDGPU_CMD_TYPE_COND_ALU,
                                  &cmd);
}

//===----------------------------------------------------------------------===//
// Fill Buffer
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_cmd_fill_buffer_t {
  // IREE_HAL_AMDGPU_CMD_TYPE_FILL_BUFFER
  iree_hal_amdgpu_cmd_type_t type : 7;
  iree_hal_amdgpu_cmd_scope_t acquire_scope : 3;
  iree_hal_amdgpu_cmd_scope_t release_scope : 3;
  // Temporal hint used for the write destination.
  iree_hal_amdgpu_cmd_temporal_hint_t dst_temporal_hint : 2;
  // Size of the element being operated on.
  iree_hal_amdgpu_cmd_element_size_t element_size : 1;
  // Destination to fill.
  // Aligned to element_size.
  iree_hal_amdgpu_va_addr_t dst_ptr;
  // Length to fill in bytes.
  // Aligned to element_size.
  iree_hal_amdgpu_va_size_t length;
  // Fill pattern. Only element_size bits are used.
  uint64_t pattern;
} iree_hal_amdgpu_cmd_fill_buffer_t;

iree_status_t iree_hal_amdgpu_cmd_fill_buffer(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Copy Buffer
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_cmd_copy_buffer_t {
  // IREE_HAL_AMDGPU_CMD_TYPE_COPY_BUFFER
  iree_hal_amdgpu_cmd_type_t type : 7;
  iree_hal_amdgpu_cmd_scope_t acquire_scope : 3;
  iree_hal_amdgpu_cmd_scope_t release_scope : 3;
  // Temporal hint used for the read source.
  iree_hal_amdgpu_cmd_temporal_hint_t src_temporal_hint : 2;
  // Temporal hint used for the write destination.
  iree_hal_amdgpu_cmd_temporal_hint_t dst_temporal_hint : 2;
  // Size of the element being operated on.
  iree_hal_amdgpu_cmd_element_size_t element_size : 1;
  // Source of the copy.
  // Aligned to element_size.
  iree_hal_amdgpu_va_addr_t src_ptr;
  // Destination of the copy.
  // Aligned to element_size.
  iree_hal_amdgpu_va_addr_t dst_ptr;
  // Length to copy in bytes.
  // Aligned to element_size.
  iree_hal_amdgpu_va_size_t length;
} iree_hal_amdgpu_cmd_copy_buffer_t;

iree_status_t iree_hal_amdgpu_cmd_copy_buffer(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Write Buffer
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_cmd_write_buffer_t {
  // IREE_HAL_AMDGPU_CMD_TYPE_WRITE_BUFFER
  iree_hal_amdgpu_cmd_type_t type : 7;
  iree_hal_amdgpu_cmd_scope_t acquire_scope : 3;
  iree_hal_amdgpu_cmd_scope_t release_scope : 3;
  // Temporal hint used for the write destination.
  iree_hal_amdgpu_cmd_temporal_hint_t dst_temporal_hint : 2;
  // Size of the element being operated on.
  iree_hal_amdgpu_cmd_element_size_t element_size : 1;
  // Source of the write stored inline in the command buffer.
  iree_hal_amdgpu_va_addr_t src_ptr;
  // Destination of the write.
  // Aligned to element_size.
  iree_hal_amdgpu_va_addr_t dst_ptr;
  // Length to write in bytes.
  // Aligned to element_size.
  iree_hal_amdgpu_va_size_t length;
} iree_hal_amdgpu_cmd_write_buffer_t;

iree_status_t iree_hal_amdgpu_cmd_write_buffer(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Dispatch (direct)
//===----------------------------------------------------------------------===//

// DO NOT SUBMIT
// precompute anything we can to keep device side simpler
//
// Kernel arguments used for fixed-size kernels.
// This must match what the kernel was compiled to support.
typedef struct iree_hal_amdgpu_device_kernel_args_t {
  // Opaque handle to the kernel object to execute.
  uint64_t kernel_object;
  // Dispatch setup parameters. Used to configure kernel dispatch parameters
  // such as the number of dimensions in the grid. The parameters are
  // described by hsa_kernel_dispatch_packet_setup_t.
  uint16_t setup;
  // XYZ dimensions of work-group, in work-items. Must be greater than 0.
  // If the grid has fewer than 3 dimensions the unused must be 1.
  uint16_t workgroup_size[3];
  // Size in bytes of private memory allocation request (per work-item).
  uint32_t private_segment_size;
  // Size in bytes of group memory allocation request (per work-group). Must
  // not be less than the sum of the group memory used by the kernel (and the
  // functions it calls directly or indirectly) and the dynamically allocated
  // group segment variables.
  uint32_t group_segment_size;
  // Size of kernarg segment memory that is required to hold the values of the
  // kernel arguments, in bytes. Must be a multiple of 16.
  uint16_t kernarg_size;
  // Alignment (in bytes) of the buffer used to pass arguments to the kernel,
  // which is the maximum of 16 and the maximum alignment of any of the kernel
  // arguments.
  uint16_t kernarg_alignment;
  // Allocated source location in host memory. Inaccessible and only here to
  // feed back to the host for trace processing.
  uint64_t trace_src_loc;
  // Total number of 4-byte constants used by the dispatch (if a HAL dispatch).
  uint16_t constant_count;
  // Total number of bindings used by the dispatch (if a HAL dispatch).
  uint16_t binding_count;
  uint32_t reserved;
} iree_hal_amdgpu_device_kernel_args_t;

// Bitfield specifying flags controlling a dispatch operation.
typedef uint64_t iree_hal_amdgpu_device_dispatch_flags_t;
enum iree_hal_amdgpu_device_dispatch_flag_bits_t {
  IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_NONE = 0,
  // Dispatch requires the iree_amdgpu_kernel_implicit_args_t kernargs to be
  // appended (with 8-byte alignment) to the kernargs.
  IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_IMPLICIT_ARGS = 1u << 0,
  // Dispatch uses an indirect workgroup count that is constant and available
  // prior to command buffer execution. The command processor will read the
  // workgroup count and embed it directly in the AQL kernel dispatch packet.
  IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_STATIC = 1u << 1,
  // Dispatch uses an indirect workgroup count that is dynamic and may change up
  // to the exact moment the dispatch is issue. The command processor will
  // enqueue a kernel that performs the indirection and updates the kernel
  // dispatch packet with the value before allowing the hardware queue to
  // continue.
  IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_DYNAMIC = 1u << 2,
};

// Either directly embedded workgroup count XYZ dimensions or a thin buffer ref
// pointing to a buffer containing the `uint32_t dims[3]` count.
typedef union iree_hal_amdgpu_device_workgroup_count_t {
  // XYZ dimensions of the grid, in workgroups. Must be greater than 0.
  // If the grid has fewer than 3 dimensions the unused ones must be 1.
  // Unused if the dispatch is indirect and instead the workgroups buffer
  // reference in the parameters is used.
  uint32_t dims[3];
  // Optional buffer containing the workgroup count.
  // Processing is controlled by the
  // IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_* flags.
  iree_hal_amdgpu_device_workgroup_count_buffer_ref_t ref;
} iree_hal_amdgpu_device_workgroup_count_t;

typedef struct iree_hal_amdgpu_cmd_dispatch_direct_t {
  // IREE_HAL_AMDGPU_CMD_TYPE_DISPATCH_DIRECT
  iree_hal_amdgpu_cmd_type_t type : 7;
  iree_hal_amdgpu_cmd_scope_t acquire_scope : 3;
  iree_hal_amdgpu_cmd_scope_t release_scope : 3;
  iree_hal_amdgpu_device_dispatch_flags_t flags : 3;
  uint64_t dynamic_lds_size : 16;
  uint64_t kernarg_offset : 32;
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args;
  // References describing how binding pointers are passed to the kernel.
  // References may include direct device pointers, allocation or slots in the
  // binding table included as part of the execution request.
  const iree_hal_amdgpu_device_buffer_ref_t* bindings /*[binding_count]*/;
  // Dispatch constants passed to the kernel.
  const uint32_t* constants /*[constant_count]*/;
} iree_hal_amdgpu_cmd_dispatch_direct_t;

iree_status_t iree_hal_amdgpu_cmd_dispatch_direct(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Dispatch (indirect)
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_cmd_dispatch_indirect_t {
  // IREE_HAL_AMDGPU_CMD_TYPE_DISPATCH_INDIRECT
  iree_hal_amdgpu_cmd_type_t type : 7;
  // iree_hal_amdgpu_cmd_scope_t acquire_scope : 3;
  // iree_hal_amdgpu_cmd_scope_t release_scope : 3;
  iree_hal_amdgpu_device_dispatch_flags_t flags : 3;
  uint64_t dynamic_lds_size : 16;
  uint64_t kernarg_offset : 32;
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args;
  iree_hal_amdgpu_device_workgroup_count_buffer_ref_t workgroup_count;
  // References describing how binding pointers are passed to the kernel.
  // References may include direct device pointers, allocation or slots in the
  // binding table included as part of the execution request.
  const iree_hal_amdgpu_device_buffer_ref_t* bindings /*[binding_count]*/;
  // Dispatch constants passed to the kernel.
  const uint32_t* constants /*[constant_count]*/;
} iree_hal_amdgpu_cmd_dispatch_indirect_t;

iree_status_t iree_hal_amdgpu_cmd_dispatch_indirect(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Conditional Execution
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_cmd_cond_exec(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Unconditional Branch
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_cmd_branch(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Conditional Branch
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_cmd_cond_branch(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Call (nested packet buffer)
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_cmd_call(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Signal
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_cmd_signal(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Enqueue Packet Buffer
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_cmd_enqueue_ib(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

//===----------------------------------------------------------------------===//
// Enqueue Scheduler
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_cmd_enqueue_scheduler(
    iree_hal_amdgpu_cmd_emitter_t* emitter,
    iree_hal_amdgpu_cmd_scope_t acquire_scope,
    iree_hal_amdgpu_cmd_scope_t release_scope);

#endif  // IREE_HAL_DRIVERS_AMDGPU_CMD_EMITTER_H_
