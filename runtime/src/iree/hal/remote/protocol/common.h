// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL remote protocol: shared wire format types.
//
// Resource identifiers, buffer parameters, dispatch configuration, and other
// building blocks used across all three protocol levels (control, queue,
// command buffer). Each level's header includes this one.
//
// ## Wire format conventions
//
//   - All multi-byte fields are little-endian.
//   - All structs are naturally aligned (no packed attributes).
//   - Variable-length data is padded to 8-byte alignment.
//   - Reserved fields MUST be zero on send; receivers MUST reject nonzero.
//
// ## Dependency policy
//
// This header depends only on iree/base/api.h. Wire format types are defined
// as fixed-width typedefs that are semantically equivalent to their HAL
// counterparts. This avoids pulling HAL headers into wire format parsing code.
// Translation is trivial (same sizes, same bit layouts) except for
// iree_device_size_t, which varies by platform — the wire format always uses
// 64-bit sizes.

#ifndef IREE_HAL_REMOTE_PROTOCOL_COMMON_H_
#define IREE_HAL_REMOTE_PROTOCOL_COMMON_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Resource IDs
//===----------------------------------------------------------------------===//

// Session-local resource identifier. 64-bit local_id within an implicit
// session scope. The full 128-bit resource ID (scope_id + local_id) is used
// in server-side resource tables; the wire format carries only the local_id
// since session scope is implicit per-connection.
//
// Encoding:
//   Type       [63:56]  8 bits   Resource type (buffer, semaphore, etc.)
//   Flags      [55:48]  8 bits   PROVISIONAL (bit 0), others reserved.
//   Generation [47:32]  16 bits  ABA prevention for slot reuse.
//   Proactor   [31:24]  8 bits   Server-assigned proactor index.
//   Slot       [23:0]   24 bits  Index in proactor's resource table.
//
// Client-assigned IDs have PROVISIONAL=1 and Proactor/Slot=0. The server
// resolves these to canonical IDs (PROVISIONAL=0, real Proactor+Slot) and
// piggybacks the resolution on ADVANCE frames.
typedef uint64_t iree_hal_remote_resource_id_t;

// Resource type codes (bits [63:56] of resource ID).
typedef enum iree_hal_remote_resource_type_e {
  IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER = 0x01,
  IREE_HAL_REMOTE_RESOURCE_TYPE_SEMAPHORE = 0x02,
  IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE = 0x03,
  IREE_HAL_REMOTE_RESOURCE_TYPE_COMMAND_BUFFER = 0x04,
  IREE_HAL_REMOTE_RESOURCE_TYPE_FILE = 0x05,
} iree_hal_remote_resource_type_t;

// Resource ID flag bits (bits [55:48] of resource ID).
#define IREE_HAL_REMOTE_RESOURCE_FLAG_PROVISIONAL (1u << 0)

// Resource ID field accessors.
#define IREE_HAL_REMOTE_RESOURCE_ID_TYPE(id) \
  ((iree_hal_remote_resource_type_t)(((id) >> 56) & 0xFF))
#define IREE_HAL_REMOTE_RESOURCE_ID_FLAGS(id) (((id) >> 48) & 0xFF)
#define IREE_HAL_REMOTE_RESOURCE_ID_GENERATION(id) \
  ((uint16_t)(((id) >> 32) & 0xFFFF))
#define IREE_HAL_REMOTE_RESOURCE_ID_PROACTOR(id) \
  ((uint8_t)(((id) >> 24) & 0xFF))
#define IREE_HAL_REMOTE_RESOURCE_ID_SLOT(id) ((uint32_t)((id) & 0xFFFFFF))
#define IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(id) \
  ((IREE_HAL_REMOTE_RESOURCE_ID_FLAGS(id) &            \
    IREE_HAL_REMOTE_RESOURCE_FLAG_PROVISIONAL) != 0)

// Constructs a provisional resource ID with the given type and generation.
// Proactor and slot are zero (filled in by server on resolution).
#define IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(type, generation)                 \
  ((iree_hal_remote_resource_id_t)(((uint64_t)(type) << 56) |                     \
                                   ((uint64_t)                                    \
                                        IREE_HAL_REMOTE_RESOURCE_FLAG_PROVISIONAL \
                                    << 48) |                                      \
                                   ((uint64_t)(generation) << 32)))

//===----------------------------------------------------------------------===//
// Shared wire format types
//===----------------------------------------------------------------------===//

// Buffer parameters for allocation requests. Fixed-width wire equivalent of
// iree_hal_buffer_params_t. All fields match their HAL counterparts in size
// and bit layout except min_alignment, which is always 64-bit on the wire
// (iree_device_size_t is platform-dependent).
typedef struct iree_hal_remote_buffer_params_t {
  uint32_t usage;           // iree_hal_buffer_usage_t
  uint16_t access;          // iree_hal_memory_access_t
  uint16_t reserved0;       // Must be 0.
  uint32_t type;            // iree_hal_memory_type_t
  uint32_t reserved1;       // Must be 0.
  uint64_t queue_affinity;  // iree_hal_queue_affinity_t
  uint64_t min_alignment;   // Always 64-bit on wire.
} iree_hal_remote_buffer_params_t;
static_assert(sizeof(iree_hal_remote_buffer_params_t) == 32, "");
static_assert(offsetof(iree_hal_remote_buffer_params_t, usage) == 0, "");
static_assert(offsetof(iree_hal_remote_buffer_params_t, access) == 4, "");
static_assert(offsetof(iree_hal_remote_buffer_params_t, type) == 8, "");
static_assert(offsetof(iree_hal_remote_buffer_params_t, queue_affinity) == 16,
              "");
static_assert(offsetof(iree_hal_remote_buffer_params_t, min_alignment) == 24,
              "");

// Buffer binding entry. Used in DISPATCH ops/cmds and COMMAND_BUFFER_EXECUTE
// binding tables.
typedef struct iree_hal_remote_binding_t {
  iree_hal_remote_resource_id_t buffer_id;  // 0 = indirect via buffer_slot.
  uint64_t offset;
  uint64_t length;
  uint32_t buffer_slot;  // Binding table index when buffer_id == 0.
  uint32_t reserved;     // Must be 0.
} iree_hal_remote_binding_t;
static_assert(sizeof(iree_hal_remote_binding_t) == 32, "");
static_assert(offsetof(iree_hal_remote_binding_t, buffer_id) == 0, "");
static_assert(offsetof(iree_hal_remote_binding_t, offset) == 8, "");
static_assert(offsetof(iree_hal_remote_binding_t, length) == 16, "");
static_assert(offsetof(iree_hal_remote_binding_t, buffer_slot) == 24, "");

// Dispatch workgroup configuration. Shared between queue DISPATCH ops and
// command buffer DISPATCH commands.
typedef struct iree_hal_remote_dispatch_config_t {
  uint32_t workgroup_size[3];
  uint32_t workgroup_count[3];
  iree_hal_remote_resource_id_t workgroup_count_buffer_id;  // 0 = static.
  uint64_t workgroup_count_offset;
  uint64_t workgroup_count_length;
  uint32_t dynamic_workgroup_local_memory;
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_dispatch_config_t;
static_assert(sizeof(iree_hal_remote_dispatch_config_t) == 56, "");
static_assert(offsetof(iree_hal_remote_dispatch_config_t,
                       workgroup_count_buffer_id) == 24,
              "");

// Memory heap description. Used in DEVICE_QUERY_INFO and BUFFER_QUERY_HEAPS
// responses. Wire equivalent of iree_hal_allocator_memory_heap_t.
typedef struct iree_hal_remote_memory_heap_t {
  uint32_t type;           // iree_hal_memory_type_t
  uint32_t allowed_usage;  // iree_hal_buffer_usage_t
  uint64_t max_allocation_size;
  uint64_t min_alignment;
} iree_hal_remote_memory_heap_t;
static_assert(sizeof(iree_hal_remote_memory_heap_t) == 24, "");
static_assert(offsetof(iree_hal_remote_memory_heap_t, max_allocation_size) == 8,
              "");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_PROTOCOL_COMMON_H_
