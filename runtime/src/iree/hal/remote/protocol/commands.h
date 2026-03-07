// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL remote protocol: command buffer serialized commands.
//
// Serialized command stream format for command buffer recordings. Commands are
// packed sequentially with self-describing headers (type + length). Two
// delivery paths:
//
//   - Inline (one-shot): Command stream appended directly to a
//     COMMAND_BUFFER_EXECUTE queue op. The hot path for compiler-generated
//     workloads. Large streams are fragmented across preceding DATA frames
//     with pages sent as they fill during recording.
//
//   - Uploaded (reusable): Complete command stream sent via
//     COMMAND_BUFFER_UPLOAD on the control channel. Executed later via
//     COMMAND_BUFFER_EXECUTE referencing the uploaded resource ID.
//
// ## Dependency policy
//
// Includes common.h for shared wire format types (resource IDs, bindings,
// dispatch config). Does not include HAL headers.

#ifndef IREE_HAL_REMOTE_PROTOCOL_COMMANDS_H_
#define IREE_HAL_REMOTE_PROTOCOL_COMMANDS_H_

#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Command buffer command types
//===----------------------------------------------------------------------===//

// Command buffer serialized command type identifiers. Commands are packed
// sequentially in a command stream (inline in COMMAND_BUFFER_EXECUTE or
// uploaded via COMMAND_BUFFER_UPLOAD). Each command is self-describing via
// its header's type and length fields.
typedef enum iree_hal_remote_cmd_type_e {
  // ── Synchronization ─────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER = 0x0001,

  // ── Buffer ──────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CMD_BUFFER_ADVISE = 0x0010,
  IREE_HAL_REMOTE_CMD_BUFFER_FILL = 0x0011,
  IREE_HAL_REMOTE_CMD_BUFFER_UPDATE = 0x0012,
  IREE_HAL_REMOTE_CMD_BUFFER_COPY = 0x0013,

  // ── Execution ───────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CMD_DISPATCH = 0x0020,

  // ── Debug ───────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CMD_DEBUG_GROUP_BEGIN = 0x0030,
  IREE_HAL_REMOTE_CMD_DEBUG_GROUP_END = 0x0031,

  // ── Extensions ──────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CMD_COMMAND_EXTENSION = 0x00F0,
} iree_hal_remote_cmd_type_t;

//===----------------------------------------------------------------------===//
// Command buffer command header
//===----------------------------------------------------------------------===//

// Common header for all serialized commands. 8 bytes: type and length plus
// padding for natural alignment of subsequent uint64_t fields.
//
// The length field covers the entire command including this header and any
// variable-length tail, rounded up to the next 8-byte boundary. The stream
// iterator advances by `length` bytes to reach the next command.
typedef struct iree_hal_remote_cmd_header_t {
  uint16_t type;      // iree_hal_remote_cmd_type_t
  uint16_t length;    // Total bytes (header + payload), multiple of 8.
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_cmd_header_t;
static_assert(sizeof(iree_hal_remote_cmd_header_t) == 8, "");

//===----------------------------------------------------------------------===//
// Barrier entries
//===----------------------------------------------------------------------===//

// Memory barrier entry for EXECUTION_BARRIER commands.
typedef struct iree_hal_remote_memory_barrier_t {
  uint32_t source_scope;  // iree_hal_access_scope_t
  uint32_t target_scope;  // iree_hal_access_scope_t
} iree_hal_remote_memory_barrier_t;
static_assert(sizeof(iree_hal_remote_memory_barrier_t) == 8, "");

// Buffer barrier entry for EXECUTION_BARRIER commands.
typedef struct iree_hal_remote_buffer_barrier_t {
  uint32_t source_scope;  // iree_hal_access_scope_t
  uint32_t target_scope;  // iree_hal_access_scope_t
  iree_hal_remote_resource_id_t buffer_id;
  uint64_t offset;
  uint64_t length;
} iree_hal_remote_buffer_barrier_t;
static_assert(sizeof(iree_hal_remote_buffer_barrier_t) == 32, "");
static_assert(offsetof(iree_hal_remote_buffer_barrier_t, buffer_id) == 8, "");

//===----------------------------------------------------------------------===//
// Command buffer command payloads
//===----------------------------------------------------------------------===//

// EXECUTION_BARRIER: Insert a pipeline barrier.
// Variable-length tail: memory barriers followed by buffer barriers.
typedef struct iree_hal_remote_execution_barrier_cmd_t {
  iree_hal_remote_cmd_header_t header;
  uint32_t source_stage_mask;  // iree_hal_execution_stage_t
  uint32_t target_stage_mask;  // iree_hal_execution_stage_t
  uint32_t barrier_flags;      // Reserved, must be 0.
  uint16_t memory_barrier_count;
  uint16_t buffer_barrier_count;
  // Followed by:
  //   iree_hal_remote_memory_barrier_t memory_barriers[memory_barrier_count]
  //   iree_hal_remote_buffer_barrier_t buffer_barriers[buffer_barrier_count]
  //   (total command padded to 8-byte alignment)
} iree_hal_remote_execution_barrier_cmd_t;
static_assert(sizeof(iree_hal_remote_execution_barrier_cmd_t) == 24, "");

// BUFFER_ADVISE: Provide a hint about buffer usage.
typedef struct iree_hal_remote_buffer_advise_cmd_t {
  iree_hal_remote_cmd_header_t header;
  iree_hal_remote_resource_id_t buffer_id;
  uint64_t offset;
  uint64_t length;
  uint32_t advise_flags;
  uint32_t reserved;   // Must be 0.
  uint64_t argument0;  // Advise-specific.
  uint64_t argument1;  // Advise-specific.
} iree_hal_remote_buffer_advise_cmd_t;
static_assert(sizeof(iree_hal_remote_buffer_advise_cmd_t) == 56, "");

// BUFFER_FILL: Fill a buffer region with a repeating pattern (within a
// command buffer recording).
typedef struct iree_hal_remote_buffer_fill_cmd_t {
  iree_hal_remote_cmd_header_t header;
  iree_hal_remote_resource_id_t target_buffer_id;
  uint64_t target_offset;
  uint64_t target_length;
  uint8_t pattern_length;  // 1, 2, or 4 bytes.
  uint8_t reserved0[3];    // Must be 0.
  uint32_t pattern;        // Zero-extended if pattern_length < 4.
  uint32_t fill_flags;
  uint32_t reserved1;  // Must be 0.
} iree_hal_remote_buffer_fill_cmd_t;
static_assert(sizeof(iree_hal_remote_buffer_fill_cmd_t) == 48, "");

// BUFFER_UPDATE: Write inline host data into a buffer region (within a
// command buffer recording). Variable-length tail: source data.
typedef struct iree_hal_remote_buffer_update_cmd_t {
  iree_hal_remote_cmd_header_t header;
  iree_hal_remote_resource_id_t target_buffer_id;
  uint64_t target_offset;
  uint64_t target_length;
  uint32_t update_flags;
  uint32_t reserved;  // Must be 0.
  // Followed by:
  //   uint8_t source_data[target_length]  (padded to 8-byte alignment)
} iree_hal_remote_buffer_update_cmd_t;
static_assert(sizeof(iree_hal_remote_buffer_update_cmd_t) == 40, "");

// BUFFER_COPY: Copy between buffer regions (within a command buffer
// recording).
typedef struct iree_hal_remote_buffer_copy_cmd_t {
  iree_hal_remote_cmd_header_t header;
  iree_hal_remote_resource_id_t source_buffer_id;
  uint64_t source_offset;
  iree_hal_remote_resource_id_t target_buffer_id;
  uint64_t target_offset;
  uint64_t length;
  uint32_t copy_flags;
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_buffer_copy_cmd_t;
static_assert(sizeof(iree_hal_remote_buffer_copy_cmd_t) == 56, "");

// DISPATCH: Execute a compute dispatch (within a command buffer recording).
// Variable-length tail: constants array followed by bindings array.
typedef struct iree_hal_remote_dispatch_cmd_t {
  iree_hal_remote_cmd_header_t header;
  iree_hal_remote_resource_id_t executable_id;
  uint32_t export_ordinal;
  uint32_t reserved0;  // Must be 0.
  iree_hal_remote_dispatch_config_t config;
  uint16_t constant_count;
  uint16_t binding_count;
  uint32_t reserved1;       // Must be 0.
  uint64_t dispatch_flags;  // iree_hal_dispatch_flags_t
  // Followed by:
  //   uint32_t constants[constant_count]  (padded to 8-byte alignment)
  //   iree_hal_remote_binding_t bindings[binding_count]
} iree_hal_remote_dispatch_cmd_t;
static_assert(sizeof(iree_hal_remote_dispatch_cmd_t) == 96, "");
static_assert(offsetof(iree_hal_remote_dispatch_cmd_t, config) == 24, "");
static_assert(offsetof(iree_hal_remote_dispatch_cmd_t, dispatch_flags) == 88,
              "");

// DEBUG_GROUP_BEGIN: Begin a named debug group. Variable-length tail:
// UTF-8 label string and optional source location.
typedef struct iree_hal_remote_debug_group_begin_cmd_t {
  iree_hal_remote_cmd_header_t header;
  uint32_t label_color;   // RGBA packed color for debug visualization.
  uint16_t label_length;  // UTF-8 byte count (not null-terminated).
  uint8_t has_location;   // Nonzero if source location follows the label.
  uint8_t reserved;       // Must be 0.
  // Followed by:
  //   uint8_t label[label_length]  (padded to 8-byte alignment)
  //   [if has_location]:
  //     uint16_t file_length
  //     uint16_t line
  //     uint32_t reserved  (must be 0)
  //     uint8_t file[file_length]  (padded to 8-byte alignment)
} iree_hal_remote_debug_group_begin_cmd_t;
static_assert(sizeof(iree_hal_remote_debug_group_begin_cmd_t) == 16, "");

// DEBUG_GROUP_END: End the current debug group. No payload.
typedef struct iree_hal_remote_debug_group_end_cmd_t {
  iree_hal_remote_cmd_header_t header;
} iree_hal_remote_debug_group_end_cmd_t;
static_assert(sizeof(iree_hal_remote_debug_group_end_cmd_t) == 8, "");

// COMMAND_EXTENSION: Device-specific command within a recording. The server
// dispatches by device_type + operation.
typedef struct iree_hal_remote_command_extension_cmd_t {
  iree_hal_remote_cmd_header_t header;
  uint32_t device_type;     // Namespace (CUDA=1, HIP=2, Vulkan=3, ...).
  uint32_t operation;       // Extension-defined operation code.
  uint32_t payload_length;  // Byte count of opaque payload.
  uint32_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t payload[payload_length]  (padded to 8-byte alignment)
} iree_hal_remote_command_extension_cmd_t;
static_assert(sizeof(iree_hal_remote_command_extension_cmd_t) == 24, "");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_PROTOCOL_COMMANDS_H_
