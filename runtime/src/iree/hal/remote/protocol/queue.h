// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL remote protocol: queue channel operations.
//
// Frontier-ordered HAL operations carried on queue channel COMMAND frames
// (client → server) and ADVANCE frames (server → client). Each COMMAND frame
// carries exactly one queue op; frontier flags bind a single wait/signal pair
// to the operation.
//
// ## Dependency policy
//
// Includes common.h for shared wire format types (resource IDs, buffer params,
// bindings, dispatch config). Does not include HAL headers.

#ifndef IREE_HAL_REMOTE_PROTOCOL_QUEUE_H_
#define IREE_HAL_REMOTE_PROTOCOL_QUEUE_H_

#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// ADVANCE frame resolution entries
//===----------------------------------------------------------------------===//

// Handle resolution entry piggybacked on ADVANCE frames (server → client).
// Maps a client-assigned provisional ID to the server's canonical ID.
typedef struct iree_hal_remote_resolution_entry_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  iree_hal_remote_resource_id_t resolved_id;     // PROVISIONAL=0
} iree_hal_remote_resolution_entry_t;
static_assert(sizeof(iree_hal_remote_resolution_entry_t) == 16, "");

// ADVANCE frame payload prefix. Follows the queue frame header (type=ADVANCE).
// The frontier itself precedes this (encoded per iree/async/frontier.h wire
// format). This prefix introduces the resolution table.
typedef struct iree_hal_remote_advance_payload_t {
  uint16_t resolution_count;
  uint16_t reserved0;  // Must be 0.
  uint32_t reserved1;  // Must be 0.
  // Followed by:
  //   iree_hal_remote_resolution_entry_t resolutions[resolution_count]
} iree_hal_remote_advance_payload_t;
static_assert(sizeof(iree_hal_remote_advance_payload_t) == 8, "");

//===----------------------------------------------------------------------===//
// Queue operation types
//===----------------------------------------------------------------------===//

// Queue operation type identifiers. Each COMMAND frame on the queue channel
// carries exactly one operation. The frame's frontier flags bind a single
// wait/signal pair to the operation.
typedef enum iree_hal_remote_queue_op_type_e {
  // ── Buffer ──────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_QUEUE_OP_BUFFER_ALLOCA = 0x0001,
  IREE_HAL_REMOTE_QUEUE_OP_BUFFER_DEALLOCA = 0x0002,
  IREE_HAL_REMOTE_QUEUE_OP_BUFFER_FILL = 0x0003,
  IREE_HAL_REMOTE_QUEUE_OP_BUFFER_UPDATE = 0x0004,
  IREE_HAL_REMOTE_QUEUE_OP_BUFFER_COPY = 0x0005,

  // ── File ────────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_QUEUE_OP_FILE_READ = 0x0006,
  IREE_HAL_REMOTE_QUEUE_OP_FILE_WRITE = 0x0007,

  // ── Execution ───────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_QUEUE_OP_DISPATCH = 0x0008,
  IREE_HAL_REMOTE_QUEUE_OP_COMMAND_BUFFER_EXECUTE = 0x0009,
  IREE_HAL_REMOTE_QUEUE_OP_HOST_CALL_INVOKE = 0x000A,

  // ── Queue Control ───────────────────────────────────────────────────────
  IREE_HAL_REMOTE_QUEUE_OP_QUEUE_FLUSH = 0x000B,

  // ── Extensions ──────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_QUEUE_OP_QUEUE_EXTENSION = 0x00F0,
} iree_hal_remote_queue_op_type_t;

//===----------------------------------------------------------------------===//
// Queue operation header
//===----------------------------------------------------------------------===//

// Common header for all queue operations. 8 bytes: 4 bytes of type+flags plus
// 4 bytes of padding to ensure subsequent uint64_t fields are naturally
// aligned.
typedef struct iree_hal_remote_queue_op_header_t {
  uint16_t type;      // iree_hal_remote_queue_op_type_t
  uint16_t flags;     // Per-op flag bits (see individual ops).
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_queue_op_header_t;
static_assert(sizeof(iree_hal_remote_queue_op_header_t) == 8, "");

// Queue op flag bits for COMMAND_BUFFER_EXECUTE.
// When set, the serialized command stream follows the binding table instead
// of using a pre-uploaded command buffer ID.
#define IREE_HAL_REMOTE_EXECUTE_FLAG_INLINE_COMMAND_STREAM (1u << 0)

//===----------------------------------------------------------------------===//
// Queue operation payloads
//===----------------------------------------------------------------------===//

// BUFFER_ALLOCA: Allocate a transient buffer from a queue-ordered pool.
// The buffer is usable in subsequent queue ops on the same stream once the
// signal frontier is reached. Client provides a provisional ID; the server
// resolves it and piggybacks the resolution on a later ADVANCE.
typedef struct iree_hal_remote_buffer_alloca_op_t {
  iree_hal_remote_queue_op_header_t header;
  uint32_t pool;       // iree_hal_allocator_pool_t
  uint32_t reserved0;  // Must be 0.
  iree_hal_remote_buffer_params_t params;
  uint64_t allocation_size;
  uint64_t alloca_flags;  // iree_hal_alloca_flags_t
  iree_hal_remote_resource_id_t provisional_buffer_id;  // PROVISIONAL=1
} iree_hal_remote_buffer_alloca_op_t;
static_assert(sizeof(iree_hal_remote_buffer_alloca_op_t) == 72, "");
static_assert(offsetof(iree_hal_remote_buffer_alloca_op_t, pool) == 8, "");
static_assert(offsetof(iree_hal_remote_buffer_alloca_op_t, params) == 16, "");
static_assert(offsetof(iree_hal_remote_buffer_alloca_op_t, allocation_size) ==
                  48,
              "");

// BUFFER_DEALLOCA: Deallocate a transient buffer. Queue-ordered: the buffer
// is freed after the wait frontier is satisfied.
typedef struct iree_hal_remote_buffer_dealloca_op_t {
  iree_hal_remote_queue_op_header_t header;
  iree_hal_remote_resource_id_t buffer_id;
  uint64_t dealloca_flags;  // iree_hal_dealloca_flags_t
} iree_hal_remote_buffer_dealloca_op_t;
static_assert(sizeof(iree_hal_remote_buffer_dealloca_op_t) == 24, "");

// BUFFER_FILL: Fill a buffer region with a repeating pattern.
typedef struct iree_hal_remote_buffer_fill_op_t {
  iree_hal_remote_queue_op_header_t header;
  iree_hal_remote_resource_id_t target_buffer_id;
  uint64_t target_offset;
  uint64_t length;
  uint8_t pattern_length;  // 1, 2, or 4 bytes.
  uint8_t reserved0[3];    // Must be 0.
  uint32_t pattern;        // Zero-extended if pattern_length < 4.
  uint64_t fill_flags;     // iree_hal_fill_flags_t
} iree_hal_remote_buffer_fill_op_t;
static_assert(sizeof(iree_hal_remote_buffer_fill_op_t) == 48, "");

// BUFFER_UPDATE: Write inline host data to a buffer region. The source data
// follows this struct, padded to 8-byte alignment. Limited to max queue frame
// payload size (~64KB). Larger host→device transfers use the bulk channel.
typedef struct iree_hal_remote_buffer_update_op_t {
  iree_hal_remote_queue_op_header_t header;
  iree_hal_remote_resource_id_t target_buffer_id;
  uint64_t target_offset;
  uint64_t length;
  uint64_t update_flags;  // Reserved, must be 0.
  // Followed by:
  //   uint8_t source_data[length]  (padded to 8-byte alignment)
} iree_hal_remote_buffer_update_op_t;
static_assert(sizeof(iree_hal_remote_buffer_update_op_t) == 40, "");

// BUFFER_COPY: Copy between two buffer regions.
typedef struct iree_hal_remote_buffer_copy_op_t {
  iree_hal_remote_queue_op_header_t header;
  iree_hal_remote_resource_id_t source_buffer_id;
  uint64_t source_offset;
  iree_hal_remote_resource_id_t target_buffer_id;
  uint64_t target_offset;
  uint64_t length;
  uint64_t copy_flags;  // iree_hal_copy_flags_t
} iree_hal_remote_buffer_copy_op_t;
static_assert(sizeof(iree_hal_remote_buffer_copy_op_t) == 56, "");

// FILE_READ: Read from a server-side file into a device buffer.
// Source file must have been opened via FILE_OPEN or registered via
// FILE_REGISTER on the control channel.
typedef struct iree_hal_remote_file_read_op_t {
  iree_hal_remote_queue_op_header_t header;
  iree_hal_remote_resource_id_t source_file_id;  // From FILE_OPEN/REGISTER.
  uint64_t source_offset;
  iree_hal_remote_resource_id_t target_buffer_id;
  uint64_t target_offset;
  uint64_t length;
  uint64_t read_flags;  // Reserved, must be 0.
} iree_hal_remote_file_read_op_t;
static_assert(sizeof(iree_hal_remote_file_read_op_t) == 56, "");

// FILE_WRITE: Write from a device buffer to a server-side file.
typedef struct iree_hal_remote_file_write_op_t {
  iree_hal_remote_queue_op_header_t header;
  iree_hal_remote_resource_id_t source_buffer_id;
  uint64_t source_offset;
  iree_hal_remote_resource_id_t target_file_id;  // From FILE_OPEN/REGISTER.
  uint64_t target_offset;
  uint64_t length;
  uint64_t write_flags;  // Reserved, must be 0.
} iree_hal_remote_file_write_op_t;
static_assert(sizeof(iree_hal_remote_file_write_op_t) == 56, "");

// DISPATCH: Execute a compute dispatch.
// Variable-length tail: constants array followed by bindings array.
typedef struct iree_hal_remote_dispatch_op_t {
  iree_hal_remote_queue_op_header_t header;
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
} iree_hal_remote_dispatch_op_t;
static_assert(sizeof(iree_hal_remote_dispatch_op_t) == 96, "");
static_assert(offsetof(iree_hal_remote_dispatch_op_t, executable_id) == 8, "");
static_assert(offsetof(iree_hal_remote_dispatch_op_t, config) == 24, "");
static_assert(offsetof(iree_hal_remote_dispatch_op_t, constant_count) == 80,
              "");
static_assert(offsetof(iree_hal_remote_dispatch_op_t, dispatch_flags) == 88,
              "");

// COMMAND_BUFFER_EXECUTE: Execute a pre-uploaded command buffer, or an
// inline one-shot command stream.
//
// Two modes controlled by op header flags:
//   - INLINE_COMMAND_STREAM clear: execute the uploaded CB identified by
//     command_buffer_id. Binding table provides indirect binding resolution.
//   - INLINE_COMMAND_STREAM set: command_buffer_id is ignored. The serialized
//     command stream follows the binding table (possibly preceded by DATA
//     frames carrying earlier pages of the stream).
typedef struct iree_hal_remote_command_buffer_execute_op_t {
  iree_hal_remote_queue_op_header_t header;
  iree_hal_remote_resource_id_t command_buffer_id;
  uint16_t binding_count;
  uint16_t reserved0;      // Must be 0.
  uint32_t reserved1;      // Must be 0.
  uint64_t execute_flags;  // iree_hal_execute_flags_t
  // Followed by:
  //   iree_hal_remote_binding_t binding_table[binding_count]
  // If INLINE_COMMAND_STREAM:
  //   Serialized command stream (iree_hal_remote_cmd_header_t sequence).
  //   For large streams, preceding DATA frames carry pages of the stream;
  //   the stream here is the final fragment (or empty if all pages were
  //   sent as DATA frames).
} iree_hal_remote_command_buffer_execute_op_t;
static_assert(sizeof(iree_hal_remote_command_buffer_execute_op_t) == 32, "");

// HOST_CALL_INVOKE: Invoke a registered host call handler on the server.
// Arguments are scalar integer values only — pointers are invalid in a remote
// context (disjoint address spaces). For structured data, pass a buffer
// resource ID and have the handler read from the buffer.
typedef struct iree_hal_remote_host_call_invoke_op_t {
  iree_hal_remote_queue_op_header_t header;
  uint64_t call_id;          // From HOST_CALL_REGISTER.
  uint64_t arguments[4];     // Scalar integer arguments passed to handler.
  uint64_t host_call_flags;  // iree_hal_host_call_flags_t
} iree_hal_remote_host_call_invoke_op_t;
static_assert(sizeof(iree_hal_remote_host_call_invoke_op_t) == 56, "");

// QUEUE_FLUSH: Hint to process pending work and report progress via ADVANCE.
// No payload beyond the header. Queue affinity is implicit from the stream.
typedef struct iree_hal_remote_queue_flush_op_t {
  iree_hal_remote_queue_op_header_t header;
} iree_hal_remote_queue_flush_op_t;
static_assert(sizeof(iree_hal_remote_queue_flush_op_t) == 8, "");

// QUEUE_EXTENSION: Device-specific queue operation. Uses the same frontier
// ordering as standard ops. The server dispatches by device_type + operation.
typedef struct iree_hal_remote_queue_extension_op_t {
  iree_hal_remote_queue_op_header_t header;
  uint32_t device_type;     // Namespace (CUDA=1, HIP=2, Vulkan=3, ...).
  uint32_t operation;       // Extension-defined operation code.
  uint32_t payload_length;  // Byte count of opaque payload.
  uint32_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t payload[payload_length]  (padded to 8-byte alignment)
} iree_hal_remote_queue_extension_op_t;
static_assert(sizeof(iree_hal_remote_queue_extension_op_t) == 24, "");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_PROTOCOL_QUEUE_H_
