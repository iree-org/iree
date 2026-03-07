// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL remote protocol: control channel messages.
//
// Request/response pairs, fire-and-forget messages, and server-initiated
// notifications on the control channel. Each message is wrapped in a control
// envelope (iree_hal_remote_control_envelope_t). Responses additionally carry
// a response prefix (iree_hal_remote_control_response_prefix_t) between the
// envelope and the message-specific payload.
//
// Naming convention:
//   _request_t   Request payload of a request/response pair.
//   _response_t  Response payload (omitted when only status is returned).
//   _t           Fire-and-forget messages and notifications.
//
// ## Dependency policy
//
// Includes common.h for shared wire format types (resource IDs, buffer params,
// memory heaps). Does not include HAL headers.

#ifndef IREE_HAL_REMOTE_PROTOCOL_CONTROL_H_
#define IREE_HAL_REMOTE_PROTOCOL_CONTROL_H_

#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Upload flags
//===----------------------------------------------------------------------===//

// Upload delivery flags shared by EXECUTABLE_UPLOAD and COMMAND_BUFFER_UPLOAD.
// Exactly one of INLINE_DATA or BULK_REFERENCE must be set.
#define IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA (1u << 0)
#define IREE_HAL_REMOTE_UPLOAD_FLAG_BULK_REFERENCE (1u << 1)

//===----------------------------------------------------------------------===//
// Control message envelope
//===----------------------------------------------------------------------===//

// Control message envelope. Prepended to every control channel DATA frame
// payload. Requests and responses share the same message_type value; the
// IS_RESPONSE flag in message_flags distinguishes direction.
typedef struct iree_hal_remote_control_envelope_t {
  uint16_t message_type;   // iree_hal_remote_control_type_t
  uint16_t message_flags;  // Combination of IREE_HAL_REMOTE_CONTROL_FLAG_*.
  uint32_t request_id;     // Client-assigned, echoed in response. 0 = notif.
  uint32_t control_epoch;  // Monotonic, incremented by CREATE-type messages.
  uint32_t reserved;       // Must be 0.
} iree_hal_remote_control_envelope_t;
static_assert(sizeof(iree_hal_remote_control_envelope_t) == 16, "");
static_assert(offsetof(iree_hal_remote_control_envelope_t, message_type) == 0,
              "");
static_assert(offsetof(iree_hal_remote_control_envelope_t, message_flags) == 2,
              "");
static_assert(offsetof(iree_hal_remote_control_envelope_t, request_id) == 4,
              "");
static_assert(offsetof(iree_hal_remote_control_envelope_t, control_epoch) == 8,
              "");

// Control message flag bits.
#define IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE (1u << 0)
#define IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET (1u << 1)

// Response prefix. Immediately follows the envelope when IS_RESPONSE is set.
// The response-specific payload (if any) follows this prefix.
typedef struct iree_hal_remote_control_response_prefix_t {
  uint32_t status_code;  // 0 = OK, else iree_status_code_t.
  uint32_t reserved;     // Must be 0.
} iree_hal_remote_control_response_prefix_t;
static_assert(sizeof(iree_hal_remote_control_response_prefix_t) == 8, "");

//===----------------------------------------------------------------------===//
// Control message types
//===----------------------------------------------------------------------===//

// Control message type identifiers. Types marked [epoch] increment the
// control_epoch counter. Queue ops depending on resources created by these
// messages include wait={control:epoch} in their frontier.
typedef enum iree_hal_remote_control_type_e {
  // ── Device ──────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_DEVICE_QUERY_INFO = 0x0001,
  IREE_HAL_REMOTE_CONTROL_DEVICE_QUERY_I64 = 0x0002,
  IREE_HAL_REMOTE_CONTROL_DEVICE_TRIM = 0x0003,

  // ── Semaphore ───────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_SEMAPHORE_CREATE = 0x0010,  // [epoch]
  IREE_HAL_REMOTE_CONTROL_SEMAPHORE_QUERY = 0x0011,
  IREE_HAL_REMOTE_CONTROL_SEMAPHORE_SIGNAL = 0x0012,  // fire-and-forget
  IREE_HAL_REMOTE_CONTROL_SEMAPHORE_WAIT = 0x0013,

  // ── Executable ──────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_EXECUTABLE_UPLOAD = 0x0020,  // [epoch]
  IREE_HAL_REMOTE_CONTROL_EXECUTABLE_QUERY_EXPORT = 0x0021,

  // ── Command Buffer ──────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_COMMAND_BUFFER_UPLOAD = 0x0030,  // [epoch]

  // ── File ────────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_FILE_OPEN = 0x0040,        // [epoch]
  IREE_HAL_REMOTE_CONTROL_FILE_CLOSE = 0x0041,       // fire-and-forget
  IREE_HAL_REMOTE_CONTROL_FILE_REGISTER = 0x0042,    // [epoch]
  IREE_HAL_REMOTE_CONTROL_FILE_UNREGISTER = 0x0043,  // fire-and-forget
  IREE_HAL_REMOTE_CONTROL_FILE_LIST = 0x0044,

  // ── Buffer ──────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_BUFFER_ALLOC = 0x0050,   // [epoch]
  IREE_HAL_REMOTE_CONTROL_BUFFER_IMPORT = 0x0051,  // [epoch]
  IREE_HAL_REMOTE_CONTROL_BUFFER_MAP = 0x0052,
  IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP = 0x0053,  // fire-and-forget
  IREE_HAL_REMOTE_CONTROL_BUFFER_QUERY_HEAPS = 0x0054,

  // ── Host Call ───────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_HOST_CALL_REGISTER = 0x0060,    // [epoch]
  IREE_HAL_REMOTE_CONTROL_HOST_CALL_UNREGISTER = 0x0061,  // fire-and-forget

  // ── Lifecycle ───────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_RESOURCE_RELEASE_BATCH = 0x0070,  // fire-and-forget

  // ── Extensions ──────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_DEVICE_EXTENSION = 0x00F0,

  // ── Notifications (server → client, request_id=0) ──────────────────────
  IREE_HAL_REMOTE_CONTROL_NOTIFY_RESOURCE_ERROR = 0x00E0,
  IREE_HAL_REMOTE_CONTROL_NOTIFY_DEVICE_LOST = 0x00E1,
  IREE_HAL_REMOTE_CONTROL_NOTIFY_MEMORY_PRESSURE = 0x00E2,
} iree_hal_remote_control_type_t;

//===----------------------------------------------------------------------===//
// Device messages
//===----------------------------------------------------------------------===//

// Queue description returned in DEVICE_QUERY_INFO responses. Describes a
// single device queue's affinity bit and command categories it supports.
typedef struct iree_hal_remote_queue_description_t {
  uint64_t queue_affinity;  // Bitmask identifying this queue.
  uint32_t categories;      // iree_hal_command_category_t
  uint32_t reserved;        // Must be 0.
} iree_hal_remote_queue_description_t;
static_assert(sizeof(iree_hal_remote_queue_description_t) == 16, "");

// DEVICE_QUERY_INFO request. Queries device topology, capabilities, and
// memory heaps.
typedef struct iree_hal_remote_device_query_info_request_t {
  uint32_t flags;     // Reserved, must be 0.
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_device_query_info_request_t;
static_assert(sizeof(iree_hal_remote_device_query_info_request_t) == 8, "");

// DEVICE_QUERY_INFO response. Variable-length tail carries device name,
// queue descriptions, and memory heap descriptions.
typedef struct iree_hal_remote_device_query_info_response_t {
  uint64_t device_id;           // Opaque server-assigned device identifier.
  uint16_t device_name_length;  // UTF-8 byte count (not null-terminated).
  uint8_t queue_count;          // Number of device queues.
  uint8_t heap_count;           // Number of memory heaps.
  uint32_t reserved;            // Must be 0.
  // Followed by:
  //   uint8_t device_name[device_name_length]  (padded to 8-byte alignment)
  //   iree_hal_remote_queue_description_t queues[queue_count]
  //   iree_hal_remote_memory_heap_t heaps[heap_count]
} iree_hal_remote_device_query_info_response_t;
static_assert(sizeof(iree_hal_remote_device_query_info_response_t) == 16, "");

// DEVICE_QUERY_I64 request. Queries a named integer device property.
// Variable-length tail carries category and key strings.
typedef struct iree_hal_remote_device_query_i64_request_t {
  uint16_t category_length;  // UTF-8 byte count.
  uint16_t key_length;       // UTF-8 byte count.
  uint32_t reserved;         // Must be 0.
  // Followed by:
  //   uint8_t category[category_length]  (padded to 8-byte alignment)
  //   uint8_t key[key_length]  (padded to 8-byte alignment)
} iree_hal_remote_device_query_i64_request_t;
static_assert(sizeof(iree_hal_remote_device_query_i64_request_t) == 8, "");

// DEVICE_QUERY_I64 response. Returns the queried value.
typedef struct iree_hal_remote_device_query_i64_response_t {
  int64_t value;
} iree_hal_remote_device_query_i64_response_t;
static_assert(sizeof(iree_hal_remote_device_query_i64_response_t) == 8, "");

// DEVICE_TRIM request. Releases unused device resources.
typedef struct iree_hal_remote_device_trim_request_t {
  uint32_t flags;     // Reserved, must be 0.
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_device_trim_request_t;
static_assert(sizeof(iree_hal_remote_device_trim_request_t) == 8, "");
// Response: status only (no _response_t struct).

//===----------------------------------------------------------------------===//
// Semaphore messages
//===----------------------------------------------------------------------===//

// SEMAPHORE_CREATE request. Creates a timeline semaphore. [epoch]
typedef struct iree_hal_remote_semaphore_create_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  uint64_t initial_value;
} iree_hal_remote_semaphore_create_request_t;
static_assert(sizeof(iree_hal_remote_semaphore_create_request_t) == 16, "");

// SEMAPHORE_CREATE response. Returns the server-assigned canonical ID.
typedef struct iree_hal_remote_semaphore_create_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
} iree_hal_remote_semaphore_create_response_t;
static_assert(sizeof(iree_hal_remote_semaphore_create_response_t) == 8, "");

// SEMAPHORE_QUERY request. Reads the current semaphore value.
typedef struct iree_hal_remote_semaphore_query_request_t {
  iree_hal_remote_resource_id_t semaphore_id;
} iree_hal_remote_semaphore_query_request_t;
static_assert(sizeof(iree_hal_remote_semaphore_query_request_t) == 8, "");

// SEMAPHORE_QUERY response. Returns the current value.
typedef struct iree_hal_remote_semaphore_query_response_t {
  uint64_t value;
} iree_hal_remote_semaphore_query_response_t;
static_assert(sizeof(iree_hal_remote_semaphore_query_response_t) == 8, "");

// SEMAPHORE_SIGNAL. Fire-and-forget: advances the semaphore to new_value.
typedef struct iree_hal_remote_semaphore_signal_t {
  iree_hal_remote_resource_id_t semaphore_id;
  uint64_t new_value;
} iree_hal_remote_semaphore_signal_t;
static_assert(sizeof(iree_hal_remote_semaphore_signal_t) == 16, "");

// SEMAPHORE_WAIT request. Waits until the semaphore reaches minimum_value
// or the timeout expires. The server MUST process this asynchronously
// (register a callback and continue processing other control messages) to
// avoid head-of-line blocking on the control channel. The response is sent
// when the wait completes; the envelope's request_id matches it to the
// original request.
typedef struct iree_hal_remote_semaphore_wait_request_t {
  iree_hal_remote_resource_id_t semaphore_id;
  uint64_t minimum_value;
  uint64_t timeout_ns;  // IREE_DURATION_INFINITE for unbounded wait.
} iree_hal_remote_semaphore_wait_request_t;
static_assert(sizeof(iree_hal_remote_semaphore_wait_request_t) == 24, "");
// Response: status only (OK or DEADLINE_EXCEEDED).

//===----------------------------------------------------------------------===//
// Executable messages
//===----------------------------------------------------------------------===//

// EXECUTABLE_UPLOAD request. Uploads a compiled executable blob. [epoch]
// Data delivery controlled by upload_flags: either inline (data follows this
// struct) or bulk (referenced by bulk_transfer_id, server defers processing
// until the bulk transfer completes).
typedef struct iree_hal_remote_executable_upload_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  uint32_t executable_format;  // Fourcc identifying the backend format.
  uint16_t constant_count;     // Specialization constant count.
  uint16_t upload_flags;       // IREE_HAL_REMOTE_UPLOAD_FLAG_*
  uint64_t data_length;        // Byte count of executable data.
  uint64_t bulk_transfer_id;   // Valid when BULK_REFERENCE is set.
  // Followed by:
  //   uint32_t constants[constant_count]  (padded to 8-byte alignment)
  //   [if INLINE_DATA]: uint8_t data[data_length]  (padded to 8-byte alignment)
} iree_hal_remote_executable_upload_request_t;
static_assert(sizeof(iree_hal_remote_executable_upload_request_t) == 32, "");
static_assert(offsetof(iree_hal_remote_executable_upload_request_t,
                       data_length) == 16,
              "");

// EXECUTABLE_UPLOAD response. Returns the resolved ID and export count.
typedef struct iree_hal_remote_executable_upload_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
  uint32_t export_count;  // Number of entry points in the executable.
  uint32_t reserved;      // Must be 0.
} iree_hal_remote_executable_upload_response_t;
static_assert(sizeof(iree_hal_remote_executable_upload_response_t) == 16, "");

// EXECUTABLE_QUERY_EXPORT request. Queries metadata for a specific entry point.
typedef struct iree_hal_remote_executable_query_export_request_t {
  iree_hal_remote_resource_id_t executable_id;
  uint32_t export_ordinal;
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_executable_query_export_request_t;
static_assert(sizeof(iree_hal_remote_executable_query_export_request_t) == 16,
              "");

// EXECUTABLE_QUERY_EXPORT response. Returns workgroup size for the export.
typedef struct iree_hal_remote_executable_query_export_response_t {
  uint32_t workgroup_size[3];
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_executable_query_export_response_t;
static_assert(sizeof(iree_hal_remote_executable_query_export_response_t) == 16,
              "");

//===----------------------------------------------------------------------===//
// Command buffer messages
//===----------------------------------------------------------------------===//

// COMMAND_BUFFER_UPLOAD request. Uploads a reusable command buffer. [epoch]
// Same upload delivery model as EXECUTABLE_UPLOAD (inline or bulk).
typedef struct iree_hal_remote_command_buffer_upload_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  uint32_t mode;              // iree_hal_command_buffer_mode_t
  uint32_t categories;        // iree_hal_command_category_t
  uint16_t binding_capacity;  // Max binding table slots.
  uint16_t upload_flags;      // IREE_HAL_REMOTE_UPLOAD_FLAG_*
  uint32_t reserved;          // Must be 0.
  uint64_t data_length;       // Byte count of serialized command stream.
  uint64_t bulk_transfer_id;  // Valid when BULK_REFERENCE is set.
  // [if INLINE_DATA]: uint8_t data[data_length]  (padded to 8-byte alignment)
} iree_hal_remote_command_buffer_upload_request_t;
static_assert(sizeof(iree_hal_remote_command_buffer_upload_request_t) == 40,
              "");
static_assert(offsetof(iree_hal_remote_command_buffer_upload_request_t,
                       data_length) == 24,
              "");

// COMMAND_BUFFER_UPLOAD response. Returns the resolved ID.
typedef struct iree_hal_remote_command_buffer_upload_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
} iree_hal_remote_command_buffer_upload_response_t;
static_assert(sizeof(iree_hal_remote_command_buffer_upload_response_t) == 8,
              "");

//===----------------------------------------------------------------------===//
// File messages
//===----------------------------------------------------------------------===//

// FILE_OPEN request. Opens a server-side file by canonical path. [epoch]
// The server's VFS layer resolves canonical names to real paths and enforces
// access control. The client never sees real filesystem paths. The client
// provides a provisional_id so queue ops (FILE_READ/FILE_WRITE) can reference
// the file immediately without waiting for the response round-trip.
typedef struct iree_hal_remote_file_open_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  uint16_t path_length;  // UTF-8 byte count (not null-terminated).
  uint16_t mode;         // Access mode (read, write, read-write).
  uint32_t flags;        // Reserved, must be 0.
  // Followed by:
  //   uint8_t path[path_length]  (padded to 8-byte alignment)
} iree_hal_remote_file_open_request_t;
static_assert(sizeof(iree_hal_remote_file_open_request_t) == 16, "");

// FILE_OPEN response. Returns the resolved file ID and metadata.
typedef struct iree_hal_remote_file_open_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
  uint64_t file_size;
  uint32_t granted_access;  // Actual access granted (may differ from request).
  uint32_t reserved;        // Must be 0.
} iree_hal_remote_file_open_response_t;
static_assert(sizeof(iree_hal_remote_file_open_response_t) == 24, "");
static_assert(offsetof(iree_hal_remote_file_open_response_t, file_size) == 8,
              "");

// FILE_CLOSE. Fire-and-forget: closes a previously opened file.
typedef struct iree_hal_remote_file_close_t {
  iree_hal_remote_resource_id_t file_id;
} iree_hal_remote_file_close_t;
static_assert(sizeof(iree_hal_remote_file_close_t) == 8, "");

// FILE_REGISTER request. Registers an external file handle for use with
// queue-ordered I/O operations (FILE_READ/FILE_WRITE). [epoch] The
// external_type identifies the handle namespace (POSIX fd, Win32 HANDLE,
// memfd, etc.); handle_payload carries the type-specific data.
typedef struct iree_hal_remote_file_register_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  uint32_t external_type;                        // File handle type identifier.
  uint32_t access_flags;           // Access mode for the registered file.
  uint32_t handle_payload_length;  // Byte count of type-specific handle data.
  uint32_t reserved;               // Must be 0.
  // Followed by:
  //   uint8_t handle_payload[handle_payload_length]  (padded to 8-byte align)
} iree_hal_remote_file_register_request_t;
static_assert(sizeof(iree_hal_remote_file_register_request_t) == 24, "");

// FILE_REGISTER response. Returns the resolved file ID and discovered size.
typedef struct iree_hal_remote_file_register_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
  uint64_t file_size;
} iree_hal_remote_file_register_response_t;
static_assert(sizeof(iree_hal_remote_file_register_response_t) == 16, "");

// FILE_UNREGISTER. Fire-and-forget: unregisters a previously registered file.
typedef struct iree_hal_remote_file_unregister_t {
  iree_hal_remote_resource_id_t file_id;
} iree_hal_remote_file_unregister_t;
static_assert(sizeof(iree_hal_remote_file_unregister_t) == 8, "");

// FILE_LIST request. Lists files available on the server, optionally filtered
// by a glob pattern. Variable-length tail carries the pattern.
typedef struct iree_hal_remote_file_list_request_t {
  uint16_t pattern_length;  // 0 = list all available files.
  uint16_t reserved0;       // Must be 0.
  uint32_t reserved1;       // Must be 0.
  // Followed by:
  //   uint8_t pattern[pattern_length]  (padded to 8-byte alignment)
} iree_hal_remote_file_list_request_t;
static_assert(sizeof(iree_hal_remote_file_list_request_t) == 8, "");

// Entry in a FILE_LIST response. Variable-length: fixed header followed by
// the file name string.
typedef struct iree_hal_remote_file_list_entry_t {
  uint64_t file_size;
  uint16_t name_length;  // UTF-8 byte count (not null-terminated).
  uint16_t reserved0;    // Must be 0.
  uint32_t reserved1;    // Must be 0.
  // Followed by:
  //   uint8_t name[name_length]  (padded to 8-byte alignment)
} iree_hal_remote_file_list_entry_t;
static_assert(sizeof(iree_hal_remote_file_list_entry_t) == 16, "");
static_assert(offsetof(iree_hal_remote_file_list_entry_t, name_length) == 8,
              "");

// FILE_LIST response. Variable-length: entry_count followed by that many
// iree_hal_remote_file_list_entry_t records (each itself variable-length).
typedef struct iree_hal_remote_file_list_response_t {
  uint32_t entry_count;
  uint32_t reserved;  // Must be 0.
  // Followed by:
  //   iree_hal_remote_file_list_entry_t entries[entry_count]
  //   (each entry is variable-length due to its name string)
} iree_hal_remote_file_list_response_t;
static_assert(sizeof(iree_hal_remote_file_list_response_t) == 8, "");

//===----------------------------------------------------------------------===//
// Buffer messages
//===----------------------------------------------------------------------===//

// BUFFER_ALLOC request. Allocates a persistent buffer (model weights, I/O
// staging). [epoch] For transient queue-ordered allocation, use the
// BUFFER_ALLOCA queue op instead.
typedef struct iree_hal_remote_buffer_alloc_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  iree_hal_remote_buffer_params_t params;        // 32 bytes.
  uint64_t allocation_size;
} iree_hal_remote_buffer_alloc_request_t;
static_assert(sizeof(iree_hal_remote_buffer_alloc_request_t) == 48, "");
static_assert(offsetof(iree_hal_remote_buffer_alloc_request_t, params) == 8,
              "");
static_assert(offsetof(iree_hal_remote_buffer_alloc_request_t,
                       allocation_size) == 40,
              "");

// BUFFER_ALLOC response. Returns the resolved buffer ID.
typedef struct iree_hal_remote_buffer_alloc_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
} iree_hal_remote_buffer_alloc_response_t;
static_assert(sizeof(iree_hal_remote_buffer_alloc_response_t) == 8, "");

// BUFFER_IMPORT request. Imports an externally-owned buffer (shared memory,
// DMA-BUF, platform handle). [epoch] The external_type identifies the import
// mechanism; the variable-length handle_payload carries type-specific metadata
// (e.g., DMA-BUF fd + plane offsets/strides, AHardwareBuffer serialization,
// SHM handle + size). This extensible layout avoids baking platform-specific
// handle structures into the fixed protocol.
typedef struct iree_hal_remote_buffer_import_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  iree_hal_remote_buffer_params_t params;        // 32 bytes.
  uint64_t allocation_size;
  uint32_t external_type;          // External buffer type identifier.
  uint32_t handle_payload_length;  // Byte count of type-specific handle data.
  // Followed by:
  //   uint8_t handle_payload[handle_payload_length]  (padded to 8-byte align)
} iree_hal_remote_buffer_import_request_t;
static_assert(sizeof(iree_hal_remote_buffer_import_request_t) == 56, "");
static_assert(offsetof(iree_hal_remote_buffer_import_request_t, params) == 8,
              "");
static_assert(offsetof(iree_hal_remote_buffer_import_request_t,
                       external_type) == 48,
              "");

// BUFFER_IMPORT response. Returns the resolved buffer ID.
typedef struct iree_hal_remote_buffer_import_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
} iree_hal_remote_buffer_import_response_t;
static_assert(sizeof(iree_hal_remote_buffer_import_response_t) == 8, "");

// BUFFER_MAP request. Establishes a host-visible mapping of a buffer region.
typedef struct iree_hal_remote_buffer_map_request_t {
  iree_hal_remote_resource_id_t buffer_id;
  uint32_t mapping_flags;  // Read/write/discard.
  uint32_t reserved;       // Must be 0.
  uint64_t offset;
  uint64_t length;
} iree_hal_remote_buffer_map_request_t;
static_assert(sizeof(iree_hal_remote_buffer_map_request_t) == 32, "");

// BUFFER_MAP response. Returns an opaque mapping handle and the actual mapped
// region (which may differ from the requested region due to alignment).
// The mapping_id is an opaque server-assigned handle — not a pointer.
typedef struct iree_hal_remote_buffer_map_response_t {
  uint64_t mapping_id;     // Opaque handle. Used in BUFFER_UNMAP to release.
  uint64_t mapped_offset;  // Actual start offset of the mapped region.
  uint64_t mapped_length;  // Actual byte count of the mapped region.
} iree_hal_remote_buffer_map_response_t;
static_assert(sizeof(iree_hal_remote_buffer_map_response_t) == 24, "");

// BUFFER_UNMAP. Fire-and-forget: releases a previously established mapping.
typedef struct iree_hal_remote_buffer_unmap_t {
  iree_hal_remote_resource_id_t buffer_id;
  uint64_t mapping_id;  // From BUFFER_MAP response.
} iree_hal_remote_buffer_unmap_t;
static_assert(sizeof(iree_hal_remote_buffer_unmap_t) == 16, "");

// BUFFER_QUERY_HEAPS request. Queries the device's memory heap topology.
typedef struct iree_hal_remote_buffer_query_heaps_request_t {
  uint32_t flags;     // Reserved, must be 0.
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_buffer_query_heaps_request_t;
static_assert(sizeof(iree_hal_remote_buffer_query_heaps_request_t) == 8, "");

// BUFFER_QUERY_HEAPS response. Returns heap descriptions.
typedef struct iree_hal_remote_buffer_query_heaps_response_t {
  uint16_t heap_count;
  uint16_t reserved0;  // Must be 0.
  uint32_t reserved1;  // Must be 0.
  // Followed by:
  //   iree_hal_remote_memory_heap_t heaps[heap_count]
} iree_hal_remote_buffer_query_heaps_response_t;
static_assert(sizeof(iree_hal_remote_buffer_query_heaps_response_t) == 8, "");

//===----------------------------------------------------------------------===//
// Host call messages
//===----------------------------------------------------------------------===//

// HOST_CALL_REGISTER request. Registers a named host-side call handler.
// [epoch] Invocation happens via HOST_CALL_INVOKE on the queue channel.
typedef struct iree_hal_remote_host_call_register_request_t {
  uint64_t call_id;      // Client-chosen ID, unique within session.
  uint16_t name_length;  // UTF-8 handler name (not null-terminated).
  uint16_t flags;        // Reserved, must be 0.
  uint32_t reserved;     // Must be 0.
  // Followed by:
  //   uint8_t name[name_length]  (padded to 8-byte alignment)
} iree_hal_remote_host_call_register_request_t;
static_assert(sizeof(iree_hal_remote_host_call_register_request_t) == 16, "");
// Response: status only (validates call_id uniqueness).

// HOST_CALL_UNREGISTER. Fire-and-forget: unregisters a call handler.
typedef struct iree_hal_remote_host_call_unregister_t {
  uint64_t call_id;
} iree_hal_remote_host_call_unregister_t;
static_assert(sizeof(iree_hal_remote_host_call_unregister_t) == 8, "");

//===----------------------------------------------------------------------===//
// Lifecycle messages
//===----------------------------------------------------------------------===//

// RESOURCE_RELEASE_BATCH. Fire-and-forget: batched release of resources.
// Client-side destroy enqueues handles into an MPSC pending-release queue,
// flushed periodically (~1ms or ~100 releases), on queue_flush, or on
// session teardown. Resource IDs may be provisional or resolved.
typedef struct iree_hal_remote_resource_release_batch_t {
  uint32_t resource_count;
  uint32_t reserved;  // Must be 0.
  // Followed by:
  //   iree_hal_remote_resource_id_t resource_ids[resource_count]
} iree_hal_remote_resource_release_batch_t;
static_assert(sizeof(iree_hal_remote_resource_release_batch_t) == 8, "");

//===----------------------------------------------------------------------===//
// Extension messages
//===----------------------------------------------------------------------===//

// DEVICE_EXTENSION request. ioctl-style escape hatch for device-specific
// control operations. The server dispatches by device_type + operation.
typedef struct iree_hal_remote_device_extension_request_t {
  uint32_t device_type;     // Namespace (CUDA=1, HIP=2, Vulkan=3, ...).
  uint32_t operation;       // Extension-defined operation code.
  uint32_t payload_length;  // Byte count of opaque payload.
  uint32_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t payload[payload_length]  (padded to 8-byte alignment)
} iree_hal_remote_device_extension_request_t;
static_assert(sizeof(iree_hal_remote_device_extension_request_t) == 16, "");

// DEVICE_EXTENSION response. Extension-defined response payload.
typedef struct iree_hal_remote_device_extension_response_t {
  uint32_t payload_length;  // Byte count of opaque response payload.
  uint32_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t payload[payload_length]  (padded to 8-byte alignment)
} iree_hal_remote_device_extension_response_t;
static_assert(sizeof(iree_hal_remote_device_extension_response_t) == 8, "");

//===----------------------------------------------------------------------===//
// Notification messages
//===----------------------------------------------------------------------===//

// NOTIFY_RESOURCE_ERROR. Server → client notification that a resource has
// entered an error state (e.g., semaphore failure, executable load error).
typedef struct iree_hal_remote_notify_resource_error_t {
  iree_hal_remote_resource_id_t resource_id;
  uint32_t error_code;      // iree_status_code_t
  uint16_t message_length;  // UTF-8 diagnostic message.
  uint16_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t message[message_length]  (padded to 8-byte alignment)
} iree_hal_remote_notify_resource_error_t;
static_assert(sizeof(iree_hal_remote_notify_resource_error_t) == 16, "");

// NOTIFY_DEVICE_LOST. Server → client notification that the device is no
// longer functional. All pending operations fail. Session must be re-created.
typedef struct iree_hal_remote_notify_device_lost_t {
  uint32_t error_code;      // iree_status_code_t
  uint16_t message_length;  // UTF-8 diagnostic message.
  uint16_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t message[message_length]  (padded to 8-byte alignment)
} iree_hal_remote_notify_device_lost_t;
static_assert(sizeof(iree_hal_remote_notify_device_lost_t) == 8, "");

// NOTIFY_MEMORY_PRESSURE. Server → client notification of memory pressure.
// Client should release unused buffers or reduce allocation rate.
typedef struct iree_hal_remote_notify_memory_pressure_t {
  uint32_t pressure_flags;  // Severity/type of pressure.
  uint32_t reserved;        // Must be 0.
} iree_hal_remote_notify_memory_pressure_t;
static_assert(sizeof(iree_hal_remote_notify_memory_pressure_t) == 8, "");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_PROTOCOL_CONTROL_H_
