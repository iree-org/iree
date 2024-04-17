// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_STATUS_PAYLOAD_H_
#define IREE_BASE_STATUS_PAYLOAD_H_

#include "iree/base/allocator.h"
#include "iree/base/status.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Status payload API
//===----------------------------------------------------------------------===//

typedef struct iree_status_storage_t iree_status_storage_t;

// Defines the type of an iree_status_payload_t.
typedef enum iree_status_payload_type_e {
  // Opaque; payload may still be formatted by a formatter but is not possible
  // to retrieve by the programmatic APIs.
  IREE_STATUS_PAYLOAD_TYPE_OPAQUE = 0,
  // A string message annotation of type iree_status_payload_message_t.
  IREE_STATUS_PAYLOAD_TYPE_MESSAGE = 1,
  // Platform-dependent stack trace in iree_status_payload_stack_trace_t.
  IREE_STATUS_PAYLOAD_TYPE_STACK_TRACE = 2,
  // Starting type ID for user payloads. IREE reserves all payloads with types
  // less than this.
  IREE_STATUS_PAYLOAD_TYPE_MIN_USER = 0x70000000u,
} iree_status_payload_type_t;

typedef struct iree_status_payload_t iree_status_payload_t;

// Function that formats a payload into a human-readable string form for logs.
typedef void(IREE_API_PTR* iree_status_payload_formatter_t)(
    const iree_status_payload_t* payload, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

// Header for optional status payloads.
// Each status may have zero or more payloads associated with it that can later
// be used to produce more detailed logging or programmatically query
// information about an error.
struct iree_status_payload_t {
  // Next payload in the status payload linked list.
  struct iree_status_payload_t* next;
  // Payload type identifier used for programmatic access to payloads. May be
  // IREE_STATUS_PAYLOAD_TYPE_OPAQUE if the payload cannot be accessed directly.
  iree_status_payload_type_t type;
  // Allocator used for the payload and associated resources.
  iree_allocator_t allocator;
  // String formatter callback used to write the payload into a string buffer.
  // If not present then the payload will be mentioned but not dumped when the
  // status is logged.
  iree_status_payload_formatter_t formatter;
};

// A string message (IREE_STATUS_PAYLOAD_TYPE_MESSAGE).
typedef struct iree_status_payload_message_t {
  iree_status_payload_t header;
  // String data reference. May point to an address immediately following this
  // struct (if copied) or a constant string reference in rodata.
  iree_string_view_t message;
} iree_status_payload_message_t;

// A platform-dependent stack trace (IREE_STATUS_PAYLOAD_TYPE_STACK_TRACE).
typedef struct iree_status_payload_stack_trace_t {
  iree_status_payload_t header;
  uint16_t skip_frames;
  uint16_t frame_count;
  uintptr_t addresses[];
} iree_status_payload_stack_trace_t;

// TODO(benvanik): expose API for appending/enumerating payloads.
// Currently this is an implementation detail.
iree_status_t iree_status_append_payload(iree_status_t status,
                                         iree_status_storage_t* storage,
                                         iree_status_payload_t* payload);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_STATUS_PAYLOAD_H_
