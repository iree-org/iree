// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/status_wire.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Alignment helpers
//===----------------------------------------------------------------------===//

static inline iree_host_size_t iree_net_align8(iree_host_size_t value) {
  return (value + 7) & ~(iree_host_size_t)7;
}

// Maximum data length for a single wire entry (uint16_t field).
#define IREE_NET_STATUS_WIRE_MAX_ENTRY_LENGTH UINT16_MAX

// Clamps a data length to the wire format's uint16_t limit.
static inline iree_host_size_t iree_net_status_wire_clamp_length(
    iree_host_size_t data_length) {
  return data_length > IREE_NET_STATUS_WIRE_MAX_ENTRY_LENGTH
             ? IREE_NET_STATUS_WIRE_MAX_ENTRY_LENGTH
             : data_length;
}

// Returns the wire size of a single entry: entry header + data + padding.
static inline iree_host_size_t iree_net_status_wire_entry_size(
    iree_host_size_t data_length) {
  return sizeof(iree_net_status_wire_entry_t) + iree_net_align8(data_length);
}

//===----------------------------------------------------------------------===//
// Sizing: walk the status to compute serialized size
//===----------------------------------------------------------------------===//

// Context for the payload sizing visitor.
typedef struct iree_net_status_wire_size_context_t {
  iree_host_size_t size;
  uint16_t entry_count;
} iree_net_status_wire_size_context_t;

static iree_status_t iree_net_status_wire_size_payload_visitor(
    void* user_data, const iree_status_payload_t* payload) {
  iree_net_status_wire_size_context_t* context =
      (iree_net_status_wire_size_context_t*)user_data;

  // Get the formatted length of this payload, clamped to the wire limit.
  iree_host_size_t payload_length = 0;
  iree_status_payload_format(payload, /*buffer_capacity=*/0, /*buffer=*/NULL,
                             &payload_length);
  if (payload_length == 0) return iree_ok_status();
  payload_length = iree_net_status_wire_clamp_length(payload_length);

  context->size += iree_net_status_wire_entry_size(payload_length);
  context->entry_count++;
  return iree_ok_status();
}

void iree_net_status_wire_size(iree_status_t status,
                               iree_host_size_t* out_size) {
  // Header is always present.
  iree_host_size_t size = sizeof(iree_net_status_wire_header_t);

  if (iree_status_is_ok(status)) {
    *out_size = size;
    return;
  }

  uint16_t entry_count = 0;

  // Source location entry.
  iree_status_source_location_t location = iree_status_source_location(status);
  iree_host_size_t file_length = 0;
  if (location.file) {
    file_length = iree_net_status_wire_clamp_length(strlen(location.file));
    size += iree_net_status_wire_entry_size(file_length);
    entry_count++;
  }

  // Primary message entry.
  iree_string_view_t message = iree_status_message(status);
  iree_host_size_t message_length =
      iree_net_status_wire_clamp_length(message.size);
  if (message_length > 0) {
    size += iree_net_status_wire_entry_size(message_length);
    entry_count++;
  }

  // Payload entries (annotations, stack traces).
  iree_net_status_wire_size_context_t context = {0, 0};
  iree_status_ignore(iree_status_enumerate_payloads(
      status, iree_net_status_wire_size_payload_visitor, &context));
  size += context.size;
  entry_count += context.entry_count;

  (void)entry_count;  // Used in serialize, computed here for consistency.
  *out_size = size;
}

//===----------------------------------------------------------------------===//
// Serialization: write status fields into wire format buffer
//===----------------------------------------------------------------------===//

// Context for the payload serialization visitor.
typedef struct iree_net_status_wire_serialize_context_t {
  uint8_t* buffer;
  iree_host_size_t buffer_capacity;
  iree_host_size_t offset;
  uint16_t entry_count;
} iree_net_status_wire_serialize_context_t;

// Writes one entry into the buffer at context->offset. Advances offset.
// |data_length| is clamped to the wire format's uint16_t limit.
static void iree_net_status_wire_write_entry(
    iree_net_status_wire_serialize_context_t* context,
    iree_net_status_entry_type_t type, uint32_t aux, const void* data,
    iree_host_size_t data_length) {
  data_length = iree_net_status_wire_clamp_length(data_length);
  iree_net_status_wire_entry_t entry;
  memset(&entry, 0, sizeof(entry));
  entry.type = (uint16_t)type;
  entry.length = (uint16_t)data_length;
  entry.aux = aux;
  memcpy(context->buffer + context->offset, &entry, sizeof(entry));
  context->offset += sizeof(entry);

  if (data_length > 0) {
    memcpy(context->buffer + context->offset, data, data_length);
    context->offset += iree_net_align8(data_length);
  }
  context->entry_count++;
}

static iree_status_t iree_net_status_wire_serialize_payload_visitor(
    void* user_data, const iree_status_payload_t* payload) {
  iree_net_status_wire_serialize_context_t* context =
      (iree_net_status_wire_serialize_context_t*)user_data;

  // Get the formatted length, clamped to the wire format limit.
  iree_host_size_t payload_length = 0;
  iree_status_payload_format(payload, /*buffer_capacity=*/0, /*buffer=*/NULL,
                             &payload_length);
  if (payload_length == 0) return iree_ok_status();
  payload_length = iree_net_status_wire_clamp_length(payload_length);

  iree_net_status_entry_type_t entry_type =
      (iree_status_payload_type(payload) ==
       IREE_STATUS_PAYLOAD_TYPE_STACK_TRACE)
          ? IREE_NET_STATUS_ENTRY_STACK_TRACE
          : IREE_NET_STATUS_ENTRY_ANNOTATION;

  // Write entry header.
  iree_net_status_wire_entry_t entry;
  memset(&entry, 0, sizeof(entry));
  entry.type = (uint16_t)entry_type;
  entry.length = (uint16_t)payload_length;
  entry.aux = 0;
  memcpy(context->buffer + context->offset, &entry, sizeof(entry));
  context->offset += sizeof(entry);

  // Format payload text directly into the buffer (clamped capacity).
  iree_host_size_t written = 0;
  iree_status_payload_format(payload, payload_length + 1,
                             (char*)(context->buffer + context->offset),
                             &written);
  context->offset += iree_net_align8(payload_length);
  context->entry_count++;

  return iree_ok_status();
}

iree_status_t iree_net_status_wire_serialize(iree_status_t status,
                                             iree_byte_span_t buffer) {
  iree_host_size_t required_size = 0;
  iree_net_status_wire_size(status, &required_size);
  if (buffer.data_length < required_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "status wire buffer too small: %" PRIhsz
                            " < %" PRIhsz,
                            buffer.data_length, required_size);
  }

  // Zero-fill for padding bytes.
  memset(buffer.data, 0, required_size);

  iree_net_status_wire_serialize_context_t context = {
      .buffer = buffer.data,
      .buffer_capacity = buffer.data_length,
      .offset = sizeof(iree_net_status_wire_header_t),
      .entry_count = 0,
  };

  if (!iree_status_is_ok(status)) {
    // Source location entry.
    iree_status_source_location_t location =
        iree_status_source_location(status);
    if (location.file) {
      iree_net_status_wire_write_entry(
          &context, IREE_NET_STATUS_ENTRY_SOURCE_LOCATION, location.line,
          location.file, strlen(location.file));
    }

    // Primary message entry.
    iree_string_view_t message = iree_status_message(status);
    if (message.size > 0) {
      iree_net_status_wire_write_entry(&context, IREE_NET_STATUS_ENTRY_MESSAGE,
                                       0, message.data, message.size);
    }

    // Payload entries (annotations, stack traces).
    iree_status_ignore(iree_status_enumerate_payloads(
        status, iree_net_status_wire_serialize_payload_visitor, &context));
  }

  // Write header (now that we know the final entry count).
  iree_net_status_wire_header_t header;
  memset(&header, 0, sizeof(header));
  header.version = IREE_NET_STATUS_WIRE_VERSION;
  header.status_code = (uint8_t)iree_status_code(status);
  header.entry_count = context.entry_count;
  header.total_size = (uint32_t)required_size;
  memcpy(buffer.data, &header, sizeof(header));

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Deserialization: reconstruct iree_status_t from wire format buffer
//===----------------------------------------------------------------------===//

iree_status_t iree_net_status_wire_deserialize(iree_const_byte_span_t data,
                                               iree_status_t* out_status) {
  if (data.data_length < sizeof(iree_net_status_wire_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "status wire data too short for header: %" PRIhsz
                            " bytes",
                            data.data_length);
  }

  iree_net_status_wire_header_t header;
  memcpy(&header, data.data, sizeof(header));

  if (header.version != IREE_NET_STATUS_WIRE_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported status wire version: %u",
                            header.version);
  }

  if (header.total_size > data.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "status wire total_size (%" PRIu32
                            ") exceeds data length (%" PRIhsz ")",
                            header.total_size, data.data_length);
  }

  iree_status_code_t status_code = (iree_status_code_t)header.status_code;
  if (status_code == IREE_STATUS_OK) {
    *out_status = iree_ok_status();
    return iree_ok_status();
  }

  // First pass: find source location and primary message.
  iree_string_view_t source_file = iree_string_view_empty();
  uint32_t source_line = 0;
  iree_string_view_t primary_message = iree_string_view_empty();

  iree_host_size_t offset = sizeof(iree_net_status_wire_header_t);
  for (uint16_t i = 0; i < header.entry_count; ++i) {
    if (offset + sizeof(iree_net_status_wire_entry_t) > header.total_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "status wire entry %u header truncated", i);
    }

    iree_net_status_wire_entry_t entry;
    memcpy(&entry, data.data + offset, sizeof(entry));
    offset += sizeof(entry);

    iree_host_size_t padded_length = iree_net_align8(entry.length);
    if (offset + padded_length > header.total_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "status wire entry %u data truncated", i);
    }

    switch ((iree_net_status_entry_type_t)entry.type) {
      case IREE_NET_STATUS_ENTRY_SOURCE_LOCATION:
        source_file = iree_make_string_view((const char*)(data.data + offset),
                                            entry.length);
        source_line = entry.aux;
        break;
      case IREE_NET_STATUS_ENTRY_MESSAGE:
        primary_message = iree_make_string_view(
            (const char*)(data.data + offset), entry.length);
        break;
      default:
        break;
    }

    offset += padded_length;
  }

  // Allocate the status with copies of the source file and message.
  // We use iree_status_allocate_copy because the source strings point into the
  // wire buffer which the caller may free after this call returns.
  iree_status_t result = iree_status_allocate_copy(
      status_code, source_file, source_line, primary_message);

  // Second pass: append annotations and stack traces.
  offset = sizeof(iree_net_status_wire_header_t);
  for (uint16_t i = 0; i < header.entry_count; ++i) {
    iree_net_status_wire_entry_t entry;
    memcpy(&entry, data.data + offset, sizeof(entry));
    offset += sizeof(entry);

    iree_net_status_entry_type_t entry_type =
        (iree_net_status_entry_type_t)entry.type;
    if (entry_type == IREE_NET_STATUS_ENTRY_ANNOTATION ||
        entry_type == IREE_NET_STATUS_ENTRY_STACK_TRACE) {
      // Use iree_status_annotate_f to copy the text into owned storage.
      // iree_status_annotate borrows the string view, but the wire buffer is
      // transient and will be freed by the caller.
      result = iree_status_annotate_f(result, "%.*s", (int)entry.length,
                                      (const char*)(data.data + offset));
    }

    offset += iree_net_align8(entry.length);
  }

  *out_status = result;
  return iree_ok_status();
}
