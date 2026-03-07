// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured serialization of iree_status_t for network transport.
//
// Serializes an iree_status_t into a self-describing wire format that preserves
// structured information: status code (programmatic), source file:line, primary
// message, annotation chain, and formatted stack traces. Deserialization
// reconstructs an iree_status_t with the same code, source location, and
// message chain.
//
// ## Wire format
//
// Header (8 bytes):
//   uint8_t  version         Format version (currently 1).
//   uint8_t  status_code     iree_status_code_t value.
//   uint16_t entry_count     Number of entries following the header.
//   uint32_t total_size      Total bytes (header + all entries). For
//   validation.
//
// Entry (repeated entry_count times):
//   uint16_t type            Entry type (see iree_net_status_entry_type_t).
//   uint16_t length          Byte count of data (excludes this header+padding).
//   uint32_t aux             Type-specific auxiliary field.
//   uint8_t  data[length]    UTF-8 text.
//   uint8_t  padding[...]    Pad to next 8-byte boundary.
//
// Entry types:
//   SOURCE_LOCATION (0): data = source filename, aux = line number.
//   MESSAGE (1): data = primary error message, aux = 0.
//   ANNOTATION (2): data = annotation text, aux = 0.
//   STACK_TRACE (3): data = formatted stack trace text, aux = 0.
//
// ## Alignment and endianness
//
// All multi-byte fields are little-endian. Entries are 8-byte aligned.
// The header is 8 bytes (naturally aligned). Each entry header is 8 bytes
// followed by data padded to the next 8-byte boundary.
//
// ## Usage
//
// Serialization (sender):
//   iree_host_size_t size = 0;
//   iree_net_status_wire_size(status, &size);
//   uint8_t* buffer = allocate(size);
//   iree_net_status_wire_serialize(status, iree_make_byte_span(buffer, size));
//   // send buffer, then free status
//
// Deserialization (receiver):
//   iree_status_t status = iree_ok_status();
//   iree_net_status_wire_deserialize(
//       iree_make_const_byte_span(buffer, length), &status);
//   // use status (caller owns it)

#ifndef IREE_NET_STATUS_WIRE_H_
#define IREE_NET_STATUS_WIRE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Wire format constants
//===----------------------------------------------------------------------===//

// Current wire format version.
#define IREE_NET_STATUS_WIRE_VERSION 1

// Entry type identifiers.
typedef enum iree_net_status_entry_type_e {
  // Source file location. data = filename (UTF-8), aux = line number.
  IREE_NET_STATUS_ENTRY_SOURCE_LOCATION = 0,
  // Primary error message. data = message text (UTF-8), aux = 0.
  IREE_NET_STATUS_ENTRY_MESSAGE = 1,
  // Annotation (from iree_status_annotate). data = text (UTF-8), aux = 0.
  IREE_NET_STATUS_ENTRY_ANNOTATION = 2,
  // Formatted stack trace. data = text (UTF-8), aux = 0.
  IREE_NET_STATUS_ENTRY_STACK_TRACE = 3,
} iree_net_status_entry_type_t;

// Wire format header. 8 bytes.
typedef struct iree_net_status_wire_header_t {
  uint8_t version;      // IREE_NET_STATUS_WIRE_VERSION
  uint8_t status_code;  // iree_status_code_t
  uint16_t entry_count;
  uint32_t total_size;  // Total bytes including this header.
} iree_net_status_wire_header_t;
static_assert(sizeof(iree_net_status_wire_header_t) == 8, "");

// Wire format entry header. 8 bytes, followed by data + padding.
typedef struct iree_net_status_wire_entry_t {
  uint16_t type;    // iree_net_status_entry_type_t
  uint16_t length;  // Byte count of data following this header.
  uint32_t aux;     // Type-specific (line number for SOURCE_LOCATION).
} iree_net_status_wire_entry_t;
static_assert(sizeof(iree_net_status_wire_entry_t) == 8, "");

//===----------------------------------------------------------------------===//
// Serialization
//===----------------------------------------------------------------------===//

// Computes the serialized size of |status| in bytes.
//
// OK statuses serialize to just the 8-byte header with status_code=0 and
// entry_count=0. Non-OK statuses include all available structured information.
void iree_net_status_wire_size(iree_status_t status,
                               iree_host_size_t* out_size);

// Serializes |status| into |buffer|.
//
// |buffer| must be at least the size returned by iree_net_status_wire_size.
// The status is not consumed (caller still owns it).
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the buffer is too small.
iree_status_t iree_net_status_wire_serialize(iree_status_t status,
                                             iree_byte_span_t buffer);

//===----------------------------------------------------------------------===//
// Deserialization
//===----------------------------------------------------------------------===//

// Deserializes a wire-format buffer into an iree_status_t.
//
// On success, |*out_status| is the reconstructed status with code, source
// location, message, and annotations preserved. Caller owns the returned
// status (must free with iree_status_free or consume).
//
// An OK status code in the wire format produces iree_ok_status().
//
// Returns IREE_STATUS_INVALID_ARGUMENT if the wire data is malformed (bad
// version, truncated entries, total_size mismatch).
iree_status_t iree_net_status_wire_deserialize(iree_const_byte_span_t data,
                                               iree_status_t* out_status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_STATUS_WIRE_H_
