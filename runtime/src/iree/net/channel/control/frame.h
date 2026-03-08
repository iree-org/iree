// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Control channel wire format.
//
// Control frames carry typed messages for session management: liveness
// detection, graceful shutdown, error reporting, and application data.
// DATA frames are the primary steady-state traffic, carrying inline command
// buffer recordings that can be many megabytes per submission. Protocol
// frames (PING/PONG/GOAWAY/ERROR) are small (tens to hundreds of bytes).
//
// ## Frame header layout (8 bytes)
//
//   ┌──────────────────────────────────────────────────────────────────────┐
//   │ Byte 0:    version (currently 1)                                     │
//   │ Byte 1:    type (see iree_net_control_frame_type_t)                  │
//   │ Byte 2:    flags (per-type, see below)                               │
//   │ Bytes 3-7: reserved (must be 0)                                      │
//   └──────────────────────────────────────────────────────────────────────┘
//
// Payload immediately follows the header. Payload length is determined by the
// enclosing message length (message.data_length - header_size), NOT by a field
// in the header. The message endpoint delivers complete messages, so no
// length-prefix is needed for frame boundary detection.
//
// ## Frame types
//
//   - PING/PONG: Liveness detection and RTT measurement.
//   - GOAWAY: Graceful shutdown initiation.
//   - ERROR: Error notification (payload is a status_wire blob).
//   - DATA: Application payload. Carries inline command buffer recordings
//     and other HAL data. Can be many megabytes per frame.
//
// ## No magic bytes
//
// The endpoint routing layer guarantees that messages delivered to the control
// channel's endpoint belong to this channel. Magic bytes for type
// discrimination are unnecessary. If transport routing is broken, magic
// validation wouldn't save us anyway.
//
// ## No stream multiplexing
//
// The previous design used a stream_id field for sub-multiplexing within the
// control channel. This is no longer needed: the connection's open_endpoint()
// provides per-channel endpoints, and the TCP mux layer handles stream routing
// transparently. If the session needs multiple logical streams, it opens
// multiple endpoints.
//
// ## Alignment and endianness
//
// All multi-byte fields are little-endian. Frame headers may arrive at
// unaligned offsets in receive buffers. The channel memcpy's the 8-byte header
// to a stack-local struct before processing.
//
// ## Reserved bytes
//
// Reserved bytes MUST be zero on send. Receivers MUST reject frames with
// nonzero reserved bytes (fail-fast). This ensures old implementations do not
// silently misparse frames from newer senders that repurpose reserved space.

#ifndef IREE_NET_CHANNEL_CONTROL_FRAME_H_
#define IREE_NET_CHANNEL_CONTROL_FRAME_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Current control frame format version.
#define IREE_NET_CONTROL_FRAME_VERSION 1

// Size of the control frame header in bytes.
#define IREE_NET_CONTROL_FRAME_HEADER_SIZE 8

//===----------------------------------------------------------------------===//
// Frame types
//===----------------------------------------------------------------------===//

// Control frame type identifiers.
//
// Types 0x05-0x7F are reserved for future channel-level extensions.
// Types 0x81-0xFF are reserved for future application-level extensions.
typedef enum iree_net_control_frame_type_e {
  IREE_NET_CONTROL_FRAME_TYPE_PING = 0x01,    // Liveness check request.
  IREE_NET_CONTROL_FRAME_TYPE_PONG = 0x02,    // Liveness check response.
  IREE_NET_CONTROL_FRAME_TYPE_GOAWAY = 0x03,  // Graceful shutdown initiation.
  IREE_NET_CONTROL_FRAME_TYPE_ERROR = 0x04,   // Error notification.
  IREE_NET_CONTROL_FRAME_TYPE_DATA = 0x80,  // Application payload (up to MBs).
} iree_net_control_frame_type_t;

//===----------------------------------------------------------------------===//
// Header
//===----------------------------------------------------------------------===//

// PONG frame flag bits.
typedef enum iree_net_control_pong_flag_bits_e {
  IREE_NET_CONTROL_PONG_FLAG_NONE = 0u,

  // Responder appended its monotonic timestamp (8 bytes, LE uint64 nanoseconds)
  // after the echoed PING payload.
  IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP = 1u << 0,
} iree_net_control_pong_flag_bits_t;

// DATA frame flag bits. Application-defined; the control channel passes these
// through to callbacks without interpretation.
typedef enum iree_net_control_data_flag_bits_e {
  IREE_NET_CONTROL_DATA_FLAG_NONE = 0u,
} iree_net_control_data_flag_bits_t;

typedef uint8_t iree_net_control_frame_flags_t;

// On-wire control frame header (8 bytes). All fields little-endian.
typedef struct iree_net_control_frame_header_t {
  uint8_t version;     // IREE_NET_CONTROL_FRAME_VERSION
  uint8_t type;        // iree_net_control_frame_type_t
  uint8_t flags;       // Per-type flag bits.
  uint8_t reserved0;   // Must be 0.
  uint32_t reserved1;  // Must be 0.
} iree_net_control_frame_header_t;

static_assert(sizeof(iree_net_control_frame_header_t) ==
                  IREE_NET_CONTROL_FRAME_HEADER_SIZE,
              "Control frame header must be exactly 8 bytes");
static_assert(offsetof(iree_net_control_frame_header_t, version) == 0, "");
static_assert(offsetof(iree_net_control_frame_header_t, type) == 1, "");
static_assert(offsetof(iree_net_control_frame_header_t, flags) == 2, "");
static_assert(offsetof(iree_net_control_frame_header_t, reserved0) == 3, "");
static_assert(offsetof(iree_net_control_frame_header_t, reserved1) == 4, "");

// Initializes a control frame header.
static inline void iree_net_control_frame_header_initialize(
    iree_net_control_frame_type_t type, iree_net_control_frame_flags_t flags,
    iree_net_control_frame_header_t* out_header) {
  out_header->version = IREE_NET_CONTROL_FRAME_VERSION;
  out_header->type = (uint8_t)type;
  out_header->flags = flags;
  out_header->reserved0 = 0;
  out_header->reserved1 = 0;
}

// Validates a control frame header.
// Returns iree_ok_status() if valid, or an error describing the problem.
// Takes header by value (caller memcpy'd from wire to aligned local).
static inline iree_status_t iree_net_control_frame_header_validate(
    iree_net_control_frame_header_t header) {
  if (header.version != IREE_NET_CONTROL_FRAME_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported control frame version: %u",
                            header.version);
  }
  if (header.reserved0 != 0 || header.reserved1 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control frame reserved fields must be 0");
  }
  return iree_ok_status();
}

// Returns the frame type.
static inline iree_net_control_frame_type_t iree_net_control_frame_header_type(
    iree_net_control_frame_header_t header) {
  return (iree_net_control_frame_type_t)header.type;
}

// Returns the frame flags.
static inline iree_net_control_frame_flags_t
iree_net_control_frame_header_flags(iree_net_control_frame_header_t header) {
  return header.flags;
}

// Returns true if the specified flag bit is set.
static inline bool iree_net_control_frame_header_has_flag(
    iree_net_control_frame_header_t header, uint8_t flag) {
  return (header.flags & flag) != 0;
}

//===----------------------------------------------------------------------===//
// Payloads
//===----------------------------------------------------------------------===//

// Fixed-size prefix of a GOAWAY frame's payload.
// Followed by a UTF-8 reason string (not null-terminated).
typedef struct iree_net_control_goaway_payload_t {
  uint32_t reason_code;  // 0 = normal shutdown, nonzero = error category.
} iree_net_control_goaway_payload_t;
static_assert(sizeof(iree_net_control_goaway_payload_t) == 4, "");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_CONTROL_FRAME_H_
