// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Remote HAL protocol message definitions.
//
// This header defines the binary message format used for communication between
// remote HAL clients and servers. Messages are designed for:
//
//   - Minimal allocation: Fixed-size headers, inline small payloads.
//   - Zero-copy where possible: Large payloads reference external data.
//   - Forward compatibility: Unknown fields are skipped.
//   - Efficient encoding: Native byte order, no serialization overhead.
//
// ## Message Structure
//
// All messages share a common header:
//
//   struct message_header {
//     uint32_t total_size;     // Including header.
//     uint16_t message_type;   // Identifies the message kind.
//     uint16_t flags;          // Message-specific flags.
//   };
//
// ## Channel Types
//
// Messages are sent on one of three channel types:
//
//   - Control channel: Session lifecycle, capability negotiation, errors.
//     Low-frequency, reliable delivery required.
//
//   - Queue channel: HAL operations with frontier ordering.
//     High-frequency, causal consistency via vector clocks.
//
//   - Bulk channel: Large data transfers.
//     High-bandwidth, may use RDMA when available.

#ifndef IREE_HAL_REMOTE_PROTOCOL_MESSAGES_H_
#define IREE_HAL_REMOTE_PROTOCOL_MESSAGES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Protocol Constants
//===----------------------------------------------------------------------===//

// REVIEW: we may not need these either, though we may want to expose a way to
// set the custom protocol version on channel creation so they can reject during
// handshaking (no need to send every packet)

// Protocol magic number for connection handshake.
// "IREE" in little-endian.
#define IREE_HAL_REMOTE_PROTOCOL_MAGIC 0x45455249u

// Current protocol version.
// Bump when making incompatible changes.
#define IREE_HAL_REMOTE_PROTOCOL_VERSION 1u

//===----------------------------------------------------------------------===//
// Message Header
//===----------------------------------------------------------------------===//

// REVIEW: do we need this message header, or can we piggyback on the frames we
// already have from the control and queue channels? that should give us size,
// have a type ID we can use, etc. The less framing layers we need the better.

// Common header for all protocol messages.
typedef struct iree_hal_remote_message_header_t {
  // Total message size in bytes, including this header.
  uint32_t total_size;
  // Message type identifier (iree_hal_remote_message_type_t).
  uint16_t message_type;
  // Message-specific flags.
  uint16_t flags;
} iree_hal_remote_message_header_t;

//===----------------------------------------------------------------------===//
// Message Types
//===----------------------------------------------------------------------===//

// REVIEW: yeah, I think these all need to be reworked - our control/queue/etc
// all handle things like this - we may have many more types of these messages,
// though, that are HAL specific: device statistics queries, synchronous
// allocation/deallocation (iree_hal_allocator_t), executable cache creation,
// remote HAL file registration, etc. maybe we remove what we don't need or
// leave them commented - this file is currently dangerous as is not the right
// design and pollutes any agent coming across it into thinking we have a
// certain shaped problem when we in fact do not.

// Control channel message types.
typedef enum iree_hal_remote_control_message_type_e {
  // Connection handshake request (client -> server).
  IREE_HAL_REMOTE_CONTROL_MESSAGE_HANDSHAKE_REQUEST = 0x0001,
  // Connection handshake response (server -> client).
  IREE_HAL_REMOTE_CONTROL_MESSAGE_HANDSHAKE_RESPONSE = 0x0002,
  // Session termination (either direction).
  IREE_HAL_REMOTE_CONTROL_MESSAGE_DISCONNECT = 0x0003,
  // Error notification (either direction).
  IREE_HAL_REMOTE_CONTROL_MESSAGE_ERROR = 0x0004,
  // Keepalive ping (either direction).
  IREE_HAL_REMOTE_CONTROL_MESSAGE_PING = 0x0005,
  // Keepalive pong (response to ping).
  IREE_HAL_REMOTE_CONTROL_MESSAGE_PONG = 0x0006,
} iree_hal_remote_control_message_type_t;

// Queue channel message types.
typedef enum iree_hal_remote_queue_message_type_e {
  // Semaphore value signal.
  IREE_HAL_REMOTE_QUEUE_MESSAGE_SEMAPHORE_SIGNAL = 0x0100,
  // Semaphore value query.
  IREE_HAL_REMOTE_QUEUE_MESSAGE_SEMAPHORE_QUERY = 0x0101,
  // Queue operation: alloca.
  IREE_HAL_REMOTE_QUEUE_MESSAGE_ALLOCA = 0x0110,
  // Queue operation: dealloca.
  IREE_HAL_REMOTE_QUEUE_MESSAGE_DEALLOCA = 0x0111,
  // Queue operation: fill.
  IREE_HAL_REMOTE_QUEUE_MESSAGE_FILL = 0x0112,
  // Queue operation: copy.
  IREE_HAL_REMOTE_QUEUE_MESSAGE_COPY = 0x0113,
  // Queue operation: update (small data inline).
  IREE_HAL_REMOTE_QUEUE_MESSAGE_UPDATE = 0x0114,
  // REVIEW: BARRIER
  // Queue operation: execute command buffer.
  IREE_HAL_REMOTE_QUEUE_MESSAGE_EXECUTE = 0x0120,
  // Queue operation: dispatch (direct, no command buffer).
  IREE_HAL_REMOTE_QUEUE_MESSAGE_DISPATCH = 0x0121,
  // Executable upload request.
  IREE_HAL_REMOTE_QUEUE_MESSAGE_EXECUTABLE_UPLOAD = 0x0130,
  // Executable upload acknowledgment.
  IREE_HAL_REMOTE_QUEUE_MESSAGE_EXECUTABLE_ACK = 0x0131,
} iree_hal_remote_queue_message_type_t;

// Bulk channel message types.
typedef enum iree_hal_remote_bulk_message_type_e {
  // Buffer data transfer (host -> device or device -> host).
  IREE_HAL_REMOTE_BULK_MESSAGE_TRANSFER = 0x0200,
  // Transfer acknowledgment.
  IREE_HAL_REMOTE_BULK_MESSAGE_TRANSFER_ACK = 0x0201,
} iree_hal_remote_bulk_message_type_t;

//===----------------------------------------------------------------------===//
// Control Channel Messages
//===----------------------------------------------------------------------===//

// Flags for handshake request.
enum iree_hal_remote_handshake_flag_bits_e {
  IREE_HAL_REMOTE_HANDSHAKE_FLAG_NONE = 0u,
  // Client supports RDMA bulk transfers.
  IREE_HAL_REMOTE_HANDSHAKE_FLAG_RDMA_CAPABLE = 1u << 0,
};
typedef uint32_t iree_hal_remote_handshake_flags_t;

// Connection handshake request.
typedef struct iree_hal_remote_handshake_request_t {
  iree_hal_remote_message_header_t header;
  // Protocol magic (IREE_HAL_REMOTE_PROTOCOL_MAGIC).
  uint32_t magic;
  // Client protocol version.
  uint32_t version;
  // Client capability flags.
  iree_hal_remote_handshake_flags_t capabilities;
  // Reserved for future use.
  uint32_t reserved;
} iree_hal_remote_handshake_request_t;

// Connection handshake response.
typedef struct iree_hal_remote_handshake_response_t {
  iree_hal_remote_message_header_t header;
  // Protocol magic (IREE_HAL_REMOTE_PROTOCOL_MAGIC).
  uint32_t magic;
  // Negotiated protocol version (min of client and server).
  uint32_t version;
  // Server capability flags (intersection with client).
  iree_hal_remote_handshake_flags_t capabilities;
  // Unique session identifier assigned by server.
  uint64_t session_id;
} iree_hal_remote_handshake_response_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_PROTOCOL_MESSAGES_H_
