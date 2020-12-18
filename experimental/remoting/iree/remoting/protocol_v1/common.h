// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Common structures and support functions for the V1 protocol.

#ifndef IREE_REMOTING_PROTOCOL_V1_COMMON_H_
#define IREE_REMOTING_PROTOCOL_V1_COMMON_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// "IREe" in big-endian. This is expected to be the magic number invariant
// of the protocol major version.
#define IREE_REMOTING_HANDSHAKE_MAGIC_NUMBER \
  (0x49 << 24) | (0x52 << 16) | (0x45 << 8) | (0x65)

// The size that an iree_remoting_handshake_v1_t should be padded to for
// transmission over the wire. This is expected to remain constant for all
// major versions of the protocol.
#define IREE_REMOTING_HANDSHAKE_PADDED_SIZE 64

// Upon initiating a protocol exchange, each side sends one of these
// handshake messages, padded to 64 bytes, establishing the transport
// parameters.
// Each multi-byte integer is represented in the byte order of the sender.
// Receivers should note the ordering of the magic number and decode
// appropriately.
typedef struct {
  // Magic number equal to IREE_REMOTING_HANDSHAKE_MAGIC_NUMBER in the
  // byte order of the sender.
  uint32_t magic_number;

  // The highest protocol version that the sender supports, represented as two
  // 16 bit quantities, representing major (high order) and minor (low order).
  // If two ends of the transport support different versions, they should
  // negotiate to the mutually supported minimum.
  uint32_t version;

  // Size in bytes of the maximum packet size for stream chunks. Both sides
  // should negotiate the actual maximum.
  uint32_t max_stream_packet_size;
} iree_remoting_v1_handshake_t;

// Handshake union, padded as it is over the wire.
typedef union {
  iree_remoting_v1_handshake_t handshake;
  char fill[IREE_REMOTING_HANDSHAKE_PADDED_SIZE];
} iree_remoting_v1_handshake_wire_t;

// Additional transport configuration decided during handshake.
// This is not part of the wire protocol.
typedef struct {
  bool byte_swapped;
} iree_remoting_transport_config_t;

// Fills a v1 handshake packet to be sent to a remote.
void iree_remoting_v1_init_handshake(iree_remoting_v1_handshake_wire_t *packet,
                                     uint32_t max_stream_packet_size);

// Merges a handshake packet that we sent with a handshake packet that a remote
// sent, updating `ours` to represent negotiated parameters.
// Returns false if the handshakes are incompatible.
bool iree_remoting_v1_merge_handshake(
    iree_remoting_transport_config_t *config,
    iree_remoting_v1_handshake_t *ours,
    const iree_remoting_v1_handshake_t *theirs);

// Packet header which precedes all data packets.
typedef struct {
  // Size of the packet, including this header, but excluding payload data.
  uint32_t packet_size;

  // Size of arbitrary payload data that follows this packet.
  uint32_t payload_size;

  // Type of the packet payload and flags:
  //   [31..16] : Bit-masked |iree_remoting_v1_flag_t| flags.
  //   [15..0]  : Packet type |iree_remoting_v1_packet_type_t| constant.
  uint32_t packet_type;
} iree_remoting_v1_packet_header_t;

// Message types that the protocol supports.
typedef enum {
  // Packet is a control message whose payload is an embedded flatbuffer.
  IREE_REMOTING_V1_PT_CONTROL = 0,
  // Packet is a stream message. An additional header
  // (|iree_remoting_v1_stream_header_t|) is present after the packet header,
  // followed by a chunk of stream data.
  IREE_REMOTING_V1_PT_STREAM = 1,
  // Packet is a no-op and should be ignored.
  IREE_REMOTING_V1_PT_NOP = 2,
} iree_remoting_v1_packet_type_t;

// Bit-mask enum for the iree_remoting_packet_v1_header_t.flags field.
typedef enum {} iree_remoting_v1_flag_t;

#ifdef __cplusplus
}
#endif

#endif  // IREE_REMOTING_PROTOCOL_V1_COMMON_H_
