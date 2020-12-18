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

#include "experimental/remoting/iree/remoting/protocol_v1/common.h"

#include <string.h>

#include <algorithm>

#include "iree/base/logging.h"

// Now that we are in C++, do some sanity asserts about the structures.
static_assert(sizeof(iree_remoting_v1_handshake_wire_t) ==
                  IREE_REMOTING_HANDSHAKE_PADDED_SIZE,
              "iree_remoting_handshake_v1_t is too large");

namespace {

uint32_t SwapUint32(uint32_t v) {
  return ((v >> 24) & 0xff) | ((v << 8) & 0xff0000) | ((v >> 8) & 0xff00) |
         ((v << 24) & 0xff000000);
}

}  // namespace

void iree_remoting_v1_init_handshake(iree_remoting_v1_handshake_wire_t *packet,
                                     uint32_t max_stream_packet_size) {
  memset(packet, 0, sizeof(*packet));
  packet->handshake.magic_number = IREE_REMOTING_HANDSHAKE_MAGIC_NUMBER;
  packet->handshake.version = (1 << 16) | (1);  // 1.1
  packet->handshake.max_stream_packet_size = max_stream_packet_size;
}

bool iree_remoting_v1_merge_handshake(
    iree_remoting_transport_config_t *config,
    iree_remoting_v1_handshake_t *ours,
    const iree_remoting_v1_handshake_t *theirs) {
  config->byte_swapped = false;

  // Detect magic number.
  if (theirs->magic_number != ours->magic_number) {
    // Is it byte swapped?
    if (theirs->magic_number != SwapUint32(ours->magic_number)) {
      IREE_DVLOG(1) << "Mismatched magic number: 0x" << std::hex
                    << theirs->magic_number;
      return false;
    }
    config->byte_swapped = true;
  }

  // Version.
  uint32_t theirs_version =
      config->byte_swapped ? SwapUint32(theirs->version) : theirs->version;
  if (theirs_version != ours->version) {
    // TODO: Implement the version negotiation when there is more than one.
    IREE_DVLOG(1) << "Version mismatch: theirs=" << std::hex << theirs_version
                  << ", ours=" << ours->version;
    return false;
  }

  // Packet sizes.
  uint32_t theirs_max_stream_packet_size =
      config->byte_swapped ? SwapUint32(theirs->max_stream_packet_size)
                           : theirs->max_stream_packet_size;
  ours->max_stream_packet_size =
      std::min(ours->max_stream_packet_size, theirs_max_stream_packet_size);

  return true;
}
