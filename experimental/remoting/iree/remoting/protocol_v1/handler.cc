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

#include "experimental/remoting/iree/remoting/protocol_v1/handler.h"

#include "iree/base/logging.h"

using namespace std::placeholders;

namespace iree {
namespace remoting {
namespace protocol_v1 {

namespace {

void ConsumeStatus(const char *message, iree_status_t status) {
  char *buffer;
  iree_host_size_t len;
  bool alloced = iree_status_to_string(status, &buffer, &len);
  IREE_DVLOG(1) << message << ": " << (alloced ? buffer : "<no information>");
  if (alloced) free(buffer);
  iree_status_ignore(status);
}

}  // namespace

constexpr size_t ProtocolHandler::kMaxControlPacketSize;
constexpr size_t ProtocolHandler::kMinControlPacketAllocSize;

ProtocolHandler::ProtocolHandler(IoBufferPool &buffer_pool,
                                 IoBufferVec::Pool &iovec_pool)
    : buffer_pool_(buffer_pool), iovec_pool_(iovec_pool) {
  IREE_CHECK_EQ(flatcc_builder_init(&outgoing_packet_builder_), 0);
}

ProtocolHandler::~ProtocolHandler() {
  iree_allocator_free(allocator_, incoming_control_packet_);
  flatcc_builder_clear(&outgoing_packet_builder_);
}

void ProtocolHandler::SetState(State new_state) {
  IREE_DVLOG(1) << "ProtocolHandler::SetState(" << static_cast<int>(state_)
                << " -> " << static_cast<int>(new_state) << ")";
  state_ = new_state;
}

void ProtocolHandler::Initiate() {
  assert(state_ == State::kNone);
  SetState(State::kHandshake);
  iree_remoting_v1_init_handshake(&handshake_, transport_buffer_size());

  auto write_iov = iovec_pool_.Get();
  write_iov->add(static_cast<void *>(&handshake_), sizeof(handshake_));
  TransportWrite(std::move(write_iov));
  TransportRead(sizeof(their_handshake_));
}

void ProtocolHandler::Abort() {
  if (state_ == State::kAbort) return;
  SetState(State::kAbort);
  TransportClose();
}

void ProtocolHandler::InitiateRecvPacketHeader() {
  SetState(State::kRecvPacketHeader);
  TransportRead(sizeof(incoming_packet_header_));
}

void ProtocolHandler::InitiateRecvControlPacketPayload() {
  SetState(State::kRecvControlPacketPayload);
  size_t size = incoming_packet_header_.packet_size;
  if (size < sizeof(incoming_packet_header_) || size > kMaxControlPacketSize) {
    IREE_DVLOG(1) << "ProtocolHandler: Illegal packet size";
    return Abort();
  }
  size -= sizeof(incoming_packet_header_);

  // Ensure the incoming_control_packet memory is sufficient.
  if (size > incoming_control_packet_capacity_) {
    size_t new_capacity = std::min(size, kMinControlPacketAllocSize);
    iree_status_t status = iree_allocator_realloc(allocator_, new_capacity,
                                                  &incoming_control_packet_);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      iree_status_ignore(status);
      IREE_DVLOG(1) << "SocketProtocolHandler: Alloc failed (" << size << ")";
      return Abort();
    }
    incoming_control_packet_capacity_ = new_capacity;
  }
  // Read payload.
  incoming_control_packet_size_ = size;
  TransportRead(incoming_control_packet_size_);
}

flatcc_builder_t *ProtocolHandler::StartPacket() {
  flatcc_builder_reset(&outgoing_packet_builder_);
  return &outgoing_packet_builder_;
}

void ProtocolHandler::TransportReceived(IoBufferVec::Ptr iovec) {
  switch (state_) {
    case State::kHandshake:
      iovec->FlattenTo(&their_handshake_, sizeof(their_handshake_));
      if (IREE_UNLIKELY(!iree_remoting_v1_merge_handshake(
              &config_, &handshake_.handshake, &their_handshake_.handshake))) {
        IREE_DVLOG(1) << "SocketProtocolHandler: Bad handshake";
        return Abort();
      }
      InitiateRecvPacketHeader();
      break;
    case State::kRecvPacketHeader: {
      iovec->FlattenTo(&incoming_packet_header_,
                       sizeof(incoming_packet_header_));
      uint32_t pt = incoming_packet_header_.packet_type & 0xffff;
      switch (pt) {
        case IREE_REMOTING_V1_PT_CONTROL:
          InitiateRecvControlPacketPayload();
          break;

        default:
          IREE_DVLOG(1) << "SocketProtocolHandler: Illegal packet type " << pt;
          return Abort();
      }
      break;
    }
    case State::kRecvControlPacketPayload: {
      IREE_DVLOG(1) << " ** TODO: Implement kRecvControlPacketPayload (size="
                    << iovec->total_iov_len() << ")";
      InitiateRecvPacketHeader();
      break;
    }
    default:
      IREE_DVLOG(1) << "SocketProtocolHandler illegal state: "
                    << static_cast<int>(state_);
      return Abort();
  }
}

// Sends a control packet as constructed into the builder most recently
// returned from |StartControlPacket|. Must not use the builder again
// following this call.
void ProtocolHandler::SendPacket() {
  flatcc_builder_t *B = &outgoing_packet_builder_;
  size_t payload_size =
      outgoing_payload_ ? outgoing_payload_->total_iov_len() : 0;

  // There are various, advanced ways to integrate with flatcc, possibly
  // emitting directly to our buffer/iov lists. However, for now, these
  // messages are small, and we accept a worst-case alloc/copy, leaving
  // optimization to posterity.
  size_t fb_size;
  uint8_t *fb_data;
  bool fb_alloced = false;
  fb_data =
      static_cast<uint8_t *>(flatcc_builder_get_direct_buffer(B, &fb_size));
  if (!fb_data) {
    fb_alloced = true;
    fb_data =
        static_cast<uint8_t *>(flatcc_builder_finalize_buffer(B, &fb_size));
  }
  IREE_CHECK_NE(fb_size, 0);

  auto iovec = iovec_pool_.Get();
  auto buffer = buffer_pool_.Get();

  // Ensure that a single buffer is at least large enough for the packet
  // header and the message.
  iree_remoting_v1_packet_header_t packet_header;
  size_t header_message_len = sizeof(packet_header) + fb_size;
  if (IREE_UNLIKELY(header_message_len > buffer->size())) {
    if (fb_alloced) flatcc_builder_free(fb_data);
    IREE_LOG(ERROR) << "Buffer size too small for message";
    return Abort();
  }

  // Populate header.
  packet_header.packet_type = IREE_REMOTING_V1_PT_CONTROL;
  packet_header.packet_size = header_message_len;
  packet_header.payload_size = payload_size;
  std::memcpy(buffer->data(), &packet_header, sizeof(packet_header));
  iovec->add(buffer->data(), header_message_len, std::move(buffer));

  // Append payload.
  if (outgoing_payload_) {
    iovec->append(*outgoing_payload_);
  }

  // Send it off.
  TransportWrite(std::move(iovec));

  if (fb_alloced) flatcc_builder_free(fb_data);
}

//===-----------------------------------------------------------------------===/
// SocketProtocolHandler
//===-----------------------------------------------------------------------===/

namespace {
// TODO: Once the shape of this is known, move it to a facility on IoLoop
// (and guard proper usage there).
template <typename RequestTy>
IoRequestPtr<RequestTy> SubmitBlocking(IoLoop &io_loop,
                                       IoRequestPtr<RequestTy> request) {
  IoRequest *done_request = nullptr;
  request->SetCompletionHandler(
      &done_request, +[](IoRequest *completed_request, void *completion_data) {
        IoRequest **inner_done_request =
            static_cast<IoRequest **>(completion_data);
        *inner_done_request = completed_request;
      });
  io_loop.Submit(std::move(request));
  io_loop.Run([&done_request]() { return done_request == nullptr; });
  return IoRequestPtr<RequestTy>(static_cast<RequestTy *>(done_request));
}
}  // namespace

SocketProtocolHandler::SocketProtocolHandler(IoLoop &io_loop,
                                             IoBufferPool &buffer_pool,
                                             IoBufferVec::Pool &iovec_pool,
                                             socket_t fd)
    : ProtocolHandler(buffer_pool, iovec_pool),
      channel_(io_loop, buffer_pool, iovec_pool, fd) {
  channel_.OnRead(std::bind(&SocketProtocolHandler::OnRead, this, _1, _2));
  channel_.OnWrite(std::bind(&SocketProtocolHandler::OnWrite, this, _1));
}

SocketProtocolHandler::~SocketProtocolHandler() {}

void SocketProtocolHandler::OnWrite(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    ConsumeStatus("SocketProtocolHandler::OnWrite() error", status);
    return Abort();
  }
}

void SocketProtocolHandler::OnRead(iree_status_t status,
                                   IoBufferVec::Ptr iovec) {
  if (iree_status_is_resource_exhausted(status)) {
    // EOF. Silently ignore and shutdown.
    iree_status_ignore(status);
    return Abort();
  }

  if (!iree_status_is_ok(status)) {
    ConsumeStatus("SocketProtocolHandler::OnWrite() error", status);
    return Abort();
  }

  TransportReceived(std::move(iovec));
}

size_t SocketProtocolHandler::transport_buffer_size() {
  return channel_.buffer_size();
}

void SocketProtocolHandler::TransportRead(size_t chunk_size) {
  channel_.Read(chunk_size, /*read_fully=*/true);
}

void SocketProtocolHandler::TransportWrite(IoBufferVec::Ptr iovec) {
  channel_.Write(std::move(iovec), /*flush=*/true);
}

void SocketProtocolHandler::TransportClose() { channel_.Close(); }

iree_status_t SocketProtocolHandler::ConnectAndWait(SocketAddress &dest_addr) {
  // Connect socket.
  IREE_DVLOG(1) << "SocketProtocolHandler::Connect() : Connecting socket";
  auto request = SubmitBlocking(
      io_loop(),
      io_loop().NewRequest<IoConnectSocketRequest>(channel_.fd(), dest_addr));
  IREE_RETURN_IF_ERROR(request->ConsumeStatus());

  // Initiate protocol and wait for ready.
  IREE_DVLOG(1) << "SocketProtocolHandler::Connect() : Initiating protocol";
  Initiate();
  io_loop().Run([this]() { return is_state_initializing(); });
  IREE_DVLOG(1) << "SocketProtocolHandler::Connect() : Protocol initiated";
  return CheckProtocolError();
}

void SocketProtocolHandler::CloseAndWait() {
  IREE_DVLOG(1) << "SocketProtocolHandler: CloseAndWait()";
  Close();
  SocketChannel *local_channel = &channel_;
  io_loop().Run([local_channel]() { return !local_channel->is_quiescent(); });
  IREE_DVLOG(1) << "SocketClient: is_quiescent";
}

}  // namespace protocol_v1
}  // namespace remoting
}  // namespace iree
