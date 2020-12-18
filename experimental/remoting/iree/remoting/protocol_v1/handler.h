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

#ifndef IREE_REMOTING_PROTOCOL_V1_HANDLER_H_
#define IREE_REMOTING_PROTOCOL_V1_HANDLER_H_

#include "experimental/remoting/iree/remoting/protocol_v1/common.h"
#include "experimental/remoting/iree/remoting/support/channel.h"
#include "iree/base/api.h"
#include "iree/base/flatcc.h"

namespace iree {
namespace remoting {
namespace protocol_v1 {

// Transport agnostic protocol handler state machine.
class ProtocolHandler {
 public:
  // A ceiling on the size of a control packet we will receive. Intended to
  // be generous but prevent abuse.
  static constexpr size_t kMaxControlPacketSize = 4096;
  static constexpr size_t kMinControlPacketAllocSize = 128;

  ProtocolHandler(IoBufferPool &buffer_pool, IoBufferVec::Pool &iovec_pool);
  virtual ~ProtocolHandler();

  //===---------------------------------------------------------------------===/
  // Protocol state.
  //===---------------------------------------------------------------------===/

  // Whether the protocol is in an initializing state, which is entered on a
  // call to Initiate and ends in either ready or abort.
  bool is_state_initializing() { return state_ == State::kHandshake; }

  // Whether the protocol is in an abort state, which is the terminal state
  // resulting from any end to the protocol (error or otherwise).
  bool is_state_abort() { return state_ == State::kAbort; }

  //===---------------------------------------------------------------------===/
  // Initiate and teardown.
  //===---------------------------------------------------------------------===/

  // Initiate IO interactions with peer.
  void Initiate();
  // Requests close of the protocol and backing channel. Complete when the
  // backing channel reports that it is quiescent.
  void Close() { Abort(); }

  //===---------------------------------------------------------------------===/
  // High level message handling.
  //===---------------------------------------------------------------------===/

  // Starts building a control packet for transmission.
  flatcc_builder_t *StartPacket();
  // Sends a control packet as constructed into the builder most recently
  // returned from |StartControlPacket|. Must not use the builder again
  // following this call.
  void SendPacket();

 protected:
  enum class State {
    kRecvPacketHeader,
    kRecvControlPacketPayload,
    kHandshake,
    kAbort,
    kNone,
  };

  IoBufferPool &buffer_pool() { return buffer_pool_; }
  IoBufferVec::Pool &iovec_pool() { return iovec_pool_; }

  // Tranport hooks.
  virtual size_t transport_buffer_size() = 0;
  virtual void TransportRead(size_t chunk_size) = 0;
  void TransportReceived(IoBufferVec::Ptr iovec);
  virtual void TransportWrite(IoBufferVec::Ptr iovec) = 0;
  virtual void TransportClose() = 0;

  void SetState(State state);
  void Abort();
  void InitiateRecvPacketHeader();
  void InitiateRecvControlPacketPayload();

 private:
  State state_ = State::kNone;
  iree_allocator_t allocator_ = iree_allocator_system();
  IoBufferPool &buffer_pool_;
  IoBufferVec::Pool &iovec_pool_;

  // Handshake.
  iree_remoting_v1_handshake_wire_t handshake_;
  iree_remoting_v1_handshake_wire_t their_handshake_;
  iree_remoting_transport_config_t config_;

  // Incoming packet.
  iree_remoting_v1_packet_header_t incoming_packet_header_;
  size_t incoming_control_packet_capacity_ = 0;
  size_t incoming_control_packet_size_ = 0;
  void *incoming_control_packet_ = nullptr;

  // Outgoing packet.
  flatcc_builder_t outgoing_packet_builder_;
  IoBufferVec::Ptr outgoing_payload_;
};

// Handler state machine for the V1 protocol.
// This class operates at the level of control packets and streams, allowing
// higher level code to build behavior on top of it and lower level code to
// bind it to a physical medium (such as a byte stream/socket).
class SocketProtocolHandler : public ProtocolHandler {
 public:
  SocketProtocolHandler(IoLoop &io_loop, IoBufferPool &buffer_pool,
                        IoBufferVec::Pool &iovec_pool, socket_t fd);
  ~SocketProtocolHandler();

  // Hooks the underlying channel's OnQuiescent callback. This will be called
  // after an Abort() is triggered once activity has ceased.
  void OnQuiescent(Channel::QuiescentCallback on_quiescent) {
    channel_.OnQuiescent(std::move(on_quiescent));
  }

  // Blocking convenience methods for client handlers.
  // Connects the socket and waits, returning the status.
  iree_status_t ConnectAndWait(SocketAddress &dest_addr);
  // Closes the socket and waits.
  void CloseAndWait();

 private:
  IoLoop &io_loop() { return channel_.io_loop(); }

  // Callbacks.
  void OnWrite(iree_status_t status);
  void OnRead(iree_status_t status, IoBufferVec::Ptr iovec);

  size_t transport_buffer_size() override;
  void TransportRead(size_t chunk_size) override;
  void TransportWrite(IoBufferVec::Ptr iovec) override;
  void TransportClose() override;

  iree_status_t CheckProtocolError() {
    if (is_state_abort()) {
      return iree_make_status(IREE_STATUS_UNKNOWN,
                              "A protocol error caused an abort");
    }
    return iree_ok_status();
  }

  SocketChannel channel_;
};

}  // namespace protocol_v1
}  // namespace remoting
}  // namespace iree

#endif  // IREE_REMOTING_PROTOCOL_V1_HANDLER_H_
