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

#include <iostream>

#include "experimental/remoting/iree/remoting/support/channel.h"
#include "experimental/remoting/iree/remoting/support/io_loop.h"
#include "experimental/remoting/iree/remoting/support/socket.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"  // TODO: Needed for IREE_CHECK_OK

using namespace iree::remoting;
using namespace std::placeholders;

class ProtocolHandler {
 public:
  ProtocolHandler(IoLoop &io_loop, IoBufferPool &buffer_pool,
                  socket_t channel_fd)
      : io_loop_(io_loop),
        socket_(io_loop, buffer_pool, iovec_pool_, channel_fd) {
    socket_.OnQuiescent(std::bind(&ProtocolHandler::OnQuiescent, this));
    socket_.OnRead(std::bind(&ProtocolHandler::OnRead, this, _1, _2));
    socket_.OnWrite(std::bind(&ProtocolHandler::OnWrite, this, _1));
  }

  void OnRead(iree_status_t status, IoBufferVec::Ptr iovec) {
    if (!iree_status_is_ok(status)) {
      auto code = iree_status_consume_code(status);
      IREE_LOG(INFO) << "Read error: " << code;
      socket_.Close();
      return;
    }

    IREE_DLOG(INFO) << "Read complete: " << (message_number_++) << " ("
                    << status << ")";
    socket_.Write(std::move(iovec), true);
    socket_.Read(256, /*read_fully=*/true);
  }

  void OnWrite(iree_status_t status) {
    if (!iree_status_is_ok(status)) {
      auto code = iree_status_consume_code(status);
      IREE_LOG(INFO) << "Write error: " << code;
      socket_.Close();
      return;
    }
    IREE_DLOG(INFO) << "Write complete";
  }

  void OnQuiescent() {
    IREE_LOG(INFO) << "ProtocolHandler::Destroy";
    delete this;
  }

  void Initiate() { socket_.Read(4096, /*read_fully=*/true); }

 private:
  IoLoop &io_loop_;
  IoBufferVec::Pool iovec_pool_;
  SocketChannel socket_;
  int message_number_ = 1;
};

int main(int argc, char **argv) {
  int port = 3951;

  // Create socket.
  SocketAddress listen_addr = SocketAddress::AnyInet(3951);
  Socket socket;
  IREE_CHECK_OK(socket.Initialize(listen_addr.family(), SOCK_STREAM, 0));
  IREE_LOG(INFO) << "Listen on [" << listen_addr.inet_addr_str()
                 << "]:" << listen_addr.inet_port();
  IREE_CHECK_OK(socket.BindAndListen(listen_addr));

  // Start IoLoop.
  std::unique_ptr<IoLoop> io_loop;
  IREE_CHECK_OK(IoLoop::Create(io_loop));

  IoBufferPool buffer_pool{4096};

  io_loop->SubmitNew<IoAcceptRequest>(
      socket.fd(), [&](IoAcceptRequest::Ptr request) {
        IREE_LOG(INFO) << "Accepted connection: " << request->client_fd();
        if (request->ok()) {
          auto handler = new ProtocolHandler(*request->io_loop(), buffer_pool,
                                             request->client_fd());
          handler->Initiate();
        }

        // Submit again to accept the next connection.
        request->io_loop()->Submit(std::move(request));
      });

  io_loop->Run();
  return 0;
}
