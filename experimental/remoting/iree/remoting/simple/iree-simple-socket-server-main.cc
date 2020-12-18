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

#include "experimental/remoting/iree/remoting/protocol_v1/handler.h"
#include "experimental/remoting/iree/remoting/support/channel.h"
#include "experimental/remoting/iree/remoting/support/io_loop.h"
#include "experimental/remoting/iree/remoting/support/socket.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"  // TODO: Needed for IREE_CHECK_OK

using namespace iree::remoting;
using namespace std::placeholders;

namespace {

class Session;
class Server;

class Server {
 public:
  static constexpr size_t kSharedBufferSize = 4096;

  Server(IoLoop &io_loop)
      : io_loop_(io_loop), shared_buffers_(kSharedBufferSize) {}

  IoLoop &io_loop() { return io_loop_; }
  IoBufferPool &shared_buffers() { return shared_buffers_; }

  void StartSession(socket_t fd);

 private:
  IoLoop &io_loop_;
  IoBufferPool shared_buffers_;
};

class Session {
 public:
  Session(Server &server, socket_t fd)
      : server_(server),
        handler_(server.io_loop(), server.shared_buffers(), iovec_pool_, fd) {
    handler_.OnQuiescent(std::bind(&Session::HandleShutdown, this));
  }

  void Initiate() { handler_.Initiate(); }

 private:
  void HandleShutdown() {
    IREE_DVLOG(1) << "Session: HandleShutdown";
    delete this;
  }
  Server &server_;
  IoBufferVec::Pool iovec_pool_;
  protocol_v1::SocketProtocolHandler handler_;
};

constexpr size_t Server::kSharedBufferSize;

inline void Server::StartSession(socket_t fd) {
  Session *session = new Session(*this, fd);
  session->Initiate();
}

}  // namespace

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
  Server server(*io_loop);

  io_loop->SubmitNew<IoAcceptRequest>(
      socket.fd(), [&](IoAcceptRequest::Ptr request) {
        IREE_LOG(INFO) << "Accepted connection: " << request->client_fd();
        if (request->ok()) {
          server.StartSession(request->client_fd());
        }

        // Submit again to accept the next connection.
        request->io_loop()->Submit(std::move(request));
      });

  io_loop->Run();
  return 0;
}
