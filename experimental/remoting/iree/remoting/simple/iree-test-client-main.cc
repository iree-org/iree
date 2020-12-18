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

#include "experimental/remoting/iree/remoting/protocol_v1/hal_stub.h"
#include "experimental/remoting/iree/remoting/protocol_v1/handler.h"
#include "experimental/remoting/iree/remoting/support/channel.h"
#include "experimental/remoting/iree/remoting/support/io_loop.h"
#include "experimental/remoting/iree/remoting/support/socket.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"  // TODO: Needed for IREE_CHECK_OK

using namespace iree::remoting;

namespace {

class SocketClient {
 public:
  static constexpr size_t kBufferSize = 4096;
  SocketClient(IoLoop &io_loop, Socket socket)
      : buffer_pool_(kBufferSize),
        protocol_(io_loop, buffer_pool_, iovec_pool_, socket.release_fd()),
        hal_stub_(protocol_) {}

  iree_status_t ConnectAndWait(SocketAddress &sa) {
    return protocol_.ConnectAndWait(sa);
  }

  void CloseAndWait() { protocol_.CloseAndWait(); }

  protocol_v1::HalClientStub &hal_stub() { return hal_stub_; }

 private:
  IoBufferPool buffer_pool_;
  IoBufferVec::Pool iovec_pool_;
  protocol_v1::SocketProtocolHandler protocol_;
  protocol_v1::HalClientStub hal_stub_;
};

constexpr size_t SocketClient::kBufferSize;

}  // namespace

int main(int argc, char **argv) {
  SocketAddress sa = SocketAddress::ParseInet("::", 3951);
  IREE_CHECK(sa.is_valid()) << "Failed to parse local address";
  IREE_LOG(INFO) << "Connecting to [" << sa.inet_addr_str()
                 << "]:" << sa.inet_port();

  // Create socket.
  Socket socket;
  IREE_CHECK_OK(socket.Initialize(sa.family(), SOCK_STREAM, 0));

  // Start IoLoop.
  std::unique_ptr<IoLoop> io_loop;
  IREE_CHECK_OK(IoLoop::Create(io_loop));

  SocketClient client(*io_loop, std::move(socket));
  IREE_CHECK_OK(client.ConnectAndWait(sa));

  sleep(1);
  IREE_LOG(INFO) << "Opening device...";
  client.hal_stub().OpenDevice();
  IREE_LOG(INFO) << "Device opened";

  sleep(1);
  IREE_LOG(INFO) << "Closing...";
  client.CloseAndWait();

  IREE_LOG(INFO) << "Done. Draining IoLoop.";
  io_loop->Run();
  return 0;
}
