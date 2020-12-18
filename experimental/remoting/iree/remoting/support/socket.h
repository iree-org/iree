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

#ifndef IREE_REMOTING_SUPPORT_SOCKET_H_
#define IREE_REMOTING_SUPPORT_SOCKET_H_

#include <string>

#include "experimental/remoting/iree/remoting/support/platform.h"
#include "iree/base/api.h"

namespace iree {
namespace remoting {

// Converts the result of the last socket error to a status in a platform
// specific way.
iree_status_t SocketErrorToStatus(const char *prefix);

class SocketAddress {
 public:
  union Addr {
    struct sockaddr sa;
    struct sockaddr_in s4;
    struct sockaddr_in6 s6;
    struct sockaddr_storage ss;
  };
  SocketAddress() : addr_len_(0) { memset(&addr_, 0, sizeof(addr_)); }
  SocketAddress(sa_family_t family);

  // Whether the SocketAddress is valid.
  bool is_valid() const { return family() != 0 && addr_len_ != 0; }

  Addr &addr() { return addr_; }
  const Addr &addr() const { return addr_; }
  socklen_t &addr_len() { return addr_len_; }
  const socklen_t &addr_len() const { return addr_len_; }

  // Pointer to the generic sockaddr struct.
  struct sockaddr *sockaddr() {
    return &addr_.sa;
  }
  const struct sockaddr *sockaddr() const { return &addr_.sa; }
  sa_family_t family() const { return addr_.sa.sa_family; }

  // ---------------------------------------------------------------------------
  // Accessors for INET family addresses. In general, these are generic for
  // IPV4 and IPV6 unless noted.
  // ---------------------------------------------------------------------------
  // Initializes to an AF_INET or AF_INET6 any address on port.
  // Optionally forces IPV4.
  static SocketAddress AnyInet(in_port_t port, bool force_ipv4 = false);

  // Initializes to an AF_INET or AF_INET6 from a textual form. The returned
  // SocketAddress will be !is_valid() if parsing is not successful.
  static SocketAddress ParseInet(const std::string &ip_text, in_port_t port,
                                 bool force_ipv4 = false);

  // Whether this is an INET family address.
  bool is_inet() const {
    auto f = family();
    return f == AF_INET || f == AF_INET6;
  }

  // Returns the INET family port or -1 if not an INET address.
  int inet_port() const {
    if (!is_inet()) return -1;
    auto f = family();
    return ntohs(f == AF_INET6 ? addr_.s6.sin6_port : addr_.s4.sin_port);
  }

  // Returns the INET address as a string or the empty string if not an INET
  // address.
  std::string inet_addr_str() const;

 private:
  Addr addr_;
  socklen_t addr_len_;
};

// Simple wrapper around a movable file descriptor for a socket with helpers
// for common setup operations.
class Socket {
 public:
  Socket() : fd_(invalid_socket) {}
  Socket(socket_t fd) : fd_(fd) {}
  Socket(const Socket &) = delete;
  Socket(Socket &&other) : fd_(other.fd_) { other.fd_ = invalid_socket; }
  Socket &operator=(const Socket &other) = delete;

  ~Socket() { CloseOrDie(); }

  // Initializes the socket for the given domain, type and protocol. See the
  // BSD socket() function for documentation.
  iree_status_t Initialize(int domain, int type, int protocol);

  // One-stop mechanism to create a socket, bind and listen.
  // The socket must not be is_valid().
  struct BindAndListenOptions {
    int backlog = 128;
  };
  iree_status_t BindAndListen(const SocketAddress &addr,
                              const BindAndListenOptions &options);
  iree_status_t BindAndListen(const SocketAddress &addr) {
    BindAndListenOptions options;
    return BindAndListen(addr, options);
  }

  // Whether the file descriptor is a valid handle.
  bool is_valid() { return fd_ != invalid_socket; }

  // Gets the file descriptor.
  socket_t fd() { return fd_; }

  // Gets the file descriptor, releasing this instance's ownership of it.
  socket_t release_fd() {
    socket_t ret = fd_;
    fd_ = invalid_socket;
    return ret;
  }

  // Closes the socket in a platform neutral way.
  iree_status_t Close();

  // Closes the socket, dieing if there is an error.
  void CloseOrDie();

 private:
  socket_t fd_;
};

}  // namespace remoting
}  // namespace iree

#endif  // IREE_REMOTING_SUPPORT_SOCKET_H_
