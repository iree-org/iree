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

#include "experimental/remoting/iree/remoting/support/socket.h"

#include "iree/base/logging.h"  // TODO: Needed for IREE_CHECK_OK

namespace iree {
namespace remoting {

iree_status_t SocketErrorToStatus(const char* message) {
  iree_status_code_t code;
#if IREE_REMOTING_IS_WINSOCK
  code = iree_status_code_from_win32_error(WSAGetLastError());
#else
  code = iree_status_code_from_errno(errno);
#endif
  return iree_make_status(code, message);
}

SocketAddress::SocketAddress(sa_family_t family) : SocketAddress() {
  switch (family) {
    case AF_INET6:
      addr_.s6.sin6_family = family;
      addr_len_ = sizeof(addr_.s6);
      break;
    case AF_INET:
      addr_.s4.sin_family = family;
      addr_len_ = sizeof(addr_.s4);
      break;
    default:
      IREE_LOG(ERROR) << "Unsupported sa_family_t = "
                      << static_cast<unsigned>(family);
  }
}

SocketAddress SocketAddress::AnyInet(in_port_t port, bool force_ipv4) {
  SocketAddress sa(force_ipv4 ? AF_INET : AF_INET6);
  if (force_ipv4) {
    sa.addr().s4.sin_addr.s_addr = INADDR_ANY;
    sa.addr().s4.sin_port = htons(port);
  } else {
    sa.addr().s6.sin6_addr = in6addr_any;
    sa.addr().s6.sin6_port = htons(port);
  }
  return sa;
}

SocketAddress SocketAddress::ParseInet(const std::string& ip_text,
                                       in_port_t port, bool force_ipv4) {
  // Try to parse as IPV6.
  if (!force_ipv4) {
    SocketAddress sa(AF_INET6);
    if (inet_pton(AF_INET6, ip_text.c_str(), &sa.addr().s6.sin6_addr)) {
      sa.addr().s6.sin6_port = htons(port);
      return sa;
    }
  }

  // Fallback to try IPV4.
  {
    SocketAddress sa(AF_INET);
    if (inet_pton(AF_INET, ip_text.c_str(), &sa.addr().s4.sin_addr)) {
      sa.addr().s4.sin_port = htons(port);
      return sa;
    }
  }

  return SocketAddress();  // Invalid
}

std::string SocketAddress::inet_addr_str() const {
  if (!is_inet()) return std::string();
  auto f = family();
  std::string s;
  if (f == AF_INET6) {
    s.resize(INET6_ADDRSTRLEN);
    if (!inet_ntop(f, &addr_.s6.sin6_addr, &s.front(), s.size())) {
      s.clear();
    }
  } else {
    s.resize(INET_ADDRSTRLEN);
    if (!inet_ntop(f, &addr_.s4.sin_addr.s_addr, &s.front(), s.size())) {
      s.clear();
    }
  }
  // inet_ntop writes a null terminated string, so resize to the actual.
  s.resize(strlen(s.data()));
  return s;
}

iree_status_t Socket::Initialize(int domain, int type, int protocol) {
  if (is_valid()) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Initialize must be called on an uninitialized socket");
  }
  fd_ = socket(domain, type, protocol);
  if (!is_valid()) {
    return SocketErrorToStatus("could not create socket");
  }
  return iree_ok_status();
}

iree_status_t Socket::BindAndListen(const SocketAddress& addr,
                                    const BindAndListenOptions& options) {
  const int val = 1;
  if (setsockopt(fd(), SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val)) ==
      socket_error) {
    return SocketErrorToStatus("could not set SO_REUSEADDR socket option");
  }

  // Bind and listen.
  if (bind(fd(), addr.sockaddr(), addr.addr_len()) == socket_error) {
    return SocketErrorToStatus("could not bind socket to address");
  }
  if (listen(fd(), options.backlog) == socket_error) {
    return SocketErrorToStatus("could not listen on socket");
  }

  return iree_ok_status();
}

iree_status_t Socket::Close() {
  if (fd_ == invalid_socket) {
    return iree_ok_status();
  }
#if IREE_REMOTING_IS_WINSOCK
  int rc = ::closesocket(fd_);
#else
  int rc = ::close(fd_);
#endif
  if (rc == socket_error) {
    return SocketErrorToStatus("could not close socket");
  }

  fd_ = invalid_socket;
  return iree_ok_status();
}

void Socket::CloseOrDie() { IREE_CHECK_OK(Close()); }

}  // namespace remoting
}  // namespace iree
