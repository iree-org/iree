// Copyright 2019 Google LLC
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

// Utilities for working with TCP sockets.
// These are (mostly) portable to systems implementing BSD sockets.

#ifndef IREE_RT_DEBUG_DEBUG_TCP_UTIL_H_
#define IREE_RT_DEBUG_DEBUG_TCP_UTIL_H_

#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cstddef>

#include "base/status.h"
#include "flatbuffers/base.h"
#include "flatbuffers/flatbuffers.h"
#include "schemas/debug_service_generated.h"

namespace iree {
namespace rt {
namespace debug {
namespace tcp {

// Toggles address reuse on a socket. Call prior to binding.
// This is useful if a socket is sitting in close_wait from a previous process
// while a new one is trying to bind to it.
inline Status ToggleSocketAddressReuse(int fd, bool is_enabled) {
  int toggle = is_enabled ? 1 : 0;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &toggle, sizeof(toggle));
  return OkStatus();
}

// Toggles the linger option on a socket.
// Enabling linger will ensure all data on the socket is sent (if it can be
// sent within N sec) prior to closing. Disabling linger will cause the socket
// to close gracefully.
inline Status ToggleSocketLinger(int fd, bool is_enabled) {
  struct linger linger;
  linger.l_onoff = is_enabled ? 1 : 0;
  linger.l_linger = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_LINGER, &linger, sizeof(linger));
  return OkStatus();
}

// Toggles Nagel's algorithm on a socket.
// Enabled by default, sockets have ~250ms delay for small packets. Disabling
// the algorithm will make socket flushes actually send data.
inline Status ToggleSocketNagelsAlgorithm(int fd, bool is_enabled) {
  int toggle = is_enabled ? 1 : 0;
  ::setsockopt(fd, SOL_TCP, TCP_NODELAY, &toggle, sizeof(toggle));
  return OkStatus();
}

// Toggles TCP keepalive on a socket.
// Assumes that the remote side is on the local machine/network and that we can
// spam it with packets.
//
// NOTE: we may want to adjust this when real debuggers are attached (to prevent
// dropping our own connections). Need to figure out how to reliably detect
// debug suspends vs. actual death.
inline Status ToggleSocketLocalKeepalive(int fd, bool is_enabled) {
  // Toggle keepalive.
  int keepalive_enable = is_enabled ? 1 : 0;
  ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive_enable,
               sizeof(keepalive_enable));
  // Begin sending keepalive probes after N sec.
  int keepalive_idle_delay = 3;
  ::setsockopt(fd, SOL_TCP, TCP_KEEPIDLE, &keepalive_idle_delay,
               sizeof(keepalive_idle_delay));
  // Try one probe and bail (faster detection).
  int keepalive_retry_count = 1;
  ::setsockopt(fd, SOL_TCP, TCP_KEEPINTVL, &keepalive_retry_count,
               sizeof(keepalive_retry_count));
  // Send keepalives every N sec.
  int keepalive_interval = 1;
  ::setsockopt(fd, SOL_TCP, TCP_KEEPINTVL, &keepalive_interval,
               sizeof(keepalive_interval));
  return OkStatus();
}

// Toggles the blocking state of a socket.
// If a socket has been set to non-blocking methods like read and write will
// return EWOULDBLOCK if they would have blocked on the specific operation.
inline Status ToggleSocketBlocking(int fd, bool is_blocking) {
  if (is_blocking) {
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) & ~O_NONBLOCK);
  } else {
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) | O_NONBLOCK);
  }
  return OkStatus();
}

// RAII wrapper for messages containing flatbuffer roots of type T.
template <typename T>
struct MessageBuffer {
 public:
  explicit MessageBuffer(std::vector<uint8_t> buffer)
      : buffer_(std::move(buffer)) {}
  MessageBuffer(const MessageBuffer&) = delete;
  MessageBuffer& operator=(const MessageBuffer&) = delete;
  MessageBuffer(MessageBuffer&&) = default;
  MessageBuffer& operator=(MessageBuffer&&) = default;

  const T& GetRoot() const {
    return *::flatbuffers::GetRoot<T>(buffer_.data());
  }

 private:
  std::vector<uint8_t> buffer_;
};

// Reads a size prefix value from the given fd.
// If |poll_only| is true then the size prefix is not consumed from the stream
// and the call will return 0 if there is no size prefix available.
// Returns CancelledError if a (probably) graceful close is detected.
inline StatusOr<size_t> ReadSizePrefix(int fd, bool poll_only) {
  ::flatbuffers::uoffset_t size_prefix = 0;
  int read_bytes = ::recv(fd, &size_prefix, sizeof(size_prefix),
                          poll_only ? (MSG_PEEK | MSG_DONTWAIT) : 0);
  if (read_bytes == 0) {
    // Remote side disconnected.
    return CancelledErrorBuilder(IREE_LOC) << "Graceful remote close";
  } else if (read_bytes < 0) {
    if (errno == ECONNRESET) {
      return CancelledErrorBuilder(IREE_LOC) << "Ungraceful remote close";
    }
    return DataLossErrorBuilder(IREE_LOC)
           << "Failed to read size prefix from socket: (" << errno << ") "
           << ::strerror(errno);
  } else if (read_bytes != sizeof(size_prefix)) {
    if (poll_only) {
      // No data available.
      return 0;
    } else {
      return DataLossErrorBuilder(IREE_LOC)
             << "Failed to read full size prefix (got " << read_bytes << "b of "
             << sizeof(size_prefix) << "b expected)";
    }
  }
  return size_prefix;
}

// Returns true if ReadBuffer will (likely) not block when called.
// Returns CancelledError if a (probably) graceful close is detected.
inline StatusOr<bool> CanReadBuffer(int fd) {
  ASSIGN_OR_RETURN(size_t size_prefix, ReadSizePrefix(fd, /*poll_only=*/true));
  return size_prefix != 0;
}

// Reads a size-prefixed message from the given fd.
// This will block until the entire message contents are available.
// Returns a buffer reference that will deallocate the buffer automatically or
// CancelledError if a (probably) graceful close is detected.
template <typename T>
StatusOr<MessageBuffer<T>> ReadBuffer(int fd) {
  // Read the size prefix (written as a uoffset_t by the Write* methods).
  ASSIGN_OR_RETURN(size_t size_prefix, ReadSizePrefix(fd, /*poll_only=*/false));

  // Allocate the buffer for the entire message.
  // We'll use the BufferRef to free() it when it's no longer required.
  std::vector<uint8_t> buffer(size_prefix);

  // Read the entire message contents.
  int full_read_bytes = ::recv(fd, buffer.data(), buffer.size(), 0);
  if (full_read_bytes < 0) {
    return DataLossErrorBuilder(IREE_LOC)
           << "Failed to read full message contents from socket: (" << errno
           << ") " << ::strerror(errno);
  } else if (full_read_bytes != buffer.size()) {
    return DataLossErrorBuilder(IREE_LOC)
           << "Failed to read full message contents (got " << full_read_bytes
           << "b of " << buffer.size() << "b expected)";
  }

  // Verify the contents. Not strictly required (as we won't ever ship this to
  // prod), but useful in ensuring our socket code isn't corrupting things.
  ::flatbuffers::Verifier verifier(buffer.data(), buffer.size());
  if (!verifier.VerifyBuffer<T>()) {
    return DataLossErrorBuilder(IREE_LOC)
           << "Verification of input buffer of type " << typeid(T).name()
           << " (" << buffer.size() << "b) failed";
  }

  // Wrap the buffer to get some RAII goodness.
  return MessageBuffer<T>(std::move(buffer));
}

// Writes a buffer to the given fd.
inline Status WriteBuffer(int fd, ::flatbuffers::DetachedBuffer buffer) {
  if (::send(fd, buffer.data(), buffer.size(), 0) < 0) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Write failed: (" << errno << ") " << ::strerror(errno);
  }
  return OkStatus();
}

}  // namespace tcp
}  // namespace debug
}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_DEBUG_DEBUG_TCP_UTIL_H_
