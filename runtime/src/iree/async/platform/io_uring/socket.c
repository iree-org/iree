// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/socket.h"

#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "iree/async/platform/io_uring/proactor.h"

//===----------------------------------------------------------------------===//
// Socket type mapping
//===----------------------------------------------------------------------===//

// Returns the socket domain (AF_*) for the given socket type.
static int iree_async_socket_type_to_domain(iree_async_socket_type_t type) {
  switch (type) {
    case IREE_ASYNC_SOCKET_TYPE_TCP:
    case IREE_ASYNC_SOCKET_TYPE_UDP:
      return AF_INET;
    case IREE_ASYNC_SOCKET_TYPE_TCP6:
    case IREE_ASYNC_SOCKET_TYPE_UDP6:
      return AF_INET6;
    case IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM:
    case IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM:
      return AF_UNIX;
    default:
      return -1;
  }
}

// Returns the socket type (SOCK_*) for the given socket type.
static int iree_async_socket_type_to_socktype(iree_async_socket_type_t type) {
  switch (type) {
    case IREE_ASYNC_SOCKET_TYPE_TCP:
    case IREE_ASYNC_SOCKET_TYPE_TCP6:
    case IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM:
      return SOCK_STREAM;
    case IREE_ASYNC_SOCKET_TYPE_UDP:
    case IREE_ASYNC_SOCKET_TYPE_UDP6:
    case IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM:
      return SOCK_DGRAM;
    default:
      return -1;
  }
}

// Returns the protocol for the given socket type.
static int iree_async_socket_type_to_protocol(iree_async_socket_type_t type) {
  switch (type) {
    case IREE_ASYNC_SOCKET_TYPE_TCP:
    case IREE_ASYNC_SOCKET_TYPE_TCP6:
      return IPPROTO_TCP;
    case IREE_ASYNC_SOCKET_TYPE_UDP:
    case IREE_ASYNC_SOCKET_TYPE_UDP6:
      return IPPROTO_UDP;
    case IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM:
    case IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM:
      return 0;  // Unix sockets don't use protocol.
    default:
      return -1;
  }
}

//===----------------------------------------------------------------------===//
// Socket options
//===----------------------------------------------------------------------===//

// Applies socket options via setsockopt.
// Returns an error status if any option fails to apply.
static iree_status_t iree_async_socket_apply_options(
    int fd, iree_async_socket_type_t type,
    iree_async_socket_options_t options) {
  int optval = 1;

  // SO_REUSEADDR: Allow binding to recently-used addresses.
  if (options & IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR) {
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "setsockopt SO_REUSEADDR failed");
    }
  }

  // SO_REUSEPORT: Allow multiple sockets on same port.
  if (options & IREE_ASYNC_SOCKET_OPTION_REUSE_PORT) {
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval)) < 0) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "setsockopt SO_REUSEPORT failed");
    }
  }

  // TCP_NODELAY: Disable Nagle's algorithm (TCP only).
  if ((options & IREE_ASYNC_SOCKET_OPTION_NO_DELAY) &&
      (type == IREE_ASYNC_SOCKET_TYPE_TCP ||
       type == IREE_ASYNC_SOCKET_TYPE_TCP6)) {
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)) < 0) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "setsockopt TCP_NODELAY failed");
    }
  }

  // SO_KEEPALIVE: Enable TCP keepalive probes.
  if (options & IREE_ASYNC_SOCKET_OPTION_KEEP_ALIVE) {
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &optval, sizeof(optval)) < 0) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "setsockopt SO_KEEPALIVE failed");
    }
  }

  // ZERO_COPY is a hint: enable SO_ZEROCOPY if available, otherwise sends use
  // the regular copy path transparently. The flag is always recorded on the
  // socket; the send path checks proactor capability to decide the actual path.
  if (options & IREE_ASYNC_SOCKET_OPTION_ZERO_COPY) {
#if defined(SO_ZEROCOPY)
    // Best-effort: if the kernel rejects it (e.g., unsupported socket type),
    // sends still work via the copy path.
    (void)setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &optval, sizeof(optval));
#endif  // SO_ZEROCOPY
  }

  // SO_LINGER with timeout=0: Force immediate RST on close.
  if (options & IREE_ASYNC_SOCKET_OPTION_LINGER_ZERO) {
    struct linger linger_opt = {
        .l_onoff = 1,
        .l_linger = 0,
    };
    if (setsockopt(fd, SOL_SOCKET, SO_LINGER, &linger_opt, sizeof(linger_opt)) <
        0) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "setsockopt SO_LINGER failed");
    }
  }

  return iree_ok_status();
}

// Sets a file descriptor to non-blocking mode.
static iree_status_t iree_async_socket_set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "fcntl F_GETFL failed");
  }
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "fcntl F_SETFL O_NONBLOCK failed");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Socket struct initialization
//===----------------------------------------------------------------------===//

// Initializes a pre-allocated socket struct.
static void iree_async_socket_initialize(
    iree_async_socket_t* socket, iree_async_proactor_io_uring_t* proactor,
    int fd, iree_async_socket_type_t type, iree_async_socket_flags_t flags) {
  iree_atomic_ref_count_init(&socket->ref_count);
  socket->proactor = &proactor->base;
  socket->primitive = iree_async_primitive_from_fd(fd);
  socket->fixed_file_index = -1;  // Not using fixed files yet.
  socket->type = type;
  socket->state = IREE_ASYNC_SOCKET_STATE_CREATED;
  socket->flags = flags;
  iree_atomic_store(&socket->failure_status, (intptr_t)iree_ok_status(),
                    iree_memory_order_release);
  IREE_TRACE({
    snprintf(socket->debug_label, sizeof(socket->debug_label), "socket:%d", fd);
  });
}

// Derives runtime flags from successfully applied socket options.
static iree_async_socket_flags_t iree_async_socket_flags_from_options(
    iree_async_socket_options_t options) {
  iree_async_socket_flags_t flags = IREE_ASYNC_SOCKET_FLAG_NONE;
  if (iree_any_bit_set(options, IREE_ASYNC_SOCKET_OPTION_ZERO_COPY)) {
    flags |= IREE_ASYNC_SOCKET_FLAG_ZERO_COPY;
  }
  return flags;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_async_io_uring_socket_create(
    iree_async_proactor_io_uring_t* proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_socket);
  *out_socket = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Map socket type to domain/type/protocol.
  int domain = iree_async_socket_type_to_domain(type);
  int socktype = iree_async_socket_type_to_socktype(type);
  int protocol = iree_async_socket_type_to_protocol(type);
  if (domain < 0 || socktype < 0 || protocol < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid socket type %d", type);
  }

  // Create the platform socket.
  // Use SOCK_NONBLOCK and SOCK_CLOEXEC for efficiency (avoids separate fcntl).
  int fd = socket(domain, socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, protocol);
  if (fd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "socket() failed");
  }

  // Apply requested options.
  iree_status_t status = iree_async_socket_apply_options(fd, type, options);

  // Allocate the socket struct.
  iree_async_socket_t* socket = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(proactor->base.allocator, sizeof(*socket),
                                   (void**)&socket);
  }

  // Initialize or cleanup.
  if (iree_status_is_ok(status)) {
    // Derive runtime flags from the successfully applied options.
    iree_async_socket_flags_t flags =
        iree_async_socket_flags_from_options(options);
    iree_async_socket_initialize(socket, proactor, fd, type, flags);
    *out_socket = socket;
  } else {
    close(fd);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_async_io_uring_socket_import(
    iree_async_proactor_io_uring_t* proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_socket);
  *out_socket = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate primitive type.
  if (primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected FD primitive for socket import");
  }

  int fd = primitive.value.fd;
  if (fd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid file descriptor %d", fd);
  }

  // Set non-blocking if not already (cheap to ensure).
  iree_status_t status = iree_async_socket_set_nonblocking(fd);

  // Allocate the socket struct.
  iree_async_socket_t* socket = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(proactor->base.allocator, sizeof(*socket),
                                   (void**)&socket);
  }

  // Initialize or leave fd alone on failure (caller still owns it).
  if (iree_status_is_ok(status)) {
    iree_async_socket_initialize(socket, proactor, fd, type, flags);
    *out_socket = socket;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_io_uring_socket_destroy(
    iree_async_proactor_io_uring_t* proactor, iree_async_socket_t* socket) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(socket);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Close the platform socket.
  int fd = socket->primitive.value.fd;
  if (fd >= 0) {
    close(fd);
  }

  // Free any stored failure status.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &socket->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    iree_status_ignore(failure);
  }

  // Free the socket struct.
  iree_allocator_free(proactor->base.allocator, socket);

  IREE_TRACE_ZONE_END(z0);
}
