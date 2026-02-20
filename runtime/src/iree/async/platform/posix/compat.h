// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// POSIX compatibility layer for Linux-specific extensions.
//
// Linux provides several useful extensions for socket programming that aren't
// available on macOS/BSD. This header provides abstractions that use native
// APIs where available and fall back to portable alternatives elsewhere:
//
// - SOCK_NONBLOCK / SOCK_CLOEXEC: Atomically set flags during socket creation.
//   Fallback: Use fcntl() after socket()/accept().
//
// - accept4(): Accept with flags to atomically set nonblock/cloexec.
//   Fallback: Use accept() + fcntl().

#ifndef IREE_ASYNC_PLATFORM_POSIX_COMPAT_H_
#define IREE_ASYNC_PLATFORM_POSIX_COMPAT_H_

// accept4() requires _GNU_SOURCE for the glibc declaration. Callers should
// define _GNU_SOURCE before their first system header; this is defense in depth
// for cases where compat.h is included before sys/socket.h.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif  // _GNU_SOURCE

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Socket creation with flags
//===----------------------------------------------------------------------===//

// Flags for socket creation and accept. On Linux these are passed directly to
// socket()/accept4(). On other platforms they're applied via fcntl() after.
#if defined(IREE_PLATFORM_LINUX)
// Linux: use native flags.
#define IREE_POSIX_SOCK_NONBLOCK SOCK_NONBLOCK
#define IREE_POSIX_SOCK_CLOEXEC SOCK_CLOEXEC
#else
// Non-Linux: flags will be applied via fcntl() in wrapper functions.
#define IREE_POSIX_SOCK_NONBLOCK 0x10000000
#define IREE_POSIX_SOCK_CLOEXEC 0x20000000
#endif  // IREE_PLATFORM_LINUX

// Creates a socket with SOCK_NONBLOCK and SOCK_CLOEXEC flags.
// On Linux this is atomic; on other platforms fcntl() is used after socket().
// Returns the fd on success, or -1 with errno set on failure.
static inline int iree_posix_socket(int domain, int type, int protocol) {
#if defined(IREE_PLATFORM_LINUX)
  return socket(domain, type | SOCK_NONBLOCK | SOCK_CLOEXEC, protocol);
#else
  int fd = socket(domain, type, protocol);
  if (fd < 0) return fd;

  // Set nonblocking. Failure would cause the poll thread to block on I/O.
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    int saved_errno = errno;
    close(fd);
    errno = saved_errno;
    return -1;
  }

  // Set close-on-exec for fd hygiene across exec.
  if (fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
    int saved_errno = errno;
    close(fd);
    errno = saved_errno;
    return -1;
  }

  return fd;
#endif  // IREE_PLATFORM_LINUX
}

//===----------------------------------------------------------------------===//
// Accept with flags
//===----------------------------------------------------------------------===//

// Accepts a connection with SOCK_NONBLOCK and SOCK_CLOEXEC flags on the new fd.
// On Linux this uses accept4(); on other platforms uses accept() + fcntl().
// Returns the fd on success, or -1 with errno set on failure.
static inline int iree_posix_accept(int sockfd, struct sockaddr* addr,
                                    socklen_t* addrlen) {
#if defined(IREE_PLATFORM_LINUX)
  return accept4(sockfd, addr, addrlen, SOCK_NONBLOCK | SOCK_CLOEXEC);
#else
  int fd = accept(sockfd, addr, addrlen);
  if (fd < 0) return fd;

  // Set nonblocking. Failure would cause the poll thread to block on I/O.
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    int saved_errno = errno;
    close(fd);
    errno = saved_errno;
    return -1;
  }

  // Set close-on-exec for fd hygiene across exec.
  if (fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
    int saved_errno = errno;
    close(fd);
    errno = saved_errno;
    return -1;
  }

  return fd;
#endif  // IREE_PLATFORM_LINUX
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_COMPAT_H_
