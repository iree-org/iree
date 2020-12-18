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

#ifndef IREE_REMOTING_SUPPORT_PLATFORM_H_
#define IREE_REMOTING_SUPPORT_PLATFORM_H_

#if defined(_WIN64) || defined(_WIN32)
// Winsock.
#define IREE_REMOTING_IS_WINSOCK 1
#include <Winerror.h>
#include <winsock2.h>
#else
// Normal berkeley sockets.
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

// Socket type and constant mapping.
#if IREE_REMOTING_IS_WINSOCK
// Winsock.
using socket_t = SOCKET;
static constexpr SOCKET invalid_socket = INVALID_SOCKET;
static constexpr int socket_error = SOCKET_ERROR;

// struct iovec is WSABUF on windows. Note that the |buf| member is a CHAR*.
// note that this is only compatible with Winsock. Scatter/gather file
// operations on Windows use a completely different (and weirder) API.
using portable_iovec_t = WSABUF;
#define IREE_REMOTING_IOVEC_BASE(iov) (iov).buf
#define IREE_REMOTING_IOVEC_LEN(iov) (iov).len
#define IREE_REMOTING_IOVEC_FROM_PTR(ptr) reinterpret_cast<CHAR *>(ptr)
#else
// Sockets are just file descriptors.
using socket_t = int;
static constexpr int invalid_socket = -1;
static constexpr int socket_error = -1;

// struct iovec is just iovec.
using portable_iovec_t = struct iovec;
#define IREE_REMOTING_IOVEC_BASE(iov) (iov).iov_base
#define IREE_REMOTING_IOVEC_LEN(iov) (iov).iov_len
#define IREE_REMOTING_IOVEC_FROM_PTR(ptr) static_cast<void *>(ptr)
#endif

#endif  // IREE_REMOTING_SUPPORT_PLATFORM_H_
