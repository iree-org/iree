// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// POSIX handshake handle exchange: passes fds via SCM_RIGHTS over sendmsg/
// recvmsg on a Unix domain socket. Each handshake message consists of the
// fixed-size header as the iovec payload and up to 3 fds as ancillary data.

#include "iree/net/carrier/shm/handshake.h"

#if !defined(IREE_PLATFORM_WINDOWS)

#include <errno.h>
#include <poll.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

// Maximum number of fds sent in a single handshake message.
// OFFER sends 3 (shm_region, wake_epoch_shm, signal_primitive).
// ACCEPT sends 2 (wake_epoch_shm, signal_primitive).
#define MAX_HANDSHAKE_FDS 3

// Size of the cmsg buffer for SCM_RIGHTS. Must be large enough for MAX_FDS.
#define CMSG_BUF_SIZE (CMSG_SPACE(MAX_HANDSHAKE_FDS * sizeof(int)))

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Extracts a POSIX fd from an iree_shm_handle_t or iree_async_primitive_t.
// Returns -1 for invalid/NONE handles.
static int iree_shm_handle_to_fd(iree_shm_handle_t handle) {
  if (!iree_shm_handle_is_valid(handle)) return -1;
  return (int)handle.value;
}

static iree_shm_handle_t iree_shm_handle_from_fd(int fd) {
  iree_shm_handle_t handle;
  handle.value = (uint64_t)fd;
  return handle;
}

static int iree_async_primitive_to_fd(iree_async_primitive_t primitive) {
  if (primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) return -1;
  return primitive.value.fd;
}

//===----------------------------------------------------------------------===//
// Send/recv with SCM_RIGHTS
//===----------------------------------------------------------------------===//

iree_status_t iree_net_shm_handshake_send(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_header_t* header,
    const iree_net_shm_handshake_handles_t* handles) {
  int channel_fd = iree_async_primitive_to_fd(channel);
  if (channel_fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake channel is not a valid POSIX fd");
  }

  // Collect fds to send. Order matters — receiver unpacks in same order.
  int fds[MAX_HANDSHAKE_FDS];
  int fd_count = 0;

  int shm_fd = iree_shm_handle_to_fd(handles->shm_region);
  if (shm_fd >= 0) {
    fds[fd_count++] = shm_fd;
  }
  int epoch_fd = iree_shm_handle_to_fd(handles->wake_epoch_shm);
  if (epoch_fd >= 0) {
    fds[fd_count++] = epoch_fd;
  }
  int signal_fd = iree_async_primitive_to_fd(handles->signal_primitive);
  if (signal_fd >= 0) {
    fds[fd_count++] = signal_fd;
  }

  // Build the message.
  struct iovec iov;
  iov.iov_base = (void*)header;
  iov.iov_len = sizeof(*header);

  struct msghdr msg;
  memset(&msg, 0, sizeof(msg));
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  // Attach SCM_RIGHTS ancillary data if we have fds.
  char cmsg_buf[CMSG_BUF_SIZE];
  if (fd_count > 0) {
    memset(cmsg_buf, 0, sizeof(cmsg_buf));
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = CMSG_SPACE(fd_count * sizeof(int));

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(fd_count * sizeof(int));
    memcpy(CMSG_DATA(cmsg), fds, fd_count * sizeof(int));
  }

  ssize_t sent;
  do {
    sent = sendmsg(channel_fd, &msg, MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  if (sent < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "handshake sendmsg failed");
  }
  if ((size_t)sent != sizeof(*header)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "handshake sendmsg short write: %zd/%zu", sent,
                            sizeof(*header));
  }

  return iree_ok_status();
}

iree_status_t iree_net_shm_handshake_recv(
    iree_async_primitive_t channel, iree_net_shm_handshake_header_t* out_header,
    iree_net_shm_handshake_handles_t* out_handles) {
  int channel_fd = iree_async_primitive_to_fd(channel);
  if (channel_fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake channel is not a valid POSIX fd");
  }

  memset(out_header, 0, sizeof(*out_header));
  memset(out_handles, 0, sizeof(*out_handles));
  out_handles->shm_region = IREE_SHM_HANDLE_INVALID;
  out_handles->wake_epoch_shm = IREE_SHM_HANDLE_INVALID;

  // Poll with timeout for the peer's message. Retry on EINTR (signal
  // delivery during the wait).
  struct pollfd pfd;
  pfd.fd = channel_fd;
  pfd.events = POLLIN;
  pfd.revents = 0;
  int poll_result;
  do {
    poll_result = poll(&pfd, 1, IREE_NET_SHM_HANDSHAKE_TIMEOUT_MS);
  } while (poll_result < 0 && errno == EINTR);
  if (poll_result < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "handshake poll failed");
  }
  if (poll_result == 0) {
    return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                            "handshake timed out waiting for peer message");
  }
  // POLLERR/POLLNVAL are fatal. POLLHUP alone (no data) means the peer closed
  // before sending. But POLLIN|POLLHUP is normal — the peer sent data and then
  // closed their end of the channel (e.g., the other handshake side finished
  // first). We proceed to recv in that case.
  if ((pfd.revents & (POLLERR | POLLNVAL)) || (!(pfd.revents & POLLIN))) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "handshake channel error during poll (revents=0x%x)", pfd.revents);
  }

  // Receive the message with ancillary data.
  struct iovec iov;
  iov.iov_base = out_header;
  iov.iov_len = sizeof(*out_header);

  char cmsg_buf[CMSG_BUF_SIZE];
  memset(cmsg_buf, 0, sizeof(cmsg_buf));

  struct msghdr msg;
  memset(&msg, 0, sizeof(msg));
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  ssize_t received;
  do {
    received = recvmsg(channel_fd, &msg, 0);
  } while (received < 0 && errno == EINTR);
  if (received < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "handshake recvmsg failed");
  }
  if ((size_t)received != sizeof(*out_header)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "handshake recvmsg short read: %zd/%zu", received,
                            sizeof(*out_header));
  }

  // Extract fds from SCM_RIGHTS ancillary data.
  int fds[MAX_HANDSHAKE_FDS];
  int fd_count = 0;
  for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL;
       cmsg = CMSG_NXTHDR(&msg, cmsg)) {
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
      int payload_size = (int)(cmsg->cmsg_len - CMSG_LEN(0));
      int received_fds = payload_size / (int)sizeof(int);
      if (received_fds > MAX_HANDSHAKE_FDS) received_fds = MAX_HANDSHAKE_FDS;
      memcpy(fds, CMSG_DATA(cmsg), received_fds * sizeof(int));
      fd_count = received_fds;
      break;
    }
  }

  // Unpack fds based on message type.
  // OFFER: 3 fds (shm_region, wake_epoch_shm, signal_primitive).
  // ACCEPT: 2 fds (wake_epoch_shm, signal_primitive).
  if (out_header->type == IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER) {
    if (fd_count != 3) {
      // Close any fds we did receive before failing.
      for (int i = 0; i < fd_count; ++i) close(fds[i]);
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "OFFER expected 3 fds, got %d", fd_count);
    }
    out_handles->shm_region = iree_shm_handle_from_fd(fds[0]);
    out_handles->wake_epoch_shm = iree_shm_handle_from_fd(fds[1]);
    out_handles->signal_primitive = iree_async_primitive_from_fd(fds[2]);
  } else if (out_header->type == IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT) {
    if (fd_count != 2) {
      for (int i = 0; i < fd_count; ++i) close(fds[i]);
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "ACCEPT expected 2 fds, got %d", fd_count);
    }
    out_handles->wake_epoch_shm = iree_shm_handle_from_fd(fds[0]);
    out_handles->signal_primitive = iree_async_primitive_from_fd(fds[1]);
  } else {
    // Unknown message type — close any received fds.
    for (int i = 0; i < fd_count; ++i) close(fds[i]);
  }

  return iree_ok_status();
}

void iree_net_shm_handshake_handles_close(
    iree_net_shm_handshake_handles_t* handles) {
  if (!handles) return;
  iree_shm_handle_close(&handles->shm_region);
  iree_shm_handle_close(&handles->wake_epoch_shm);
  iree_async_primitive_close(&handles->signal_primitive);
}

#endif  // !IREE_PLATFORM_WINDOWS
