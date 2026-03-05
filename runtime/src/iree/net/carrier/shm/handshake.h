// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-process SHM handshake: bootstraps carrier pairs over a connected
// channel (Unix domain socket on POSIX, named pipe on Windows).
//
// The handshake is a two-message synchronous exchange. The server creates the
// SHM region, sends an OFFER with its handles, and receives an ACCEPT with
// the client's handles. The client receives the OFFER, maps the SHM region,
// and sends its ACCEPT.
//
// After the handshake, each side has everything needed to create an
// iree_net_shm_carrier_t: a mapping of the shared ring buffers, a proxy
// notification for waking the peer, and armed flag pointers.
//
// Platform-specific handle exchange (the only divergent code):
//   POSIX:   fd passing via SCM_RIGHTS over sendmsg/recvmsg on Unix sockets.
//   Windows: DuplicateHandle over ReadFile/WriteFile on named pipes.
//
// The handshake is synchronous (blocking with timeout). Over a local channel,
// it completes in microseconds. The channel is closed on return.

#ifndef IREE_NET_CARRIER_SHM_HANDSHAKE_H_
#define IREE_NET_CARRIER_SHM_HANDSHAKE_H_

#include "iree/async/primitive.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/internal/shm.h"
#include "iree/net/carrier/shm/carrier.h"
#include "iree/net/carrier/shm/shared_wake.h"
#include "iree/net/carrier/shm/xproc_context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Protocol constants
//===----------------------------------------------------------------------===//

#define IREE_NET_SHM_HANDSHAKE_MAGIC 0x49524853u  // "IRHS" (IREE Handshake SHM)
#define IREE_NET_SHM_HANDSHAKE_VERSION 1u

// Handshake timeout in milliseconds. If the peer doesn't respond within this
// window, the handshake fails with DEADLINE_EXCEEDED.
#define IREE_NET_SHM_HANDSHAKE_TIMEOUT_MS 5000

enum iree_net_shm_handshake_message_type_e {
  // Server → Client: offering the SHM region and server's wake handles.
  IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER = 1u,
  // Client → Server: accepting with client's wake handles.
  IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT = 2u,
};
typedef uint8_t iree_net_shm_handshake_message_type_t;

// Fixed-size message header. Sent as the primary payload on the socket.
// Handles are sent alongside (POSIX: SCM_RIGHTS ancillary data; Windows:
// named object strings appended after the header).
typedef struct iree_net_shm_handshake_header_t {
  uint32_t magic;
  uint32_t version;
  iree_net_shm_handshake_message_type_t type;
  uint8_t reserved[3];
  // OFFER only: total size of the SHM region in bytes.
  uint32_t region_size;
  // OFFER only: SPSC ring data capacity in bytes (power of two).
  uint32_t ring_capacity;
  // Size of the wake epoch SHM region (always one page, but sent for
  // validation).
  uint32_t wake_epoch_size;
  uint8_t padding[8];
} iree_net_shm_handshake_header_t;

//===----------------------------------------------------------------------===//
// Internal platform interface
//===----------------------------------------------------------------------===//
// Implemented in handshake_posix.c and handshake_win32.c.

// Handles exchanged alongside a handshake message. The OFFER includes the
// SHM region handle plus wake handles; the ACCEPT includes only wake handles.
typedef struct iree_net_shm_handshake_handles_t {
  // SHM region handle (OFFER only; zero/invalid for ACCEPT).
  iree_shm_handle_t shm_region;
  // Wake epoch SHM handle.
  iree_shm_handle_t wake_epoch_shm;
  // Signal primitive (eventfd/pipe write end/Event HANDLE).
  iree_async_primitive_t signal_primitive;
} iree_net_shm_handshake_handles_t;

// Sends a handshake message with attached handles over the channel.
// Platform-specific: POSIX uses SCM_RIGHTS over sendmsg; Windows uses
// DuplicateHandle over WriteFile.
iree_status_t iree_net_shm_handshake_send(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_header_t* header,
    const iree_net_shm_handshake_handles_t* handles);

// Receives a handshake message with attached handles from the channel.
// Blocks with timeout until data is available.
iree_status_t iree_net_shm_handshake_recv(
    iree_async_primitive_t channel, iree_net_shm_handshake_header_t* out_header,
    iree_net_shm_handshake_handles_t* out_handles);

// Closes all handles in a handshake_handles_t. Used for error cleanup.
void iree_net_shm_handshake_handles_close(
    iree_net_shm_handshake_handles_t* handles);

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Handshake result: everything needed to create one carrier.
typedef struct iree_net_shm_handshake_result_t {
  // Carrier creation parameters. Ready to pass to
  // iree_net_shm_carrier_create().
  iree_net_shm_carrier_create_params_t carrier_params;
  // Cross-process context (set as carrier's release_context). Owns the SHM
  // mapping, peer notification proxy, and peer signal primitive.
  iree_net_shm_xproc_context_t* context;
  // Region 0 info. Stored here so the create_params.regions pointer (which
  // points to this field) remains valid until the result is consumed.
  iree_net_shm_region_info_t region;
} iree_net_shm_handshake_result_t;

// Server side: create SHM region, send OFFER, receive ACCEPT, assemble
// carrier params.
//
// |channel| is a connected channel primitive (Unix domain socket fd on POSIX,
// named pipe HANDLE on Windows). The handshake is synchronous with a timeout;
// the channel is closed on return (success or failure).
//
// |shared_wake| must have been created with
// iree_net_shm_shared_wake_create_shared().
//
// On success, |out_result| contains carrier params ready for
// iree_net_shm_carrier_create(), with the xproc context set as release_context.
IREE_API_EXPORT iree_status_t iree_net_shm_handshake_server(
    iree_async_primitive_t channel, iree_net_shm_shared_wake_t* shared_wake,
    iree_net_shm_carrier_options_t options, iree_async_proactor_t* proactor,
    iree_allocator_t host_allocator,
    iree_net_shm_handshake_result_t* out_result);

// Client side: receive OFFER, map SHM region, send ACCEPT, assemble carrier
// params.
//
// Same semantics as server: synchronous with timeout, channel closed on return.
IREE_API_EXPORT iree_status_t iree_net_shm_handshake_client(
    iree_async_primitive_t channel, iree_net_shm_shared_wake_t* shared_wake,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_net_shm_handshake_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_HANDSHAKE_H_
