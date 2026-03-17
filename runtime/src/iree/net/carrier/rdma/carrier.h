// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA carrier: kernel-bypass transport with one-sided operations.
//
// The RDMA carrier provides high-performance networking using InfiniBand verbs:
//
//   - Kernel bypass: No syscalls on the data path.
//   - Zero-copy: Data moves directly between registered memory regions.
//   - One-sided operations: RDMA WRITE/READ without remote CPU involvement.
//   - Low latency: Single-digit microsecond round trips.
//
// ## Capabilities
//
// RDMA carrier reports these capabilities:
//   - RELIABLE: RC (Reliable Connection) QP mode guarantees delivery.
//   - ZERO_COPY_TX: All sends are zero-copy from registered memory.
//   - ZERO_COPY_RX: All receives land in registered memory.
//   - REGISTERED_REGIONS: Memory registration required for DMA.
//   - DIRECT_WRITE: RDMA WRITE to remote memory.
//   - DIRECT_READ: RDMA READ from remote memory.
//
// RDMA carrier does NOT support:
//   - ORDERED: Completion ordering is not guaranteed for one-sided ops.
//   - DATAGRAM: UD mode is not implemented (use RC for reliability).
//
// ## Connection setup
//
// RDMA connection requires out-of-band exchange of connection parameters:
//   1. Each side creates a QP and gets its local address (GID, QP number).
//   2. Exchange addresses via TCP, shared memory, or other mechanism.
//   3. Call iree_net_rdma_carrier_connect() with remote parameters.
//
// The carrier handles QP state transitions (INIT -> RTR -> RTS).
//
// ## Memory registration
//
// Buffers used for RDMA operations must be registered with the device:
//   - Use iree_async_region_t with RDMA device capabilities.
//   - For direct_write/read, publish the region to get a remote handle.
//   - Send the handle to the peer via control channel.
//
// ## Work request flow
//
// The carrier submits work requests (WRs) to the QP:
//   - send(): Submits IBV_WR_SEND.
//   - recv(): Posts receive buffers (IBV_WR_RECV).
//   - direct_write(): Submits IBV_WR_RDMA_WRITE.
//   - direct_read(): Submits IBV_WR_RDMA_READ.
//
// Completions are polled from the CQ and delivered via callback.
//
// ## Backpressure
//
// query_send_budget() returns available send queue depth. RDMA has strict
// limits on outstanding WRs; exceeding them causes RNR (receiver not ready)
// errors. The record layer respects this budget.

#ifndef IREE_NET_CARRIER_RDMA_CARRIER_H_
#define IREE_NET_CARRIER_RDMA_CARRIER_H_

#include "iree/async/api.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// RDMA connection parameters
//===----------------------------------------------------------------------===//

// Local RDMA endpoint address. Exchanged out-of-band for connection setup.
typedef struct iree_net_rdma_address_t {
  // Global identifier (GID) for routing.
  uint8_t gid[16];

  // Queue pair number.
  uint32_t qp_number;

  // Partition key (for InfiniBand subnets).
  uint16_t pkey;

  // Local identifier (LID) for InfiniBand (0 for RoCE).
  uint16_t lid;

  // Packet sequence number for reliability.
  uint32_t psn;
} iree_net_rdma_address_t;

//===----------------------------------------------------------------------===//
// RDMA carrier options
//===----------------------------------------------------------------------===//

// Options for RDMA carrier creation.
typedef struct iree_net_rdma_carrier_options_t {
  // Proactor for completion polling. The carrier retains a reference.
  // The proactor must support RDMA completion channels.
  iree_async_proactor_t* proactor;

  // RDMA device name (e.g., "mlx5_0"). NULL uses the first available device.
  const char* device_name;

  // Port number on the device (1-based). 0 uses the first active port.
  uint8_t port_number;

  // Completion callback for carrier operations.
  iree_net_carrier_callback_t callback;

  // Send queue depth. 0 uses the default (256).
  uint32_t send_queue_depth;

  // Receive queue depth. 0 uses the default (256).
  uint32_t recv_queue_depth;

  // Maximum send scatter-gather entries. 0 uses the default (16).
  uint32_t max_send_sge;

  // Maximum receive scatter-gather entries. 0 uses the default (16).
  uint32_t max_recv_sge;

  // Maximum inline data size for small sends. 0 uses the default (64).
  // Small sends below this threshold are inlined in the WQE.
  uint32_t max_inline_data;

  // GID index for RoCE. 0 uses the default GID.
  uint8_t gid_index;
} iree_net_rdma_carrier_options_t;

// Returns default options with reasonable values.
static inline iree_net_rdma_carrier_options_t
iree_net_rdma_carrier_options_default(void) {
  iree_net_rdma_carrier_options_t options;
  memset(&options, 0, sizeof(options));
  options.device_name = NULL;
  options.port_number = 0;
  options.send_queue_depth = 256;
  options.recv_queue_depth = 256;
  options.max_send_sge = 16;
  options.max_recv_sge = 16;
  options.max_inline_data = 64;
  options.gid_index = 0;
  return options;
}

//===----------------------------------------------------------------------===//
// RDMA carrier
//===----------------------------------------------------------------------===//

// Creates an RDMA carrier.
//
// |options| configures the carrier. The proactor is required.
//
// After creation, the carrier is in INIT state. Call
// iree_net_rdma_carrier_local_address() to get the local address for
// out-of-band exchange, then iree_net_rdma_carrier_connect() to complete
// the connection.
//
// On success, |*out_carrier| receives the new carrier with ref count 1.
IREE_API_EXPORT iree_status_t iree_net_rdma_carrier_create(
    iree_net_rdma_carrier_options_t options, iree_allocator_t host_allocator,
    iree_net_carrier_t** out_carrier);

// Returns the local RDMA address for out-of-band exchange.
// Call after creation to get the address to send to the peer.
IREE_API_EXPORT iree_status_t iree_net_rdma_carrier_local_address(
    iree_net_rdma_carrier_t* carrier, iree_net_rdma_address_t* out_address);

// Connects to a remote RDMA endpoint.
//
// |remote_address| is the peer's address obtained via out-of-band exchange.
//
// This transitions the QP through INIT -> RTR -> RTS and enables
// send/recv/direct operations.
//
// Must be called exactly once after creation.
IREE_API_EXPORT iree_status_t
iree_net_rdma_carrier_connect(iree_net_rdma_carrier_t* carrier,
                              const iree_net_rdma_address_t* remote_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_CARRIER_H_
