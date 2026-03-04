// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared memory carrier: high-performance IPC via SPSC rings in shared memory.
//
// The SHM carrier provides zero-kernel-copy data transfer between two endpoints
// (typically in different processes) using SPSC ring buffers in a shared memory
// region. A single shared notification per carrier drives the proactor poll
// loop to drain received data and fire send completion callbacks.
//
// Design:
//   - Two carriers share a single SHM region (connected via create_pair or a
//     cross-process factory). The region contains two SPSC rings (one per
//     direction) plus shared epoch counters and armed flags for adaptive
//     signaling.
//   - send() writes data inline into the TX ring (memcpy) under a slim mutex
//     to serialize producers. Completion fires when the consumer advances past
//     the written entry.
//   - A single NOTIFICATION_WAIT per carrier wakes the proactor when the peer
//     signals. The drain callback checks both RX data and TX completions —
//     each check is one acquire-load, so spurious wakes cost ~50ns.
//
// Signaling:
//   Each carrier has one shared notification (epoch in SHM, event pair for
//   proactor integration). The armed flag in SHM controls adaptive wake:
//   a Dekker-fence protocol ensures no missed wakes between the producer's
//   write and the consumer's arm-then-sleep sequence.
//
//   This is the same code path for both create_pair (in-process testing) and
//   cross-process operation via a factory. The only difference is how handles
//   and SHM are exchanged — the carrier-level code is identical.
//
// Ring entry framing (layered on SPSC queue's length-prefixed entries):
//   length == 0: shutdown marker (peer initiated graceful shutdown)
//   length > 0:  [uint8_t type][type-specific payload]
//     type 0x00 (INLINE):    data bytes (memcpy'd from scatter-gather spans)
//     type 0x01 (REFERENCE): region descriptor for zero-copy direct_write
//
// Capabilities:
//   - RELIABLE: no drops (shared memory, single producer/consumer per ring).
//   - ORDERED: FIFO (SPSC ring preserves insertion order).
//   - ZERO_COPY_TX: advertised for large-data CTS tests; inline sends copy but
//     reference sends (register_buffer) avoid the copy entirely.
//
// Thread safety:
//   - send() is thread-safe (serialized by slim mutex).
//   - All RX delivery and TX completion callbacks fire from the proactor
//   thread.

#ifndef IREE_NET_CARRIER_SHM_CARRIER_H_
#define IREE_NET_CARRIER_SHM_CARRIER_H_

#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/internal/spsc_queue.h"
#include "iree/net/carrier.h"

typedef struct iree_net_shm_shared_wake_t iree_net_shm_shared_wake_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Magic number identifying an SHM carrier region header ("SHMC" in LE).
#define IREE_NET_SHM_CARRIER_MAGIC ((uint32_t)0x434D4853)

// Current SHM carrier region ABI version.
#define IREE_NET_SHM_CARRIER_VERSION ((uint32_t)1)

// Default ring capacity in bytes (256KB). Must be a power of two.
#define IREE_NET_SHM_CARRIER_DEFAULT_RING_CAPACITY ((uint32_t)(256 * 1024))

// Maximum number of in-flight send completions tracked per carrier.
#define IREE_NET_SHM_CARRIER_MAX_COMPLETIONS 256

// Number of consecutive empty progress callback iterations before the carrier
// transitions from poll mode back to notification-based sleep mode.
// At ~50ns per poll iteration, 256 iterations ≈ 13µs of spinning.
#define IREE_NET_SHM_IDLE_SPIN_THRESHOLD 256

// AIMD adaptive window bounds for progress callback batching.
// Controls how many RX entries are drained per progress callback invocation.
// Additive increase (+1) after two consecutive full windows; multiplicative
// decrease (/2) on underconsumed polls. Matches UCX mm transport behavior.
#define IREE_NET_SHM_POLL_WINDOW_MIN 1
#define IREE_NET_SHM_POLL_WINDOW_MAX 256
#define IREE_NET_SHM_POLL_WINDOW_DEFAULT IREE_NET_SHM_POLL_WINDOW_MAX

// Inline data: payload bytes follow the type tag directly.
#define IREE_NET_SHM_ENTRY_TYPE_INLINE ((uint8_t)0x00)

// Reference descriptor: a region ID + offset + length identifying data in a
// registered shared memory region. The consumer resolves the reference to a
// pointer without any data copy.
#define IREE_NET_SHM_ENTRY_TYPE_REFERENCE ((uint8_t)0x01)

// Reference descriptor for zero-copy sends via registered shared memory
// regions. Written into the ring as the payload of a REFERENCE entry.
// Uses 64-bit offset and length to match iree_async_span_t's iree_host_size_t
// fields and avoid artificial 4GB caps on registered region ranges.
typedef struct iree_net_shm_reference_descriptor_t {
  uint32_t region_id;  // Assigned during register_buffer.
  uint32_t reserved;
  uint64_t offset;  // Byte offset within the registered region.
  uint64_t length;  // Byte length of the referenced data.
} iree_net_shm_reference_descriptor_t;

// Byte offsets of shared state fields within the SHM region.
// Each field is cache-line aligned (64 bytes) to prevent false sharing.
//
// Layout:
//   0x0000  Region header (64B): magic, version, ring_capacity, reserved
//   0x0040  epoch_a (64B): notification epoch for Ring A consumer (client)
//   0x0080  epoch_b (64B): notification epoch for Ring B consumer (server)
//   0x00C0  consumer_a_armed (64B): client armed flag (server checks on write)
//   0x0100  consumer_b_armed (64B): server armed flag (client checks on write)
//   0x0140  Reserved padding (192B) to 0x0200
//   0x0200  Ring A: SPSC ring (256 + ring_capacity bytes)
//   Ring B follows Ring A: SPSC ring (256 + ring_capacity bytes)
//
// Epochs are in SHM because shared notifications require a
// cross-process-visible epoch address. Both processes atomically increment and
// observe the epoch:
//   epoch_a: incremented when the client should wake (server wrote to Ring A or
//     server consumed from Ring B)
//   epoch_b: incremented when the server should wake (client wrote to Ring B or
//     client consumed from Ring A)
//
// Armed flags control adaptive signaling. When a consumer has no work, it sets
// its armed flag before sleeping. The producer checks the peer's armed flag
// after writing; if set, it signals the peer's notification to wake it.
//
// Ring assignment:
//   Client TX = Ring B, Client RX = Ring A
//   Server TX = Ring A, Server RX = Ring B
#define IREE_NET_SHM_OFFSET_HEADER ((iree_host_size_t)0x0000)
#define IREE_NET_SHM_OFFSET_EPOCH_A ((iree_host_size_t)0x0040)
#define IREE_NET_SHM_OFFSET_EPOCH_B ((iree_host_size_t)0x0080)
#define IREE_NET_SHM_OFFSET_CONSUMER_A_ARMED ((iree_host_size_t)0x00C0)
#define IREE_NET_SHM_OFFSET_CONSUMER_B_ARMED ((iree_host_size_t)0x0100)
#define IREE_NET_SHM_OFFSET_RINGS ((iree_host_size_t)0x0200)

// Immutable header at offset 0 of the SHM region. Written once by the creator.
typedef struct iree_net_shm_region_header_t {
  uint32_t magic;
  uint32_t version;
  // SPSC ring data capacity in bytes (power of two).
  uint32_t ring_capacity;
  uint8_t reserved[52];
} iree_net_shm_region_header_t;

// Describes a memory region known to the carrier for buffer registration
// and direct read/write. Both sides of a carrier pair must populate matching
// region IDs — the region_id in a remote handle is the array index.
typedef struct iree_net_shm_region_info_t {
  void* base_ptr;
  iree_host_size_t size;
} iree_net_shm_region_info_t;

// Mode bits for SHM carrier behavior. Bitfield for future extension.
// New modes are added as new bit values — no API churn.
typedef uint32_t iree_net_shm_carrier_mode_t;
enum iree_net_shm_carrier_mode_bits_e {
  IREE_NET_SHM_CARRIER_MODE_DEFAULT = 0u,
};

typedef struct iree_net_shm_carrier_options_t {
  // Ring buffer data capacity in bytes. Must be a power of two.
  // 0 = use IREE_NET_SHM_CARRIER_DEFAULT_RING_CAPACITY.
  uint32_t ring_capacity;
  // Mode bitfield controlling carrier behavior.
  iree_net_shm_carrier_mode_t mode;
} iree_net_shm_carrier_options_t;

// Returns default options (zero-initialized, which selects default capacity
// and default mode).
static inline iree_net_shm_carrier_options_t
iree_net_shm_carrier_options_default(void) {
  iree_net_shm_carrier_options_t options;
  memset(&options, 0, sizeof(options));
  return options;
}

//===----------------------------------------------------------------------===//
// iree_net_shm_carrier_create
//===----------------------------------------------------------------------===//

// Parameters for creating a single SHM carrier from pre-assembled
// dependencies. All pointer fields must remain valid for the lifetime of the
// carrier (the carrier retains the shared_wake, peer_wake_notification,
// and calls release_context_fn during destroy).
typedef struct iree_net_shm_carrier_create_params_t {
  bool is_client;
  iree_spsc_queue_t tx_queue;
  iree_spsc_queue_t rx_queue;
  // Our proactor's shared wake (owns notification + sleeping carrier list).
  // Retained by the carrier. Provides the proactor reference.
  iree_net_shm_shared_wake_t* shared_wake;
  // Peer proactor's shared notification (signaled to wake the peer's scan).
  // Retained by the carrier.
  iree_async_notification_t* peer_wake_notification;
  // Adaptive signaling flag pointers into the SHM region. Each must be on its
  // own cache line (64-byte aligned) in the shared region.
  iree_atomic_int32_t* our_armed;
  iree_atomic_int32_t* peer_armed;
  // Opaque ownership context released when the carrier is destroyed. Called
  // after the carrier allocation is freed. For create_pair: a ref-counted pair
  // context holding SHM mappings. For factory-created carriers: a per-carrier
  // resource bundle. May be NULL if no cleanup is needed.
  void* release_context;
  void (*release_context_fn)(void* context);

  // Minimum number of RX entries drained in a single notification callback
  // before the carrier transitions from sleep mode to inline poll mode.
  // 0 or 1 = transition on first data arrival (lowest latency).
  // Higher values avoid spinning for low-rate traffic — e.g., 64 means "only
  // spin-poll when receiving bursts of 64+ entries per wake."
  // Default: 1 (immediate transition to poll mode on first data).
  int32_t poll_mode_threshold;

  // Known SHM regions for buffer registration and direct read/write.
  // Region 0 is typically the carrier's main SHM region. Both sides of a
  // carrier pair must populate matching region IDs. When region_count > 0,
  // the carrier advertises REGISTERED_REGIONS | DIRECT_WRITE | DIRECT_READ
  // capabilities.
  const iree_net_shm_region_info_t* regions;
  iree_host_size_t region_count;
} iree_net_shm_carrier_create_params_t;

// Creates a single SHM carrier from pre-assembled dependencies.
//
// The carrier retains |params->shared_wake| and
// |params->peer_wake_notification|. It takes ownership of one reference to
// |params->release_context| (will call release_context_fn during destroy).
//
// Capabilities: RELIABLE | ORDERED | ZERO_COPY_TX (always), plus
// REGISTERED_REGIONS | DIRECT_WRITE | DIRECT_READ when regions are provided.
//
// |callback| receives send completion notifications (bytes confirmed consumed
// by the peer). Pass a callback with fn=NULL to skip completion callbacks.
IREE_API_EXPORT iree_status_t iree_net_shm_carrier_create(
    const iree_net_shm_carrier_create_params_t* params,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator,
    iree_net_carrier_t** out_carrier);

// Returns the base pointer and size of a registered region by index.
// This is an SHM-specific API (not on the base carrier vtable) used by tests
// and buffer pool creation to discover the SHM layout.
//
// Returns IREE_STATUS_OUT_OF_RANGE if |region_id| >= the carrier's region
// count.
IREE_API_EXPORT iree_status_t iree_net_shm_carrier_query_region(
    iree_net_carrier_t* base_carrier, uint32_t region_id,
    iree_net_shm_region_info_t* out_region_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_CARRIER_H_
