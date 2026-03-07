// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/carrier_pair.h"

#include "iree/base/internal/shm.h"
#include "iree/base/internal/spsc_queue.h"
#include "iree/net/carrier/shm/carrier.h"
#include "iree/net/carrier/shm/shared_wake.h"

//===----------------------------------------------------------------------===//
// Pair context
//===----------------------------------------------------------------------===//

// Shared resources for an in-process carrier pair. Both carriers reference
// this via their release_context, keeping the SHM region alive until both
// carriers are destroyed.
typedef struct iree_net_shm_pair_context_t {
  iree_atomic_ref_count_t ref_count;
  iree_shm_mapping_t creator_mapping;
  iree_shm_mapping_t opener_mapping;
  iree_allocator_t allocator;
} iree_net_shm_pair_context_t;

static void iree_net_shm_pair_context_release(void* context) {
  iree_net_shm_pair_context_t* pair_context =
      (iree_net_shm_pair_context_t*)context;
  if (iree_atomic_ref_count_dec(&pair_context->ref_count) == 1) {
    iree_shm_close(&pair_context->creator_mapping);
    iree_shm_close(&pair_context->opener_mapping);
    iree_allocator_t allocator = pair_context->allocator;
    iree_allocator_free(allocator, pair_context);
  }
}

static void iree_net_shm_pair_context_retain(
    iree_net_shm_pair_context_t* context) {
  iree_atomic_ref_count_inc(&context->ref_count);
}

//===----------------------------------------------------------------------===//
// Pair resource creation helpers
//===----------------------------------------------------------------------===//

// SPSC ring handles for both mappings of the SHM region. Value types (no
// ownership semantics) — the underlying ring storage is in the SHM region.
typedef struct iree_net_shm_pair_rings_t {
  iree_spsc_queue_t ring_a;         // Server TX (creator mapping).
  iree_spsc_queue_t ring_b;         // Client TX (creator mapping).
  iree_spsc_queue_t ring_a_opener;  // Client RX (opener mapping).
  iree_spsc_queue_t ring_b_opener;  // Server RX (opener mapping).
} iree_net_shm_pair_rings_t;

// All mode bits that the current implementation supports.
#define IREE_NET_SHM_CARRIER_SUPPORTED_MODES IREE_NET_SHM_CARRIER_MODE_DEFAULT

// Creates the SHM region, writes the immutable header, initializes SPSC rings
// on both mappings, and returns a pair context holding the SHM mappings. On
// failure, all partial resources are cleaned up.
static iree_status_t iree_net_shm_pair_create_context(
    iree_net_shm_carrier_options_t options, iree_allocator_t host_allocator,
    iree_net_shm_pair_context_t** out_pair_context,
    iree_net_shm_pair_rings_t* out_rings) {
  *out_pair_context = NULL;
  memset(out_rings, 0, sizeof(*out_rings));

  // Validate options.
  if (options.mode & ~IREE_NET_SHM_CARRIER_SUPPORTED_MODES) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED, "unsupported SHM carrier mode bits: 0x%08x",
        (unsigned)(options.mode & ~IREE_NET_SHM_CARRIER_SUPPORTED_MODES));
  }
  uint32_t ring_capacity = options.ring_capacity;
  if (ring_capacity < IREE_SPSC_QUEUE_MIN_CAPACITY ||
      (ring_capacity & (ring_capacity - 1)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ring_capacity must be a power of two >= %" PRIu32,
                            IREE_SPSC_QUEUE_MIN_CAPACITY);
  }

  // Compute total SHM region size with overflow checking. On 32-bit platforms
  // a large ring_capacity could overflow the size arithmetic.
  iree_host_size_t ring_size = iree_spsc_queue_required_size(ring_capacity);
  iree_host_size_t double_ring_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(2, ring_size, &double_ring_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ring size overflow for capacity %" PRIu32,
                            ring_capacity);
  }
  iree_host_size_t total_region_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          IREE_NET_SHM_OFFSET_RINGS, double_ring_size, &total_region_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM region size overflow for capacity %" PRIu32,
                            ring_capacity);
  }

  // Create the SHM region.
  iree_shm_mapping_t creator_mapping;
  memset(&creator_mapping, 0, sizeof(creator_mapping));
  creator_mapping.handle = IREE_SHM_HANDLE_INVALID;
  iree_status_t status = iree_shm_create(iree_shm_options_default(),
                                         total_region_size, &creator_mapping);

  // Write the immutable header and initialize creator-side SPSC rings.
  if (iree_status_is_ok(status)) {
    memset(creator_mapping.base, 0, creator_mapping.size);
    iree_net_shm_region_header_t* header =
        (iree_net_shm_region_header_t*)((uint8_t*)creator_mapping.base +
                                        IREE_NET_SHM_OFFSET_HEADER);
    header->magic = IREE_NET_SHM_CARRIER_MAGIC;
    header->version = IREE_NET_SHM_CARRIER_VERSION;
    header->ring_capacity = ring_capacity;
  }
  if (iree_status_is_ok(status)) {
    void* ring_a_base =
        (uint8_t*)creator_mapping.base + IREE_NET_SHM_OFFSET_RINGS;
    status = iree_spsc_queue_initialize(ring_a_base, ring_size, ring_capacity,
                                        &out_rings->ring_a);
  }
  if (iree_status_is_ok(status)) {
    void* ring_b_base =
        (uint8_t*)creator_mapping.base + IREE_NET_SHM_OFFSET_RINGS + ring_size;
    status = iree_spsc_queue_initialize(ring_b_base, ring_size, ring_capacity,
                                        &out_rings->ring_b);
  }

  // Open a second mapping (simulates cross-process handle exchange).
  iree_shm_mapping_t opener_mapping;
  memset(&opener_mapping, 0, sizeof(opener_mapping));
  opener_mapping.handle = IREE_SHM_HANDLE_INVALID;
  if (iree_status_is_ok(status)) {
    status =
        iree_shm_open_handle(creator_mapping.handle, iree_shm_options_default(),
                             creator_mapping.size, &opener_mapping);
  }
  if (iree_status_is_ok(status)) {
    void* ring_a_base =
        (uint8_t*)opener_mapping.base + IREE_NET_SHM_OFFSET_RINGS;
    status =
        iree_spsc_queue_open(ring_a_base, ring_size, &out_rings->ring_a_opener);
  }
  if (iree_status_is_ok(status)) {
    void* ring_b_base =
        (uint8_t*)opener_mapping.base + IREE_NET_SHM_OFFSET_RINGS + ring_size;
    status =
        iree_spsc_queue_open(ring_b_base, ring_size, &out_rings->ring_b_opener);
  }

  // Bundle both mappings into a pair context.
  if (iree_status_is_ok(status)) {
    iree_net_shm_pair_context_t* pair_context = NULL;
    status = iree_allocator_malloc(host_allocator, sizeof(*pair_context),
                                   (void**)&pair_context);
    if (iree_status_is_ok(status)) {
      iree_atomic_ref_count_init(&pair_context->ref_count);
      pair_context->creator_mapping = creator_mapping;
      pair_context->opener_mapping = opener_mapping;
      pair_context->allocator = host_allocator;
      *out_pair_context = pair_context;
      return iree_ok_status();
    }
  }

  // Cleanup on failure: close SHM mappings directly (no pair_context yet).
  iree_shm_close(&opener_mapping);
  iree_shm_close(&creator_mapping);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_net_shm_carrier_create_pair
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_shm_carrier_create_pair(
    iree_net_shm_shared_wake_t* client_shared_wake,
    iree_net_shm_shared_wake_t* server_shared_wake,
    iree_net_shm_carrier_options_t options,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator,
    iree_net_carrier_t** out_client, iree_net_carrier_t** out_server) {
  IREE_ASSERT_ARGUMENT(client_shared_wake);
  IREE_ASSERT_ARGUMENT(server_shared_wake);
  IREE_ASSERT_ARGUMENT(out_client);
  IREE_ASSERT_ARGUMENT(out_server);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_client = NULL;
  *out_server = NULL;

  // Phase 1: Create shared memory region with SPSC rings.
  iree_net_shm_pair_context_t* pair_context = NULL;
  iree_net_shm_pair_rings_t rings;
  iree_status_t status = iree_net_shm_pair_create_context(
      options, host_allocator, &pair_context, &rings);

  // Phase 2: Create carriers.
  // Resolve armed flag pointers from both SHM mappings. Each carrier reads its
  // own armed flag from the opener mapping and the peer's flag from the creator
  // mapping (simulating separate process address spaces).
  if (iree_status_is_ok(status)) {
    uint8_t* creator_base = (uint8_t*)pair_context->creator_mapping.base;
    uint8_t* opener_base = (uint8_t*)pair_context->opener_mapping.base;
    iree_atomic_int32_t* consumer_a_armed =
        (iree_atomic_int32_t*)(creator_base +
                               IREE_NET_SHM_OFFSET_CONSUMER_A_ARMED);
    iree_atomic_int32_t* consumer_b_armed =
        (iree_atomic_int32_t*)(creator_base +
                               IREE_NET_SHM_OFFSET_CONSUMER_B_ARMED);
    iree_atomic_int32_t* consumer_a_armed_opener =
        (iree_atomic_int32_t*)(opener_base +
                               IREE_NET_SHM_OFFSET_CONSUMER_A_ARMED);
    iree_atomic_int32_t* consumer_b_armed_opener =
        (iree_atomic_int32_t*)(opener_base +
                               IREE_NET_SHM_OFFSET_CONSUMER_B_ARMED);

    // Both carriers get both SHM mappings as regions so that buffers from
    // either mapping can be registered on either carrier.
    iree_net_shm_region_info_t pair_regions[2];
    pair_regions[0].base_ptr = pair_context->creator_mapping.base;
    pair_regions[0].size = pair_context->creator_mapping.size;
    pair_regions[1].base_ptr = pair_context->opener_mapping.base;
    pair_regions[1].size = pair_context->opener_mapping.size;

    // Create client: TX=Ring B (creator), RX=Ring A (opener).
    // Client's armed flag = consumer_a (from opener); checks consumer_b (from
    // creator).
    iree_net_shm_carrier_create_params_t client_params;
    memset(&client_params, 0, sizeof(client_params));
    client_params.is_client = true;
    client_params.tx_queue = rings.ring_b;
    client_params.rx_queue = rings.ring_a_opener;
    client_params.shared_wake = client_shared_wake;
    client_params.peer_wake_notification = server_shared_wake->notification;
    client_params.our_armed = consumer_a_armed_opener;
    client_params.peer_armed = consumer_b_armed;
    client_params.release_context = pair_context;
    client_params.release_context_fn = iree_net_shm_pair_context_release;
    client_params.regions = pair_regions;
    client_params.region_count = IREE_ARRAYSIZE(pair_regions);
    status = iree_net_shm_carrier_create(&client_params, callback,
                                         host_allocator, out_client);

    if (iree_status_is_ok(status)) {
      // Retain pair_context for the server carrier (client already took one
      // ref).
      iree_net_shm_pair_context_retain(pair_context);

      // Create server: TX=Ring A (creator), RX=Ring B (opener).
      // Server's armed flag = consumer_b (from opener); checks consumer_a
      // (from creator).
      iree_net_shm_carrier_create_params_t server_params;
      memset(&server_params, 0, sizeof(server_params));
      server_params.is_client = false;
      server_params.tx_queue = rings.ring_a;
      server_params.rx_queue = rings.ring_b_opener;
      server_params.shared_wake = server_shared_wake;
      server_params.peer_wake_notification = client_shared_wake->notification;
      server_params.our_armed = consumer_b_armed_opener;
      server_params.peer_armed = consumer_a_armed;
      server_params.release_context = pair_context;
      server_params.release_context_fn = iree_net_shm_pair_context_release;
      server_params.regions = pair_regions;
      server_params.region_count = IREE_ARRAYSIZE(pair_regions);
      status = iree_net_shm_carrier_create(&server_params, callback,
                                           host_allocator, out_server);

      if (!iree_status_is_ok(status)) {
        // Release the retain we did for server — client still owns its ref.
        iree_net_shm_pair_context_release(pair_context);
      }
    }
  }

  if (!iree_status_is_ok(status)) {
    if (*out_client) {
      // Client carrier owns a pair_context ref and shared_wake/notification
      // refs. Destroying it releases all of them.
      iree_net_carrier_release(*out_client);
      *out_client = NULL;
    } else if (pair_context) {
      // No carrier was created to take ownership of pair_context.
      iree_net_shm_pair_context_release(pair_context);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
