// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/handshake.h"

#include "iree/base/internal/spsc_queue.h"

//===----------------------------------------------------------------------===//
// Header validation
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_shm_handshake_validate_header(
    const iree_net_shm_handshake_header_t* header,
    iree_net_shm_handshake_message_type_t expected_type) {
  if (header->magic != IREE_NET_SHM_HANDSHAKE_MAGIC) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "handshake magic mismatch: got 0x%08x, expected 0x%08x", header->magic,
        IREE_NET_SHM_HANDSHAKE_MAGIC);
  }
  if (header->version != IREE_NET_SHM_HANDSHAKE_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake version mismatch: got %u, expected %u",
                            header->version, IREE_NET_SHM_HANDSHAKE_VERSION);
  }
  if (header->type != expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected handshake message type: got %u, "
                            "expected %u",
                            (unsigned)header->type, (unsigned)expected_type);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Peer notification proxy creation
//===----------------------------------------------------------------------===//

// Creates a shared notification that proxies signals to a remote peer.
// When signaled, it increments the epoch in the peer's SHM page (visible to
// their futex) and writes to the peer's signal primitive (wakes their
// proactor).
static iree_status_t iree_net_shm_handshake_create_peer_notification(
    iree_async_proactor_t* proactor, iree_shm_mapping_t* peer_epoch_mapping,
    iree_async_primitive_t peer_signal_primitive,
    iree_async_notification_t** out_notification) {
  iree_async_notification_shared_options_t options;
  memset(&options, 0, sizeof(options));
  options.epoch_address = (iree_atomic_int32_t*)peer_epoch_mapping->base;
  // For the proxy notification: we don't monitor the peer's signal primitive
  // for POLLIN (that's the peer's proactor's job). We only need the signal
  // primitive to write to. The wake_primitive can be NONE since we never
  // submit NOTIFICATION_WAITs on this proxy.
  options.wake_primitive = iree_async_primitive_none();
  options.signal_primitive = peer_signal_primitive;
  return iree_async_notification_create_shared(proactor, &options,
                                               out_notification);
}

//===----------------------------------------------------------------------===//
// Carrier assembly
//===----------------------------------------------------------------------===//

// Initializes SPSC ring handles from a SHM mapping for either server or client.
// Server: TX=Ring A, RX=Ring B.  Client: TX=Ring B, RX=Ring A.
static iree_status_t iree_net_shm_handshake_init_rings(
    iree_shm_mapping_t* mapping, uint32_t ring_capacity, bool is_client,
    iree_spsc_queue_t* out_tx, iree_spsc_queue_t* out_rx) {
  iree_host_size_t ring_size = iree_spsc_queue_required_size(ring_capacity);
  void* ring_a_base = (uint8_t*)mapping->base + IREE_NET_SHM_OFFSET_RINGS;
  void* ring_b_base =
      (uint8_t*)mapping->base + IREE_NET_SHM_OFFSET_RINGS + ring_size;

  // The server created the rings (initialize), we open them. In cross-process
  // mode, the server's process has already called iree_spsc_queue_initialize()
  // so we always open here (both server and client in their own process).
  iree_spsc_queue_t ring_a, ring_b;
  iree_status_t status = iree_spsc_queue_open(ring_a_base, ring_size, &ring_a);
  if (!iree_status_is_ok(status)) return status;
  status = iree_spsc_queue_open(ring_b_base, ring_size, &ring_b);
  if (!iree_status_is_ok(status)) return status;

  if (is_client) {
    // Client: TX=Ring B (writes to B, reads from A).
    *out_tx = ring_b;
    *out_rx = ring_a;
  } else {
    // Server: TX=Ring A (writes to A, reads from B).
    *out_tx = ring_a;
    *out_rx = ring_b;
  }
  return iree_ok_status();
}

// Assembles carrier params from handshake results. Common to both server and
// client — only the is_client flag and armed flag assignments differ.
//
// Cross-process carriers have a single SHM mapping each (as opposed to the
// in-process pair which has two). Region 0 is populated with this mapping.
static void iree_net_shm_handshake_assemble_params(
    iree_net_shm_handshake_result_t* result, iree_shm_mapping_t* mapping,
    bool is_client, iree_spsc_queue_t tx_queue, iree_spsc_queue_t rx_queue,
    iree_net_shm_shared_wake_t* shared_wake,
    iree_async_notification_t* peer_notification,
    iree_net_shm_xproc_context_t* context) {
  uint8_t* base = (uint8_t*)mapping->base;
  iree_atomic_int32_t* consumer_a_armed =
      (iree_atomic_int32_t*)(base + IREE_NET_SHM_OFFSET_CONSUMER_A_ARMED);
  iree_atomic_int32_t* consumer_b_armed =
      (iree_atomic_int32_t*)(base + IREE_NET_SHM_OFFSET_CONSUMER_B_ARMED);

  // Populate region 0 with this process's SHM mapping. The handshake_result_t
  // carries the region info so it outlives this function.
  result->region.base_ptr = mapping->base;
  result->region.size = mapping->size;

  iree_net_shm_carrier_create_params_t* params = &result->carrier_params;
  memset(params, 0, sizeof(*params));
  params->is_client = is_client;
  params->tx_queue = tx_queue;
  params->rx_queue = rx_queue;
  params->shared_wake = shared_wake;
  params->peer_wake_notification = peer_notification;
  params->regions = &result->region;
  params->region_count = 1;

  if (is_client) {
    // Client reads from Ring A (consumer A), writes to Ring B.
    // Our armed = consumer_a, peer's armed = consumer_b.
    params->our_armed = consumer_a_armed;
    params->peer_armed = consumer_b_armed;
  } else {
    // Server reads from Ring B (consumer B), writes to Ring A.
    // Our armed = consumer_b, peer's armed = consumer_a.
    params->our_armed = consumer_b_armed;
    params->peer_armed = consumer_a_armed;
  }

  params->release_context = context;
  params->release_context_fn = iree_net_shm_xproc_context_release;
  result->context = context;
}

//===----------------------------------------------------------------------===//
// Server handshake
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_shm_handshake_server(
    iree_async_primitive_t socket, iree_net_shm_shared_wake_t* shared_wake,
    iree_net_shm_carrier_options_t options, iree_async_proactor_t* proactor,
    iree_allocator_t host_allocator,
    iree_net_shm_handshake_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(shared_wake);
  IREE_ASSERT_ARGUMENT(out_result);
  memset(out_result, 0, sizeof(*out_result));
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate ring capacity.
  uint32_t ring_capacity = options.ring_capacity;
  if (ring_capacity == 0) {
    ring_capacity = IREE_NET_SHM_CARRIER_DEFAULT_RING_CAPACITY;
  }
  if (ring_capacity < IREE_SPSC_QUEUE_MIN_CAPACITY ||
      (ring_capacity & (ring_capacity - 1)) != 0) {
    iree_async_primitive_close(&socket);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ring_capacity must be a power of two >= %" PRIu32,
                            IREE_SPSC_QUEUE_MIN_CAPACITY);
  }

  // Compute SHM region size.
  iree_host_size_t ring_size = iree_spsc_queue_required_size(ring_capacity);
  iree_host_size_t double_ring_size = 0;
  iree_host_size_t total_region_size = 0;
  if (!iree_host_size_checked_mul(2, ring_size, &double_ring_size) ||
      !iree_host_size_checked_add(IREE_NET_SHM_OFFSET_RINGS, double_ring_size,
                                  &total_region_size)) {
    iree_async_primitive_close(&socket);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM region size overflow for capacity %" PRIu32,
                            ring_capacity);
  }

  // Create the SHM region.
  iree_shm_mapping_t region_mapping;
  memset(&region_mapping, 0, sizeof(region_mapping));
  region_mapping.handle = IREE_SHM_HANDLE_INVALID;
  iree_status_t status = iree_shm_create(iree_shm_options_default(),
                                         total_region_size, &region_mapping);

  // Write the immutable header and initialize SPSC rings.
  if (iree_status_is_ok(status)) {
    memset(region_mapping.base, 0, region_mapping.size);
    iree_net_shm_region_header_t* header =
        (iree_net_shm_region_header_t*)((uint8_t*)region_mapping.base +
                                        IREE_NET_SHM_OFFSET_HEADER);
    header->magic = IREE_NET_SHM_CARRIER_MAGIC;
    header->version = IREE_NET_SHM_CARRIER_VERSION;
    header->ring_capacity = ring_capacity;

    // Initialize SPSC rings in the SHM region (server is the creator).
    void* ring_a_base =
        (uint8_t*)region_mapping.base + IREE_NET_SHM_OFFSET_RINGS;
    iree_spsc_queue_t ring_a;
    status = iree_spsc_queue_initialize(ring_a_base, ring_size, ring_capacity,
                                        &ring_a);
    if (iree_status_is_ok(status)) {
      void* ring_b_base = (uint8_t*)ring_a_base + ring_size;
      iree_spsc_queue_t ring_b;
      status = iree_spsc_queue_initialize(ring_b_base, ring_size, ring_capacity,
                                          &ring_b);
      (void)ring_a;
      (void)ring_b;
    }
  }

  // Export our shared_wake handles.
  iree_net_shm_shared_wake_export_t our_export;
  memset(&our_export, 0, sizeof(our_export));
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_shared_wake_export(shared_wake, &our_export);
  }

  // Duplicate the SHM region handle for the OFFER.
  iree_shm_handle_t region_handle_dup = IREE_SHM_HANDLE_INVALID;
  if (iree_status_is_ok(status)) {
    status = iree_shm_handle_dup(region_mapping.handle, &region_handle_dup);
  }

  // Send OFFER: SHM region + our wake handles.
  if (iree_status_is_ok(status)) {
    iree_net_shm_handshake_header_t offer_header;
    memset(&offer_header, 0, sizeof(offer_header));
    offer_header.magic = IREE_NET_SHM_HANDSHAKE_MAGIC;
    offer_header.version = IREE_NET_SHM_HANDSHAKE_VERSION;
    offer_header.type = IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER;
    offer_header.region_size = (uint32_t)region_mapping.size;
    offer_header.ring_capacity = ring_capacity;
    offer_header.wake_epoch_size = (uint32_t)our_export.epoch_shm_size;

    iree_net_shm_handshake_handles_t offer_handles;
    memset(&offer_handles, 0, sizeof(offer_handles));
    offer_handles.shm_region = region_handle_dup;
    offer_handles.wake_epoch_shm = our_export.epoch_shm_handle;
    offer_handles.signal_primitive = our_export.signal_primitive;

    status = iree_net_shm_handshake_send(socket, &offer_header, &offer_handles);
    if (iree_status_is_ok(status)) {
      // Handles ownership transferred to the socket layer — don't close them.
      region_handle_dup = IREE_SHM_HANDLE_INVALID;
      our_export.epoch_shm_handle = IREE_SHM_HANDLE_INVALID;
      our_export.signal_primitive = iree_async_primitive_none();
    }
  }

  // Receive ACCEPT: client's wake handles.
  iree_net_shm_handshake_header_t accept_header;
  iree_net_shm_handshake_handles_t accept_handles;
  memset(&accept_header, 0, sizeof(accept_header));
  memset(&accept_handles, 0, sizeof(accept_handles));
  if (iree_status_is_ok(status)) {
    status =
        iree_net_shm_handshake_recv(socket, &accept_header, &accept_handles);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_validate_header(
        &accept_header, IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT);
  }

  // Close the socket — handshake is done.
  iree_async_primitive_close(&socket);

  // Map the peer's wake epoch SHM.
  iree_shm_mapping_t peer_epoch_mapping;
  memset(&peer_epoch_mapping, 0, sizeof(peer_epoch_mapping));
  peer_epoch_mapping.handle = IREE_SHM_HANDLE_INVALID;
  if (iree_status_is_ok(status)) {
    status = iree_shm_open_handle(
        accept_handles.wake_epoch_shm, iree_shm_options_default(),
        accept_header.wake_epoch_size, &peer_epoch_mapping);
  }

  // Create peer notification proxy.
  iree_async_notification_t* peer_notification = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_create_peer_notification(
        proactor, &peer_epoch_mapping, accept_handles.signal_primitive,
        &peer_notification);
  }

  // Assemble carrier params.
  iree_net_shm_xproc_context_t* context = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_xproc_context_create(host_allocator, &context);
  }
  iree_spsc_queue_t tx_queue, rx_queue;
  memset(&tx_queue, 0, sizeof(tx_queue));
  memset(&rx_queue, 0, sizeof(rx_queue));
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_init_rings(&region_mapping, ring_capacity,
                                               /*is_client=*/false, &tx_queue,
                                               &rx_queue);
  }
  if (iree_status_is_ok(status)) {
    context->shm_mapping = region_mapping;
    context->peer_wake_epoch_mapping = peer_epoch_mapping;
    context->peer_notification = peer_notification;
    context->peer_signal_primitive = accept_handles.signal_primitive;

    iree_net_shm_handshake_assemble_params(
        out_result, &region_mapping, /*is_client=*/false, tx_queue, rx_queue,
        shared_wake, peer_notification, context);
    iree_shm_handle_close(&accept_handles.wake_epoch_shm);
  } else {
    if (context) iree_net_shm_xproc_context_release(context);
    iree_async_notification_release(peer_notification);
    iree_shm_close(&peer_epoch_mapping);
    iree_net_shm_handshake_handles_close(&accept_handles);
    iree_shm_handle_close(&region_handle_dup);
    iree_shm_handle_close(&our_export.epoch_shm_handle);
    iree_async_primitive_close(&our_export.signal_primitive);
    iree_shm_close(&region_mapping);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Client handshake
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_shm_handshake_client(
    iree_async_primitive_t socket, iree_net_shm_shared_wake_t* shared_wake,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_net_shm_handshake_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(shared_wake);
  IREE_ASSERT_ARGUMENT(out_result);
  memset(out_result, 0, sizeof(*out_result));
  IREE_TRACE_ZONE_BEGIN(z0);

  // Receive OFFER from server.
  iree_net_shm_handshake_header_t offer_header;
  iree_net_shm_handshake_handles_t offer_handles;
  memset(&offer_header, 0, sizeof(offer_header));
  memset(&offer_handles, 0, sizeof(offer_handles));
  iree_status_t status =
      iree_net_shm_handshake_recv(socket, &offer_header, &offer_handles);
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_validate_header(
        &offer_header, IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER);
  }

  // Map the SHM region from the received handle.
  iree_shm_mapping_t region_mapping;
  memset(&region_mapping, 0, sizeof(region_mapping));
  region_mapping.handle = IREE_SHM_HANDLE_INVALID;
  if (iree_status_is_ok(status)) {
    status = iree_shm_open_handle(offer_handles.shm_region,
                                  iree_shm_options_default(),
                                  offer_header.region_size, &region_mapping);
  }

  // Validate the SHM region header.
  if (iree_status_is_ok(status)) {
    iree_net_shm_region_header_t* region_header =
        (iree_net_shm_region_header_t*)((uint8_t*)region_mapping.base +
                                        IREE_NET_SHM_OFFSET_HEADER);
    if (region_header->magic != IREE_NET_SHM_CARRIER_MAGIC) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SHM region magic mismatch: 0x%08x",
                                region_header->magic);
    } else if (region_header->ring_capacity != offer_header.ring_capacity) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "ring_capacity mismatch: header says %" PRIu32
                                " but region says %" PRIu32,
                                offer_header.ring_capacity,
                                region_header->ring_capacity);
    }
  }

  // Map the server's wake epoch SHM.
  iree_shm_mapping_t peer_epoch_mapping;
  memset(&peer_epoch_mapping, 0, sizeof(peer_epoch_mapping));
  peer_epoch_mapping.handle = IREE_SHM_HANDLE_INVALID;
  if (iree_status_is_ok(status)) {
    status = iree_shm_open_handle(
        offer_handles.wake_epoch_shm, iree_shm_options_default(),
        offer_header.wake_epoch_size, &peer_epoch_mapping);
  }

  // Export our shared_wake handles for the ACCEPT.
  iree_net_shm_shared_wake_export_t our_export;
  memset(&our_export, 0, sizeof(our_export));
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_shared_wake_export(shared_wake, &our_export);
  }

  // Send ACCEPT: our wake handles.
  if (iree_status_is_ok(status)) {
    iree_net_shm_handshake_header_t accept_header;
    memset(&accept_header, 0, sizeof(accept_header));
    accept_header.magic = IREE_NET_SHM_HANDSHAKE_MAGIC;
    accept_header.version = IREE_NET_SHM_HANDSHAKE_VERSION;
    accept_header.type = IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT;
    accept_header.wake_epoch_size = (uint32_t)our_export.epoch_shm_size;

    iree_net_shm_handshake_handles_t accept_handles;
    memset(&accept_handles, 0, sizeof(accept_handles));
    accept_handles.shm_region = IREE_SHM_HANDLE_INVALID;
    accept_handles.wake_epoch_shm = our_export.epoch_shm_handle;
    accept_handles.signal_primitive = our_export.signal_primitive;

    status =
        iree_net_shm_handshake_send(socket, &accept_header, &accept_handles);
    if (iree_status_is_ok(status)) {
      our_export.epoch_shm_handle = IREE_SHM_HANDLE_INVALID;
      our_export.signal_primitive = iree_async_primitive_none();
    }
  }

  // Close the socket — handshake is done.
  iree_async_primitive_close(&socket);

  // Create peer notification proxy (to signal the server).
  iree_async_notification_t* peer_notification = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_create_peer_notification(
        proactor, &peer_epoch_mapping, offer_handles.signal_primitive,
        &peer_notification);
  }

  // Assemble carrier params.
  iree_net_shm_xproc_context_t* context = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_xproc_context_create(host_allocator, &context);
  }
  iree_spsc_queue_t tx_queue, rx_queue;
  memset(&tx_queue, 0, sizeof(tx_queue));
  memset(&rx_queue, 0, sizeof(rx_queue));
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_init_rings(
        &region_mapping, offer_header.ring_capacity, /*is_client=*/true,
        &tx_queue, &rx_queue);
  }
  if (iree_status_is_ok(status)) {
    // Transfer ownership of resources to the xproc context. The context
    // release function handles cleanup of all of these.
    context->shm_mapping = region_mapping;
    context->peer_wake_epoch_mapping = peer_epoch_mapping;
    context->peer_notification = peer_notification;
    context->peer_signal_primitive = offer_handles.signal_primitive;

    iree_net_shm_handshake_assemble_params(
        out_result, &region_mapping, /*is_client=*/true, tx_queue, rx_queue,
        shared_wake, peer_notification, context);

    // Close received handles we no longer need directly (mapped already).
    iree_shm_handle_close(&offer_handles.shm_region);
    iree_shm_handle_close(&offer_handles.wake_epoch_shm);
  } else {
    // Cleanup on failure. All close/release functions are NULL/invalid-safe.
    // Context fields are zero-initialized — resources haven't been transferred
    // yet, so release the context (if allocated) and each resource separately.
    if (context) iree_net_shm_xproc_context_release(context);
    iree_async_notification_release(peer_notification);
    iree_shm_close(&peer_epoch_mapping);
    iree_shm_close(&region_mapping);
    iree_net_shm_handshake_handles_close(&offer_handles);
    iree_shm_handle_close(&our_export.epoch_shm_handle);
    iree_async_primitive_close(&our_export.signal_primitive);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
