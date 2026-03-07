// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/framing_adapter.h"

#include <string.h>

struct iree_net_framing_adapter_t {
  // Owned carrier - released when adapter is freed.
  iree_net_carrier_t* carrier;

  // Referenced pool for copy-path lease bridging - not owned.
  iree_async_buffer_pool_t* reassembly_pool;

  // Message handlers (set_callbacks for atomic handoff).
  iree_net_message_endpoint_callbacks_t callbacks;

  // Deactivate callback (set during deactivate).
  struct {
    iree_net_message_endpoint_deactivate_fn_t fn;
    void* user_data;
  } deactivate_callback;

  bool activated;
  iree_allocator_t host_allocator;

  // Embedded frame accumulator with FAM for internal buffer - must be last.
  iree_net_frame_accumulator_t accumulator;
};

// Called by frame_accumulator when a complete frame is ready.
// Bridges NULL lease (copy path) to valid lease using reassembly_pool.
static iree_status_t iree_net_framing_adapter_on_frame_complete(
    void* user_data, iree_const_byte_span_t frame,
    iree_async_buffer_lease_t* lease) {
  iree_net_framing_adapter_t* adapter = (iree_net_framing_adapter_t*)user_data;

  if (lease) {
    return adapter->callbacks.on_message(adapter->callbacks.user_data, frame,
                                         lease);
  }

  // Copy path: frame is in accumulator's internal buffer with NULL lease.
  // message_endpoint contract requires non-NULL lease, so acquire from pool.
  iree_async_buffer_lease_t bridged_lease;
  IREE_RETURN_IF_ERROR(
      iree_async_buffer_pool_acquire(adapter->reassembly_pool, &bridged_lease));

  uint8_t* dest = iree_async_span_ptr(bridged_lease.span);
  memcpy(dest, frame.data, frame.data_length);
  iree_const_byte_span_t bridged_frame =
      iree_make_const_byte_span(dest, frame.data_length);

  iree_status_t status = adapter->callbacks.on_message(
      adapter->callbacks.user_data, bridged_frame, &bridged_lease);

  // Release our handle. Handler may have copied the lease to retain it.
  iree_async_buffer_lease_release(&bridged_lease);
  return status;
}

static iree_status_t iree_net_framing_adapter_on_recv(
    void* user_data, iree_async_span_t data, iree_async_buffer_lease_t* lease) {
  iree_net_framing_adapter_t* adapter = (iree_net_framing_adapter_t*)user_data;
  return iree_net_frame_accumulator_push_lease(&adapter->accumulator, lease,
                                               data.length);
}

static void iree_net_framing_adapter_on_carrier_deactivated(void* user_data) {
  iree_net_framing_adapter_t* adapter = (iree_net_framing_adapter_t*)user_data;
  if (adapter->deactivate_callback.fn) {
    adapter->deactivate_callback.fn(adapter->deactivate_callback.user_data);
  }
}

static void iree_net_framing_adapter_set_callbacks(
    void* self, iree_net_message_endpoint_callbacks_t callbacks) {
  iree_net_framing_adapter_t* adapter = (iree_net_framing_adapter_t*)self;
  adapter->callbacks = callbacks;
}

static iree_status_t iree_net_framing_adapter_activate(void* self) {
  iree_net_framing_adapter_t* adapter = (iree_net_framing_adapter_t*)self;
  if (!adapter->callbacks.on_message) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callbacks must be set before activate");
  }
  if (adapter->activated) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "adapter already activated");
  }

  iree_net_carrier_recv_handler_t recv_handler = {
      .fn = iree_net_framing_adapter_on_recv,
      .user_data = adapter,
  };
  iree_net_carrier_set_recv_handler(adapter->carrier, recv_handler);

  iree_status_t status = iree_net_carrier_activate(adapter->carrier);
  if (iree_status_is_ok(status)) {
    adapter->activated = true;
  }
  return status;
}

static iree_status_t iree_net_framing_adapter_deactivate(
    void* self, iree_net_message_endpoint_deactivate_fn_t callback,
    void* user_data) {
  iree_net_framing_adapter_t* adapter = (iree_net_framing_adapter_t*)self;
  adapter->deactivate_callback.fn = callback;
  adapter->deactivate_callback.user_data = user_data;
  return iree_net_carrier_deactivate(
      adapter->carrier, iree_net_framing_adapter_on_carrier_deactivated,
      adapter);
}

static iree_status_t iree_net_framing_adapter_send(
    void* self, const iree_net_message_endpoint_send_params_t* params) {
  iree_net_framing_adapter_t* adapter = (iree_net_framing_adapter_t*)self;
  iree_net_send_params_t carrier_params = {
      .data = params->data,
      .flags = IREE_NET_SEND_FLAG_NONE,
      .user_data = params->user_data,
  };
  return iree_net_carrier_send(adapter->carrier, &carrier_params);
}

static iree_net_carrier_send_budget_t
iree_net_framing_adapter_query_send_budget(void* self) {
  iree_net_framing_adapter_t* adapter = (iree_net_framing_adapter_t*)self;
  return iree_net_carrier_query_send_budget(adapter->carrier);
}

iree_status_t iree_net_framing_adapter_allocate(
    iree_net_carrier_t* carrier, iree_net_frame_length_callback_t frame_length,
    iree_host_size_t max_frame_size, iree_async_buffer_pool_t* reassembly_pool,
    iree_allocator_t host_allocator, iree_net_framing_adapter_t** out_adapter) {
  *out_adapter = NULL;

  if (!carrier) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "carrier is required");
  }
  if (!frame_length.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "frame_length.fn is required");
  }
  if (!reassembly_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reassembly_pool is required");
  }
  if (max_frame_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "max_frame_size must be > 0");
  }
  if (iree_net_carrier_state(carrier) != IREE_NET_CARRIER_STATE_CREATED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must not be activated before wrapping");
  }
  iree_host_size_t pool_buffer_size =
      iree_async_buffer_pool_buffer_size(reassembly_pool);
  if (pool_buffer_size < max_frame_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reassembly_pool buffer_size (%" PRIhsz
                            ") must be >= max_frame_size (%" PRIhsz ")",
                            pool_buffer_size, max_frame_size);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t accumulator_storage =
      iree_net_frame_accumulator_storage_size(max_frame_size);
  iree_host_size_t total_size = sizeof(iree_net_framing_adapter_t) -
                                sizeof(iree_net_frame_accumulator_t) +
                                accumulator_storage;

  iree_net_framing_adapter_t* adapter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&adapter));
  memset(adapter, 0, total_size);
  adapter->host_allocator = host_allocator;
  adapter->carrier = carrier;
  adapter->reassembly_pool = reassembly_pool;

  iree_net_frame_complete_callback_t on_frame_complete = {
      .fn = iree_net_framing_adapter_on_frame_complete,
      .user_data = adapter,
  };
  iree_status_t status = iree_net_frame_accumulator_initialize(
      &adapter->accumulator, max_frame_size, frame_length, on_frame_complete);

  if (iree_status_is_ok(status)) {
    *out_adapter = adapter;
  } else {
    iree_net_framing_adapter_free(adapter);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_net_framing_adapter_free(iree_net_framing_adapter_t* adapter) {
  if (!adapter) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = adapter->host_allocator;
  iree_net_frame_accumulator_deinitialize(&adapter->accumulator);
  iree_net_carrier_release(adapter->carrier);
  iree_allocator_free(allocator, adapter);
  IREE_TRACE_ZONE_END(z0);
}

iree_net_message_endpoint_t iree_net_framing_adapter_as_endpoint(
    iree_net_framing_adapter_t* adapter) {
  static const iree_net_message_endpoint_vtable_t vtable = {
      .set_callbacks = iree_net_framing_adapter_set_callbacks,
      .activate = iree_net_framing_adapter_activate,
      .deactivate = iree_net_framing_adapter_deactivate,
      .send = iree_net_framing_adapter_send,
      .query_send_budget = iree_net_framing_adapter_query_send_budget,
  };
  iree_net_message_endpoint_t endpoint = {
      .self = adapter,
      .vtable = &vtable,
  };
  return endpoint;
}

iree_net_carrier_t* iree_net_framing_adapter_carrier(
    iree_net_framing_adapter_t* adapter) {
  return adapter->carrier;
}
