// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/factory.h"

#include <string.h>

#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/carrier/shm/carrier_pair.h"
#include "iree/net/connection.h"
#include "iree/net/message_endpoint.h"

//===----------------------------------------------------------------------===//
// Internal types
//===----------------------------------------------------------------------===//

typedef struct iree_net_shm_listener_t iree_net_shm_listener_t;

typedef struct iree_net_shm_factory_t {
  iree_net_transport_factory_t base;
  iree_slim_mutex_t mutex;
  // Dynamic array of active listener pointers. Listeners add/remove themselves
  // during create_listener and stop.
  iree_net_shm_listener_t** listeners;
  iree_host_size_t listener_count;
  iree_host_size_t listener_capacity;
  iree_net_shm_carrier_options_t options;
  iree_allocator_t host_allocator;
} iree_net_shm_factory_t;

//===----------------------------------------------------------------------===//
// SHM endpoint adapter
//===----------------------------------------------------------------------===//

// Thin adapter that bridges SHM carrier recv_handler to message_endpoint
// callbacks. SHM carrier delivers NULL lease (data is a view into the SPSC
// ring, valid only during the callback), but message_endpoint consumers expect
// valid leases. This adapter acquires from recv_pool, copies, and delivers
// with a valid lease.
typedef struct iree_net_shm_endpoint_adapter_t {
  // Owned carrier.
  iree_net_carrier_t* carrier;
  // Referenced recv_pool for lease bridging.
  iree_async_buffer_pool_t* recv_pool;
  // Message and error handlers from the endpoint consumer.
  iree_net_message_endpoint_callbacks_t callbacks;
  // Deactivation callback.
  struct {
    iree_net_message_endpoint_deactivate_fn_t fn;
    void* user_data;
  } deactivate;
  bool activated;
} iree_net_shm_endpoint_adapter_t;

// Carrier recv handler that bridges NULL lease to valid lease.
static iree_status_t iree_net_shm_endpoint_on_recv(
    void* user_data, iree_async_span_t data, iree_async_buffer_lease_t* lease) {
  iree_net_shm_endpoint_adapter_t* adapter =
      (iree_net_shm_endpoint_adapter_t*)user_data;
  if (lease) {
    // Non-NULL lease: pass through directly.
    iree_const_byte_span_t message =
        iree_make_const_byte_span(iree_async_span_ptr(data), data.length);
    return adapter->callbacks.on_message(adapter->callbacks.user_data, message,
                                         lease);
  }
  // NULL lease: acquire from pool, copy, deliver with valid lease.
  iree_async_buffer_lease_t bridged;
  memset(&bridged, 0, sizeof(bridged));
  IREE_RETURN_IF_ERROR(
      iree_async_buffer_pool_acquire(adapter->recv_pool, &bridged));
  uint8_t* destination = iree_async_span_ptr(bridged.span);
  memcpy(destination, iree_async_span_ptr(data), data.length);
  iree_const_byte_span_t message =
      iree_make_const_byte_span(destination, data.length);
  iree_status_t status = adapter->callbacks.on_message(
      adapter->callbacks.user_data, message, &bridged);
  iree_async_buffer_lease_release(&bridged);
  return status;
}

// Carrier deactivation callback forwarded to endpoint consumer.
static void iree_net_shm_endpoint_carrier_deactivated(void* user_data) {
  iree_net_shm_endpoint_adapter_t* adapter =
      (iree_net_shm_endpoint_adapter_t*)user_data;
  adapter->activated = false;
  if (adapter->deactivate.fn) {
    adapter->deactivate.fn(adapter->deactivate.user_data);
  }
}

// Carrier error callback forwarded to endpoint consumer.
static void iree_net_shm_endpoint_carrier_error(void* user_data,
                                                iree_status_t status) {
  iree_net_shm_endpoint_adapter_t* adapter =
      (iree_net_shm_endpoint_adapter_t*)user_data;
  if (adapter->callbacks.on_error) {
    adapter->callbacks.on_error(adapter->callbacks.user_data, status);
  } else {
    iree_status_ignore(status);
  }
}

static void iree_net_shm_endpoint_set_callbacks(
    void* self, iree_net_message_endpoint_callbacks_t callbacks) {
  iree_net_shm_endpoint_adapter_t* adapter =
      (iree_net_shm_endpoint_adapter_t*)self;
  adapter->callbacks = callbacks;
}

static iree_status_t iree_net_shm_endpoint_activate(void* self) {
  iree_net_shm_endpoint_adapter_t* adapter =
      (iree_net_shm_endpoint_adapter_t*)self;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!adapter->callbacks.on_message) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callbacks must be set before activation");
  }
  // Set carrier recv handler to our bridge function.
  iree_net_carrier_set_recv_handler(adapter->carrier,
                                    (iree_net_carrier_recv_handler_t){
                                        .fn = iree_net_shm_endpoint_on_recv,
                                        .user_data = adapter,
                                    });
  adapter->activated = true;
  iree_status_t status = iree_net_carrier_activate(adapter->carrier);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_shm_endpoint_deactivate(
    void* self, iree_net_message_endpoint_deactivate_fn_t callback,
    void* user_data) {
  iree_net_shm_endpoint_adapter_t* adapter =
      (iree_net_shm_endpoint_adapter_t*)self;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!adapter->activated) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "endpoint not active");
  }
  adapter->deactivate.fn = callback;
  adapter->deactivate.user_data = user_data;
  iree_status_t status = iree_net_carrier_deactivate(
      adapter->carrier, iree_net_shm_endpoint_carrier_deactivated, adapter);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_shm_endpoint_send(
    void* self, const iree_net_message_endpoint_send_params_t* params) {
  iree_net_shm_endpoint_adapter_t* adapter =
      (iree_net_shm_endpoint_adapter_t*)self;
  iree_net_send_params_t carrier_params;
  memset(&carrier_params, 0, sizeof(carrier_params));
  carrier_params.data = params->data;
  return iree_net_carrier_send(adapter->carrier, &carrier_params);
}

static iree_net_carrier_send_budget_t iree_net_shm_endpoint_query_send_budget(
    void* self) {
  iree_net_shm_endpoint_adapter_t* adapter =
      (iree_net_shm_endpoint_adapter_t*)self;
  return iree_net_carrier_query_send_budget(adapter->carrier);
}

static const iree_net_message_endpoint_vtable_t iree_net_shm_endpoint_vtable = {
    .set_callbacks = iree_net_shm_endpoint_set_callbacks,
    .activate = iree_net_shm_endpoint_activate,
    .deactivate = iree_net_shm_endpoint_deactivate,
    .send = iree_net_shm_endpoint_send,
    .query_send_budget = iree_net_shm_endpoint_query_send_budget,
};

static iree_status_t iree_net_shm_endpoint_adapter_allocate(
    iree_net_carrier_t* carrier, iree_async_buffer_pool_t* recv_pool,
    iree_allocator_t host_allocator,
    iree_net_shm_endpoint_adapter_t** out_adapter) {
  *out_adapter = NULL;
  iree_net_shm_endpoint_adapter_t* adapter = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*adapter),
                                             (void**)&adapter));
  memset(adapter, 0, sizeof(*adapter));
  adapter->carrier = carrier;
  adapter->recv_pool = recv_pool;
  *out_adapter = adapter;
  return iree_ok_status();
}

static void iree_net_shm_endpoint_adapter_free(
    iree_net_shm_endpoint_adapter_t* adapter, iree_allocator_t host_allocator) {
  iree_net_carrier_release(adapter->carrier);
  iree_allocator_free(host_allocator, adapter);
}

//===----------------------------------------------------------------------===//
// Connection
//===----------------------------------------------------------------------===//

typedef struct iree_net_shm_connection_t {
  iree_net_connection_t base;
  iree_async_proactor_t* proactor;
  // Carrier from the initial connect/accept. Consumed by open_endpoint (set to
  // NULL when the adapter takes ownership).
  iree_net_carrier_t* initial_carrier;
  // Referenced recv_pool for endpoint adapter creation. May be NULL for
  // connections that never open endpoints (e.g., factory-level tests).
  iree_async_buffer_pool_t* recv_pool;
  // Created lazily at open_endpoint time. Owns the carrier after creation.
  iree_net_shm_endpoint_adapter_t* adapter;
} iree_net_shm_connection_t;

static const iree_net_connection_vtable_t iree_net_shm_connection_vtable;

// Creates an SHM connection with the given initial carrier. The connection
// takes ownership of |initial_carrier| — the caller must not release it on
// success. On failure, the caller retains ownership.
static iree_status_t iree_net_shm_connection_create(
    iree_async_proactor_t* proactor, iree_net_carrier_t* initial_carrier,
    iree_async_buffer_pool_t* recv_pool, iree_allocator_t host_allocator,
    iree_net_connection_t** out_connection) {
  *out_connection = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_net_shm_connection_t* connection = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*connection),
                                (void**)&connection));
  memset(connection, 0, sizeof(*connection));
  iree_net_connection_initialize(&iree_net_shm_connection_vtable,
                                 host_allocator, &connection->base);
  connection->proactor = proactor;
  connection->initial_carrier = initial_carrier;
  connection->recv_pool = recv_pool;
  *out_connection = &connection->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_net_shm_connection_destroy(
    iree_net_connection_t* base_connection) {
  iree_net_shm_connection_t* connection =
      (iree_net_shm_connection_t*)base_connection;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = connection->base.host_allocator;
  // Adapter owns the carrier if it was created. Otherwise the connection still
  // owns initial_carrier.
  if (connection->adapter) {
    iree_net_shm_endpoint_adapter_free(connection->adapter, host_allocator);
  } else if (connection->initial_carrier) {
    iree_net_carrier_release(connection->initial_carrier);
  }
  iree_allocator_free(host_allocator, connection);
  IREE_TRACE_ZONE_END(z0);
}

// Heap-allocated state for deferred async endpoint delivery via NOP.
typedef struct iree_net_shm_endpoint_deferred_t {
  iree_async_nop_operation_t nop;
  struct {
    iree_net_endpoint_ready_callback_t fn;
    void* user_data;
  } endpoint_ready;
  iree_net_message_endpoint_t endpoint;
  iree_allocator_t host_allocator;
} iree_net_shm_endpoint_deferred_t;

// NOP completion for async endpoint delivery.
static void iree_net_shm_endpoint_deferred_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_endpoint_deferred_t* deferred =
      (iree_net_shm_endpoint_deferred_t*)user_data;
  iree_status_ignore(status);
  deferred->endpoint_ready.fn(deferred->endpoint_ready.user_data,
                              iree_ok_status(), deferred->endpoint);
  iree_allocator_free(deferred->host_allocator, deferred);
}

// Allocates, populates, and submits a deferred endpoint delivery NOP.
// On success the proactor owns the allocation and fires |callback| on the next
// poll with the given |endpoint|. On failure the allocation is cleaned up.
static iree_status_t iree_net_shm_endpoint_deferred_submit(
    iree_async_proactor_t* proactor, iree_net_message_endpoint_t endpoint,
    iree_net_endpoint_ready_callback_t callback, void* user_data,
    iree_allocator_t host_allocator) {
  iree_net_shm_endpoint_deferred_t* deferred = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*deferred),
                                             (void**)&deferred));
  memset(deferred, 0, sizeof(*deferred));
  deferred->endpoint_ready.fn = callback;
  deferred->endpoint_ready.user_data = user_data;
  deferred->endpoint = endpoint;
  deferred->host_allocator = host_allocator;
  iree_async_operation_initialize(
      &deferred->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_shm_endpoint_deferred_complete,
      deferred);
  iree_status_t status =
      iree_async_proactor_submit_one(proactor, &deferred->nop.base);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, deferred);
  }
  return status;
}

static iree_status_t iree_net_shm_connection_open_endpoint(
    iree_net_connection_t* base_connection,
    iree_net_endpoint_ready_callback_t callback, void* user_data) {
  iree_net_shm_connection_t* connection =
      (iree_net_shm_connection_t*)base_connection;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (connection->adapter) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SHM supports one endpoint per connection");
  }
  if (!connection->initial_carrier) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier already consumed");
  }
  if (!connection->recv_pool) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "recv_pool required for endpoint creation");
  }

  iree_allocator_t host_allocator = connection->base.host_allocator;

  // Create adapter, transferring carrier ownership.
  iree_status_t status = iree_net_shm_endpoint_adapter_allocate(
      connection->initial_carrier, connection->recv_pool, host_allocator,
      &connection->adapter);
  if (iree_status_is_ok(status)) {
    connection->initial_carrier = NULL;  // Ownership transferred.
    iree_net_message_endpoint_t endpoint = {
        .self = connection->adapter,
        .vtable = &iree_net_shm_endpoint_vtable,
    };
    status = iree_net_shm_endpoint_deferred_submit(
        connection->proactor, endpoint, callback, user_data, host_allocator);
    if (!iree_status_is_ok(status)) {
      // Submit failed. Free the adapter through the normal ownership path
      // (releases carrier, frees adapter struct). The connection is terminal:
      // the carrier has been consumed and released.
      iree_net_shm_endpoint_adapter_free(connection->adapter, host_allocator);
      connection->adapter = NULL;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_net_connection_vtable_t iree_net_shm_connection_vtable = {
    .destroy = iree_net_shm_connection_destroy,
    .open_endpoint = iree_net_shm_connection_open_endpoint,
};

//===----------------------------------------------------------------------===//
// Listener
//===----------------------------------------------------------------------===//

struct iree_net_shm_listener_t {
  iree_net_listener_t base;
  iree_net_shm_factory_t* factory;
  iree_async_proactor_t* proactor;
  iree_async_buffer_pool_t* recv_pool;
  struct {
    iree_net_listener_accept_callback_t fn;
    void* user_data;
  } accept;
  iree_allocator_t host_allocator;
  iree_host_size_t name_length;
  char name[];
};

static const iree_net_listener_vtable_t iree_net_shm_listener_vtable;

// Finds an active listener by name. Caller must hold factory->mutex.
static iree_net_shm_listener_t* iree_net_shm_factory_find_listener_unsafe(
    iree_net_shm_factory_t* factory, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < factory->listener_count; ++i) {
    iree_net_shm_listener_t* listener = factory->listeners[i];
    if (listener->name_length == name.size &&
        memcmp(listener->name, name.data, name.size) == 0) {
      return listener;
    }
  }
  return NULL;
}

// Removes a listener from the table (swap-remove). Caller must hold mutex.
static void iree_net_shm_factory_remove_listener_unsafe(
    iree_net_shm_factory_t* factory, iree_net_shm_listener_t* listener) {
  for (iree_host_size_t i = 0; i < factory->listener_count; ++i) {
    if (factory->listeners[i] == listener) {
      factory->listeners[i] = factory->listeners[factory->listener_count - 1];
      factory->listener_count--;
      return;
    }
  }
}

static void iree_net_shm_listener_free(iree_net_listener_t* base_listener) {
  iree_net_shm_listener_t* listener = (iree_net_shm_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = listener->host_allocator;
  iree_allocator_free(host_allocator, listener);
  IREE_TRACE_ZONE_END(z0);
}

// Heap-allocated state for deferred listener stopped notification via NOP.
typedef struct iree_net_shm_stop_deferred_t {
  iree_async_nop_operation_t nop;
  iree_net_listener_stopped_callback_t stopped;
  iree_allocator_t host_allocator;
} iree_net_shm_stop_deferred_t;

static void iree_net_shm_stop_deferred_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_stop_deferred_t* deferred =
      (iree_net_shm_stop_deferred_t*)user_data;
  iree_status_ignore(status);
  deferred->stopped.fn(deferred->stopped.user_data);
  iree_allocator_free(deferred->host_allocator, deferred);
}

// Allocates, populates, and submits a deferred stopped notification NOP.
// On success the proactor owns the allocation and fires |callback| on the next
// poll. On failure the allocation is cleaned up.
static iree_status_t iree_net_shm_stop_deferred_submit(
    iree_async_proactor_t* proactor,
    iree_net_listener_stopped_callback_t callback,
    iree_allocator_t host_allocator) {
  iree_net_shm_stop_deferred_t* deferred = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*deferred),
                                             (void**)&deferred));
  memset(deferred, 0, sizeof(*deferred));
  deferred->stopped = callback;
  deferred->host_allocator = host_allocator;
  iree_async_operation_initialize(
      &deferred->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_shm_stop_deferred_complete,
      deferred);
  iree_status_t status =
      iree_async_proactor_submit_one(proactor, &deferred->nop.base);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, deferred);
  }
  return status;
}

static iree_status_t iree_net_shm_listener_stop(
    iree_net_listener_t* base_listener,
    iree_net_listener_stopped_callback_t callback) {
  iree_net_shm_listener_t* listener = (iree_net_shm_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Remove from factory table so no new connections route here.
  iree_slim_mutex_lock(&listener->factory->mutex);
  iree_net_shm_factory_remove_listener_unsafe(listener->factory, listener);
  iree_slim_mutex_unlock(&listener->factory->mutex);

  iree_status_t status = iree_net_shm_stop_deferred_submit(
      listener->proactor, callback, listener->host_allocator);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_shm_listener_query_bound_address(
    iree_net_listener_t* base_listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  iree_net_shm_listener_t* listener = (iree_net_shm_listener_t*)base_listener;
  if (buffer_capacity < listener->name_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer too small for bound address");
  }
  memcpy(buffer, listener->name, listener->name_length);
  *out_address = iree_make_string_view(buffer, listener->name_length);
  return iree_ok_status();
}

static const iree_net_listener_vtable_t iree_net_shm_listener_vtable = {
    .free = iree_net_shm_listener_free,
    .stop = iree_net_shm_listener_stop,
    .query_bound_address = iree_net_shm_listener_query_bound_address,
};

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

static const iree_net_transport_factory_vtable_t iree_net_shm_factory_vtable;

IREE_API_EXPORT iree_status_t iree_net_shm_factory_allocate(
    iree_net_shm_carrier_options_t options, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory) {
  IREE_ASSERT_ARGUMENT(out_factory);
  *out_factory = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_shm_factory_t* factory = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*factory),
                                (void**)&factory));
  memset(factory, 0, sizeof(*factory));
  factory->base.vtable = &iree_net_shm_factory_vtable;
  factory->options = options;
  factory->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&factory->mutex);

  *out_factory = &factory->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_net_shm_factory_destroy(
    iree_net_transport_factory_t* base_factory) {
  iree_net_shm_factory_t* factory = (iree_net_shm_factory_t*)base_factory;
  IREE_ASSERT(factory->listener_count == 0,
              "factory destroyed with active listeners; stop and free all "
              "listeners before destroying the factory");
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = factory->host_allocator;
  iree_allocator_free(host_allocator, factory->listeners);
  iree_slim_mutex_deinitialize(&factory->mutex);
  iree_allocator_free(host_allocator, factory);
  IREE_TRACE_ZONE_END(z0);
}

static iree_net_transport_capabilities_t
iree_net_shm_factory_query_capabilities(
    iree_net_transport_factory_t* base_factory) {
  (void)base_factory;
  return IREE_NET_TRANSPORT_CAPABILITY_RELIABLE |
         IREE_NET_TRANSPORT_CAPABILITY_ORDERED;
}

// NOP-deferred connect: fires accept_callback then connect_callback on the
// next proactor poll, ensuring asynchronous delivery.
typedef struct iree_net_shm_connect_deferred_t {
  iree_async_nop_operation_t nop;
  struct {
    iree_net_transport_connect_callback_t fn;
    void* user_data;
  } connect;
  iree_net_connection_t* client_connection;
  struct {
    iree_net_listener_accept_callback_t fn;
    void* user_data;
  } accept;
  iree_net_connection_t* server_connection;
  iree_status_t error_status;
  iree_allocator_t host_allocator;
} iree_net_shm_connect_deferred_t;

static void iree_net_shm_connect_deferred_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_connect_deferred_t* deferred =
      (iree_net_shm_connect_deferred_t*)user_data;
  iree_status_ignore(status);
  if (iree_status_is_ok(deferred->error_status)) {
    deferred->accept.fn(deferred->accept.user_data, iree_ok_status(),
                        deferred->server_connection);
    deferred->connect.fn(deferred->connect.user_data, iree_ok_status(),
                         deferred->client_connection);
  } else {
    deferred->connect.fn(deferred->connect.user_data, deferred->error_status,
                         NULL);
  }
  iree_allocator_free(deferred->host_allocator, deferred);
}

static iree_status_t iree_net_shm_factory_connect(
    iree_net_transport_factory_t* base_factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data) {
  iree_net_shm_factory_t* factory = (iree_net_shm_factory_t*)base_factory;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_shm_connect_deferred_t* deferred = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(factory->host_allocator, sizeof(*deferred),
                                (void**)&deferred));
  memset(deferred, 0, sizeof(*deferred));
  deferred->connect.fn = callback;
  deferred->connect.user_data = user_data;
  deferred->host_allocator = factory->host_allocator;

  // Look up listener and create carrier pair + connections under lock.
  iree_slim_mutex_lock(&factory->mutex);
  iree_net_shm_listener_t* listener =
      iree_net_shm_factory_find_listener_unsafe(factory, address);

  iree_status_t status = iree_ok_status();
  iree_net_carrier_t* client_carrier = NULL;
  iree_net_carrier_t* server_carrier = NULL;
  if (listener) {
    iree_net_carrier_callback_t no_callback = {0};
    status = iree_net_shm_carrier_create_pair(
        proactor, listener->proactor, factory->options, no_callback,
        factory->host_allocator, &client_carrier, &server_carrier);
    if (iree_status_is_ok(status)) {
      status = iree_net_shm_connection_create(
          proactor, client_carrier, recv_pool, factory->host_allocator,
          &deferred->client_connection);
      if (iree_status_is_ok(status)) client_carrier = NULL;
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_shm_connection_create(
          listener->proactor, server_carrier, listener->recv_pool,
          factory->host_allocator, &deferred->server_connection);
      if (iree_status_is_ok(status)) server_carrier = NULL;
    }
    if (iree_status_is_ok(status)) {
      deferred->accept.fn = listener->accept.fn;
      deferred->accept.user_data = listener->accept.user_data;
    }
  }
  iree_slim_mutex_unlock(&factory->mutex);

  // Route errors to deferred delivery. connect never returns application errors
  // synchronously — all results go through the callback.
  if (iree_status_is_ok(status) && !listener) {
    deferred->error_status = iree_make_status(
        IREE_STATUS_UNAVAILABLE, "no listener registered for '%.*s'",
        (int)address.size, address.data);
  } else if (!iree_status_is_ok(status)) {
    if (deferred->client_connection) {
      iree_net_connection_release(deferred->client_connection);
      deferred->client_connection = NULL;
    }
    if (deferred->server_connection) {
      iree_net_connection_release(deferred->server_connection);
      deferred->server_connection = NULL;
    }
    if (client_carrier) iree_net_carrier_release(client_carrier);
    if (server_carrier) iree_net_carrier_release(server_carrier);
    deferred->error_status = status;
    status = iree_ok_status();
  }

  // Submit NOP to deliver callback on next poll().
  iree_async_operation_initialize(
      &deferred->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_shm_connect_deferred_complete,
      deferred);
  status = iree_async_proactor_submit_one(proactor, &deferred->nop.base);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(deferred->error_status);
    if (deferred->client_connection) {
      iree_net_connection_release(deferred->client_connection);
    }
    if (deferred->server_connection) {
      iree_net_connection_release(deferred->server_connection);
    }
    iree_allocator_free(factory->host_allocator, deferred);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_shm_factory_create_listener(
    iree_net_transport_factory_t* base_factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  iree_net_shm_factory_t* factory = (iree_net_shm_factory_t*)base_factory;
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_listener = NULL;

  iree_host_size_t total_size =
      sizeof(iree_net_shm_listener_t) + bind_address.size;
  iree_net_shm_listener_t* listener = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&listener));
  memset(listener, 0, total_size);
  listener->base.vtable = &iree_net_shm_listener_vtable;
  listener->factory = factory;
  listener->proactor = proactor;
  listener->recv_pool = recv_pool;
  listener->accept.fn = accept_callback;
  listener->accept.user_data = user_data;
  listener->host_allocator = host_allocator;
  listener->name_length = bind_address.size;
  memcpy(listener->name, bind_address.data, bind_address.size);

  // Register under lock.
  iree_slim_mutex_lock(&factory->mutex);
  iree_status_t status = iree_ok_status();
  if (iree_net_shm_factory_find_listener_unsafe(factory, bind_address)) {
    status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "listener '%.*s' already registered",
                              (int)bind_address.size, bind_address.data);
  }
  if (iree_status_is_ok(status) &&
      factory->listener_count >= factory->listener_capacity) {
    status = iree_allocator_grow_array(
        factory->host_allocator, factory->listener_count + 1,
        sizeof(*factory->listeners), &factory->listener_capacity,
        (void**)&factory->listeners);
  }
  if (iree_status_is_ok(status)) {
    factory->listeners[factory->listener_count++] = listener;
  }
  iree_slim_mutex_unlock(&factory->mutex);

  if (iree_status_is_ok(status)) {
    *out_listener = &listener->base;
  } else {
    iree_allocator_free(host_allocator, listener);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_net_transport_factory_vtable_t iree_net_shm_factory_vtable = {
    .free = iree_net_shm_factory_destroy,
    .query_capabilities = iree_net_shm_factory_query_capabilities,
    .connect = iree_net_shm_factory_connect,
    .create_listener = iree_net_shm_factory_create_listener,
};
