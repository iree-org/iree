// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/memory.h"

#include <stdlib.h>
#include <string.h>

#include "iree/tooling/profile/model.h"
#include "iree/tooling/profile/reader.h"

const char* iree_profile_memory_event_type_name(
    iree_hal_profile_memory_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      return "slab_acquire";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      return "slab_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      return "pool_reserve";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      return "pool_materialize";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      return "pool_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      return "pool_wait";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return "queue_alloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      return "queue_dealloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      return "buffer_allocate";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      return "buffer_free";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
      return "buffer_import";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      return "buffer_unimport";
    default:
      return "unknown";
  }
}

static const char* iree_profile_memory_lifecycle_kind_name(
    iree_profile_memory_lifecycle_kind_t kind) {
  switch (kind) {
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB:
      return "slab";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION:
      return "pool_reservation";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION:
      return "queue_allocation";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION:
      return "buffer_allocation";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_IMPORTED_BUFFER:
      return "imported_buffer";
    default:
      return "unknown";
  }
}

static bool iree_profile_memory_event_allocation_kind(
    const iree_hal_profile_memory_event_t* event,
    iree_profile_memory_lifecycle_kind_t* out_kind) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      if (iree_all_bits_set(
              event->flags,
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION)) {
        *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION;
        return true;
      }
      return false;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_IMPORTED_BUFFER;
      return true;
    default:
      return false;
  }
}

static bool iree_profile_memory_event_pool_kind(
    const iree_hal_profile_memory_event_t* event,
    iree_profile_memory_lifecycle_kind_t* out_kind) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_IMPORTED_BUFFER;
      return true;
    default:
      return false;
  }
}

static bool iree_profile_memory_event_opens_lifecycle(
    const iree_hal_profile_memory_event_t* event) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      return iree_all_bits_set(
          event->flags, IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION);
    default:
      return false;
  }
}

static bool iree_profile_memory_event_closes_lifecycle(
    const iree_hal_profile_memory_event_t* event) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      return true;
    default:
      return false;
  }
}

void iree_profile_memory_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_memory_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
}

void iree_profile_memory_context_deinitialize(
    iree_profile_memory_context_t* context) {
  iree_allocator_free(context->host_allocator, context->devices);
  iree_allocator_free(context->host_allocator, context->pools);
  iree_allocator_free(context->host_allocator, context->allocations);
  memset(context, 0, sizeof(*context));
}

static iree_status_t iree_profile_memory_get_device(
    iree_profile_memory_context_t* context, uint32_t physical_device_ordinal,
    iree_profile_memory_device_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    if (context->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      *out_device = &context->devices[i];
      return iree_ok_status();
    }
  }

  if (context->device_count + 1 > context->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->device_count + 1),
        sizeof(context->devices[0]), &context->device_capacity,
        (void**)&context->devices));
  }

  iree_profile_memory_device_t* device =
      &context->devices[context->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_get_pool(
    iree_profile_memory_context_t* context,
    iree_profile_memory_lifecycle_kind_t kind, uint32_t physical_device_ordinal,
    uint64_t pool_id, uint64_t memory_type,
    iree_profile_memory_pool_t** out_pool) {
  *out_pool = NULL;

  for (iree_host_size_t i = context->pool_count; i > 0; --i) {
    iree_profile_memory_pool_t* pool = &context->pools[i - 1];
    if (pool->kind == kind &&
        pool->physical_device_ordinal == physical_device_ordinal &&
        pool->pool_id == pool_id && pool->memory_type == memory_type) {
      *out_pool = pool;
      return iree_ok_status();
    }
  }

  if (context->pool_count + 1 > context->pool_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)8, context->pool_count + 1),
        sizeof(context->pools[0]), &context->pool_capacity,
        (void**)&context->pools));
  }

  iree_profile_memory_pool_t* pool = &context->pools[context->pool_count++];
  memset(pool, 0, sizeof(*pool));
  pool->kind = kind;
  pool->physical_device_ordinal = physical_device_ordinal;
  pool->pool_id = pool_id;
  pool->memory_type = memory_type;
  *out_pool = pool;
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_get_allocation(
    iree_profile_memory_context_t* context,
    iree_profile_memory_lifecycle_kind_t kind, uint32_t physical_device_ordinal,
    uint64_t allocation_id, uint64_t pool_id,
    iree_profile_memory_allocation_t** out_allocation) {
  *out_allocation = NULL;

  for (iree_host_size_t i = context->allocation_count; i > 0; --i) {
    iree_profile_memory_allocation_t* allocation = &context->allocations[i - 1];
    if (allocation->kind == kind &&
        allocation->physical_device_ordinal == physical_device_ordinal &&
        allocation->allocation_id == allocation_id &&
        allocation->pool_id == pool_id) {
      *out_allocation = allocation;
      return iree_ok_status();
    }
  }

  if (context->allocation_count + 1 > context->allocation_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->allocation_count + 1),
        sizeof(context->allocations[0]), &context->allocation_capacity,
        (void**)&context->allocations));
  }

  iree_profile_memory_allocation_t* allocation =
      &context->allocations[context->allocation_count++];
  memset(allocation, 0, sizeof(*allocation));
  allocation->kind = kind;
  allocation->physical_device_ordinal = physical_device_ordinal;
  allocation->allocation_id = allocation_id;
  allocation->pool_id = pool_id;
  allocation->first_queue_ordinal = UINT32_MAX;
  allocation->last_queue_ordinal = UINT32_MAX;
  *out_allocation = allocation;
  return iree_ok_status();
}

static const iree_profile_memory_allocation_t*
iree_profile_memory_find_allocation(
    const iree_profile_memory_context_t* context,
    iree_profile_memory_lifecycle_kind_t kind, uint32_t physical_device_ordinal,
    uint64_t allocation_id, uint64_t pool_id) {
  for (iree_host_size_t i = context->allocation_count; i > 0; --i) {
    const iree_profile_memory_allocation_t* allocation =
        &context->allocations[i - 1];
    if (allocation->kind == kind &&
        allocation->physical_device_ordinal == physical_device_ordinal &&
        allocation->allocation_id == allocation_id &&
        allocation->pool_id == pool_id) {
      return allocation;
    }
  }
  return NULL;
}

static uint64_t iree_profile_memory_resolve_pool_id(
    const iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event,
    iree_profile_memory_lifecycle_kind_t kind) {
  if (event->pool_id != 0) return event->pool_id;
  if (kind != IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION) {
    return event->pool_id;
  }

  for (iree_host_size_t i = context->allocation_count; i > 0; --i) {
    const iree_profile_memory_allocation_t* allocation =
        &context->allocations[i - 1];
    if (allocation->kind == kind &&
        allocation->physical_device_ordinal == event->physical_device_ordinal &&
        allocation->allocation_id == event->allocation_id) {
      return allocation->pool_id;
    }
  }
  return event->pool_id;
}

static bool iree_profile_memory_event_matches(
    const iree_hal_profile_memory_event_t* event, int64_t id_filter,
    iree_string_view_t filter) {
  if (id_filter >= 0 && event->event_id != (uint64_t)id_filter &&
      event->allocation_id != (uint64_t)id_filter) {
    return false;
  }
  iree_string_view_t type_name =
      iree_make_cstring_view(iree_profile_memory_event_type_name(event->type));
  return iree_profile_key_matches(type_name, filter);
}

static bool iree_profile_memory_event_has_pool_stats(
    const iree_hal_profile_memory_event_t* event) {
  return iree_all_bits_set(event->flags,
                           IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_STATS);
}

static bool iree_profile_memory_event_closes_materialization(
    const iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event) {
  if (event->type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE) {
    return false;
  }
  if (!iree_all_bits_set(event->flags,
                         IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION)) {
    return false;
  }

  const uint64_t pool_id = iree_profile_memory_resolve_pool_id(
      context, event, IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION);
  const iree_profile_memory_allocation_t* allocation =
      iree_profile_memory_find_allocation(
          context, IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION,
          event->physical_device_ordinal, event->allocation_id, pool_id);
  return allocation &&
         (allocation->materialization_balance.current_count != 0 ||
          allocation->materialization_balance.current_bytes != 0);
}

static bool iree_profile_memory_add_u64(uint64_t* target, uint64_t value) {
  if (value > UINT64_MAX - *target) return false;
  *target += value;
  return true;
}

static bool iree_profile_memory_balance_open(
    iree_profile_memory_balance_t* balance, uint64_t length) {
  if (!iree_profile_memory_add_u64(&balance->total_open_count, 1) ||
      !iree_profile_memory_add_u64(&balance->current_count, 1) ||
      !iree_profile_memory_add_u64(&balance->total_open_bytes, length) ||
      !iree_profile_memory_add_u64(&balance->current_bytes, length)) {
    return false;
  }
  balance->high_water_count =
      iree_max(balance->high_water_count, balance->current_count);
  balance->high_water_bytes =
      iree_max(balance->high_water_bytes, balance->current_bytes);
  return true;
}

static bool iree_profile_memory_balance_close(
    iree_profile_memory_balance_t* balance, uint64_t length) {
  if (!iree_profile_memory_add_u64(&balance->total_close_count, 1) ||
      !iree_profile_memory_add_u64(&balance->total_close_bytes, length)) {
    return false;
  }

  const bool partial_count = balance->current_count == 0;
  if (balance->current_count != 0) {
    --balance->current_count;
  }

  uint64_t partial_bytes = 0;
  if (length > balance->current_bytes) {
    partial_bytes = length - balance->current_bytes;
    balance->current_bytes = 0;
  } else {
    balance->current_bytes -= length;
  }

  if (partial_count || partial_bytes != 0) {
    if (!iree_profile_memory_add_u64(&balance->partial_close_count, 1) ||
        !iree_profile_memory_add_u64(&balance->partial_close_bytes,
                                     partial_bytes)) {
      return false;
    }
  }
  return true;
}

static void iree_profile_memory_count_device_event(
    iree_profile_memory_device_t* device,
    const iree_hal_profile_memory_event_t* event) {
  ++device->event_count;
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      ++device->slab_acquire_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      ++device->slab_release_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      ++device->pool_reserve_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      ++device->pool_materialize_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      ++device->pool_release_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      ++device->pool_wait_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      ++device->buffer_allocate_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      ++device->buffer_free_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
      ++device->buffer_import_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      ++device->buffer_unimport_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      ++device->queue_alloca_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      ++device->queue_dealloca_count;
      break;
    default:
      break;
  }
}

static bool iree_profile_memory_apply_device_event_balance(
    iree_profile_memory_device_t* device,
    const iree_hal_profile_memory_event_t* event, bool close_materialization) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      return iree_profile_memory_balance_open(&device->slab_allocation_balance,
                                              event->length);
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      return iree_profile_memory_balance_close(&device->slab_allocation_balance,
                                               event->length);
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      if (iree_all_bits_set(
              event->flags,
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION)) {
        return iree_profile_memory_balance_open(
            &device->pool_reservation_balance, event->length);
      }
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      return iree_profile_memory_balance_open(
          &device->pool_materialization_balance, event->length);
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE: {
      bool accounted = iree_profile_memory_balance_close(
          &device->pool_reservation_balance, event->length);
      if (accounted && close_materialization) {
        accounted = iree_profile_memory_balance_close(
            &device->pool_materialization_balance, event->length);
      }
      return accounted;
    }
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      return iree_profile_memory_balance_open(
          &device->buffer_allocation_balance, event->length);
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      return iree_profile_memory_balance_close(
          &device->buffer_allocation_balance, event->length);
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
      return iree_profile_memory_balance_open(&device->buffer_import_balance,
                                              event->length);
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      return iree_profile_memory_balance_close(&device->buffer_import_balance,
                                               event->length);
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return iree_profile_memory_balance_open(&device->queue_inflight_balance,
                                              event->length);
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      return iree_profile_memory_balance_close(&device->queue_inflight_balance,
                                               event->length);
    default:
      return true;
  }
}

static iree_status_t iree_profile_memory_record_device_event(
    iree_profile_memory_device_t* device,
    const iree_hal_profile_memory_event_t* event, bool close_materialization) {
  iree_profile_memory_count_device_event(device, event);
  const bool accounted = iree_profile_memory_apply_device_event_balance(
      device, event, close_materialization);
  if (!accounted) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "memory accounting overflow for event_id=%" PRIu64,
                            event->event_id);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_record_pool_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event, bool close_materialization) {
  iree_profile_memory_lifecycle_kind_t kind = 0;
  if (!iree_profile_memory_event_pool_kind(event, &kind)) {
    return iree_ok_status();
  }

  const uint64_t pool_id =
      iree_profile_memory_resolve_pool_id(context, event, kind);
  iree_profile_memory_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_memory_get_pool(
      context, kind, event->physical_device_ordinal, pool_id,
      event->memory_type, &pool));
  ++pool->event_count;
  pool->buffer_usage |= event->buffer_usage;
  if (event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT) {
    ++pool->wait_count;
  } else if (event->type ==
             IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE) {
    ++pool->materialize_count;
  }

  bool accounted = true;
  if (iree_profile_memory_event_opens_lifecycle(event)) {
    accounted = iree_profile_memory_balance_open(&pool->lifecycle_balance,
                                                 event->length);
  } else if (iree_profile_memory_event_closes_lifecycle(event)) {
    accounted = iree_profile_memory_balance_close(&pool->lifecycle_balance,
                                                  event->length);
  }
  if (accounted &&
      event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE) {
    accounted = iree_profile_memory_balance_open(&pool->materialization_balance,
                                                 event->length);
  } else if (accounted && close_materialization &&
             event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE) {
    accounted = iree_profile_memory_balance_close(
        &pool->materialization_balance, event->length);
  }
  if (!accounted) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "memory pool accounting overflow for event_id=%" PRIu64,
        event->event_id);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_record_pool_stats_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event) {
  if (!iree_profile_memory_event_has_pool_stats(event) || event->pool_id == 0) {
    return iree_ok_status();
  }

  iree_profile_memory_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_memory_get_pool(
      context, IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION,
      event->physical_device_ordinal, event->pool_id, event->memory_type,
      &pool));
  const bool first_sample = pool->pool_stats_sample_count == 0;
  if (!iree_profile_memory_add_u64(&pool->pool_stats_sample_count, 1)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "memory pool stats sample overflow for event_id=%" PRIu64,
        event->event_id);
  }
  pool->buffer_usage |= event->buffer_usage;
  pool->pool_bytes_reserved = event->pool_bytes_reserved;
  pool->pool_bytes_reserved_high_water = iree_max(
      pool->pool_bytes_reserved_high_water, event->pool_bytes_reserved);
  pool->pool_bytes_free = event->pool_bytes_free;
  pool->pool_bytes_free_low_water =
      first_sample
          ? event->pool_bytes_free
          : iree_min(pool->pool_bytes_free_low_water, event->pool_bytes_free);
  pool->pool_bytes_committed = event->pool_bytes_committed;
  pool->pool_bytes_committed_high_water = iree_max(
      pool->pool_bytes_committed_high_water, event->pool_bytes_committed);
  pool->pool_budget_limit = event->pool_budget_limit;
  pool->pool_reservation_count = event->pool_reservation_count;
  pool->pool_reservation_high_water_count = iree_max(
      pool->pool_reservation_high_water_count, event->pool_reservation_count);
  pool->pool_slab_count = event->pool_slab_count;
  pool->pool_slab_high_water_count =
      iree_max(pool->pool_slab_high_water_count, event->pool_slab_count);
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_record_allocation_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event, bool close_materialization) {
  iree_profile_memory_lifecycle_kind_t kind = 0;
  if (!iree_profile_memory_event_allocation_kind(event, &kind)) {
    return iree_ok_status();
  }

  const uint64_t pool_id =
      iree_profile_memory_resolve_pool_id(context, event, kind);
  iree_profile_memory_allocation_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_memory_get_allocation(
      context, kind, event->physical_device_ordinal, event->allocation_id,
      pool_id, &allocation));
  if (allocation->first_event_id == 0) {
    allocation->first_event_id = event->event_id;
    allocation->first_host_time_ns = event->host_time_ns;
  }
  allocation->last_event_id = event->event_id;
  allocation->last_host_time_ns = event->host_time_ns;
  if (allocation->first_submission_id == 0 && event->submission_id != 0) {
    allocation->first_submission_id = event->submission_id;
    allocation->first_queue_ordinal = event->queue_ordinal;
  }
  if (event->submission_id != 0) {
    allocation->last_submission_id = event->submission_id;
    allocation->last_queue_ordinal = event->queue_ordinal;
  }
  allocation->backing_id =
      allocation->backing_id ? allocation->backing_id : event->backing_id;
  allocation->flags |= event->flags;
  allocation->memory_type |= event->memory_type;
  allocation->buffer_usage |= event->buffer_usage;
  ++allocation->event_count;
  if (event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT) {
    ++allocation->wait_count;
  } else if (event->type ==
             IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE) {
    ++allocation->materialize_count;
  }

  bool accounted = true;
  if (iree_profile_memory_event_opens_lifecycle(event)) {
    accounted = iree_profile_memory_balance_open(&allocation->lifecycle_balance,
                                                 event->length);
  } else if (iree_profile_memory_event_closes_lifecycle(event)) {
    accounted = iree_profile_memory_balance_close(
        &allocation->lifecycle_balance, event->length);
  }
  if (accounted &&
      event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE) {
    accounted = iree_profile_memory_balance_open(
        &allocation->materialization_balance, event->length);
  } else if (accounted && close_materialization &&
             event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE) {
    accounted = iree_profile_memory_balance_close(
        &allocation->materialization_balance, event->length);
  }
  if (!accounted) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "memory allocation accounting overflow for event_id=%" PRIu64,
        event->event_id);
  }
  return iree_ok_status();
}

static void iree_profile_memory_print_event_jsonl(
    const iree_hal_profile_memory_event_t* event, FILE* file) {
  fprintf(file,
          "{\"type\":\"memory_event\",\"event_id\":%" PRIu64 ",\"event_type\":",
          event->event_id);
  iree_profile_fprint_json_string(
      file,
      iree_make_cstring_view(iree_profile_memory_event_type_name(event->type)));
  fprintf(file,
          ",\"event_type_value\":%u,\"flags\":%u,\"result\":%u"
          ",\"host_time_ns\":%" PRId64 ",\"allocation_id\":%" PRIu64
          ",\"host_time_domain\":\"iree_host_time_ns\""
          ",\"pool_id\":%" PRIu64 ",\"backing_id\":%" PRIu64
          ",\"submission_id\":%" PRIu64
          ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
          ",\"frontier_entry_count\":%u,\"memory_type\":%" PRIu64
          ",\"buffer_usage\":%" PRIu64 ",\"offset\":%" PRIu64
          ",\"length\":%" PRIu64 ",\"alignment\":%" PRIu64
          ",\"externally_owned\":%s,\"pool_stats_available\":%s"
          ",\"pool_bytes_reserved\":%" PRIu64 ",\"pool_bytes_free\":%" PRIu64
          ",\"pool_bytes_committed\":%" PRIu64 ",\"pool_budget_limit\":%" PRIu64
          ",\"pool_reservation_count\":%u"
          ",\"pool_slab_count\":%u}\n",
          event->type, event->flags, event->result, event->host_time_ns,
          event->allocation_id, event->pool_id, event->backing_id,
          event->submission_id, event->physical_device_ordinal,
          event->queue_ordinal, event->frontier_entry_count, event->memory_type,
          event->buffer_usage, event->offset, event->length, event->alignment,
          iree_all_bits_set(event->flags,
                            IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED)
              ? "true"
              : "false",
          iree_profile_memory_event_has_pool_stats(event) ? "true" : "false",
          event->pool_bytes_reserved, event->pool_bytes_free,
          event->pool_bytes_committed, event->pool_budget_limit,
          event->pool_reservation_count, event->pool_slab_count);
}

static iree_status_t iree_profile_memory_emit_event_jsonl(
    void* user_data, const iree_profile_memory_event_row_t* row) {
  FILE* file = (FILE*)user_data;
  iree_profile_memory_print_event_jsonl(row->event, file);
  return iree_ok_status();
}

static bool iree_profile_memory_allocation_open_at_end(
    const iree_profile_memory_allocation_t* allocation) {
  return allocation->lifecycle_balance.current_bytes != 0 ||
         allocation->materialization_balance.current_bytes != 0;
}

static uint64_t iree_profile_memory_allocation_partial_close_count(
    const iree_profile_memory_allocation_t* allocation) {
  return allocation->lifecycle_balance.partial_close_count +
         allocation->materialization_balance.partial_close_count;
}

typedef struct iree_profile_memory_overview_t {
  // Allocation lifecycles with bytes or materializations still open.
  uint64_t open_allocation_lifecycle_count;
  // Partial close transitions across all allocation lifecycles.
  uint64_t partial_lifecycle_close_count;
} iree_profile_memory_overview_t;

static iree_profile_memory_overview_t iree_profile_memory_compute_overview(
    const iree_profile_memory_context_t* context) {
  iree_profile_memory_overview_t overview;
  memset(&overview, 0, sizeof(overview));
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    if (iree_profile_memory_allocation_open_at_end(&context->allocations[i])) {
      ++overview.open_allocation_lifecycle_count;
    }
    overview.partial_lifecycle_close_count +=
        iree_profile_memory_allocation_partial_close_count(
            &context->allocations[i]);
  }
  return overview;
}

static void iree_profile_memory_print_balance_text(
    FILE* file, const char* name,
    const iree_profile_memory_balance_t* balance) {
  fprintf(file,
          "  %s: open_at_end=%" PRIu64 " peak_open=%" PRIu64
          " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
          " opened_bytes=%" PRIu64 " closed_bytes=%" PRIu64
          " partial_closes=%" PRIu64 " partial_close_bytes=%" PRIu64 "\n",
          name, balance->current_count, balance->high_water_count,
          balance->current_bytes, balance->high_water_bytes,
          balance->total_open_bytes, balance->total_close_bytes,
          balance->partial_close_count, balance->partial_close_bytes);
}

static void iree_profile_memory_fprint_balance_json_fields(
    FILE* file, const char* prefix,
    const iree_profile_memory_balance_t* balance) {
  fprintf(
      file,
      ",\"%s_current_count\":%" PRIu64 ",\"%s_high_water_count\":%" PRIu64
      ",\"%s_total_open_count\":%" PRIu64 ",\"%s_total_close_count\":%" PRIu64
      ",\"%s_partial_close_count\":%" PRIu64 ",\"%s_current_bytes\":%" PRIu64
      ",\"%s_high_water_bytes\":%" PRIu64 ",\"%s_total_open_bytes\":%" PRIu64
      ",\"%s_total_close_bytes\":%" PRIu64
      ",\"%s_partial_close_bytes\":%" PRIu64,
      prefix, balance->current_count, prefix, balance->high_water_count, prefix,
      balance->total_open_count, prefix, balance->total_close_count, prefix,
      balance->partial_close_count, prefix, balance->current_bytes, prefix,
      balance->high_water_bytes, prefix, balance->total_open_bytes, prefix,
      balance->total_close_bytes, prefix, balance->partial_close_bytes);
}

typedef struct iree_profile_memory_device_lifetime_t {
  // Device event for the queue alloca operation, if available.
  const iree_hal_profile_queue_device_event_t* alloca_event;
  // Device event for the queue dealloca operation, if available.
  const iree_hal_profile_queue_device_event_t* dealloca_event;
  // True when |start_tick|, |end_tick|, and |duration_ticks| are valid.
  bool is_valid;
  // Device tick at which the queue allocation lifetime begins.
  uint64_t start_tick;
  // Device tick at which the queue allocation lifetime ends.
  uint64_t end_tick;
  // Device tick duration from alloca start through dealloca completion.
  uint64_t duration_ticks;
  // True when |duration_ns| was scaled through a clock fit.
  bool has_duration_ns;
  // Duration in nanoseconds when |has_duration_ns| is true.
  int64_t duration_ns;
} iree_profile_memory_device_lifetime_t;

typedef struct iree_profile_memory_queue_device_event_index_t {
  // Profile model owning queue metadata and clock-correlation samples.
  const iree_profile_model_t* model;
  // Host allocator used for |events|.
  iree_allocator_t host_allocator;
  // Queue alloca/dealloca device events sorted by lookup key.
  iree_hal_profile_queue_device_event_t* events;
  // Number of valid entries in |events|.
  iree_host_size_t event_count;
  // Capacity of |events| in entries.
  iree_host_size_t event_capacity;
} iree_profile_memory_queue_device_event_index_t;

typedef struct iree_profile_memory_queue_device_event_key_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal within |physical_device_ordinal|.
  uint32_t queue_ordinal;
  // Queue event type being resolved.
  iree_hal_profile_queue_event_type_t type;
  // Queue submission epoch containing the device event.
  uint64_t submission_id;
  // Producer-defined allocation identifier carried by the event.
  uint64_t allocation_id;
} iree_profile_memory_queue_device_event_key_t;

static int iree_profile_memory_compare_u32(uint32_t lhs, uint32_t rhs) {
  return (lhs > rhs) - (lhs < rhs);
}

static int iree_profile_memory_compare_u64(uint64_t lhs, uint64_t rhs) {
  return (lhs > rhs) - (lhs < rhs);
}

static int iree_profile_memory_compare_queue_device_event_to_key(
    const iree_hal_profile_queue_device_event_t* event,
    const iree_profile_memory_queue_device_event_key_t* key) {
  int cmp = iree_profile_memory_compare_u32(event->physical_device_ordinal,
                                            key->physical_device_ordinal);
  if (cmp != 0) return cmp;
  cmp =
      iree_profile_memory_compare_u32(event->queue_ordinal, key->queue_ordinal);
  if (cmp != 0) return cmp;
  cmp = iree_profile_memory_compare_u32(event->type, key->type);
  if (cmp != 0) return cmp;
  cmp =
      iree_profile_memory_compare_u64(event->submission_id, key->submission_id);
  if (cmp != 0) return cmp;
  return iree_profile_memory_compare_u64(event->allocation_id,
                                         key->allocation_id);
}

static int iree_profile_memory_compare_queue_device_events(
    const void* lhs_ptr, const void* rhs_ptr) {
  const iree_hal_profile_queue_device_event_t* lhs =
      (const iree_hal_profile_queue_device_event_t*)lhs_ptr;
  const iree_hal_profile_queue_device_event_t* rhs =
      (const iree_hal_profile_queue_device_event_t*)rhs_ptr;
  iree_profile_memory_queue_device_event_key_t rhs_key = {
      .physical_device_ordinal = rhs->physical_device_ordinal,
      .queue_ordinal = rhs->queue_ordinal,
      .type = rhs->type,
      .submission_id = rhs->submission_id,
      .allocation_id = rhs->allocation_id,
  };
  int cmp =
      iree_profile_memory_compare_queue_device_event_to_key(lhs, &rhs_key);
  if (cmp != 0) return cmp;
  return iree_profile_memory_compare_u64(lhs->event_id, rhs->event_id);
}

static void iree_profile_memory_queue_device_event_index_initialize(
    const iree_profile_model_t* model, iree_allocator_t host_allocator,
    iree_profile_memory_queue_device_event_index_t* out_index) {
  memset(out_index, 0, sizeof(*out_index));
  out_index->model = model;
  out_index->host_allocator = host_allocator;
}

static void iree_profile_memory_queue_device_event_index_deinitialize(
    iree_profile_memory_queue_device_event_index_t* index) {
  iree_allocator_free(index->host_allocator, index->events);
  memset(index, 0, sizeof(*index));
}

static bool iree_profile_memory_queue_device_event_is_memory_lifetime_event(
    const iree_hal_profile_queue_device_event_t* event) {
  return event->type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA ||
         event->type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA;
}

static iree_status_t iree_profile_memory_queue_device_event_index_append(
    iree_profile_memory_queue_device_event_index_t* index,
    const iree_hal_profile_queue_device_event_t* event) {
  if (index->event_count + 1 > index->event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        index->host_allocator,
        iree_max((iree_host_size_t)64, index->event_count + 1),
        sizeof(index->events[0]), &index->event_capacity,
        (void**)&index->events));
  }
  index->events[index->event_count++] = *event;
  return iree_ok_status();
}

static iree_status_t
iree_profile_memory_queue_device_event_index_process_record(
    iree_profile_memory_queue_device_event_index_t* index,
    const iree_hal_profile_file_record_t* record) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(
          record->content_type,
          IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
    return iree_ok_status();
  }

  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_queue_device_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_queue_device_event_t event;
    memcpy(&event, typed_record.contents.data, sizeof(event));
    if (!iree_profile_memory_queue_device_event_is_memory_lifetime_event(
            &event)) {
      continue;
    }
    if (!iree_profile_model_find_queue(index->model,
                                       event.physical_device_ordinal,
                                       event.queue_ordinal, event.stream_id)) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "queue device event references missing queue "
          "metadata device=%u queue=%u stream=%" PRIu64 " submission=%" PRIu64,
          event.physical_device_ordinal, event.queue_ordinal, event.stream_id,
          event.submission_id);
    }
    status = iree_profile_memory_queue_device_event_index_append(index, &event);
  }
  return status;
}

static void iree_profile_memory_queue_device_event_index_sort(
    iree_profile_memory_queue_device_event_index_t* index) {
  if (index->event_count <= 1) return;
  qsort(index->events, index->event_count, sizeof(index->events[0]),
        iree_profile_memory_compare_queue_device_events);
}

static bool iree_profile_memory_queue_device_event_is_valid(
    const iree_hal_profile_queue_device_event_t* event) {
  return event && event->start_tick != 0 && event->end_tick != 0 &&
         event->end_tick >= event->start_tick;
}

static const iree_hal_profile_queue_device_event_t*
iree_profile_memory_find_queue_device_event(
    const iree_profile_memory_queue_device_event_index_t* index,
    const iree_profile_memory_allocation_t* allocation, uint64_t submission_id,
    uint32_t queue_ordinal, iree_hal_profile_queue_event_type_t event_type) {
  if (!index || submission_id == 0 || queue_ordinal == UINT32_MAX) {
    return NULL;
  }
  iree_profile_memory_queue_device_event_key_t key = {
      .physical_device_ordinal = allocation->physical_device_ordinal,
      .queue_ordinal = queue_ordinal,
      .type = event_type,
      .submission_id = submission_id,
      .allocation_id = allocation->allocation_id,
  };

  iree_host_size_t low = 0;
  iree_host_size_t high = index->event_count;
  while (low < high) {
    const iree_host_size_t mid = low + (high - low) / 2;
    const int cmp = iree_profile_memory_compare_queue_device_event_to_key(
        &index->events[mid], &key);
    if (cmp < 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  if (low == index->event_count) return NULL;
  return iree_profile_memory_compare_queue_device_event_to_key(
             &index->events[low], &key) == 0
             ? &index->events[low]
             : NULL;
}

static iree_profile_memory_device_lifetime_t
iree_profile_memory_resolve_device_lifetime(
    const iree_profile_memory_queue_device_event_index_t* queue_device_index,
    const iree_profile_memory_allocation_t* allocation) {
  iree_profile_memory_device_lifetime_t lifetime;
  memset(&lifetime, 0, sizeof(lifetime));
  if (allocation->kind != IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION) {
    return lifetime;
  }

  lifetime.alloca_event = iree_profile_memory_find_queue_device_event(
      queue_device_index, allocation, allocation->first_submission_id,
      allocation->first_queue_ordinal,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA);
  lifetime.dealloca_event = iree_profile_memory_find_queue_device_event(
      queue_device_index, allocation, allocation->last_submission_id,
      allocation->last_queue_ordinal,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA);
  if (!iree_profile_memory_queue_device_event_is_valid(lifetime.alloca_event) ||
      !iree_profile_memory_queue_device_event_is_valid(
          lifetime.dealloca_event)) {
    return lifetime;
  }
  if (lifetime.dealloca_event->end_tick < lifetime.alloca_event->start_tick) {
    return lifetime;
  }

  lifetime.is_valid = true;
  lifetime.start_tick = lifetime.alloca_event->start_tick;
  lifetime.end_tick = lifetime.dealloca_event->end_tick;
  lifetime.duration_ticks = lifetime.end_tick - lifetime.start_tick;
  const iree_profile_model_device_t* device = iree_profile_model_find_device(
      queue_device_index ? queue_device_index->model : NULL,
      allocation->physical_device_ordinal);
  iree_profile_model_clock_fit_t clock_fit;
  if (device &&
      iree_profile_model_device_try_fit_clock_exact(
          device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
          &clock_fit)) {
    lifetime.has_duration_ns = iree_profile_model_clock_fit_scale_ticks_to_ns(
        &clock_fit, lifetime.duration_ticks, &lifetime.duration_ns);
  }
  return lifetime;
}

static iree_status_t iree_profile_memory_context_accumulate_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event, bool is_truncated,
    iree_profile_memory_event_callback_t event_callback) {
  ++context->matched_event_count;
  if (is_truncated) ++context->truncated_event_count;

  const bool close_materialization =
      iree_profile_memory_event_closes_materialization(context, event);
  iree_profile_memory_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_memory_get_device(
      context, event->physical_device_ordinal, &device));
  IREE_RETURN_IF_ERROR(iree_profile_memory_record_device_event(
      device, event, close_materialization));
  IREE_RETURN_IF_ERROR(iree_profile_memory_record_pool_event(
      context, event, close_materialization));
  IREE_RETURN_IF_ERROR(
      iree_profile_memory_record_pool_stats_event(context, event));
  IREE_RETURN_IF_ERROR(iree_profile_memory_record_allocation_event(
      context, event, close_materialization));

  if (event_callback.fn) {
    const iree_profile_memory_event_row_t event_row = {
        .event = event,
        .is_truncated = is_truncated,
    };
    return event_callback.fn(event_callback.user_data, &event_row);
  }
  return iree_ok_status();
}

iree_status_t iree_profile_memory_context_accumulate_record(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, iree_profile_memory_event_callback_t event_callback) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    return iree_ok_status();
  }

  const bool is_truncated = iree_any_bit_set(
      record->header.chunk_flags, IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_memory_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_memory_event_t event;
    memcpy(&event, typed_record.contents.data, sizeof(event));
    ++context->total_event_count;
    if (iree_profile_memory_event_matches(&event, id_filter, filter)) {
      status = iree_profile_memory_context_accumulate_event(
          context, &event, is_truncated, event_callback);
    }
  }
  return status;
}

static void iree_profile_memory_print_text_header(
    const iree_profile_memory_context_t* context,
    const iree_profile_memory_overview_t* overview, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "IREE HAL profile memory summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  if (id_filter >= 0) {
    fprintf(file, "id_filter: %" PRId64 "\n", id_filter);
  }
  fprintf(file,
          "events: total=%" PRIu64 " matched=%" PRIu64
          " truncated_matched=%" PRIu64 " devices=%" PRIhsz " pools=%" PRIhsz
          " allocation_lifecycles=%" PRIhsz " open_lifecycles=%" PRIu64
          " partial_closes=%" PRIu64 "\n",
          context->total_event_count, context->matched_event_count,
          context->truncated_event_count, context->device_count,
          context->pool_count, context->allocation_count,
          overview->open_allocation_lifecycle_count,
          overview->partial_lifecycle_close_count);
}

static void iree_profile_memory_print_text_devices(
    const iree_profile_memory_context_t* context, FILE* file) {
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    const iree_profile_memory_device_t* device = &context->devices[i];
    fprintf(file,
            "device[%u]: events=%" PRIu64 " slab_acquire/release=%" PRIu64
            "/%" PRIu64 " pool_reserve/materialize/release/wait=%" PRIu64
            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " queue_alloca/dealloca=%" PRIu64 "/%" PRIu64
            " buffer_allocate/free=%" PRIu64 "/%" PRIu64
            " buffer_import/unimport=%" PRIu64 "/%" PRIu64 "\n",
            device->physical_device_ordinal, device->event_count,
            device->slab_acquire_count, device->slab_release_count,
            device->pool_reserve_count, device->pool_materialize_count,
            device->pool_release_count, device->pool_wait_count,
            device->queue_alloca_count, device->queue_dealloca_count,
            device->buffer_allocate_count, device->buffer_free_count,
            device->buffer_import_count, device->buffer_unimport_count);
    iree_profile_memory_print_balance_text(file, "slab_provider_events",
                                           &device->slab_allocation_balance);
    iree_profile_memory_print_balance_text(file, "pool_reservations",
                                           &device->pool_reservation_balance);
    iree_profile_memory_print_balance_text(
        file, "pool_materializations", &device->pool_materialization_balance);
    iree_profile_memory_print_balance_text(file, "queue_inflight_allocations",
                                           &device->queue_inflight_balance);
    iree_profile_memory_print_balance_text(file, "buffer_allocations",
                                           &device->buffer_allocation_balance);
    iree_profile_memory_print_balance_text(file, "buffer_imports",
                                           &device->buffer_import_balance);
  }
}

static void iree_profile_memory_print_text_pools(
    const iree_profile_memory_context_t* context, FILE* file) {
  for (iree_host_size_t i = 0; i < context->pool_count; ++i) {
    const iree_profile_memory_pool_t* pool = &context->pools[i];
    fprintf(file,
            "pool[%s device=%u id=%" PRIu64 " memory_type=%" PRIu64
            "]: events=%" PRIu64 " waits=%" PRIu64 " materializes=%" PRIu64
            " open_at_end=%" PRIu64 " peak_open=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " opened_bytes=%" PRIu64 " closed_bytes=%" PRIu64
            " partial_closes=%" PRIu64 " partial_close_bytes=%" PRIu64,
            iree_profile_memory_lifecycle_kind_name(pool->kind),
            pool->physical_device_ordinal, pool->pool_id, pool->memory_type,
            pool->event_count, pool->wait_count, pool->materialize_count,
            pool->lifecycle_balance.current_count,
            pool->lifecycle_balance.high_water_count,
            pool->lifecycle_balance.current_bytes,
            pool->lifecycle_balance.high_water_bytes,
            pool->lifecycle_balance.total_open_bytes,
            pool->lifecycle_balance.total_close_bytes,
            pool->lifecycle_balance.partial_close_count,
            pool->lifecycle_balance.partial_close_bytes);
    if (pool->kind == IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION) {
      fprintf(file,
              " materialized_at_end=%" PRIu64
              " materialized_high_water_bytes=%" PRIu64,
              pool->materialization_balance.current_count,
              pool->materialization_balance.high_water_bytes);
    }
    if (pool->pool_stats_sample_count != 0) {
      fprintf(
          file,
          " pool_stats_samples=%" PRIu64 " pool_bytes_reserved=%" PRIu64
          " pool_bytes_reserved_high_water=%" PRIu64 " pool_bytes_free=%" PRIu64
          " pool_bytes_free_low_water=%" PRIu64 " pool_bytes_committed=%" PRIu64
          " pool_bytes_committed_high_water=%" PRIu64
          " pool_budget_limit=%" PRIu64
          " pool_reservations=%u"
          " pool_reservation_high_water=%u"
          " pool_slabs=%u"
          " pool_slab_high_water=%u",
          pool->pool_stats_sample_count, pool->pool_bytes_reserved,
          pool->pool_bytes_reserved_high_water, pool->pool_bytes_free,
          pool->pool_bytes_free_low_water, pool->pool_bytes_committed,
          pool->pool_bytes_committed_high_water, pool->pool_budget_limit,
          pool->pool_reservation_count, pool->pool_reservation_high_water_count,
          pool->pool_slab_count, pool->pool_slab_high_water_count);
    }
    fputc('\n', file);
  }
}

static void iree_profile_memory_print_text_allocations(
    const iree_profile_memory_context_t* context,
    const iree_profile_memory_queue_device_event_index_t* queue_device_index,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    const iree_profile_memory_allocation_t* allocation =
        &context->allocations[i];
    const int64_t duration_ns =
        allocation->last_host_time_ns >= allocation->first_host_time_ns
            ? allocation->last_host_time_ns - allocation->first_host_time_ns
            : 0;
    const iree_profile_memory_device_lifetime_t device_lifetime =
        iree_profile_memory_resolve_device_lifetime(queue_device_index,
                                                    allocation);
    fprintf(
        file,
        "allocation[%s device=%u id=%" PRIu64 " pool=%" PRIu64
        " backing=%" PRIu64 " externally_owned=%s]: events=%" PRIu64
        " waits=%" PRIu64 " materializes=%" PRIu64
        " open_at_end=%s current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
        " opened_bytes=%" PRIu64 " closed_bytes=%" PRIu64
        " partial_closes=%" PRIu64 " partial_close_bytes=%" PRIu64
        " first_event=%" PRIu64 " last_event=%" PRIu64 " duration_ns=%" PRId64
        "\n",
        iree_profile_memory_lifecycle_kind_name(allocation->kind),
        allocation->physical_device_ordinal, allocation->allocation_id,
        allocation->pool_id, allocation->backing_id,
        iree_all_bits_set(allocation->flags,
                          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED)
            ? "true"
            : "false",
        allocation->event_count, allocation->wait_count,
        allocation->materialize_count,
        iree_profile_memory_allocation_open_at_end(allocation) ? "true"
                                                               : "false",
        allocation->lifecycle_balance.current_bytes,
        allocation->lifecycle_balance.high_water_bytes,
        allocation->lifecycle_balance.total_open_bytes,
        allocation->lifecycle_balance.total_close_bytes,
        allocation->lifecycle_balance.partial_close_count,
        allocation->lifecycle_balance.partial_close_bytes,
        allocation->first_event_id, allocation->last_event_id, duration_ns);
    if (allocation->kind ==
        IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION) {
      fprintf(file,
              "  device_lifetime: available=%s alloca_event=%" PRIu64
              " dealloca_event=%" PRIu64 " duration_ticks=%" PRIu64,
              device_lifetime.is_valid ? "true" : "false",
              device_lifetime.alloca_event
                  ? device_lifetime.alloca_event->event_id
                  : 0,
              device_lifetime.dealloca_event
                  ? device_lifetime.dealloca_event->event_id
                  : 0,
              device_lifetime.duration_ticks);
      if (device_lifetime.has_duration_ns) {
        fprintf(file, " duration_ns=%" PRId64, device_lifetime.duration_ns);
      }
      fputc('\n', file);
    }
  }
}

static iree_status_t iree_profile_memory_print_text(
    const iree_profile_memory_context_t* context,
    const iree_profile_memory_queue_device_event_index_t* queue_device_index,
    iree_string_view_t filter, int64_t id_filter, FILE* file) {
  const iree_profile_memory_overview_t overview =
      iree_profile_memory_compute_overview(context);
  iree_profile_memory_print_text_header(context, &overview, filter, id_filter,
                                        file);
  iree_profile_memory_print_text_devices(context, file);
  iree_profile_memory_print_text_pools(context, file);
  if (id_filter >= 0) {
    iree_profile_memory_print_text_allocations(context, queue_device_index,
                                               file);
  }
  return iree_ok_status();
}

static void iree_profile_memory_print_jsonl_header(
    const iree_profile_memory_context_t* context,
    const iree_profile_memory_overview_t* overview, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "{\"type\":\"memory_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"id_filter\":%" PRId64 ",\"total_events\":%" PRIu64
          ",\"matched_events\":%" PRIu64
          ",\"truncated_matched_events\":%" PRIu64 ",\"devices\":%" PRIhsz
          ",\"pools\":%" PRIhsz ",\"allocation_lifecycles\":%" PRIhsz
          ",\"open_allocation_lifecycles\":%" PRIu64
          ",\"partial_lifecycle_closes\":%" PRIu64 "}\n",
          id_filter, context->total_event_count, context->matched_event_count,
          context->truncated_event_count, context->device_count,
          context->pool_count, context->allocation_count,
          overview->open_allocation_lifecycle_count,
          overview->partial_lifecycle_close_count);
}

static void iree_profile_memory_print_jsonl_devices(
    const iree_profile_memory_context_t* context, FILE* file) {
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    const iree_profile_memory_device_t* device = &context->devices[i];
    fprintf(file,
            "{\"type\":\"memory_device\",\"physical_device_ordinal\":%u"
            ",\"events\":%" PRIu64 ",\"slab_acquires\":%" PRIu64
            ",\"slab_releases\":%" PRIu64 ",\"pool_reserves\":%" PRIu64
            ",\"pool_materializes\":%" PRIu64 ",\"pool_releases\":%" PRIu64
            ",\"pool_waits\":%" PRIu64 ",\"queue_allocas\":%" PRIu64
            ",\"queue_deallocas\":%" PRIu64 ",\"buffer_allocates\":%" PRIu64
            ",\"buffer_frees\":%" PRIu64 ",\"buffer_imports\":%" PRIu64
            ",\"buffer_unimports\":%" PRIu64,
            device->physical_device_ordinal, device->event_count,
            device->slab_acquire_count, device->slab_release_count,
            device->pool_reserve_count, device->pool_materialize_count,
            device->pool_release_count, device->pool_wait_count,
            device->queue_alloca_count, device->queue_dealloca_count,
            device->buffer_allocate_count, device->buffer_free_count,
            device->buffer_import_count, device->buffer_unimport_count);
    iree_profile_memory_fprint_balance_json_fields(
        file, "slab", &device->slab_allocation_balance);
    iree_profile_memory_fprint_balance_json_fields(
        file, "pool_reserved", &device->pool_reservation_balance);
    iree_profile_memory_fprint_balance_json_fields(
        file, "pool_materialized", &device->pool_materialization_balance);
    iree_profile_memory_fprint_balance_json_fields(
        file, "queue_inflight", &device->queue_inflight_balance);
    iree_profile_memory_fprint_balance_json_fields(
        file, "buffer", &device->buffer_allocation_balance);
    iree_profile_memory_fprint_balance_json_fields(
        file, "imported_buffer", &device->buffer_import_balance);
    fprintf(file, "}\n");
  }
}

static void iree_profile_memory_print_jsonl_pools(
    const iree_profile_memory_context_t* context, FILE* file) {
  for (iree_host_size_t i = 0; i < context->pool_count; ++i) {
    const iree_profile_memory_pool_t* pool = &context->pools[i];
    fprintf(file, "{\"type\":\"memory_pool\",\"kind\":");
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_memory_lifecycle_kind_name(pool->kind)));
    fprintf(file,
            ",\"physical_device_ordinal\":%u,\"pool_id\":%" PRIu64
            ",\"memory_type\":%" PRIu64 ",\"buffer_usage\":%" PRIu64
            ",\"events\":%" PRIu64 ",\"waits\":%" PRIu64
            ",\"materializes\":%" PRIu64,
            pool->physical_device_ordinal, pool->pool_id, pool->memory_type,
            pool->buffer_usage, pool->event_count, pool->wait_count,
            pool->materialize_count);
    iree_profile_memory_fprint_balance_json_fields(file, "lifecycle",
                                                   &pool->lifecycle_balance);
    iree_profile_memory_fprint_balance_json_fields(
        file, "materialized", &pool->materialization_balance);
    fprintf(
        file,
        ",\"pool_stats_available\":%s"
        ",\"pool_stats_samples\":%" PRIu64 ",\"pool_bytes_reserved\":%" PRIu64
        ",\"pool_bytes_reserved_high_water\":%" PRIu64
        ",\"pool_bytes_free\":%" PRIu64
        ",\"pool_bytes_free_low_water\":%" PRIu64
        ",\"pool_bytes_committed\":%" PRIu64
        ",\"pool_bytes_committed_high_water\":%" PRIu64
        ",\"pool_budget_limit\":%" PRIu64
        ",\"pool_reservation_count\":%u"
        ",\"pool_reservation_high_water_count\":%u"
        ",\"pool_slab_count\":%u"
        ",\"pool_slab_high_water_count\":%u",
        pool->pool_stats_sample_count != 0 ? "true" : "false",
        pool->pool_stats_sample_count, pool->pool_bytes_reserved,
        pool->pool_bytes_reserved_high_water, pool->pool_bytes_free,
        pool->pool_bytes_free_low_water, pool->pool_bytes_committed,
        pool->pool_bytes_committed_high_water, pool->pool_budget_limit,
        pool->pool_reservation_count, pool->pool_reservation_high_water_count,
        pool->pool_slab_count, pool->pool_slab_high_water_count);
    fprintf(file, "}\n");
  }
}

static void iree_profile_memory_print_jsonl_allocations(
    const iree_profile_memory_context_t* context,
    const iree_profile_memory_queue_device_event_index_t* queue_device_index,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    const iree_profile_memory_allocation_t* allocation =
        &context->allocations[i];
    const int64_t duration_ns =
        allocation->last_host_time_ns >= allocation->first_host_time_ns
            ? allocation->last_host_time_ns - allocation->first_host_time_ns
            : 0;
    const iree_profile_memory_device_lifetime_t device_lifetime =
        iree_profile_memory_resolve_device_lifetime(queue_device_index,
                                                    allocation);
    fprintf(file, "{\"type\":\"memory_allocation\",\"kind\":");
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_memory_lifecycle_kind_name(allocation->kind)));
    fprintf(
        file,
        ",\"physical_device_ordinal\":%u,\"allocation_id\":%" PRIu64
        ",\"pool_id\":%" PRIu64 ",\"backing_id\":%" PRIu64
        ",\"externally_owned\":%s"
        ",\"memory_type\":%" PRIu64 ",\"buffer_usage\":%" PRIu64
        ",\"events\":%" PRIu64 ",\"waits\":%" PRIu64
        ",\"materializes\":%" PRIu64
        ",\"open_at_end\":%s"
        ",\"first_event_id\":%" PRIu64 ",\"last_event_id\":%" PRIu64
        ",\"first_host_time_ns\":%" PRId64 ",\"last_host_time_ns\":%" PRId64
        ",\"host_time_domain\":\"iree_host_time_ns\""
        ",\"duration_ns\":%" PRId64 ",\"first_submission_id\":%" PRIu64
        ",\"last_submission_id\":%" PRIu64
        ",\"first_queue_ordinal\":%u,\"last_queue_ordinal\":%u"
        ",\"alloca_device_event_id\":%" PRIu64
        ",\"dealloca_device_event_id\":%" PRIu64
        ",\"device_lifetime_available\":%s"
        ",\"device_lifetime_start_tick\":%" PRIu64
        ",\"device_lifetime_end_tick\":%" PRIu64
        ",\"device_lifetime_duration_ticks\":%" PRIu64
        ",\"device_lifetime_time_ns_available\":%s"
        ",\"device_lifetime_duration_ns\":%" PRId64,
        allocation->physical_device_ordinal, allocation->allocation_id,
        allocation->pool_id, allocation->backing_id,
        iree_all_bits_set(allocation->flags,
                          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED)
            ? "true"
            : "false",
        allocation->memory_type, allocation->buffer_usage,
        allocation->event_count, allocation->wait_count,
        allocation->materialize_count,
        iree_profile_memory_allocation_open_at_end(allocation) ? "true"
                                                               : "false",
        allocation->first_event_id, allocation->last_event_id,
        allocation->first_host_time_ns, allocation->last_host_time_ns,
        duration_ns, allocation->first_submission_id,
        allocation->last_submission_id, allocation->first_queue_ordinal,
        allocation->last_queue_ordinal,
        device_lifetime.alloca_event ? device_lifetime.alloca_event->event_id
                                     : 0,
        device_lifetime.dealloca_event
            ? device_lifetime.dealloca_event->event_id
            : 0,
        device_lifetime.is_valid ? "true" : "false", device_lifetime.start_tick,
        device_lifetime.end_tick, device_lifetime.duration_ticks,
        device_lifetime.has_duration_ns ? "true" : "false",
        device_lifetime.has_duration_ns ? device_lifetime.duration_ns : 0);
    iree_profile_memory_fprint_balance_json_fields(
        file, "lifecycle", &allocation->lifecycle_balance);
    iree_profile_memory_fprint_balance_json_fields(
        file, "materialized", &allocation->materialization_balance);
    fprintf(file, "}\n");
  }
}

static void iree_profile_memory_print_jsonl_summary(
    const iree_profile_memory_context_t* context,
    const iree_profile_memory_queue_device_event_index_t* queue_device_index,
    iree_string_view_t filter, int64_t id_filter, FILE* file) {
  const iree_profile_memory_overview_t overview =
      iree_profile_memory_compute_overview(context);
  iree_profile_memory_print_jsonl_header(context, &overview, filter, id_filter,
                                         file);
  iree_profile_memory_print_jsonl_devices(context, file);
  iree_profile_memory_print_jsonl_pools(context, file);
  iree_profile_memory_print_jsonl_allocations(context, queue_device_index,
                                              file);
}

typedef struct iree_profile_memory_parse_context_t {
  // Aggregation state populated by memory-event records.
  iree_profile_memory_context_t* memory_context;
  // Metadata used to validate queues and scale device ticks.
  iree_profile_model_t* model;
  // Compact queue-device event index used by memory lifetime joins.
  iree_profile_memory_queue_device_event_index_t* queue_device_index;
  // Optional glob filter applied to memory event rows.
  iree_string_view_t filter;
  // Optional event/allocation identifier filter, or -1 when disabled.
  int64_t id_filter;
  // Optional callback receiving matched raw memory events.
  iree_profile_memory_event_callback_t event_callback;
  // True when queue-device rows are needed for allocation lifetime joins.
  bool capture_queue_device_events;
} iree_profile_memory_parse_context_t;

static iree_status_t iree_profile_memory_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_memory_parse_context_t* context =
      (iree_profile_memory_parse_context_t*)user_data;
  if (context->capture_queue_device_events) {
    IREE_RETURN_IF_ERROR(
        iree_profile_model_process_metadata_record(context->model, record));
    IREE_RETURN_IF_ERROR(
        iree_profile_memory_queue_device_event_index_process_record(
            context->queue_device_index, record));
  }
  return iree_profile_memory_context_accumulate_record(
      context->memory_context, record, context->filter, context->id_filter,
      context->event_callback);
}

iree_status_t iree_profile_memory_report_file(iree_string_view_t path,
                                              iree_string_view_t format,
                                              iree_string_view_t filter,
                                              int64_t id_filter, FILE* file,
                                              iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, host_allocator, &profile_file));

  const bool capture_queue_device_events = is_jsonl || id_filter >= 0;
  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(host_allocator, &context);
  iree_profile_model_t model;
  iree_profile_model_initialize(host_allocator, &model);
  iree_profile_memory_queue_device_event_index_t queue_device_index;
  iree_profile_memory_queue_device_event_index_initialize(
      &model, host_allocator, &queue_device_index);
  const iree_profile_memory_event_callback_t event_callback = {
      .fn = is_jsonl ? iree_profile_memory_emit_event_jsonl : NULL,
      .user_data = file,
  };
  iree_profile_memory_parse_context_t parse_context = {
      .memory_context = &context,
      .model = &model,
      .queue_device_index = &queue_device_index,
      .filter = filter,
      .id_filter = id_filter,
      .event_callback = event_callback,
      .capture_queue_device_events = capture_queue_device_events,
  };
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_memory_record,
      .user_data = &parse_context,
  };
  iree_status_t status =
      iree_profile_file_for_each_record(&profile_file, record_callback);

  if (iree_status_is_ok(status) && capture_queue_device_events) {
    iree_profile_memory_queue_device_event_index_sort(&queue_device_index);
  }
  if (iree_status_is_ok(status)) {
    if (is_text) {
      status = iree_profile_memory_print_text(&context, &queue_device_index,
                                              filter, id_filter, file);
    } else {
      iree_profile_memory_print_jsonl_summary(&context, &queue_device_index,
                                              filter, id_filter, file);
    }
  }

  iree_profile_memory_queue_device_event_index_deinitialize(
      &queue_device_index);
  iree_profile_model_deinitialize(&model);
  iree_profile_memory_context_deinitialize(&context);
  iree_profile_file_close(&profile_file);
  return status;
}
