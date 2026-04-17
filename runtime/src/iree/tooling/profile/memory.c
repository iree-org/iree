// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/memory.h"

#include <string.h>

#include "iree/tooling/profile/dispatch.h"
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
    default:
      return false;
  }
}

static bool iree_profile_memory_event_increases_live_bytes(
    const iree_hal_profile_memory_event_t* event) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      return iree_all_bits_set(
          event->flags, IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION);
    default:
      return false;
  }
}

static bool iree_profile_memory_event_decreases_live_bytes(
    const iree_hal_profile_memory_event_t* event) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
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
  *out_allocation = allocation;
  return iree_ok_status();
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
  return iree_profile_dispatch_key_matches(type_name, filter);
}

static void iree_profile_memory_record_event(
    iree_profile_memory_device_t* device,
    const iree_hal_profile_memory_event_t* event) {
  ++device->event_count;
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      ++device->slab_acquire_count;
      ++device->current_slab_allocation_count;
      device->high_water_slab_allocation_count =
          iree_max(device->high_water_slab_allocation_count,
                   device->current_slab_allocation_count);
      device->total_slab_acquired_bytes += event->length;
      device->current_slab_bytes += event->length;
      device->high_water_slab_bytes =
          iree_max(device->high_water_slab_bytes, device->current_slab_bytes);
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      ++device->slab_release_count;
      device->current_slab_allocation_count =
          device->current_slab_allocation_count == 0
              ? 0
              : device->current_slab_allocation_count - 1;
      device->total_slab_released_bytes += event->length;
      device->current_slab_bytes =
          event->length > device->current_slab_bytes
              ? 0
              : device->current_slab_bytes - event->length;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      ++device->pool_reserve_count;
      if (iree_all_bits_set(
              event->flags,
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION)) {
        ++device->current_pool_reservation_count;
        device->high_water_pool_reservation_count =
            iree_max(device->high_water_pool_reservation_count,
                     device->current_pool_reservation_count);
        device->total_pool_reserved_bytes += event->length;
        device->current_pool_reserved_bytes += event->length;
        device->high_water_pool_reserved_bytes =
            iree_max(device->high_water_pool_reserved_bytes,
                     device->current_pool_reserved_bytes);
      }
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      ++device->pool_materialize_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      ++device->pool_release_count;
      device->current_pool_reservation_count =
          device->current_pool_reservation_count == 0
              ? 0
              : device->current_pool_reservation_count - 1;
      device->total_pool_released_bytes += event->length;
      device->current_pool_reserved_bytes =
          event->length > device->current_pool_reserved_bytes
              ? 0
              : device->current_pool_reserved_bytes - event->length;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      ++device->pool_wait_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      ++device->buffer_allocate_count;
      ++device->current_buffer_allocation_count;
      device->high_water_buffer_allocation_count =
          iree_max(device->high_water_buffer_allocation_count,
                   device->current_buffer_allocation_count);
      device->total_buffer_allocate_bytes += event->length;
      device->current_buffer_bytes += event->length;
      device->high_water_buffer_bytes = iree_max(
          device->high_water_buffer_bytes, device->current_buffer_bytes);
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      ++device->buffer_free_count;
      device->current_buffer_allocation_count =
          device->current_buffer_allocation_count == 0
              ? 0
              : device->current_buffer_allocation_count - 1;
      device->total_buffer_free_bytes += event->length;
      device->current_buffer_bytes =
          event->length > device->current_buffer_bytes
              ? 0
              : device->current_buffer_bytes - event->length;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      ++device->queue_alloca_count;
      ++device->current_queue_allocation_count;
      device->high_water_queue_allocation_count =
          iree_max(device->high_water_queue_allocation_count,
                   device->current_queue_allocation_count);
      device->total_queue_alloca_bytes += event->length;
      device->current_queue_bytes += event->length;
      device->high_water_queue_bytes =
          iree_max(device->high_water_queue_bytes, device->current_queue_bytes);
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      ++device->queue_dealloca_count;
      device->current_queue_allocation_count =
          device->current_queue_allocation_count == 0
              ? 0
              : device->current_queue_allocation_count - 1;
      device->total_queue_dealloca_bytes += event->length;
      device->current_queue_bytes =
          event->length > device->current_queue_bytes
              ? 0
              : device->current_queue_bytes - event->length;
      break;
    default:
      break;
  }
}

static void iree_profile_memory_apply_live_increase(
    uint64_t length, uint64_t* current_count, uint64_t* high_water_count,
    uint64_t* current_bytes, uint64_t* high_water_bytes,
    uint64_t* total_allocate_bytes) {
  *current_count += 1;
  *high_water_count = iree_max(*high_water_count, *current_count);
  *current_bytes += length;
  *high_water_bytes = iree_max(*high_water_bytes, *current_bytes);
  *total_allocate_bytes += length;
}

static void iree_profile_memory_apply_live_decrease(
    uint64_t length, uint64_t* current_count, uint64_t* current_bytes,
    uint64_t* total_free_bytes) {
  *current_count = *current_count == 0 ? 0 : *current_count - 1;
  *current_bytes = length > *current_bytes ? 0 : *current_bytes - length;
  *total_free_bytes += length;
}

static iree_status_t iree_profile_memory_record_pool_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event) {
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

  if (iree_profile_memory_event_increases_live_bytes(event)) {
    iree_profile_memory_apply_live_increase(
        event->length, &pool->current_allocation_count,
        &pool->high_water_allocation_count, &pool->current_bytes,
        &pool->high_water_bytes, &pool->total_allocate_bytes);
  } else if (iree_profile_memory_event_decreases_live_bytes(event)) {
    iree_profile_memory_apply_live_decrease(
        event->length, &pool->current_allocation_count, &pool->current_bytes,
        &pool->total_free_bytes);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_record_allocation_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event) {
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
  }
  if (event->submission_id != 0) {
    allocation->last_submission_id = event->submission_id;
  }
  allocation->backing_id =
      allocation->backing_id ? allocation->backing_id : event->backing_id;
  allocation->memory_type |= event->memory_type;
  allocation->buffer_usage |= event->buffer_usage;
  ++allocation->event_count;
  if (event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT) {
    ++allocation->wait_count;
  } else if (event->type ==
             IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE) {
    ++allocation->materialize_count;
  }

  if (iree_profile_memory_event_increases_live_bytes(event)) {
    allocation->total_allocate_bytes += event->length;
    allocation->current_bytes += event->length;
    allocation->high_water_bytes =
        iree_max(allocation->high_water_bytes, allocation->current_bytes);
  } else if (iree_profile_memory_event_decreases_live_bytes(event)) {
    allocation->total_free_bytes += event->length;
    allocation->current_bytes = event->length > allocation->current_bytes
                                    ? 0
                                    : allocation->current_bytes - event->length;
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
          ",\"length\":%" PRIu64 ",\"alignment\":%" PRIu64 "}\n",
          event->type, event->flags, event->result, event->host_time_ns,
          event->allocation_id, event->pool_id, event->backing_id,
          event->submission_id, event->physical_device_ordinal,
          event->queue_ordinal, event->frontier_entry_count, event->memory_type,
          event->buffer_usage, event->offset, event->length, event->alignment);
}

iree_status_t iree_profile_memory_process_event_records(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, bool emit_events, FILE* file) {
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
      ++context->matched_event_count;
      if (is_truncated) ++context->truncated_event_count;
      iree_profile_memory_device_t* device = NULL;
      status = iree_profile_memory_get_device(
          context, event.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        iree_profile_memory_record_event(device, &event);
        status = iree_profile_memory_record_pool_event(context, &event);
      }
      if (iree_status_is_ok(status)) {
        status = iree_profile_memory_record_allocation_event(context, &event);
      }
      if (iree_status_is_ok(status)) {
        if (emit_events) {
          iree_profile_memory_print_event_jsonl(&event, file);
        }
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_memory_print_text(
    const iree_profile_memory_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  uint64_t live_allocation_count = 0;
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    if (context->allocations[i].current_bytes != 0) {
      ++live_allocation_count;
    }
  }

  fprintf(file, "IREE HAL profile memory summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  if (id_filter >= 0) {
    fprintf(file, "id_filter: %" PRId64 "\n", id_filter);
  }
  fprintf(file,
          "events: total=%" PRIu64 " matched=%" PRIu64
          " truncated_matched=%" PRIu64 " devices=%" PRIhsz " pools=%" PRIhsz
          " allocation_lifecycles=%" PRIhsz " live_lifecycles=%" PRIu64 "\n",
          context->total_event_count, context->matched_event_count,
          context->truncated_event_count, context->device_count,
          context->pool_count, context->allocation_count,
          live_allocation_count);
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    const iree_profile_memory_device_t* device = &context->devices[i];
    fprintf(file,
            "device[%u]: events=%" PRIu64 " slab_acquire/release=%" PRIu64
            "/%" PRIu64 " pool_reserve/materialize/release/wait=%" PRIu64
            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " queue_alloca/dealloca=%" PRIu64 "/%" PRIu64
            " buffer_allocate/free=%" PRIu64 "/%" PRIu64 "\n",
            device->physical_device_ordinal, device->event_count,
            device->slab_acquire_count, device->slab_release_count,
            device->pool_reserve_count, device->pool_materialize_count,
            device->pool_release_count, device->pool_wait_count,
            device->queue_alloca_count, device->queue_dealloca_count,
            device->buffer_allocate_count, device->buffer_free_count);
    fprintf(file,
            "  slab_provider_events: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " acquired_bytes=%" PRIu64 " released_bytes=%" PRIu64 "\n",
            device->current_slab_allocation_count,
            device->high_water_slab_allocation_count,
            device->current_slab_bytes, device->high_water_slab_bytes,
            device->total_slab_acquired_bytes,
            device->total_slab_released_bytes);
    fprintf(file,
            "  pool_reservations: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " reserved_bytes=%" PRIu64 " released_bytes=%" PRIu64 "\n",
            device->current_pool_reservation_count,
            device->high_water_pool_reservation_count,
            device->current_pool_reserved_bytes,
            device->high_water_pool_reserved_bytes,
            device->total_pool_reserved_bytes,
            device->total_pool_released_bytes);
    fprintf(file,
            "  queue_allocations: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " alloca_bytes=%" PRIu64 " dealloca_bytes=%" PRIu64 "\n",
            device->current_queue_allocation_count,
            device->high_water_queue_allocation_count,
            device->current_queue_bytes, device->high_water_queue_bytes,
            device->total_queue_alloca_bytes,
            device->total_queue_dealloca_bytes);
    fprintf(file,
            "  buffer_allocations: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " allocate_bytes=%" PRIu64 " free_bytes=%" PRIu64 "\n",
            device->current_buffer_allocation_count,
            device->high_water_buffer_allocation_count,
            device->current_buffer_bytes, device->high_water_buffer_bytes,
            device->total_buffer_allocate_bytes,
            device->total_buffer_free_bytes);
  }
  for (iree_host_size_t i = 0; i < context->pool_count; ++i) {
    const iree_profile_memory_pool_t* pool = &context->pools[i];
    fprintf(file,
            "pool[%s device=%u id=%" PRIu64 " memory_type=%" PRIu64
            "]: events=%" PRIu64 " waits=%" PRIu64 " materializes=%" PRIu64
            " live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " allocate_bytes=%" PRIu64 " free_bytes=%" PRIu64 "\n",
            iree_profile_memory_lifecycle_kind_name(pool->kind),
            pool->physical_device_ordinal, pool->pool_id, pool->memory_type,
            pool->event_count, pool->wait_count, pool->materialize_count,
            pool->current_allocation_count, pool->high_water_allocation_count,
            pool->current_bytes, pool->high_water_bytes,
            pool->total_allocate_bytes, pool->total_free_bytes);
  }
  if (id_filter >= 0) {
    for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
      const iree_profile_memory_allocation_t* allocation =
          &context->allocations[i];
      const int64_t duration_ns =
          allocation->last_host_time_ns >= allocation->first_host_time_ns
              ? allocation->last_host_time_ns - allocation->first_host_time_ns
              : 0;
      fprintf(
          file,
          "allocation[%s device=%u id=%" PRIu64 " pool=%" PRIu64
          " backing=%" PRIu64 "]: events=%" PRIu64 " waits=%" PRIu64
          " materializes=%" PRIu64 " live_at_end=%s current_bytes=%" PRIu64
          " high_water_bytes=%" PRIu64 " allocate_bytes=%" PRIu64
          " free_bytes=%" PRIu64 " first_event=%" PRIu64 " last_event=%" PRIu64
          " duration_ns=%" PRId64 "\n",
          iree_profile_memory_lifecycle_kind_name(allocation->kind),
          allocation->physical_device_ordinal, allocation->allocation_id,
          allocation->pool_id, allocation->backing_id, allocation->event_count,
          allocation->wait_count, allocation->materialize_count,
          allocation->current_bytes != 0 ? "true" : "false",
          allocation->current_bytes, allocation->high_water_bytes,
          allocation->total_allocate_bytes, allocation->total_free_bytes,
          allocation->first_event_id, allocation->last_event_id, duration_ns);
    }
  }
  return iree_ok_status();
}

static void iree_profile_memory_print_jsonl_summary(
    const iree_profile_memory_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  uint64_t live_allocation_count = 0;
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    if (context->allocations[i].current_bytes != 0) {
      ++live_allocation_count;
    }
  }

  fprintf(file, "{\"type\":\"memory_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"id_filter\":%" PRId64 ",\"total_events\":%" PRIu64
          ",\"matched_events\":%" PRIu64
          ",\"truncated_matched_events\":%" PRIu64 ",\"devices\":%" PRIhsz
          ",\"pools\":%" PRIhsz ",\"allocation_lifecycles\":%" PRIhsz
          ",\"live_allocation_lifecycles\":%" PRIu64 "}\n",
          id_filter, context->total_event_count, context->matched_event_count,
          context->truncated_event_count, context->device_count,
          context->pool_count, context->allocation_count,
          live_allocation_count);
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    const iree_profile_memory_device_t* device = &context->devices[i];
    fprintf(
        file,
        "{\"type\":\"memory_device\",\"physical_device_ordinal\":%u"
        ",\"events\":%" PRIu64 ",\"slab_acquires\":%" PRIu64
        ",\"slab_releases\":%" PRIu64 ",\"pool_reserves\":%" PRIu64
        ",\"pool_materializes\":%" PRIu64 ",\"pool_releases\":%" PRIu64
        ",\"pool_waits\":%" PRIu64 ",\"queue_allocas\":%" PRIu64
        ",\"queue_deallocas\":%" PRIu64 ",\"buffer_allocates\":%" PRIu64
        ",\"buffer_frees\":%" PRIu64 ",\"current_slab_allocations\":%" PRIu64
        ",\"high_water_slab_allocations\":%" PRIu64
        ",\"current_slab_bytes\":%" PRIu64 ",\"high_water_slab_bytes\":%" PRIu64
        ",\"total_slab_acquired_bytes\":%" PRIu64
        ",\"total_slab_released_bytes\":%" PRIu64
        ",\"current_pool_reservations\":%" PRIu64
        ",\"high_water_pool_reservations\":%" PRIu64
        ",\"current_pool_reserved_bytes\":%" PRIu64
        ",\"high_water_pool_reserved_bytes\":%" PRIu64
        ",\"total_pool_reserved_bytes\":%" PRIu64
        ",\"total_pool_released_bytes\":%" PRIu64
        ",\"current_queue_allocations\":%" PRIu64
        ",\"high_water_queue_allocations\":%" PRIu64
        ",\"current_queue_bytes\":%" PRIu64
        ",\"high_water_queue_bytes\":%" PRIu64
        ",\"total_queue_alloca_bytes\":%" PRIu64
        ",\"total_queue_dealloca_bytes\":%" PRIu64
        ",\"current_buffer_allocations\":%" PRIu64
        ",\"high_water_buffer_allocations\":%" PRIu64
        ",\"current_buffer_bytes\":%" PRIu64
        ",\"high_water_buffer_bytes\":%" PRIu64
        ",\"total_buffer_allocate_bytes\":%" PRIu64
        ",\"total_buffer_free_bytes\":%" PRIu64 "}\n",
        device->physical_device_ordinal, device->event_count,
        device->slab_acquire_count, device->slab_release_count,
        device->pool_reserve_count, device->pool_materialize_count,
        device->pool_release_count, device->pool_wait_count,
        device->queue_alloca_count, device->queue_dealloca_count,
        device->buffer_allocate_count, device->buffer_free_count,
        device->current_slab_allocation_count,
        device->high_water_slab_allocation_count, device->current_slab_bytes,
        device->high_water_slab_bytes, device->total_slab_acquired_bytes,
        device->total_slab_released_bytes,
        device->current_pool_reservation_count,
        device->high_water_pool_reservation_count,
        device->current_pool_reserved_bytes,
        device->high_water_pool_reserved_bytes,
        device->total_pool_reserved_bytes, device->total_pool_released_bytes,
        device->current_queue_allocation_count,
        device->high_water_queue_allocation_count, device->current_queue_bytes,
        device->high_water_queue_bytes, device->total_queue_alloca_bytes,
        device->total_queue_dealloca_bytes,
        device->current_buffer_allocation_count,
        device->high_water_buffer_allocation_count,
        device->current_buffer_bytes, device->high_water_buffer_bytes,
        device->total_buffer_allocate_bytes, device->total_buffer_free_bytes);
  }
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
            ",\"materializes\":%" PRIu64 ",\"current_allocations\":%" PRIu64
            ",\"high_water_allocations\":%" PRIu64 ",\"current_bytes\":%" PRIu64
            ",\"high_water_bytes\":%" PRIu64
            ",\"total_allocate_bytes\":%" PRIu64
            ",\"total_free_bytes\":%" PRIu64 "}\n",
            pool->physical_device_ordinal, pool->pool_id, pool->memory_type,
            pool->buffer_usage, pool->event_count, pool->wait_count,
            pool->materialize_count, pool->current_allocation_count,
            pool->high_water_allocation_count, pool->current_bytes,
            pool->high_water_bytes, pool->total_allocate_bytes,
            pool->total_free_bytes);
  }
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    const iree_profile_memory_allocation_t* allocation =
        &context->allocations[i];
    const int64_t duration_ns =
        allocation->last_host_time_ns >= allocation->first_host_time_ns
            ? allocation->last_host_time_ns - allocation->first_host_time_ns
            : 0;
    fprintf(file, "{\"type\":\"memory_allocation\",\"kind\":");
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_memory_lifecycle_kind_name(allocation->kind)));
    fprintf(
        file,
        ",\"physical_device_ordinal\":%u,\"allocation_id\":%" PRIu64
        ",\"pool_id\":%" PRIu64 ",\"backing_id\":%" PRIu64
        ",\"memory_type\":%" PRIu64 ",\"buffer_usage\":%" PRIu64
        ",\"events\":%" PRIu64 ",\"waits\":%" PRIu64
        ",\"materializes\":%" PRIu64
        ",\"live_at_end\":%s"
        ",\"current_bytes\":%" PRIu64 ",\"high_water_bytes\":%" PRIu64
        ",\"total_allocate_bytes\":%" PRIu64 ",\"total_free_bytes\":%" PRIu64
        ",\"first_event_id\":%" PRIu64 ",\"last_event_id\":%" PRIu64
        ",\"first_host_time_ns\":%" PRId64 ",\"last_host_time_ns\":%" PRId64
        ",\"host_time_domain\":\"iree_host_time_ns\""
        ",\"duration_ns\":%" PRId64 ",\"first_submission_id\":%" PRIu64
        ",\"last_submission_id\":%" PRIu64 "}\n",
        allocation->physical_device_ordinal, allocation->allocation_id,
        allocation->pool_id, allocation->backing_id, allocation->memory_type,
        allocation->buffer_usage, allocation->event_count,
        allocation->wait_count, allocation->materialize_count,
        allocation->current_bytes != 0 ? "true" : "false",
        allocation->current_bytes, allocation->high_water_bytes,
        allocation->total_allocate_bytes, allocation->total_free_bytes,
        allocation->first_event_id, allocation->last_event_id,
        allocation->first_host_time_ns, allocation->last_host_time_ns,
        duration_ns, allocation->first_submission_id,
        allocation->last_submission_id);
  }
}

typedef struct iree_profile_memory_parse_context_t {
  // Aggregation state populated by memory-event records.
  iree_profile_memory_context_t* memory_context;
  // Optional glob filter applied to memory event rows.
  iree_string_view_t filter;
  // Optional event/allocation identifier filter, or -1 when disabled.
  int64_t id_filter;
  // True when raw event rows should be streamed while parsing.
  bool emit_events;
  // Output stream receiving raw event rows when enabled.
  FILE* file;
} iree_profile_memory_parse_context_t;

static iree_status_t iree_profile_memory_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_memory_parse_context_t* context =
      (iree_profile_memory_parse_context_t*)user_data;
  return iree_profile_memory_process_event_records(
      context->memory_context, record, context->filter, context->id_filter,
      context->emit_events, context->file);
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

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(host_allocator, &context);
  iree_profile_memory_parse_context_t parse_context = {
      .memory_context = &context,
      .filter = filter,
      .id_filter = id_filter,
      .emit_events = is_jsonl,
      .file = file,
  };
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_memory_record,
      .user_data = &parse_context,
  };
  iree_status_t status =
      iree_profile_file_for_each_record(&profile_file, record_callback);

  if (iree_status_is_ok(status)) {
    if (is_text) {
      status =
          iree_profile_memory_print_text(&context, filter, id_filter, file);
    } else {
      iree_profile_memory_print_jsonl_summary(&context, filter, id_filter,
                                              file);
    }
  }

  iree_profile_memory_context_deinitialize(&context);
  iree_profile_file_close(&profile_file);
  return status;
}
