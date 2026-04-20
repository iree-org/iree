// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_allocator.h"

#include <string.h>

#include "iree/hal/replay/recorder_buffer.h"
#include "iree/hal/replay/recorder_record.h"

#define IREE_HAL_REPLAY_VTABLE_DISPATCH(resource, type_prefix, method_name) \
  ((const type_prefix##_vtable_t*)((const iree_hal_resource_t*)(resource))  \
       ->vtable)                                                            \
      ->method_name

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_allocator_t {
  // HAL resource header for the recording wrapper allocator.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying allocator receiving forwarded HAL calls.
  iree_hal_allocator_t* base_allocator;
  // Device used when preserving placement metadata on wrapped buffers.
  iree_hal_device_t* placement_device;
  // Session-local device object id associated with this allocator.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this allocator.
  iree_hal_replay_object_id_t allocator_id;
} iree_hal_replay_recorder_allocator_t;

static const iree_hal_allocator_vtable_t
    iree_hal_replay_recorder_allocator_vtable;

static iree_hal_replay_recorder_allocator_t*
iree_hal_replay_recorder_allocator_cast(iree_hal_allocator_t* base_allocator) {
  IREE_HAL_ASSERT_TYPE(base_allocator,
                       &iree_hal_replay_recorder_allocator_vtable);
  return (iree_hal_replay_recorder_allocator_t*)base_allocator;
}

static iree_hal_allocator_t* iree_hal_replay_recorder_allocator_base_or_self(
    iree_hal_allocator_t* allocator) {
  return iree_hal_resource_is(allocator,
                              &iree_hal_replay_recorder_allocator_vtable)
             ? iree_hal_replay_recorder_allocator_cast(allocator)
                   ->base_allocator
             : allocator;
}

void iree_hal_replay_recorder_allocator_make_allocate_buffer_payload(
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_replay_allocator_allocate_buffer_payload_t* out_payload) {
  memset(out_payload, 0, sizeof(*out_payload));
  out_payload->allocation_size = allocation_size;
  out_payload->queue_affinity = params->queue_affinity;
  out_payload->min_alignment = params->min_alignment;
  out_payload->usage = params->usage;
  out_payload->type = params->type;
  out_payload->access = params->access;
}

static iree_status_t iree_hal_replay_recorder_allocator_begin_operation(
    iree_hal_replay_recorder_allocator_t* allocator,
    iree_hal_replay_object_id_t related_object_id,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      allocator->recorder, allocator->device_id, allocator->allocator_id,
      related_object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR, operation_code,
      payload_type, out_pending_record);
}

static void iree_hal_replay_recorder_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(allocator->base_allocator);
  iree_hal_replay_recorder_release(allocator->recorder);
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_replay_recorder_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  const iree_hal_replay_recorder_allocator_t* allocator =
      (const iree_hal_replay_recorder_allocator_t*)base_allocator;
  return iree_hal_allocator_host_allocator(allocator->base_allocator);
}

static iree_status_t iree_hal_replay_recorder_allocator_trim(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_TRIM,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_allocator_trim(allocator->base_allocator));
}

static void iree_hal_replay_recorder_allocator_query_statistics(
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_allocator_query_statistics(allocator->base_allocator,
                                      out_statistics);
}

static iree_status_t iree_hal_replay_recorder_allocator_query_memory_heaps(
    iree_hal_allocator_t* base_allocator, iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* heaps, iree_host_size_t* out_count) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_QUERY_MEMORY_HEAPS,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_allocator_query_memory_heaps(allocator->base_allocator, capacity,
                                            heaps, out_count));
}

static iree_hal_buffer_compatibility_t
iree_hal_replay_recorder_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t* params,
    iree_device_size_t* allocation_size) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  return IREE_HAL_REPLAY_VTABLE_DISPATCH(allocator->base_allocator,
                                         iree_hal_allocator,
                                         query_buffer_compatibility)(
      allocator->base_allocator, params, allocation_size);
}

static iree_status_t iree_hal_replay_recorder_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  *out_buffer = NULL;

  iree_hal_replay_object_id_t buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
      allocator->recorder, &buffer_id));

  iree_hal_replay_allocator_allocate_buffer_payload_t operation_payload;
  iree_hal_replay_recorder_allocator_make_allocate_buffer_payload(
      params, allocation_size, &operation_payload);
  iree_const_byte_span_t operation_iovec = iree_make_const_byte_span(
      (const uint8_t*)&operation_payload, sizeof(operation_payload));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, buffer_id,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_ALLOCATE_BUFFER, &pending_record));

  iree_hal_buffer_t* base_buffer = NULL;
  iree_hal_buffer_t* replay_buffer = NULL;
  iree_status_t status = IREE_HAL_REPLAY_VTABLE_DISPATCH(
      allocator->base_allocator, iree_hal_allocator, allocate_buffer)(
      allocator->base_allocator, params, allocation_size, &base_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_create_proxy(
        allocator->recorder, allocator->device_id, buffer_id,
        allocator->placement_device, base_buffer, allocator->host_allocator,
        &replay_buffer);
  }

  iree_hal_replay_buffer_object_payload_t object_payload;
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_buffer_make_object_payload(base_buffer,
                                                        &object_payload);
  } else {
    memset(&object_payload, 0, sizeof(object_payload));
  }
  iree_const_byte_span_t object_iovec = iree_make_const_byte_span(
      (const uint8_t*)&object_payload, sizeof(object_payload));
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, 1, &operation_iovec,
      IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, buffer_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT, 1, &object_iovec);

  if (iree_status_is_ok(status)) {
    *out_buffer = replay_buffer;
  } else {
    iree_hal_buffer_release(replay_buffer);
  }
  iree_hal_buffer_release(base_buffer);
  return status;
}

static void iree_hal_replay_recorder_allocator_deallocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* buffer) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_buffer_t* base_buffer =
      iree_hal_replay_recorder_buffer_base_or_self(buffer);
  if (base_buffer != buffer) {
    iree_hal_buffer_destroy(buffer);
  } else {
    IREE_HAL_REPLAY_VTABLE_DISPATCH(allocator->base_allocator,
                                    iree_hal_allocator, deallocate_buffer)(
        allocator->base_allocator, base_buffer);
  }
}

static iree_status_t iree_hal_replay_recorder_allocator_import_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params,
    iree_hal_external_buffer_t* external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  *out_buffer = NULL;

  iree_hal_replay_object_id_t buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
      allocator->recorder, &buffer_id));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, buffer_id,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));

  iree_hal_buffer_t* base_buffer = NULL;
  iree_hal_buffer_t* replay_buffer = NULL;
  iree_status_t status = IREE_HAL_REPLAY_VTABLE_DISPATCH(
      allocator->base_allocator, iree_hal_allocator, import_buffer)(
      allocator->base_allocator, params, external_buffer, release_callback,
      &base_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_create_proxy(
        allocator->recorder, allocator->device_id, buffer_id,
        allocator->placement_device, base_buffer, allocator->host_allocator,
        &replay_buffer);
  }

  iree_hal_replay_buffer_object_payload_t object_payload;
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_buffer_make_object_payload(base_buffer,
                                                        &object_payload);
  } else {
    memset(&object_payload, 0, sizeof(object_payload));
  }
  iree_const_byte_span_t object_iovec = iree_make_const_byte_span(
      (const uint8_t*)&object_payload, sizeof(object_payload));
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, 0, NULL, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      buffer_id, IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT, 1, &object_iovec);

  if (iree_status_is_ok(status)) {
    *out_buffer = replay_buffer;
  } else {
    iree_hal_buffer_release(replay_buffer);
  }
  iree_hal_buffer_release(base_buffer);
  return status;
}

static iree_status_t iree_hal_replay_recorder_allocator_export_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* out_external_buffer) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_EXPORT_BUFFER,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_allocator_export_buffer(
          allocator->base_allocator,
          iree_hal_replay_recorder_buffer_base_or_self(buffer), requested_type,
          requested_flags, out_external_buffer));
}

static bool iree_hal_replay_recorder_allocator_supports_virtual_memory(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  return iree_hal_allocator_supports_virtual_memory(allocator->base_allocator);
}

static iree_status_t
iree_hal_replay_recorder_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t* out_minimum_page_size,
    iree_device_size_t* out_recommended_page_size) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_QUERY_GRANULARITY,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_allocator_virtual_memory_query_granularity(
                           allocator->base_allocator, params,
                           out_minimum_page_size, out_recommended_page_size));
}

static iree_status_t iree_hal_replay_recorder_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t** out_virtual_buffer) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  *out_virtual_buffer = NULL;

  iree_hal_replay_object_id_t buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
      allocator->recorder, &buffer_id));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, buffer_id,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));

  iree_hal_buffer_t* base_buffer = NULL;
  iree_hal_buffer_t* replay_buffer = NULL;
  iree_status_t status = iree_hal_allocator_virtual_memory_reserve(
      allocator->base_allocator, queue_affinity, size, &base_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_create_proxy(
        allocator->recorder, allocator->device_id, buffer_id,
        allocator->placement_device, base_buffer, allocator->host_allocator,
        &replay_buffer);
  }

  iree_hal_replay_buffer_object_payload_t object_payload;
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_buffer_make_object_payload(base_buffer,
                                                        &object_payload);
  } else {
    memset(&object_payload, 0, sizeof(object_payload));
  }
  iree_const_byte_span_t object_iovec = iree_make_const_byte_span(
      (const uint8_t*)&object_payload, sizeof(object_payload));
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, 0, NULL, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      buffer_id, IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT, 1, &object_iovec);

  if (iree_status_is_ok(status)) {
    *out_virtual_buffer = replay_buffer;
  } else {
    iree_hal_buffer_release(replay_buffer);
  }
  iree_hal_buffer_release(base_buffer);
  return status;
}

static iree_status_t iree_hal_replay_recorder_allocator_virtual_memory_release(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_allocator_virtual_memory_release(
          allocator->base_allocator,
          iree_hal_replay_recorder_buffer_base_or_self(virtual_buffer)));
}

static iree_status_t
iree_hal_replay_recorder_allocator_physical_memory_allocate(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_allocator_physical_memory_allocate(
                           allocator->base_allocator, params, size,
                           host_allocator, out_physical_memory));
}

static iree_status_t iree_hal_replay_recorder_allocator_physical_memory_free(
    iree_hal_allocator_t* base_allocator,
    iree_hal_physical_memory_t* physical_memory) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_allocator_physical_memory_free(
                           allocator->base_allocator, physical_memory));
}

static iree_status_t iree_hal_replay_recorder_allocator_virtual_memory_map(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_allocator_virtual_memory_map(
          allocator->base_allocator,
          iree_hal_replay_recorder_buffer_base_or_self(virtual_buffer),
          virtual_offset, physical_memory, physical_offset, size));
}

static iree_status_t iree_hal_replay_recorder_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_allocator_virtual_memory_unmap(
          allocator->base_allocator,
          iree_hal_replay_recorder_buffer_base_or_self(virtual_buffer),
          virtual_offset, size));
}

static iree_status_t iree_hal_replay_recorder_allocator_virtual_memory_protect(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_allocator_virtual_memory_protect(
          allocator->base_allocator,
          iree_hal_replay_recorder_buffer_base_or_self(virtual_buffer),
          virtual_offset, size, queue_affinity, protection));
}

static iree_status_t iree_hal_replay_recorder_allocator_virtual_memory_advise(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_allocator_begin_operation(
      allocator, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_allocator_virtual_memory_advise(
          allocator->base_allocator,
          iree_hal_replay_recorder_buffer_base_or_self(virtual_buffer),
          virtual_offset, size, queue_affinity, advice));
}

iree_status_t iree_hal_replay_recorder_wrap_allocator(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_device_t* placement_device, iree_hal_allocator_t* base_allocator,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;

  iree_hal_replay_recorder_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                             (void**)&allocator));
  memset(allocator, 0, sizeof(*allocator));
  iree_hal_resource_initialize(&iree_hal_replay_recorder_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->recorder = recorder;
  iree_hal_replay_recorder_retain(allocator->recorder);
  allocator->base_allocator =
      iree_hal_replay_recorder_allocator_base_or_self(base_allocator);
  iree_hal_allocator_retain(allocator->base_allocator);
  allocator->placement_device = placement_device;
  allocator->device_id = device_id;

  iree_status_t status = iree_hal_replay_recorder_record_object(
      recorder, device_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, 0, NULL, &allocator->allocator_id);
  if (iree_status_is_ok(status)) {
    *out_allocator = (iree_hal_allocator_t*)allocator;
  } else {
    iree_hal_allocator_release(allocator->base_allocator);
    iree_hal_replay_recorder_release(allocator->recorder);
    iree_allocator_free(host_allocator, allocator);
  }
  return status;
}

static const iree_hal_allocator_vtable_t
    iree_hal_replay_recorder_allocator_vtable = {
        .destroy = iree_hal_replay_recorder_allocator_destroy,
        .host_allocator = iree_hal_replay_recorder_allocator_host_allocator,
        .trim = iree_hal_replay_recorder_allocator_trim,
        .query_statistics = iree_hal_replay_recorder_allocator_query_statistics,
        .query_memory_heaps =
            iree_hal_replay_recorder_allocator_query_memory_heaps,
        .query_buffer_compatibility =
            iree_hal_replay_recorder_allocator_query_buffer_compatibility,
        .allocate_buffer = iree_hal_replay_recorder_allocator_allocate_buffer,
        .deallocate_buffer =
            iree_hal_replay_recorder_allocator_deallocate_buffer,
        .import_buffer = iree_hal_replay_recorder_allocator_import_buffer,
        .export_buffer = iree_hal_replay_recorder_allocator_export_buffer,
        .supports_virtual_memory =
            iree_hal_replay_recorder_allocator_supports_virtual_memory,
        .virtual_memory_query_granularity =
            iree_hal_replay_recorder_allocator_virtual_memory_query_granularity,
        .virtual_memory_reserve =
            iree_hal_replay_recorder_allocator_virtual_memory_reserve,
        .virtual_memory_release =
            iree_hal_replay_recorder_allocator_virtual_memory_release,
        .physical_memory_allocate =
            iree_hal_replay_recorder_allocator_physical_memory_allocate,
        .physical_memory_free =
            iree_hal_replay_recorder_allocator_physical_memory_free,
        .virtual_memory_map =
            iree_hal_replay_recorder_allocator_virtual_memory_map,
        .virtual_memory_unmap =
            iree_hal_replay_recorder_allocator_virtual_memory_unmap,
        .virtual_memory_protect =
            iree_hal_replay_recorder_allocator_virtual_memory_protect,
        .virtual_memory_advise =
            iree_hal_replay_recorder_allocator_virtual_memory_advise,
};
