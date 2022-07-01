// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/level_zero_allocator.h"

#include <stddef.h>

#include "experimental/level_zero/dynamic_symbols.h"
#include "experimental/level_zero/level_zero_buffer.h"
#include "experimental/level_zero/status_util.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_level_zero_allocator_t {
  iree_hal_resource_t resource;
  iree_hal_device_t* base_device;
  ze_device_handle_t level_zero_device;
  iree_hal_level_zero_context_wrapper_t* context;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_level_zero_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_level_zero_allocator_vtable;

static iree_hal_level_zero_allocator_t* iree_hal_level_zero_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_level_zero_allocator_vtable);
  return (iree_hal_level_zero_allocator_t*)base_value;
}

iree_status_t iree_hal_level_zero_allocator_create(
    iree_hal_device_t* base_device, ze_device_handle_t level_zero_device,
    iree_hal_level_zero_context_wrapper_t* context,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(base_device);
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_level_zero_allocator_t* allocator = NULL;
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, sizeof(*allocator), (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_level_zero_allocator_vtable,
                                 &allocator->resource);
    allocator->context = context;
    allocator->base_device = base_device;
    allocator->level_zero_device = level_zero_device;
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_level_zero_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_level_zero_allocator_t* allocator =
      iree_hal_level_zero_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_level_zero_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_level_zero_allocator_t* allocator =
      (iree_hal_level_zero_allocator_t*)base_allocator;
  return allocator->context->host_allocator;
}

static iree_status_t iree_hal_level_zero_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_level_zero_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_level_zero_allocator_t* allocator =
        iree_hal_level_zero_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_hal_buffer_compatibility_t
iree_hal_level_zero_allocator_query_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size) {
  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // LevelZero supports host <-> device for all copies.
  if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_all_bits_set(params->usage,
                          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  return compatibility;
}

static void iree_hal_level_zero_buffer_free(
    iree_hal_level_zero_context_wrapper_t* context,
    iree_hal_memory_type_t memory_type,
    iree_hal_level_zero_device_ptr_t device_ptr, void* host_ptr) {
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local.
    LEVEL_ZERO_IGNORE_ERROR(context->syms,
                            zeMemFree(context->level_zero_context, device_ptr));
  } else {
    // Host local.
    LEVEL_ZERO_IGNORE_ERROR(context->syms,
                            zeMemFree(context->level_zero_context, host_ptr));
  }
}

static iree_status_t iree_hal_level_zero_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_level_zero_allocator_t* allocator =
      iree_hal_level_zero_allocator_cast(base_allocator);
  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (allocation_size == 0) allocation_size = 4;
  size_t alloc_alignment = 32;

  iree_status_t status = iree_ok_status();
  // Defining device memory alloc.
  ze_device_mem_alloc_desc_t memAllocDesc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  memAllocDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
  memAllocDesc.ordinal = 0;
  // Defining host memory alloc.
  ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};

  // Defining memalloc limits.
  ze_relaxed_allocation_limits_exp_desc_t exceedCapacity = {
      ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC, NULL,
      ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE};
  hostDesc.pNext = &exceedCapacity;
  memAllocDesc.pNext = &exceedCapacity;

  void* host_ptr = NULL;
  iree_hal_level_zero_device_ptr_t device_ptr = NULL;
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local case.
    if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      status = LEVEL_ZERO_RESULT_TO_STATUS(
          allocator->context->syms,
          zeMemAllocShared(allocator->context->level_zero_context,
                           &memAllocDesc, &hostDesc, allocation_size,
                           alloc_alignment, allocator->level_zero_device,
                           (void**)&device_ptr));
      host_ptr = (void*)device_ptr;
    } else {
      // Device only.
      status = LEVEL_ZERO_RESULT_TO_STATUS(
          allocator->context->syms,
          zeMemAllocDevice(allocator->context->level_zero_context,
                           &memAllocDesc, allocation_size, alloc_alignment,
                           allocator->level_zero_device, (void**)&device_ptr));
    }
  } else {
    // Since in Level Zero host memory is visible to device, we can simply
    // allocate on host and set device_ptr to point to same data.
    status = LEVEL_ZERO_RESULT_TO_STATUS(
        allocator->context->syms,
        zeMemAllocHost(allocator->context->level_zero_context, &hostDesc,
                       allocation_size, 64, &host_ptr));
    device_ptr = (iree_hal_level_zero_device_ptr_t)host_ptr;
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_level_zero_buffer_wrap(
        (iree_hal_allocator_t*)allocator, params->type, params->access,
        params->usage, allocation_size,
        /*byte_offset=*/0,
        /*byte_length=*/allocation_size, device_ptr, host_ptr, &buffer);
  }

  // Copy the initial contents into the buffer. This may require staging.
  if (iree_status_is_ok(status) &&
      !iree_const_byte_span_is_empty(initial_data)) {
    status = iree_hal_device_transfer_range(
        allocator->base_device,
        iree_hal_make_host_transfer_buffer_span((void*)initial_data.data,
                                                initial_data.data_length),
        0, iree_hal_make_device_transfer_buffer(buffer), 0,
        initial_data.data_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }

  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, params->type, allocation_size));
    *out_buffer = buffer;
  } else {
    if (!buffer) {
      iree_hal_level_zero_buffer_free(allocator->context, params->type,
                                      device_ptr, host_ptr);
    } else {
      iree_hal_buffer_release(buffer);
    }
  }
  return status;
}

static void iree_hal_level_zero_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_level_zero_allocator_t* allocator =
      iree_hal_level_zero_allocator_cast(base_allocator);

  iree_hal_memory_type_t memory_type = iree_hal_buffer_memory_type(base_buffer);
  iree_hal_level_zero_buffer_free(
      allocator->context, memory_type,
      iree_hal_level_zero_buffer_device_pointer(base_buffer),
      iree_hal_level_zero_buffer_host_pointer(base_buffer));

  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, memory_type,
      iree_hal_buffer_allocation_size(base_buffer)));

  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_level_zero_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "importing from external buffers not supported");
}

static iree_status_t iree_hal_level_zero_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "exporting to external buffers not supported");
}

static const iree_hal_allocator_vtable_t iree_hal_level_zero_allocator_vtable =
    {
        .destroy = iree_hal_level_zero_allocator_destroy,
        .host_allocator = iree_hal_level_zero_allocator_host_allocator,
        .trim = iree_hal_level_zero_allocator_trim,
        .query_statistics = iree_hal_level_zero_allocator_query_statistics,
        .query_compatibility =
            iree_hal_level_zero_allocator_query_compatibility,
        .allocate_buffer = iree_hal_level_zero_allocator_allocate_buffer,
        .deallocate_buffer = iree_hal_level_zero_allocator_deallocate_buffer,
        .import_buffer = iree_hal_level_zero_allocator_import_buffer,
        .export_buffer = iree_hal_level_zero_allocator_export_buffer,
};
