// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_allocator.h"

#include "iree/hal/drivers/webgpu/webgpu_buffer.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_allocator_t
//===----------------------------------------------------------------------===//

// WebGPU exposes three memory heaps corresponding to the three buffer
// categories defined by the GPUBufferUsage constraints:
//
// 1. Device-local: STORAGE | COPY_SRC | COPY_DST (plus UNIFORM, INDIRECT).
//    Not host-mappable. Used for GPU compute buffers.
//
// 2. Staging write: MAP_WRITE | COPY_SRC. Created with mappedAtCreation:true.
//    Host writes data, unmaps to trigger GPU upload, then copies to
//    device-local. This is the bd-bqa path for HOST_LOCAL|DEVICE_VISIBLE.
//
// 3. Staging read: MAP_READ | COPY_DST. Created normally, mapped after GPU
//    copies results into it via mapAsync. Host reads results from the mapped
//    range.
//
// WebGPU constrains mappable buffer usage: MAP_READ can only combine with
// COPY_DST, and MAP_WRITE can only combine with COPY_SRC. This means a
// single buffer cannot be both mappable and usable as a storage binding.

#define IREE_HAL_WEBGPU_ALLOCATOR_HEAP_COUNT 3

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_WEBGPU_ALLOCATOR_ID = "WebGPU";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_webgpu_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Bridge handle for the GPUDevice. Not retained.
  iree_hal_webgpu_handle_t device_handle;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_webgpu_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_webgpu_allocator_vtable;

static iree_hal_webgpu_allocator_t* iree_hal_webgpu_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_allocator_vtable);
  return (iree_hal_webgpu_allocator_t*)base_value;
}

iree_status_t iree_hal_webgpu_allocator_create(
    iree_hal_webgpu_handle_t device_handle, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;

  iree_hal_webgpu_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_webgpu_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->device_handle = device_handle;

  *out_allocator = (iree_hal_allocator_t*)allocator;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_webgpu_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_webgpu_allocator_t* allocator =
      iree_hal_webgpu_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_webgpu_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_webgpu_allocator_t* allocator =
      (iree_hal_webgpu_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_webgpu_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  // No pooling yet — nothing to trim.
  return iree_ok_status();
}

static void iree_hal_webgpu_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_webgpu_allocator_t* allocator =
        iree_hal_webgpu_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_status_t iree_hal_webgpu_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  IREE_ASSERT_ARGUMENT(out_count);

  // WebGPU devices do not expose heap size information. We report a large max
  // allocation size; the actual limit depends on the device's maxBufferSize
  // limit (typically 256MB–2GB), which is checked at buffer creation time by
  // the WebGPU implementation itself.
  const iree_device_size_t max_allocation_size = ~(iree_device_size_t)0;
  const iree_device_size_t min_alignment = 256;

  *out_count = IREE_HAL_WEBGPU_ALLOCATOR_HEAP_COUNT;
  if (capacity < IREE_HAL_WEBGPU_ALLOCATOR_HEAP_COUNT) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  int i = 0;

  // Heap 0: Device-local memory.
  // GPU-optimal storage for compute dispatch. Not host-mappable.
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .allowed_usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  // Heap 1: Staging write (host→GPU upload).
  // HOST_LOCAL | DEVICE_VISIBLE with MAP_WRITE | COPY_SRC.
  // Created with mappedAtCreation:true for immediate host write access.
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .allowed_usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  // Heap 2: Staging read (GPU→host download).
  // HOST_VISIBLE | DEVICE_VISIBLE with MAP_READ | COPY_DST.
  // Mapped via mapAsync after GPU writes results into this buffer.
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .allowed_usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  IREE_ASSERT(i == IREE_HAL_WEBGPU_ALLOCATOR_HEAP_COUNT);
  return iree_ok_status();
}

// Determines whether the requested memory type and usage correspond to a
// staging buffer (host-mappable with transfer usage) vs a device-local buffer
// (GPU compute storage). Returns true for staging buffers.
static bool iree_hal_webgpu_allocator_is_staging(
    iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t usage) {
  return iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
         iree_any_bit_set(usage, IREE_HAL_BUFFER_USAGE_MAPPING);
}

// Determines whether the staging buffer is for writing (host→GPU upload) vs
// reading (GPU→host download). Must only be called when is_staging is true.
static bool iree_hal_webgpu_allocator_is_staging_write(
    iree_hal_memory_type_t memory_type) {
  return iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL);
}

// Maps IREE buffer parameters to WebGPU GPUBufferUsage flags.
static iree_hal_webgpu_buffer_usage_t
iree_hal_webgpu_allocator_compute_gpu_usage(iree_hal_memory_type_t memory_type,
                                            iree_hal_buffer_usage_t usage) {
  if (iree_hal_webgpu_allocator_is_staging(memory_type, usage)) {
    if (iree_hal_webgpu_allocator_is_staging_write(memory_type)) {
      // Staging write: MAP_WRITE | COPY_SRC (data flows host → staging → GPU).
      return IREE_HAL_WEBGPU_BUFFER_USAGE_MAP_WRITE |
             IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_SRC;
    } else {
      // Staging read: MAP_READ | COPY_DST (data flows GPU → staging → host).
      return IREE_HAL_WEBGPU_BUFFER_USAGE_MAP_READ |
             IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_DST;
    }
  }

  // Device-local buffer: build usage from IREE flags.
  iree_hal_webgpu_buffer_usage_t gpu_usage = 0;
  if (iree_any_bit_set(usage, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE)) {
    gpu_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_SRC;
  }
  if (iree_any_bit_set(usage, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET)) {
    gpu_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_DST;
  }
  if (iree_any_bit_set(usage, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
    gpu_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_STORAGE;
  }
  if (iree_any_bit_set(usage, IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ)) {
    gpu_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_UNIFORM;
  }
  if (iree_any_bit_set(usage,
                       IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS)) {
    gpu_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_INDIRECT;
  }

  // Ensure at least COPY_SRC|COPY_DST for device-local buffers so they can
  // participate in transfers even if the caller didn't explicitly request it.
  gpu_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_SRC |
               IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_DST;

  return gpu_usage;
}

static iree_hal_buffer_compatibility_t
iree_hal_webgpu_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // If the buffer is device-visible it can participate in queue operations.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) ||
      iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  // WebGPU cannot have a buffer that is both a storage binding and
  // host-mappable. If someone requests DEVICE_LOCAL | HOST_VISIBLE with
  // DISPATCH_STORAGE, we can allocate it as device-local (dropping
  // HOST_VISIBLE) but warn that it will require staging for host access.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
      iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
    // Coerce to pure device-local — host access requires staging copies.
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE;
    params->type &= ~IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  }

  // Handle OPTIMAL: choose the best heap based on usage.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
    params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
    if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_MAPPING)) {
      // Mapping requested → staging buffer.
      params->type |=
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    } else {
      // Default to device-local for compute buffers.
      params->type |= IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    }
  } else {
    // Not OPTIMAL — clear the bit if it somehow snuck in.
    params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  }

  // WebGPU only has host-visible buffers in the form of mappable staging
  // buffers (MAP_READ|COPY_DST or MAP_WRITE|COPY_SRC). If someone requests
  // host-visible memory without MAPPING usage, there is no WebGPU buffer type
  // that matches — coerce to device-local since the caller doesn't need host
  // access. This happens when e.g. HOST_LOCAL is requested for a storage
  // buffer without explicit MAPPING_SCOPED/MAPPING_PERSISTENT usage.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
      !iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_MAPPING)) {
    params->type &= ~(
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
        IREE_HAL_MEMORY_TYPE_HOST_COHERENT | IREE_HAL_MEMORY_TYPE_HOST_CACHED);
    params->type |= IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  }

  // Ensure HOST_LOCAL staging buffers have DEVICE_VISIBLE set (implied by the
  // staging buffer model — the GPU must be able to copy from/to the buffer).
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    params->type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  }

  // WebGPU buffer sizes must be multiples of 4 bytes.
  *allocation_size = iree_host_align(*allocation_size, 4);
  if (*allocation_size == 0) *allocation_size = 4;

  return compatibility;
}

static iree_status_t iree_hal_webgpu_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_webgpu_allocator_t* allocator =
      iree_hal_webgpu_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Coerce parameters to match device capabilities.
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_webgpu_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1, temp2;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    iree_string_view_t compatibility_str =
        iree_hal_buffer_compatibility_format(compatibility, &temp2);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  // Compute WebGPU usage flags from IREE parameters.
  iree_hal_webgpu_buffer_usage_t gpu_usage =
      iree_hal_webgpu_allocator_compute_gpu_usage(compat_params.type,
                                                  compat_params.usage);

  // Staging write buffers (HOST_LOCAL|DEVICE_VISIBLE) are created with
  // mappedAtCreation:true for immediate host write access.
  bool mapped_at_creation =
      iree_hal_webgpu_allocator_is_staging(compat_params.type,
                                           compat_params.usage) &&
      iree_hal_webgpu_allocator_is_staging_write(compat_params.type);

  // Create the GPU buffer via the bridge.
  iree_hal_webgpu_handle_t buffer_handle =
      iree_hal_webgpu_import_device_create_buffer(
          allocator->device_handle, gpu_usage, (uint64_t)allocation_size,
          mapped_at_creation ? 1 : 0);
  if (buffer_handle == IREE_HAL_WEBGPU_HANDLE_NULL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "WebGPU device.createBuffer failed for %" PRIu64
                            " bytes",
                            (uint64_t)allocation_size);
  }

  // Wrap in an IREE HAL buffer.
  iree_hal_buffer_placement_t placement = {
      .device = NULL,  // Set by the device after allocation.
      .queue_affinity = compat_params.queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_webgpu_buffer_create(
      placement, compat_params.type, compat_params.access, compat_params.usage,
      allocation_size, buffer_handle, mapped_at_creation,
      allocator->host_allocator, &buffer);

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_WEBGPU_ALLOCATOR_ID,
                           (void*)(uintptr_t)buffer_handle, allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    *out_buffer = buffer;
  } else {
    // Destroy the GPU buffer we created.
    iree_hal_webgpu_import_buffer_destroy(buffer_handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_webgpu_allocator_t* allocator =
      iree_hal_webgpu_allocator_cast(base_allocator);

  IREE_TRACE({
    iree_hal_webgpu_handle_t buffer_handle =
        iree_hal_webgpu_buffer_handle(base_buffer);
    IREE_TRACE_FREE_NAMED(IREE_HAL_WEBGPU_ALLOCATOR_ID,
                          (void*)(uintptr_t)buffer_handle);
  });
  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allocation_size(base_buffer)));
  (void)allocator;

  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_webgpu_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // WebGPU does not support importing external buffer handles.
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support external buffer import");
}

static iree_status_t iree_hal_webgpu_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  // WebGPU does not support exporting buffer handles.
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support external buffer export");
}

static const iree_hal_allocator_vtable_t iree_hal_webgpu_allocator_vtable = {
    .destroy = iree_hal_webgpu_allocator_destroy,
    .host_allocator = iree_hal_webgpu_allocator_host_allocator,
    .trim = iree_hal_webgpu_allocator_trim,
    .query_statistics = iree_hal_webgpu_allocator_query_statistics,
    .query_memory_heaps = iree_hal_webgpu_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_webgpu_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_webgpu_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_webgpu_allocator_deallocate_buffer,
    .import_buffer = iree_hal_webgpu_allocator_import_buffer,
    .export_buffer = iree_hal_webgpu_allocator_export_buffer,
};
