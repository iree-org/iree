// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_buffer.h"

#include "iree/hal/drivers/webgpu/webgpu_imports.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;

  // Bridge handle for the WebGPU GPUBuffer.
  iree_hal_webgpu_handle_t buffer_handle;

  // Shadow buffer in wasm linear memory for host mapping. WebGPU mapped ranges
  // live in JS memory (not wasm linear memory), so we maintain a shadow that
  // the C code reads/writes. Data is synchronized between the shadow and the
  // GPU mapped range on map/unmap transitions.
  void* shadow_buffer;
  iree_device_size_t shadow_buffer_size;

  // Whether the GPU buffer is currently in the mapped state.
  bool is_mapped;

  // Whether the buffer was created with mappedAtCreation:true. These buffers
  // start in the mapped state and the shadow buffer is allocated at creation.
  bool mapped_at_creation;
} iree_hal_webgpu_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_webgpu_buffer_vtable;

static iree_hal_webgpu_buffer_t* iree_hal_webgpu_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_buffer_vtable);
  return (iree_hal_webgpu_buffer_t*)base_value;
}

static const iree_hal_webgpu_buffer_t* iree_hal_webgpu_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_buffer_vtable);
  return (const iree_hal_webgpu_buffer_t*)base_value;
}

iree_status_t iree_hal_webgpu_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_hal_webgpu_handle_t buffer_handle, bool mapped_at_creation,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_webgpu_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));

  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size,
                             /*byte_offset=*/0, /*byte_length=*/allocation_size,
                             memory_type, allowed_access, allowed_usage,
                             &iree_hal_webgpu_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->buffer_handle = buffer_handle;
  buffer->shadow_buffer = NULL;
  buffer->shadow_buffer_size = 0;
  buffer->is_mapped = mapped_at_creation;
  buffer->mapped_at_creation = mapped_at_creation;

  // For mappedAtCreation buffers, allocate the shadow buffer immediately.
  // The host can write into the shadow, and on unmap we flush to the GPU.
  iree_status_t status = iree_ok_status();
  if (mapped_at_creation) {
    status =
        iree_allocator_malloc(host_allocator, (iree_host_size_t)allocation_size,
                              &buffer->shadow_buffer);
    if (iree_status_is_ok(status)) {
      buffer->shadow_buffer_size = allocation_size;
      // Pull the initial mapped contents from the GPU buffer into the shadow.
      // For mappedAtCreation the buffer is zero-initialized by WebGPU, so
      // this copies zeroes — but it ensures the shadow matches the GPU state.
      iree_hal_webgpu_import_buffer_get_mapped_range(
          buffer_handle, /*offset=*/0, /*size=*/(uint64_t)allocation_size,
          (uint32_t)(uintptr_t)buffer->shadow_buffer);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base;
  } else {
    iree_hal_buffer_release(&buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_webgpu_buffer_create_stub(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_webgpu_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));

  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size,
                             /*byte_offset=*/0, /*byte_length=*/allocation_size,
                             memory_type, allowed_access, allowed_usage,
                             &iree_hal_webgpu_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->buffer_handle = IREE_HAL_WEBGPU_HANDLE_NULL;
  buffer->shadow_buffer = NULL;
  buffer->shadow_buffer_size = 0;
  buffer->is_mapped = false;
  buffer->mapped_at_creation = false;

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Maps IREE buffer parameters to WebGPU GPUBufferUsage flags for buffer
// creation. Used by buffer_bind to compute GPU usage from stored params.
static iree_hal_webgpu_buffer_usage_t iree_hal_webgpu_buffer_compute_gpu_usage(
    iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t usage) {
  // Staging buffers: constrained to MAP+COPY usage.
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
      iree_any_bit_set(usage, IREE_HAL_BUFFER_USAGE_MAPPING)) {
    if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
      return IREE_HAL_WEBGPU_BUFFER_USAGE_MAP_WRITE |
             IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_SRC;
    } else {
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
  // participate in transfers.
  gpu_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_SRC |
               IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_DST;

  return gpu_usage;
}

iree_status_t iree_hal_webgpu_buffer_bind(
    iree_hal_buffer_t* base_buffer, iree_hal_webgpu_handle_t device_handle) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (buffer->buffer_handle != IREE_HAL_WEBGPU_HANDLE_NULL) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "buffer already bound to a GPU buffer");
  }

  iree_hal_webgpu_handle_t new_handle = IREE_HAL_WEBGPU_HANDLE_NULL;
  bool mapped_at_creation = false;
  if (iree_status_is_ok(status)) {
    iree_hal_memory_type_t memory_type =
        iree_hal_buffer_memory_type(base_buffer);
    iree_hal_buffer_usage_t usage = iree_hal_buffer_allowed_usage(base_buffer);
    iree_device_size_t allocation_size =
        iree_hal_buffer_allocation_size(base_buffer);

    iree_hal_webgpu_buffer_usage_t gpu_usage =
        iree_hal_webgpu_buffer_compute_gpu_usage(memory_type, usage);

    // Staging write buffers use mappedAtCreation for immediate host access.
    mapped_at_creation =
        iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
        iree_any_bit_set(usage, IREE_HAL_BUFFER_USAGE_MAPPING) &&
        iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL);

    new_handle = iree_hal_webgpu_import_device_create_buffer(
        device_handle, gpu_usage, (uint64_t)allocation_size,
        mapped_at_creation ? 1 : 0);
    if (new_handle == IREE_HAL_WEBGPU_HANDLE_NULL) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "WebGPU device.createBuffer failed for "
                                "%" PRIu64 " bytes",
                                (uint64_t)allocation_size);
    }
  }

  if (iree_status_is_ok(status)) {
    buffer->buffer_handle = new_handle;
    buffer->mapped_at_creation = mapped_at_creation;

    // For mappedAtCreation buffers, allocate the shadow buffer.
    if (mapped_at_creation) {
      iree_device_size_t allocation_size =
          iree_hal_buffer_allocation_size(base_buffer);
      buffer->is_mapped = true;
      status = iree_allocator_malloc(buffer->host_allocator,
                                     (iree_host_size_t)allocation_size,
                                     &buffer->shadow_buffer);
      if (iree_status_is_ok(status)) {
        buffer->shadow_buffer_size = allocation_size;
        iree_hal_webgpu_import_buffer_get_mapped_range(
            new_handle, /*offset=*/0, /*size=*/(uint64_t)allocation_size,
            (uint32_t)(uintptr_t)buffer->shadow_buffer);
      }
    }
  }

  if (!iree_status_is_ok(status)) {
    // Clean up: destroy the GPU buffer if we created one, reset state.
    if (new_handle != IREE_HAL_WEBGPU_HANDLE_NULL) {
      iree_hal_webgpu_import_buffer_destroy(new_handle);
      buffer->buffer_handle = IREE_HAL_WEBGPU_HANDLE_NULL;
      buffer->is_mapped = false;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_webgpu_buffer_unbind(iree_hal_buffer_t* base_buffer) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (buffer->buffer_handle != IREE_HAL_WEBGPU_HANDLE_NULL) {
    // Unmap if still mapped.
    if (buffer->is_mapped) {
      iree_hal_webgpu_import_buffer_unmap(buffer->buffer_handle);
      buffer->is_mapped = false;
    }

    // Free the shadow buffer.
    if (buffer->shadow_buffer) {
      iree_allocator_free(buffer->host_allocator, buffer->shadow_buffer);
      buffer->shadow_buffer = NULL;
      buffer->shadow_buffer_size = 0;
    }

    // Destroy the GPU buffer via the bridge.
    iree_hal_webgpu_import_buffer_destroy(buffer->buffer_handle);
    buffer->buffer_handle = IREE_HAL_WEBGPU_HANDLE_NULL;
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_hal_webgpu_handle_t iree_hal_webgpu_buffer_handle(
    const iree_hal_buffer_t* base_buffer) {
  const iree_hal_webgpu_buffer_t* buffer =
      iree_hal_webgpu_buffer_const_cast(base_buffer);
  return buffer->buffer_handle;
}

static void iree_hal_webgpu_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (buffer->buffer_handle != IREE_HAL_WEBGPU_HANDLE_NULL) {
    // If the buffer is still mapped, unmap it.
    if (buffer->is_mapped) {
      iree_hal_webgpu_import_buffer_unmap(buffer->buffer_handle);
      buffer->is_mapped = false;
    }

    // Free the shadow buffer.
    if (buffer->shadow_buffer) {
      iree_allocator_free(host_allocator, buffer->shadow_buffer);
      buffer->shadow_buffer = NULL;
    }

    // Destroy the GPU buffer via the bridge.
    iree_hal_webgpu_import_buffer_destroy(buffer->buffer_handle);
  }

  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_webgpu_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  if (!buffer->is_mapped) {
    // The GPU buffer is not in mapped state. For mappedAtCreation buffers this
    // means the buffer was already unmapped (the buffer was flushed to the GPU
    // and cannot be re-mapped for writing). For read-back buffers, mapAsync
    // must have been called and completed before map_range.
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer is not in mapped state; for staging-write buffers "
        "mappedAtCreation mapping is single-use, for staging-read buffers "
        "mapAsync must complete before mapping");
  }

  // Ensure the shadow buffer covers the requested range.
  if (!buffer->shadow_buffer ||
      local_byte_offset + local_byte_length > buffer->shadow_buffer_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "requested mapping range [%" PRIu64 ", %" PRIu64
                            ") exceeds shadow buffer size %" PRIu64,
                            (uint64_t)local_byte_offset,
                            (uint64_t)(local_byte_offset + local_byte_length),
                            (uint64_t)buffer->shadow_buffer_size);
  }

  // For read access, pull data from the GPU mapped range into the shadow.
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ)) {
    iree_hal_webgpu_import_buffer_get_mapped_range(
        buffer->buffer_handle, (uint64_t)local_byte_offset,
        (uint64_t)local_byte_length,
        (uint32_t)(uintptr_t)((uint8_t*)buffer->shadow_buffer +
                              local_byte_offset));
  }

  // Return a pointer into the shadow buffer.
  mapping->contents = iree_make_byte_span(
      (uint8_t*)buffer->shadow_buffer + local_byte_offset, local_byte_length);

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);

  if (!buffer->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer is not mapped");
  }

  // For write access, flush the shadow buffer contents to the GPU mapped range.
  // We always flush the entire mapped region to keep the GPU state consistent.
  // The cost is a single memcpy through the bridge — the mapped range is JS
  // memory and will be uploaded to the GPU when the buffer is unmapped.
  iree_hal_webgpu_import_buffer_set_mapped_range(
      buffer->buffer_handle, (uint64_t)local_byte_offset,
      (uint64_t)local_byte_length,
      (uint32_t)(uintptr_t)((uint8_t*)buffer->shadow_buffer +
                            local_byte_offset));

  // For mappedAtCreation buffers, unmap transitions the buffer to the
  // unmapped state permanently (it cannot be re-mapped for writing).
  // Unmap triggers the GPU upload of the staged data.
  if (buffer->mapped_at_creation) {
    iree_hal_webgpu_import_buffer_unmap(buffer->buffer_handle);
    buffer->is_mapped = false;

    // Free the shadow buffer — the data has been flushed to the GPU.
    iree_allocator_free(buffer->host_allocator, buffer->shadow_buffer);
    buffer->shadow_buffer = NULL;
    buffer->shadow_buffer_size = 0;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // WebGPU buffers are coherent — no explicit cache invalidation needed.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // WebGPU buffers are coherent — no explicit cache flush needed.
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_webgpu_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_webgpu_buffer_destroy,
    .map_range = iree_hal_webgpu_buffer_map_range,
    .unmap_range = iree_hal_webgpu_buffer_unmap_range,
    .invalidate_range = iree_hal_webgpu_buffer_invalidate_range,
    .flush_range = iree_hal_webgpu_buffer_flush_range,
};
