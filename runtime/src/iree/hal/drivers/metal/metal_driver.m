// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/api.h"
#include "iree/hal/drivers/metal/metal_device.h"

// Maximum device path length we support. The path is always a 16 character hex string.
#define IREE_HAL_METAL_MAX_DEVICE_PATH_LENGTH 32
// Maximum device name length we support. Example names: "Apple M1 Pro".
#define IREE_HAL_METAL_MAX_DEVICE_NAME_LENGTH 64

// Cast utilities between Metal id<MTLDevice> and IREE opaque iree_hal_device_id_t.
#define METAL_DEVICE_TO_DEVICE_ID(device) (iree_hal_device_id_t)((__bridge void*)device)
#define DEVICE_ID_TO_METAL_DEVICE(device_id) (__bridge id<MTLDevice>)(device_id)

typedef struct iree_hal_metal_driver_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  // Identifier used for the driver in the IREE driver registry. We allow overriding so that
  // multiple Metal versions can be exposed in the same process.
  iree_string_view_t identifier;

  // Parameters used to control device behavior.
  iree_hal_metal_device_params_t device_params;

  // The list of GPUs available when creating the driver. We retain them here to make sure
  // id<MTLDevice>, which is used for creating devices and such, remains valid.
  NSArray<id<MTLDevice>>* devices;
} iree_hal_metal_driver_t;

static const iree_hal_driver_vtable_t iree_hal_metal_driver_vtable;

static iree_hal_metal_driver_t* iree_hal_metal_driver_cast(iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_driver_vtable);
  return (iree_hal_metal_driver_t*)base_value;
}

static const iree_hal_metal_driver_t* iree_hal_metal_driver_const_cast(
    const iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_driver_vtable);
  return (const iree_hal_metal_driver_t*)base_value;
}

// Returns an retained array of available Metal GPU devices; the caller should release later.
static NSArray<id<MTLDevice>>* iree_hal_metal_device_copy() {
#if defined(IREE_PLATFORM_MACOS)
  // For macOS, we might have more then one GPU devices.
  return MTLCopyAllDevices();  // +1
#else
  // For other Apple platforms, we only have one GPU device.
  @autoreleasepool {  // Use @autorelasepool to trigger the autorelease carried in NSArray literal.
    return [@[ MTLCreateSystemDefaultDevice() ] retain];  // +1
  }
#endif  // IREE_PLATFORM_MACOS
}

static iree_status_t iree_hal_metal_device_check_params(
    const iree_hal_metal_device_params_t* params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_driver_create_internal(
    iree_string_view_t identifier, const iree_hal_metal_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  iree_hal_metal_driver_t* driver = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_metal_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                    (char*)driver + iree_sizeof_struct(*driver));
  driver->device_params = *device_params;

  // Get all available Metal devices.
  driver->devices = iree_hal_metal_device_copy();

  *out_driver = (iree_hal_driver_t*)driver;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_metal_driver_create(
    iree_string_view_t identifier, const iree_hal_metal_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_metal_device_check_params(device_params));
  iree_status_t status =
      iree_hal_metal_driver_create_internal(identifier, device_params, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_metal_driver_t* driver = iree_hal_metal_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  [driver->devices release];  // -1
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

// Populates device information from the given Metal physical device handle. |out_device_info| must
// point to valid memory and additional data will be appended to |buffer_ptr| and the new pointer is
// returned.
static iree_status_t iree_hal_metal_populate_device_info(id<MTLDevice> device, uint8_t* buffer_ptr,
                                                         uint8_t** out_buffer_ptr,
                                                         iree_hal_device_info_t* out_device_info) {
  *out_buffer_ptr = buffer_ptr;

  memset(out_device_info, 0, sizeof(*out_device_info));

  out_device_info->device_id = METAL_DEVICE_TO_DEVICE_ID(device);

  // For Metal devices, we don't have a 128-bit UUID; so just use the 64-bit registry ID here.
  char device_path[16 + 1] = {0};
  snprintf(device_path, sizeof(device_path), "%016" PRIx64, device.registryID);
  buffer_ptr += iree_string_view_append_to_buffer(
      iree_make_string_view(device_path, IREE_ARRAYSIZE(device_path) - 1), &out_device_info->path,
      (char*)buffer_ptr);

  const char* device_name = [device.name UTF8String];
  const size_t name_len = strlen(device_name);
  if (name_len >= IREE_HAL_METAL_MAX_DEVICE_NAME_LENGTH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "device name out of range");
  }
  buffer_ptr += iree_string_view_append_to_buffer(iree_make_string_view(device_name, name_len),
                                                  &out_device_info->name, (char*)buffer_ptr);

  *out_buffer_ptr = buffer_ptr;

  return iree_ok_status();
}

static iree_status_t iree_hal_metal_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count, iree_hal_device_info_t** out_device_infos) {
  iree_hal_metal_driver_t* driver = iree_hal_metal_driver_cast(base_driver);
  NSArray<id<MTLDevice>>* devices = driver->devices;
  unsigned device_count = devices.count;

  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t single_info_size =
      sizeof(iree_hal_device_info_t) +
      (IREE_HAL_METAL_MAX_DEVICE_PATH_LENGTH + IREE_HAL_METAL_MAX_DEVICE_NAME_LENGTH) *
          sizeof(char);
  iree_host_size_t total_size = device_count * single_info_size;
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);

  if (iree_status_is_ok(status)) {
    // Append all path and name strings at the end of the struct.
    uint8_t* buffer_ptr = (uint8_t*)device_infos + device_count * sizeof(iree_hal_device_info_t);
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      status = iree_hal_metal_populate_device_info(devices[i], buffer_ptr, &buffer_ptr,
                                                   &device_infos[i]);
      if (!iree_status_is_ok(status)) break;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_device_info_count = device_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }
  return status;
}

// Returns the GPU family the given |device| supports. Returns 0 if the given |device| does not
// belong to a GPU family considered by IREE right now.
static MTLGPUFamily iree_hal_metal_apple_gpu_family_query(id<MTLDevice> device) {
  // Inspect whether the given device is a specific Apple GPU.
  if ([device supportsFamily:MTLGPUFamilyApple8]) return MTLGPUFamilyApple8;
  if ([device supportsFamily:MTLGPUFamilyApple7]) return MTLGPUFamilyApple7;
  if ([device supportsFamily:MTLGPUFamilyApple6]) return MTLGPUFamilyApple6;
  if ([device supportsFamily:MTLGPUFamilyApple5]) return MTLGPUFamilyApple5;
  if ([device supportsFamily:MTLGPUFamilyApple4]) return MTLGPUFamilyApple4;
  if ([device supportsFamily:MTLGPUFamilyApple3]) return MTLGPUFamilyApple3;
  if ([device supportsFamily:MTLGPUFamilyApple2]) return MTLGPUFamilyApple2;
  if ([device supportsFamily:MTLGPUFamilyApple1]) return MTLGPUFamilyApple1;

  // Inspect whether whether the given GPU falls into some common family.
  if ([device supportsFamily:MTLGPUFamilyCommon3]) return MTLGPUFamilyCommon3;
  if ([device supportsFamily:MTLGPUFamilyCommon2]) return MTLGPUFamilyCommon2;
  if ([device supportsFamily:MTLGPUFamilyCommon1]) return MTLGPUFamilyCommon1;

  return 0;
}

static const char* iree_hal_metal_get_gpu_family_name(MTLGPUFamily family) {
  switch (family) {
    case MTLGPUFamilyApple8:
      return "apple8(a15/m2)";
    case MTLGPUFamilyApple7:
      return "apple7(a14/m1)";
    case MTLGPUFamilyApple6:
      return "apple6(a13)";
    case MTLGPUFamilyApple5:
      return "apple5(a12)";
    case MTLGPUFamilyApple4:
      return "apple4(a11)";
    case MTLGPUFamilyApple3:
      return "apple3(a9/a10)";
    case MTLGPUFamilyApple2:
      return "apple2(a8)";
    case MTLGPUFamilyApple1:
      return "apple1(a7)";

    case MTLGPUFamilyCommon3:
      return "common3";
    case MTLGPUFamilyCommon2:
      return "common2";
    case MTLGPUFamilyCommon1:
      return "common1";

    default:
      return "";
  }
}

static inline const char* iree_hal_metal_get_argument_buffer_tier_str(MTLGPUFamily family) {
  if (family >= MTLGPUFamilyApple6 && family <= MTLGPUFamilyApple8) return "2";
  if (family >= MTLGPUFamilyApple2 && family <= MTLGPUFamilyApple5) return "1";
  return "unknown";
}

static inline bool iree_hal_metal_support_simd_matrix_multiply(MTLGPUFamily family) {
  return family >= MTLGPUFamilyApple7 && family <= MTLGPUFamilyApple8;
}

static inline bool iree_hal_metal_support_simd_reduce(MTLGPUFamily family) {
  return family >= MTLGPUFamilyApple7 && family <= MTLGPUFamilyApple8;
}

static inline bool iree_hal_metal_support_simd_permute(MTLGPUFamily family) {
  return family >= MTLGPUFamilyApple6 && family <= MTLGPUFamilyApple8;
}

static inline bool iree_hal_metal_support_simd_shift_and_fill(MTLGPUFamily family) {
  return family >= MTLGPUFamilyApple8 && family <= MTLGPUFamilyApple8;
}

static iree_status_t iree_hal_metal_driver_dump_device_info(iree_hal_driver_t* base_driver,
                                                            iree_hal_device_id_t device_id,
                                                            iree_string_builder_t* builder) {
  id<MTLDevice> device = DEVICE_ID_TO_METAL_DEVICE(device_id);
  MTLGPUFamily apple_gpu_family = iree_hal_metal_apple_gpu_family_query(device);

  // Dump GPU family information.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "- gpu-family:"));
  const char* apple_family_str = iree_hal_metal_get_gpu_family_name(apple_gpu_family);
  if (apple_family_str) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " "));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, apple_family_str));
  }
  if ([device supportsFamily:MTLGPUFamilyMetal3]) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " metal3"));
  }

  // Dump memory information.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(builder, "\n- unified-memory: %d",
                                                         device.hasUnifiedMemory));

  // Dump argument buffer tier.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n- argument-buffer-tier: "));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder, iree_hal_metal_get_argument_buffer_tier_str(apple_gpu_family)));

  // Dump resource limits.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n- max-buffer-size: "));
  {
    uint32_t max_buffer_mb = device.maxBufferLength / 1024u / 1024u;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(builder, "%uMB", max_buffer_mb));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\n- max-threadgroup-memory-size: "));
  {
    uint32_t max_memory_kb = device.maxThreadgroupMemoryLength / 1024u;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(builder, "%uKB", max_memory_kb));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\n- max-threads-per-threadgroup: "));
  {
    MTLSize threads = device.maxThreadsPerThreadgroup;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "(%lu, %lu, %lu)", threads.width, threads.height, threads.depth));
  }

  // Dump SIMD-scoped operation features.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n- simd-scoped-operation:"));
  if (iree_hal_metal_support_simd_matrix_multiply(apple_gpu_family)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " matmul"));
  }
  if (iree_hal_metal_support_simd_reduce(apple_gpu_family)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " reduce"));
  }
  if (iree_hal_metal_support_simd_permute(apple_gpu_family)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " permute"));
  }
  if (iree_hal_metal_support_simd_shift_and_fill(apple_gpu_family)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " shift-and-fill"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));

  return iree_ok_status();
}

static iree_status_t iree_hal_metal_driver_find_device_by_index(iree_hal_driver_t* base_driver,
                                                                uint32_t device_index,
                                                                iree_allocator_t host_allocator,
                                                                id<MTLDevice>* found_device) {
  iree_hal_metal_driver_t* driver = iree_hal_metal_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (uint64_t)device_index);

  NSArray<id<MTLDevice>>* devices = driver->devices;
  if (device_index >= devices.count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND, "%d devices enumerated; device #%d not found",
                            (int)devices.count, device_index);
  }
  *found_device = devices[device_index];

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_driver_create_device_by_id(iree_hal_driver_t* base_driver,
                                                               iree_hal_device_id_t device_id,
                                                               iree_host_size_t param_count,
                                                               const iree_string_pair_t* params,
                                                               iree_allocator_t host_allocator,
                                                               iree_hal_device_t** out_device) {
  iree_hal_metal_driver_t* driver = iree_hal_metal_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  id<MTLDevice> device = nil;
  if (device_id == IREE_HAL_DEVICE_ID_DEFAULT) {
    // Default to the first Metal device in the list.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_metal_driver_find_device_by_index(
                                              base_driver, device_id, host_allocator, &device));
  } else {
    device = DEVICE_ID_TO_METAL_DEVICE(device_id);
  }

  iree_string_view_t device_name = iree_make_cstring_view("metal");

  iree_status_t status = iree_hal_metal_device_create(device_name, &driver->device_params, device,
                                                      host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_driver_create_device_by_registry_id(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name, uint64_t device_registry_id,
    iree_host_size_t param_count, const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_metal_driver_t* driver = iree_hal_metal_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Scan the devices and find the one with the matching registry ID.
  NSArray<id<MTLDevice>>* devices = driver->devices;
  id<MTLDevice> found_device = nil;
  for (iree_host_size_t i = 0, e = devices.count; i < e; ++i) {
    if (device_registry_id == devices[i].registryID) {
      found_device = devices[i];
      break;
    }
  }

  if (!found_device) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Metal device with device registry ID %016" PRIx64 " not found",
                            device_registry_id);
  }

  iree_status_t status = iree_hal_metal_driver_create_device_by_id(
      base_driver, METAL_DEVICE_TO_DEVICE_ID(found_device), param_count, params, host_allocator,
      out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name, iree_string_view_t device_path,
    iree_host_size_t param_count, const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  if (iree_string_view_is_empty(device_path)) {
    return iree_hal_metal_driver_create_device_by_id(
        base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params, host_allocator, out_device);
  }

  // Try parsing as a device ID.
  uint64_t device_registry_id = 0;
  if (iree_string_view_atoi_uint64_base(device_path, 16, &device_registry_id)) {
    return iree_hal_metal_driver_create_device_by_registry_id(base_driver, driver_name,
                                                              device_registry_id, param_count,
                                                              params, host_allocator, out_device);
  }

  // Fallback and try to parse as a device index.
  uint32_t device_index = 0;
  if (iree_string_view_atoi_uint32(device_path, &device_index)) {
    id<MTLDevice> found_device;
    IREE_RETURN_IF_ERROR(iree_hal_metal_driver_find_device_by_index(base_driver, device_index,
                                                                    host_allocator, &found_device));
    return iree_hal_metal_driver_create_device_by_id(
        base_driver, METAL_DEVICE_TO_DEVICE_ID(found_device), param_count, params, host_allocator,
        out_device);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported device path");
}

static const iree_hal_driver_vtable_t iree_hal_metal_driver_vtable = {
    .destroy = iree_hal_metal_driver_destroy,
    .query_available_devices = iree_hal_metal_driver_query_available_devices,
    .dump_device_info = iree_hal_metal_driver_dump_device_info,
    .create_device_by_id = iree_hal_metal_driver_create_device_by_id,
    .create_device_by_path = iree_hal_metal_driver_create_device_by_path,
};
