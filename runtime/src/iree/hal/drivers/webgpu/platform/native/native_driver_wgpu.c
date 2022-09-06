// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <wgpu.h>  // wgpu-native implementation only

#include "iree/base/tracing.h"
#include "iree/hal/drivers/webgpu/platform/native/native_driver.h"

#define IREE_HAL_WEBGPU_DEVICE_ID_DEFAULT 0

//===----------------------------------------------------------------------===//
// wgpu-native callbacks
//===----------------------------------------------------------------------===//

static void iree_hal_webgpu_native_log_callback(WGPULogLevel level,
                                                const char* message) {
  fprintf(stderr, "WGPU: %s", message);
}

static void iree_hal_webgpu_native_setup_logging(
    iree_hal_webgpu_driver_log_level_t log_level) {
  WGPULogLevel wgpu_log_level = WGPULogLevel_Off;
  switch (log_level) {
    default:
    case IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_OFF:
      wgpu_log_level = WGPULogLevel_Off;
      break;
    case IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_ERROR:
      wgpu_log_level = WGPULogLevel_Error;
      break;
    case IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_WARNING:
      wgpu_log_level = WGPULogLevel_Warn;
      break;
    case IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_INFO:
      wgpu_log_level = WGPULogLevel_Info;
      break;
    case IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_DEBUG:
      wgpu_log_level = WGPULogLevel_Debug;
      break;
    case IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_TRACE:
      wgpu_log_level = WGPULogLevel_Trace;
      break;
  }
  if (wgpu_log_level == WGPULogLevel_Off) return;
  wgpuSetLogCallback(iree_hal_webgpu_native_log_callback);
  wgpuSetLogLevel(wgpu_log_level);
}

typedef struct iree_hal_webgpu_native_adapter_data_t {
  iree_status_t status;
  WGPUAdapter handle;
} iree_hal_webgpu_native_adapter_data_t;
static void iree_hal_webgpu_native_request_adapter_callback(
    WGPURequestAdapterStatus status, WGPUAdapter received, const char* message,
    void* user_data_ptr) {
  iree_hal_webgpu_native_adapter_data_t* data =
      (iree_hal_webgpu_native_adapter_data_t*)user_data_ptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  switch (status) {
    case WGPURequestAdapterStatus_Success:
      IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(z0, "success");
      data->status = iree_ok_status();
      data->handle = received;
      break;
    case WGPURequestAdapterStatus_Unavailable:
      IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(z0, "unavailable");
      data->status =
          iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "wgpuInstanceRequestAdapter failed: %s", message);
      break;
    case WGPURequestAdapterStatus_Error:
      IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(z0, "error");
      data->status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "wgpuInstanceRequestAdapter failed: %s", message);
      break;
    default:
    case WGPURequestAdapterStatus_Unknown:
      IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(z0, "unknown");
      data->status =
          iree_make_status(IREE_STATUS_UNKNOWN,
                           "wgpuInstanceRequestAdapter failed: %s", message);
      break;
  }

  IREE_TRACE_ZONE_END(z0);
}

typedef struct iree_hal_webgpu_native_device_data_t {
  iree_status_t status;
  WGPUDevice handle;
} iree_hal_webgpu_native_device_data_t;
static void iree_hal_webgpu_native_request_device_callback(
    WGPURequestDeviceStatus status, WGPUDevice received, const char* message,
    void* user_data_ptr) {
  iree_hal_webgpu_native_device_data_t* data =
      (iree_hal_webgpu_native_device_data_t*)user_data_ptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  switch (status) {
    case WGPURequestDeviceStatus_Success:
      IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(z0, "success");
      data->status = iree_ok_status();
      data->handle = received;
      break;
    case WGPURequestDeviceStatus_Error:
      IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(z0, "error");
      data->status = iree_make_status(
          IREE_STATUS_INTERNAL, "wgpuAdapterRequestDevice failed: %s", message);
      break;
    default:
    case WGPURequestDeviceStatus_Unknown:
      IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(z0, "unknown");
      data->status = iree_make_status(
          IREE_STATUS_UNKNOWN, "wgpuAdapterRequestDevice failed: %s", message);
      break;
  }

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_native_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_native_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_webgpu_device_options_t default_options;

  WGPUInstance instance;
  WGPUAdapter adapter;
} iree_hal_webgpu_native_driver_t;

static const iree_hal_driver_vtable_t iree_hal_webgpu_native_driver_vtable;

static iree_hal_webgpu_native_driver_t* iree_hal_webgpu_native_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_native_driver_vtable);
  return (iree_hal_webgpu_native_driver_t*)base_value;
}

iree_status_t iree_hal_webgpu_native_driver_create(
    iree_string_view_t identifier,
    const iree_hal_webgpu_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_native_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size + /*NUL=*/1;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_webgpu_native_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;

  iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                    (char*)driver + sizeof(*driver));
  memcpy(&driver->default_options, &options->device_options,
         sizeof(driver->default_options));

  // wgpu-native doesn't currently expose (or need) this. Other implementations
  // may and they may require it in the future.
  const WGPUInstanceDescriptor instance_descriptor = {
      .nextInChain = NULL,
  };
  driver->instance = wgpuCreateInstance(&instance_descriptor);
  if (!driver->instance) {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "WebGPU implementation not present or failed to load");
  }

  // Setup logging first as the adapter queries and such we perform immediately
  // after this may log things.
  iree_hal_webgpu_native_setup_logging(options->log_level);

  // Request an adapter from the implementation. We only get one of these and it
  // may expose multiple devices so it's effectively what we consider a driver.
  const WGPURequestAdapterOptions adapter_options = {
      .nextInChain = NULL,
      .compatibleSurface = NULL,
      .powerPreference = options->power_preference,
      .forceFallbackAdapter = false,
  };
  iree_hal_webgpu_native_adapter_data_t adapter_data;
  memset(&adapter_data, 0, sizeof(adapter_data));
  wgpuInstanceRequestAdapter(driver->instance, &adapter_options,
                             iree_hal_webgpu_native_request_adapter_callback,
                             (void*)&adapter_data);
  if (iree_status_is_ok(adapter_data.status)) {
    IREE_ASSERT_NE(adapter_data.handle, NULL);
    driver->adapter = adapter_data.handle;

    WGPUAdapterProperties adapter_props;
    memset(&adapter_props, 0, sizeof(adapter_props));
    wgpuAdapterGetProperties(adapter_data.handle, &adapter_props);

    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }

  IREE_TRACE_ZONE_END(z0);
  return adapter_data.status;
}

static void iree_hal_webgpu_native_driver_destroy(
    iree_hal_driver_t* base_driver) {
  iree_hal_webgpu_native_driver_t* driver =
      iree_hal_webgpu_native_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: there's no wgpu-native teardown for adapters and instances.
  driver->adapter = NULL;
  driver->instance = NULL;

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_webgpu_native_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t allocator,
    iree_hal_device_info_t** out_device_infos,
    iree_host_size_t* out_device_info_count) {
  // Unfortunately no queries in WebGPU; we can only request a single device.
  static const iree_hal_device_info_t device_infos[1] = {
      {
          .device_id = IREE_HAL_WEBGPU_DEVICE_ID_DEFAULT,
          .name = iree_string_view_literal("default"),
      },
  };
  *out_device_info_count = IREE_ARRAYSIZE(device_infos);
  return iree_allocator_clone(
      allocator, iree_make_const_byte_span(device_infos, sizeof(device_infos)),
      (void**)out_device_infos);
}

static iree_status_t iree_hal_webgpu_native_driver_create_device(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_webgpu_native_driver_t* driver =
      iree_hal_webgpu_native_driver_cast(base_driver);

  const WGPURequiredLimits required_limits = {
      .nextInChain = NULL,
      .limits =
          {
              .maxBindGroups = 4,
              .maxStorageBuffersPerShaderStage = 8,
          },
  };
  const WGPUDeviceDescriptor device_descriptor = {
      .nextInChain = NULL,
      // TODO(benvanik): pull from NUL terminated driver storage.
      // .label = device->identifier,
      .requiredFeaturesCount = 0,
      .requiredFeatures = NULL,
      .requiredLimits = &required_limits,
  };
  iree_hal_webgpu_native_device_data_t device_data;
  memset(&device_data, 0, sizeof(device_data));
  wgpuAdapterRequestDevice(driver->adapter, &device_descriptor,
                           iree_hal_webgpu_native_request_device_callback,
                           (void*)&device_data);
  IREE_RETURN_IF_ERROR(device_data.status);
  IREE_ASSERT_NE(device_data.handle, NULL);

  return iree_hal_webgpu_wrap_device(
      driver->identifier, &driver->default_options, device_data.handle,
      driver->host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_webgpu_native_driver_vtable = {
    .destroy = iree_hal_webgpu_native_driver_destroy,
    .query_available_devices =
        iree_hal_webgpu_native_driver_query_available_devices,
    .create_device = iree_hal_webgpu_native_driver_create_device,
};
