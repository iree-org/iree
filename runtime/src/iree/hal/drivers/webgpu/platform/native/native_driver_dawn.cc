// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dawn/dawn_proc.h>
#include <dawn/webgpu_cpp.h>
#include <dawn_native/DawnNative.h>

#include <memory>

#include "iree/base/tracing.h"
#include "iree/hal/drivers/webgpu/platform/native/native_driver.h"

#define IREE_HAL_WEBGPU_DEVICE_ID_DEFAULT 0

extern "C" {

//===----------------------------------------------------------------------===//
// wgpu-native callbacks
//===----------------------------------------------------------------------===//

/*
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
*/

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
  dawn_native::Instance* dawn_instance;
  dawn_native::Adapter* dawn_adapter;
} iree_hal_webgpu_native_driver_t;

extern const iree_hal_driver_vtable_t iree_hal_webgpu_native_driver_vtable;

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

  // Dawn requires this to happen before we can call any WebGPU function,
  // including wgpuCreateInstance. Hopefully that changes.
  auto procs = dawn_native::GetProcs();
  dawnProcSetProcs(&procs);

  WGPUInstanceDescriptor instance_descriptor = {0};
  driver->instance = wgpuCreateInstance(&instance_descriptor);
  if (!driver->instance) {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "WebGPU implementation not present or failed to load");
  }

  auto instance = std::make_unique<dawn_native::Instance>();
  instance->DiscoverDefaultAdapters();
  std::vector<dawn_native::Adapter> adapters = instance->GetAdapters();
  dawn_native::Adapter chosenAdapter;
  for (dawn_native::Adapter& adapter : adapters) {
    wgpu::AdapterProperties properties;
    adapter.GetProperties(&properties);
    if (properties.backendType != wgpu::BackendType::Null) {
      chosenAdapter = adapter;
      break;
    }
  }

  driver->adapter = (WGPUAdapter)((void*)0x1u);  // HACK
  driver->dawn_instance = instance.release();
  driver->dawn_adapter = new dawn_native::Adapter(chosenAdapter);

  // Request an adapter from the implementation. We only get one of these and it
  // may expose multiple devices so it's effectively what we consider a driver.
  // DO NOT SUBMIT
  // WGPURequestAdapterOptions adapter_options = {0};
  // adapter_options.powerPreference = options->power_preference;
  // iree_hal_webgpu_native_adapter_data_t adapter_data;
  // memset(&adapter_data, 0, sizeof(adapter_data));
  // wgpuInstanceRequestAdapter(driver->instance, &adapter_options,
  //                            iree_hal_webgpu_native_request_adapter_callback,
  //                            (void*)&adapter_data);
  // if (iree_status_is_ok(adapter_data.status)) {
  //   IREE_ASSERT_NE(adapter_data.handle, NULL);
  //   driver->adapter = adapter_data.handle;

  //   WGPUAdapterProperties adapter_props;
  //   memset(&adapter_props, 0, sizeof(adapter_props));
  //   wgpuAdapterGetProperties(adapter_data.handle, &adapter_props);

  *out_driver = (iree_hal_driver_t*)driver;
  // } else {
  //   iree_hal_driver_release((iree_hal_driver_t*)driver);
  // }

  IREE_TRACE_ZONE_END(z0);
  // return adapter_data.status;
  return iree_ok_status();
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
  delete driver->dawn_adapter;
  delete driver->dawn_instance;

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
          IREE_HAL_WEBGPU_DEVICE_ID_DEFAULT,
          {"default", IREE_ARRAYSIZE("default")},
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

  WGPURequiredLimits required_limits = {0};
  required_limits.limits.maxBindGroups = 4;
  required_limits.limits.maxStorageBuffersPerShaderStage = 8;
  // WGPUDeviceDescriptor device_descriptor = {0};
  // device_descriptor.requiredLimits = &required_limits;
  // iree_hal_webgpu_native_device_data_t device_data;
  // memset(&device_data, 0, sizeof(device_data));
  // wgpuAdapterRequestDevice(driver->adapter, &device_descriptor,
  //                          iree_hal_webgpu_native_request_device_callback,
  //                          (void*)&device_data);
  // IREE_RETURN_IF_ERROR(device_data.status);
  // IREE_ASSERT_NE(device_data.handle, NULL);

  WGPUDevice handle = driver->dawn_adapter->CreateDevice();
  auto device = wgpu::Device::Acquire(handle);
  device.SetUncapturedErrorCallback(
      [](WGPUErrorType error_type, const char* message, void*) {
        const char* error_type_name = "";
        switch (error_type) {
          case WGPUErrorType_Validation:
            error_type_name = "Validation";
            break;
          case WGPUErrorType_OutOfMemory:
            error_type_name = "Out of memory";
            break;
          default:
          case WGPUErrorType_Unknown:
            error_type_name = "Unknown";
            break;
          case WGPUErrorType_DeviceLost:
            error_type_name = "Device lost";
            break;
        }
        fprintf(stderr, "[DAWN %s] %s", error_type_name, message);
      },
      nullptr);
  device.Release();

  return iree_hal_webgpu_wrap_device(driver->identifier,
                                     &driver->default_options, handle,
                                     driver->host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_webgpu_native_driver_vtable = {
    /*.destroy=*/iree_hal_webgpu_native_driver_destroy,
    /*.query_available_devices=*/
    iree_hal_webgpu_native_driver_query_available_devices,
    /*.create_device=*/iree_hal_webgpu_native_driver_create_device,
};

}  // extern "C"
