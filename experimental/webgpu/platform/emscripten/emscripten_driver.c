// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/platform/emscripten/emscripten_driver.h"

#include <emscripten.h>

#include "iree/base/tracing.h"

#define IREE_HAL_WEBGPU_DEVICE_ID_DEFAULT 0

//===----------------------------------------------------------------------===//
// Driver and device options
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_webgpu_driver_options_initialize(
    iree_hal_webgpu_driver_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));

  out_options->backend_preference = IREE_HAL_WEBGPU_DRIVER_BACKEND_ANY;
  out_options->log_level = IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_OFF;

  // TODO(benvanik): coming in future spec update. For now go high-perf.
  // out_options->power_preference = WGPUPowerPreference_Undefined;
  out_options->power_preference = WGPUPowerPreference_HighPerformance;

  iree_hal_webgpu_device_options_initialize(&out_options->device_options);
}

//===----------------------------------------------------------------------===//
// Synchronous adapter and device request functions
//===----------------------------------------------------------------------===//

// We use Asyncify for convenience - for now.
//   https://emscripten.org/docs/porting/asyncify.html
//   https://github.com/emscripten-core/emscripten/issues/15746
//   https://github.com/juj/wasm_webgpu/blob/main/lib/lib_webgpu.h
//
// The requestAdapter and requestDevice functions are asynchronous, while the
// HAL API has synchronous driver and device creation. We want to support both
// platform-independent tests (CTS tests, 'check' framework tests) and user
// applications. For platform-independent tests, we keep the APIs synchronous
// by using Asyncify. For user applications, we could pass in a loop or add
// asynchronous APIs that let those applications use async/await directly.
//
// An even simpler solution for user applications is to request an adapter and
// and device purely up in JavaScript, then to pass the already created device
// in via preinitializedWebGPUDevice / emscripten_webgpu_get_device().

#ifdef EM_ASYNC_JS

EM_ASYNC_JS(WGPUAdapter, wgpuInstanceRequestAdapterSync, (), {
  // TODO(scotttodd): WGPURequestAdapterOptions struct
  const adapter = await navigator['gpu']['requestAdapter']();
  // WARNING: this calls functions directly on Emscripten's library_webgpu.js.
  // This is not a stable API!
  const adapterId = WebGPU.mgrAdapter.create(adapter);
  return adapterId;
});

EM_ASYNC_JS(WGPUDevice, wgpuAdapterRequestDeviceSync, (WGPUAdapter adapterId), {
  // WARNING: this calls functions directly on Emscripten's library_webgpu.js.
  // This is not a stable API!
  const adapter = WebGPU.mgrAdapter.get(adapterId);

  // TODO(scotttodd): WGPUDeviceDescriptor struct
  const descriptor = {};
  const device = await adapter['requestDevice'](descriptor);

  const deviceWrapper = {queueId : WebGPU.mgrQueue.create(device["queue"])};
  const deviceId = WebGPU.mgrDevice.create(device, deviceWrapper);
  return deviceId;
});

#else

WGPUAdapter wgpuInstanceRequestAdapterSync() {
  fprintf(stderr, "wgpuInstanceRequestAdapterSync requires -sASYNCIFY\n");
  return NULL;
}

WGPUDevice wgpuAdapterRequestDeviceSync(WGPUAdapter adapterId) {
  fprintf(stderr, "wgpuAdapterRequestDeviceSync requires -sASYNCIFY\n");
  return NULL;
}

#endif  // EM_ASYNC_JS

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_emscripten_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_emscripten_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_webgpu_device_options_t default_options;

  WGPUInstance instance;
  WGPUAdapter adapter;
} iree_hal_webgpu_emscripten_driver_t;

static const iree_hal_driver_vtable_t iree_hal_webgpu_emscripten_driver_vtable;

static iree_hal_webgpu_emscripten_driver_t*
iree_hal_webgpu_emscripten_driver_cast(iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_emscripten_driver_vtable);
  return (iree_hal_webgpu_emscripten_driver_t*)base_value;
}

iree_status_t iree_hal_webgpu_emscripten_driver_create(
    iree_string_view_t identifier,
    const iree_hal_webgpu_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_emscripten_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size + /*NUL=*/1;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_webgpu_emscripten_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;

  iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                    (char*)driver + sizeof(*driver));
  memcpy(&driver->default_options, &options->device_options,
         sizeof(driver->default_options));

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

  // Request an adapter from the implementation. We only get one of these and it
  // may expose multiple devices so it's effectively what we consider a driver.
  // HACKS: sync via Asyncify
  WGPUAdapter adapter = wgpuInstanceRequestAdapterSync();
  if (!adapter) {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "WebGPU requestAdapter() failed to return a WGPUAdapter");
  }
  driver->adapter = adapter;

  WGPUAdapterProperties adapter_props;
  memset(&adapter_props, 0, sizeof(adapter_props));
  wgpuAdapterGetProperties(driver->adapter, &adapter_props);

  *out_driver = (iree_hal_driver_t*)driver;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_webgpu_emscripten_driver_destroy(
    iree_hal_driver_t* base_driver) {
  iree_hal_webgpu_emscripten_driver_t* driver =
      iree_hal_webgpu_emscripten_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(scotttodd): emscripten teardown?
  // driver->adapter = NULL;
  // driver->instance = NULL;

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_webgpu_emscripten_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
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

static iree_status_t iree_hal_webgpu_emscripten_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_webgpu_emscripten_driver_t* driver =
      iree_hal_webgpu_emscripten_driver_cast(base_driver);
  // TODO(scotttodd): dump detailed device info.
  (void)driver;
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_emscripten_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_webgpu_emscripten_driver_t* driver =
      iree_hal_webgpu_emscripten_driver_cast(base_driver);

  // HACKS: sync via Asyncify
  WGPUDevice device = wgpuAdapterRequestDeviceSync(driver->adapter);
  if (!device) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "WebGPU requestDevice() failed to return a WGPUDevice");
  }

  return iree_hal_webgpu_wrap_device(driver->identifier,
                                     &driver->default_options, device,
                                     driver->host_allocator, out_device);
}

static iree_status_t iree_hal_webgpu_emscripten_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  if (!iree_string_view_is_empty(device_path)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "device paths not yet implemented");
  }
  return iree_hal_webgpu_emscripten_driver_create_device_by_id(
      base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
      host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_webgpu_emscripten_driver_vtable =
    {
        .destroy = iree_hal_webgpu_emscripten_driver_destroy,
        .query_available_devices =
            iree_hal_webgpu_emscripten_driver_query_available_devices,
        .dump_device_info = iree_hal_webgpu_emscripten_driver_dump_device_info,
        .create_device_by_id =
            iree_hal_webgpu_emscripten_driver_create_device_by_id,
        .create_device_by_path =
            iree_hal_webgpu_emscripten_driver_create_device_by_path,
};
