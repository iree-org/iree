// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_driver.h"

#include "iree/async/operation.h"
#include "iree/async/platform/js/proactor.h"
#include "iree/async/proactor.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/webgpu/api.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_driver_t
//===----------------------------------------------------------------------===//

// WebGPU has a single default device (one adapter → one device → one queue).
#define IREE_HAL_WEBGPU_DEVICE_ID_DEFAULT 0

typedef struct iree_hal_webgpu_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_webgpu_driver_options_t options;

  // + trailing identifier string storage
} iree_hal_webgpu_driver_t;

static const iree_hal_driver_vtable_t iree_hal_webgpu_driver_vtable;

static iree_hal_webgpu_driver_t* iree_hal_webgpu_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_driver_vtable);
  return (iree_hal_webgpu_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_webgpu_driver_options_initialize(
    iree_hal_webgpu_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

IREE_API_EXPORT iree_status_t iree_hal_webgpu_driver_create(
    iree_string_view_t identifier,
    const iree_hal_webgpu_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_driver = NULL;

  iree_hal_webgpu_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_webgpu_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);
  memcpy(&driver->options, options, sizeof(*options));

  *out_driver = (iree_hal_driver_t*)driver;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_webgpu_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_webgpu_driver_t* driver = iree_hal_webgpu_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_webgpu_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  // WebGPU exposes a single "default" device. The actual adapter/device
  // request happens asynchronously during device creation via bridge imports.
  static const iree_hal_device_info_t device_infos[1] = {
      {
          .device_id = IREE_HAL_WEBGPU_DEVICE_ID_DEFAULT,
          .name = iree_string_view_literal("default"),
      },
  };
  *out_device_info_count = IREE_ARRAYSIZE(device_infos);
  return iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(device_infos, sizeof(device_infos)),
      (void**)out_device_infos);
}

static iree_status_t iree_hal_webgpu_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  // Device info queries (adapter_get_info, device_get_limits) require a live
  // device handle. Since we don't have one at enumeration time, return an
  // empty string. Users can query device limits after device creation.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Async device creation (proactor-driven adapter/device request)
//===----------------------------------------------------------------------===//

// State for a single async handle request (adapter or device). Stack-allocated
// by the caller, submitted to the JS proactor, and polled until completion.
typedef struct iree_hal_webgpu_handle_request_t {
  iree_async_operation_t operation;
  // Written by JS via the out_handle_ptr parameter before posting completion.
  iree_hal_webgpu_handle_t result_handle;
  // Set by the completion callback.
  iree_status_t result_status;
  bool completed;
} iree_hal_webgpu_handle_request_t;

static void iree_hal_webgpu_handle_request_completion_fn(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_handle_request_t* request =
      (iree_hal_webgpu_handle_request_t*)base_operation;
  request->result_status = status;
  request->completed = true;
}

// Submits an async handle request and polls the proactor until it completes.
// This blocks the calling thread via the proactor's poll_wait, which uses
// Atomics.wait on workers.
static iree_status_t iree_hal_webgpu_poll_until_complete(
    iree_async_proactor_t* proactor,
    iree_hal_webgpu_handle_request_t* request) {
  while (!request->completed) {
    iree_status_t poll_status =
        iree_async_proactor_poll(proactor, iree_infinite_timeout(), NULL);
    // DEADLINE_EXCEEDED means poll found nothing this iteration — retry.
    if (!iree_status_is_ok(poll_status) &&
        !iree_status_is_deadline_exceeded(poll_status)) {
      return poll_status;
    }
  }
  return request->result_status;
}

// Requests a GPU adapter through the proactor completion ring.
// Blocks until the adapter handle is available.
static iree_status_t iree_hal_webgpu_driver_request_adapter(
    iree_async_proactor_t* proactor,
    iree_hal_webgpu_handle_t* out_adapter_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_handle_request_t request;
  memset(&request, 0, sizeof(request));
  iree_async_operation_initialize(
      &request.operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_webgpu_handle_request_completion_fn, /*user_data=*/NULL);

  uint32_t token = UINT32_MAX;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_js_submit_external(proactor, &request.operation,
                                                 &token));

  iree_hal_webgpu_import_request_adapter(
      /*options_flags=*/0, (uint32_t)(uintptr_t)&request.result_handle, token);

  iree_status_t status =
      iree_hal_webgpu_poll_until_complete(proactor, &request);
  if (iree_status_is_ok(status)) {
    if (request.result_handle == 0) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "WebGPU adapter request succeeded but "
                                "returned a null handle");
    } else {
      *out_adapter_handle = request.result_handle;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Requests a GPU device from an adapter through the proactor completion ring.
// Blocks until the device handle is available.
static iree_status_t iree_hal_webgpu_driver_request_device(
    iree_async_proactor_t* proactor, iree_hal_webgpu_handle_t adapter_handle,
    iree_hal_webgpu_handle_t* out_device_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_handle_request_t request;
  memset(&request, 0, sizeof(request));
  iree_async_operation_initialize(
      &request.operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_webgpu_handle_request_completion_fn, /*user_data=*/NULL);

  uint32_t token = UINT32_MAX;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_js_submit_external(proactor, &request.operation,
                                                 &token));

  iree_hal_webgpu_import_adapter_request_device(
      adapter_handle, (uint32_t)(uintptr_t)&request.result_handle, token);

  iree_status_t status =
      iree_hal_webgpu_poll_until_complete(proactor, &request);
  if (iree_status_is_ok(status)) {
    if (request.result_handle == 0) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "WebGPU device request succeeded but "
                                "returned a null handle");
    } else {
      *out_device_handle = request.result_handle;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Creates a WebGPU device by driving request_adapter → adapter_request_device
// through the proactor completion ring. This blocks the calling thread via
// Atomics.wait (worker mode only).
static iree_status_t iree_hal_webgpu_driver_create_device_async(
    iree_hal_webgpu_driver_t* driver,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Async device creation requires blocking waits (Atomics.wait), which are
  // only available on Web Workers. On the main thread, callers must obtain
  // the device handle from JS and use iree_hal_webgpu_device_create() directly.
  if (!iree_hal_webgpu_import_can_block()) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "WebGPU device creation through the driver requires blocking waits "
        "(Atomics.wait), which are not available on the browser main thread; "
        "use iree_hal_webgpu_device_create() with a pre-obtained device "
        "handle from JS");
  }

  // Get a proactor from the pool for the async operations.
  iree_async_proactor_t* proactor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_async_proactor_pool_get(create_params->proactor_pool, 0, &proactor));

  iree_hal_webgpu_handle_t adapter_handle = 0;
  iree_status_t status =
      iree_hal_webgpu_driver_request_adapter(proactor, &adapter_handle);

  // Once we have an adapter, request a device from it.
  iree_hal_webgpu_handle_t device_handle = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_driver_request_device(proactor, adapter_handle,
                                                   &device_handle);
  }

  // Release the adapter handle — the device holds its own reference to the
  // underlying GPU adapter internally.
  iree_hal_webgpu_import_handle_release(adapter_handle);

  // Wrap the obtained device handle in a HAL device. The driver owns the
  // device handle — if creation fails, we destroy it ourselves.
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_device_create(
        driver->identifier, device_handle,
        IREE_HAL_WEBGPU_DEVICE_FLAG_OWNS_DEVICE_HANDLE, create_params,
        host_allocator, out_device);
    if (!iree_status_is_ok(status) && device_handle) {
      iree_hal_webgpu_import_device_destroy(device_handle);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Creates a device, checking first for a pre-configured device handle provided
// by the JS host. This supports inline-mode deployments (browser main thread,
// node.js) where the host creates the GPUDevice before wasm starts and passes
// it in via the bridge. If no pre-configured device is available, falls through
// to the async proactor-driven adapter → device request path.
static iree_status_t iree_hal_webgpu_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_webgpu_driver_t* driver = iree_hal_webgpu_driver_cast(base_driver);

  // Check for a pre-configured device provided by the JS host.
  iree_hal_webgpu_handle_t preconfigured =
      iree_hal_webgpu_import_get_preconfigured_device();
  if (preconfigured != 0) {
    // The host owns the device handle — do not set OWNS_DEVICE_HANDLE.
    return iree_hal_webgpu_device_create(
        driver->identifier, preconfigured, IREE_HAL_WEBGPU_DEVICE_FLAG_NONE,
        create_params, host_allocator, out_device);
  }

  return iree_hal_webgpu_driver_create_device_async(driver, create_params,
                                                    host_allocator, out_device);
}

static iree_status_t iree_hal_webgpu_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_webgpu_driver_t* driver = iree_hal_webgpu_driver_cast(base_driver);

  // Check for pre-configured device (same as by_id — WebGPU has a single
  // default device so path is ignored).
  iree_hal_webgpu_handle_t preconfigured =
      iree_hal_webgpu_import_get_preconfigured_device();
  if (preconfigured != 0) {
    return iree_hal_webgpu_device_create(
        driver->identifier, preconfigured, IREE_HAL_WEBGPU_DEVICE_FLAG_NONE,
        create_params, host_allocator, out_device);
  }

  return iree_hal_webgpu_driver_create_device_async(driver, create_params,
                                                    host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_webgpu_driver_vtable = {
    .destroy = iree_hal_webgpu_driver_destroy,
    .query_available_devices = iree_hal_webgpu_driver_query_available_devices,
    .dump_device_info = iree_hal_webgpu_driver_dump_device_info,
    .create_device_by_id = iree_hal_webgpu_driver_create_device_by_id,
    .create_device_by_path = iree_hal_webgpu_driver_create_device_by_path,
};
