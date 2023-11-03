// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "experimental/cuda2/api.h"
#include "iree/base/api.h"
#include "iree/base/internal/flags.h"

IREE_FLAG(
    bool, cuda2_use_streams, false,
    "Use CUDA streams (instead of graphs) for executing command buffers.");

IREE_FLAG(bool, cuda2_allow_inline_execution, false,
          "Allow command buffers to execute inline against CUDA streams when\n"
          "possible.");

IREE_FLAG(
    bool, cuda2_async_allocations, true,
    "Enables CUDA asynchronous stream-ordered allocations when supported.");

IREE_FLAG(
    bool, cuda2_tracing, true,
    "Enables tracing of stream events when Tracy instrumentation is enabled.\n"
    "Severely impacts benchmark timings and should only be used when\n"
    "analyzing dispatch timings.");

IREE_FLAG(int32_t, cuda2_default_index, 0,
          "Specifies the index of the default CUDA device to use");

IREE_FLAG(bool, cuda2_default_index_from_mpi, true,
          "Infers the default CUDA device index from the PMI_RANK or\n"
          "OMPI_COMM_WORLD_LOCAL_RANK environment variables when set");

static bool iree_try_parse_env_i32(const char* var_name, int32_t* out_value) {
  const char* var_value = getenv(var_name);
  if (!var_value || strlen(var_value) == 0) return false;
  return iree_string_view_atoi_int32(iree_make_cstring_view(var_value),
                                     out_value);
}

// Tries to infer the device index using the local MPI rank from environment
// variables; otherwise returns |default_index|.
//
// This makes it easy to use N devices on a single system when running via
// `mpiexec`.
static int32_t iree_hal_cuda2_infer_device_index_from_env(
    int32_t default_index) {
  // TODO: try more env vars from other implementations. This covers Intel/MS
  // and OpenMPI today.
  int32_t result = 0;
  if (iree_try_parse_env_i32("PMI_RANK", &result) ||
      iree_try_parse_env_i32("OMPI_COMM_WORLD_LOCAL_RANK", &result)) {
    return result;
  }
  return default_index;
}

static iree_status_t iree_hal_cuda2_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  IREE_ASSERT_ARGUMENT(out_driver_info_count);
  IREE_ASSERT_ARGUMENT(out_driver_infos);
  IREE_TRACE_ZONE_BEGIN(z0);

  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_name = IREE_SVL("cuda2"),
      .full_name = IREE_SVL("next-gen NVIDIA CUDA HAL driver (via dylib)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);

  if (!iree_string_view_equal(driver_name, IREE_SV("cuda2"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda2_driver_options_t driver_options;
  iree_hal_cuda2_driver_options_initialize(&driver_options);

  iree_hal_cuda2_device_params_t device_params;
  iree_hal_cuda2_device_params_initialize(&device_params);
  if (FLAG_cuda2_use_streams) {
    device_params.command_buffer_mode =
        IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM;
  }
  device_params.stream_tracing = FLAG_cuda2_tracing;
  device_params.async_allocations = FLAG_cuda2_async_allocations;

  driver_options.default_device_index = FLAG_cuda2_default_index;
  if (FLAG_cuda2_default_index_from_mpi) {
    driver_options.default_device_index =
        iree_hal_cuda2_infer_device_index_from_env(
            driver_options.default_device_index);
  }

  iree_status_t status = iree_hal_cuda2_driver_create(
      driver_name, &driver_options, &device_params, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);

  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_cuda2_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_cuda2_driver_factory_enumerate,
      .try_create = iree_hal_cuda2_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
