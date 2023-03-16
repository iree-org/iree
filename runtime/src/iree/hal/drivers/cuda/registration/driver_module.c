// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/cuda/api.h"

// Force using CUDA streams until we support command buffer caching to avoid the
// overhead of graph creation.
IREE_FLAG(
    bool, cuda_use_streams, true,
    "Use CUDA streams for executing command buffers (instead of graphs).");

IREE_FLAG(bool, cuda_allow_inline_execution, false,
          "Allow command buffers to execute inline against CUDA streams when "
          "possible.");

IREE_FLAG(bool, cuda_tracing, true,
          "Enables tracing of stream events when Tracy instrumentation is "
          "enabled. Severely impacts benchmark timings and should only be used "
          "when analyzing dispatch timings.");

IREE_FLAG(int32_t, cuda_default_index, 0, "Index of the default CUDA device.");

static iree_status_t iree_hal_cuda_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  // NOTE: we could query supported cuda versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_name = iree_string_view_literal("cuda"),
      .full_name = iree_string_view_literal("CUDA (dynamic)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_init_nccl_rank_and_count(
    iree_hal_cuda_device_params_t* params) {
  params->nccl_default_count = 0;
  params->nccl_default_rank = 0;

  char* nprocs_str = getenv("IREE_CUDA_NCCL_NPROCS");
  if (!nprocs_str) {
    return iree_ok_status();
  }
  int32_t nprocs = -1;
  bool parsed =
      iree_string_view_atoi_int32(iree_make_cstring_view(nprocs_str), &nprocs);
  if (!parsed || nprocs <= 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "IREE_CUDA_NCCL_NPROCS has invalid value '%s'; expected integer > 0",
        nprocs_str);
  }
  params->nccl_default_count = nprocs;

  char* procid_str = getenv("IREE_CUDA_NCCL_PROCID");
  if (!procid_str) {
    // Expected PROCID when NPROCS is set.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "IREE_CUDA_NCCL_PROCID must be set when IREE_CUDA_NCCL_NPROCS is set.");
  }
  int32_t procid = -1;
  parsed =
      iree_string_view_atoi_int32(iree_make_cstring_view(procid_str), &procid);
  if (!parsed || procid < 0 || procid >= nprocs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "IREE_CUDA_NCCL_PROCID has invalid value '%s'; expected integer >= 0",
        procid_str);
  }
  params->nccl_default_rank = procid;
  return iree_status_from_code(IREE_STATUS_OK);
}

static iree_status_t iree_hal_cuda_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  if (!iree_string_view_equal(driver_name, IREE_SV("cuda"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_device_params_t default_params;
  iree_hal_cuda_device_params_initialize(&default_params);
  if (FLAG_cuda_use_streams) {
    default_params.command_buffer_mode =
        IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM;
  }
  default_params.allow_inline_execution = FLAG_cuda_allow_inline_execution;
  default_params.stream_tracing = FLAG_cuda_tracing;

  iree_status_t status =
      iree_hal_cuda_init_nccl_rank_and_count(&default_params);

  if (iree_status_is_ok(status)) {
    // Note that nccl_default_id can't be initalized until the driver imports
    // the NCCL symbols from the dynamic library.

    iree_hal_cuda_driver_options_t driver_options;
    iree_hal_cuda_driver_options_initialize(&driver_options);
    driver_options.default_device_index = FLAG_cuda_default_index;

    status = iree_hal_cuda_driver_create(driver_name, &default_params,
                                         &driver_options, host_allocator,
                                         out_driver);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_cuda_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_cuda_driver_factory_enumerate,
      .try_create = iree_hal_cuda_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
