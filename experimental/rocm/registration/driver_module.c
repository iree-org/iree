// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/registration/driver_module.h"

#include <stddef.h>

#include "experimental/rocm/api.h"
#include "iree/base/api.h"
#include "iree/base/internal/flags.h"

// Force using ROCM direct until we support command buffer caching to avoid the
// overhead of graph creation.
IREE_FLAG(
  bool, rocm_use_graph, false,
  "Use ROCM graph for executing command buffers (instead of direct).");

IREE_FLAG(
  bool, rocm_tracing, true,
  "Enables tracing of stream events when Tracy instrumentation is enabled.\n"
  "Severely impacts benchmark timings and should only be used when\n"
  "analyzing dispatch timings.");

static iree_status_t iree_hal_rocm_driver_factory_enumerate(
    void *self, iree_host_size_t *out_driver_info_count,
    const iree_hal_driver_info_t **out_driver_infos) {
  // NOTE: we could query supported ROCM versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_name = iree_string_view_literal("rocm"),
      .full_name = iree_string_view_literal("ROCM (dynamic)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_driver_factory_try_create(
    void *self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t **out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  if (!iree_string_view_equal(driver_name, IREE_SV("rocm"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }
  IREE_TRACE_ZONE_BEGIN(z0);

    iree_hal_rocm_device_params_t default_params;
  iree_hal_rocm_device_params_initialize(&default_params);
  if (FLAG_rocm_use_graph) {
    default_params.command_buffer_mode =
        IREE_HAL_ROCM_COMMAND_BUFFER_MODE_GRAPH;
  }
  default_params.stream_tracing = FLAG_rocm_tracing;
  iree_hal_rocm_driver_options_t driver_options;
  iree_hal_rocm_driver_options_initialize(&driver_options);
  iree_status_t status = iree_hal_rocm_driver_create(
      driver_name, &default_params, &driver_options, host_allocator, out_driver);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_rocm_driver_module_register(iree_hal_driver_registry_t *registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_rocm_driver_factory_enumerate,
      .try_create = iree_hal_rocm_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
