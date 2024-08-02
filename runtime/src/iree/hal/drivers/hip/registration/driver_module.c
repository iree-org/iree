// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/hip/api.h"

IREE_FLAG_LIST(
    string, hip_dylib_path,
    "Path to search for an appropriate libamdhip64.so / amdhip64.dll. If any \n"
    "paths are provided, then only the given paths are searched. Otherwise, \n"
    "system heuristics are used to find the dylib. By default, each path is \n"
    "treated as a directory name, but a distinct file can be given which \n"
    "must match exactly by prefixing with 'file:'.")

IREE_FLAG(bool, hip_use_streams, true,
          "Use HIP streams (instead of graphs) for executing command buffers.");

IREE_FLAG(bool, hip_allow_inline_execution, false,
          "Allow command buffers to execute inline against HIP streams when \n"
          "possible.");

IREE_FLAG(
    bool, hip_async_allocations, true,
    "Enables HIP asynchronous stream-ordered allocations when supported.");

IREE_FLAG(
    bool, hip_tracing, true,
    "Enables tracing of stream events when Tracy instrumentation is enabled.\n"
    "Severely impacts benchmark timings and should only be used when\n"
    "analyzing dispatch timings.");

IREE_FLAG(int32_t, hip_default_index, 0,
          "Specifies the index of the default HIP device to use");

static iree_status_t iree_hal_hip_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  IREE_ASSERT_ARGUMENT(out_driver_info_count);
  IREE_ASSERT_ARGUMENT(out_driver_infos);
  IREE_TRACE_ZONE_BEGIN(z0);

  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_name = IREE_SVL("hip"),
      .full_name = IREE_SVL("HIP HAL driver (via dylib)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Option key constants.
static const iree_string_view_t key_hip_dylib_path =
    iree_string_view_literal("hip_dylib_path");
static const iree_string_view_t key_hip_use_streams =
    iree_string_view_literal("hip_use_streams");
static const iree_string_view_t key_hip_allow_inline_execution =
    iree_string_view_literal("hip_allow_inline_execution");
static const iree_string_view_t key_hip_async_allocations =
    iree_string_view_literal("hip_async_allocations");
static const iree_string_view_t key_hip_tracing =
    iree_string_view_literal("hip_tracing");
static const iree_string_view_t key_hip_default_index =
    iree_string_view_literal("hip_default_index");

// Parses flags and environment variables into a string pair builder.
static iree_status_t iree_hal_hip_driver_parse_flags(
    iree_string_pair_builder_t* builder) {
  const iree_flag_string_list_t dylib_path_list = FLAG_hip_dylib_path_list();

  // hip_dylib_path
  for (iree_host_size_t i = 0; i < dylib_path_list.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_string_pair_builder_add(
        builder,
        iree_make_string_pair(key_hip_dylib_path, dylib_path_list.values[i])));
  }

  // bool and int flags
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, key_hip_use_streams, FLAG_hip_use_streams));
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, key_hip_allow_inline_execution,
      FLAG_hip_allow_inline_execution));
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, key_hip_async_allocations, FLAG_hip_async_allocations));
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, key_hip_tracing, FLAG_hip_tracing));
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, key_hip_default_index, FLAG_hip_default_index));

  // If there were no flag-based dylib paths, consult the environment
  // variable.
  // Read from the IREE_HIP_DYLIB_PATH env var and split by ';' (regardless
  // of platform).
  if (dylib_path_list.count == 0) {
    char* raw_dylib_path_env = getenv("IREE_HIP_DYLIB_PATH");
    if (raw_dylib_path_env) {
      iree_string_view_t dylib_path_env =
          iree_make_cstring_view(raw_dylib_path_env);
      IREE_RETURN_IF_ERROR(
          iree_string_pair_builder_emplace_string(builder, &dylib_path_env));
      while (true) {
        iree_string_view_t first, rest;
        intptr_t index =
            iree_string_view_split(dylib_path_env, ';', &first, &rest);
        IREE_RETURN_IF_ERROR(iree_string_pair_builder_add(
            builder, iree_make_string_pair(key_hip_dylib_path, first)));

        if (index < 0) {
          break;
        }

        dylib_path_env = rest;
      }
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_hip_driver_populate_options(
    iree_allocator_t host_allocator,
    iree_hal_hip_driver_options_t* driver_options,
    iree_hal_hip_device_params_t* device_params, iree_host_size_t pairs_size,
    iree_string_pair_t* pairs) {
  // On the first loop, we just count dynamic sized values.
  int dylib_path_count = 0;
  for (iree_host_size_t i = 0; i < pairs_size; ++i) {
    iree_string_view_t key = pairs[i].key;
    iree_string_view_t value = pairs[i].value;
    int32_t ivalue;

    if (iree_string_view_equal(key, key_hip_dylib_path)) {
      ++dylib_path_count;
    } else if (iree_string_view_equal(key, key_hip_use_streams)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'hip_use_streams' expected to be int. Got: '%.*s'",
            (int)value.size, value.data);
      }
      device_params->command_buffer_mode =
          ivalue ? IREE_HAL_HIP_COMMAND_BUFFER_MODE_STREAM
                 : IREE_HAL_HIP_COMMAND_BUFFER_MODE_GRAPH;
    } else if (iree_string_view_equal(key, key_hip_allow_inline_execution)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'hip_allow_inline_execution' expected to be "
            "int. Got: '%.*s'",
            (int)value.size, value.data);
      }
      if (ivalue) {
        device_params->allow_inline_execution = ivalue ? true : false;
      }
    } else if (iree_string_view_equal(key, key_hip_async_allocations)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'hip_async_allocations' expected to be int Got: '%.*s'",
            (int)value.size, value.data);
      }
      device_params->async_allocations = ivalue ? true : false;
    } else if (iree_string_view_equal(key, key_hip_tracing)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'hip_tracing' expected to be int. Got: '%.*s'",
            (int)value.size, value.data);
      }
      device_params->stream_tracing = ivalue ? true : false;
    } else if (iree_string_view_equal(key, key_hip_default_index)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'hip_default_index' expected to be int. Got: '%.*s'",
            (int)value.size, value.data);
        ;
      }
      driver_options->default_device_index = ivalue;
    } else {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Unrecognized options: %.*s", (int)key.size,
                              key.data);
    }
  }

  // Populate dynamic sized values.
  if (dylib_path_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator,
        dylib_path_count * sizeof(driver_options->hip_lib_search_paths[0]),
        (void**)&driver_options->hip_lib_search_paths));
    for (iree_host_size_t i = 0; i < pairs_size; ++i) {
      iree_string_view_t key = pairs[i].key;
      iree_string_view_t value = pairs[i].value;
      if (iree_string_view_equal(key, key_hip_dylib_path)) {
        driver_options->hip_lib_search_paths
            [driver_options->hip_lib_search_path_count++] = value;
      }
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_hip_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);

  if (!iree_string_view_equal(driver_name, IREE_SV("hip"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Option structs.
  iree_hal_hip_driver_options_t driver_options;
  iree_hal_hip_driver_options_initialize(&driver_options);
  iree_hal_hip_device_params_t device_params;
  iree_hal_hip_device_params_initialize(&device_params);

  // Parse flags.
  // TODO: In the future, we will only do this step if not given explicit
  // options.
  iree_string_pair_builder_t flag_option_builder;
  iree_string_pair_builder_initialize(host_allocator, &flag_option_builder);
  iree_status_t status = iree_hal_hip_driver_parse_flags(&flag_option_builder);

  // Parse key/values into option structs.
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_driver_populate_options(
        host_allocator, &driver_options, &device_params,
        iree_string_pair_builder_size(&flag_option_builder),
        iree_string_pair_builder_pairs(&flag_option_builder));
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_hal_hip_driver_create(driver_name, &driver_options, &device_params,
                                   host_allocator, out_driver);
  }

  iree_allocator_free(host_allocator, driver_options.hip_lib_search_paths);
  iree_string_pair_builder_deinitialize(&flag_option_builder);

  IREE_TRACE_ZONE_END(z0);

  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_hip_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_hip_driver_factory_enumerate,
      .try_create = iree_hal_hip_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
