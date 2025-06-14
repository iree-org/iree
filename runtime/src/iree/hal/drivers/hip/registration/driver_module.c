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

IREE_FLAG(bool, hip_async_caching, true,
          "Enables caching for stream-ordered allocations")

IREE_FLAG(
    int32_t, hip_tracing, 2,
    "Controls the verbosity of tracing when Tracy instrumentation is enabled.\n"
    "The impact to benchmark timing becomes more severe as the verbosity\n"
    "increases, and thus should be only enabled when needed.\n"
    "Permissible values are:\n"
    "   0 : stream tracing disabled.\n"
    "   1 : coarse command buffer level tracing enabled.\n"
    "   2 : fine-grained kernel level tracing enabled.\n");

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

// Parses flags and environment variables into a string pair builder.
static iree_status_t iree_hal_hip_driver_parse_flags(
    iree_string_pair_builder_t* builder) {
  const iree_flag_string_list_t dylib_path_list = FLAG_hip_dylib_path_list();

  // hip_dylib_path
  // TODO: make this a single key-value pair (semicolon separated).
  // Repeated fields don't work in things like python dictionaries/JSON and we
  // want to match the environment variable formatting.
  for (iree_host_size_t i = 0; i < dylib_path_list.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_string_pair_builder_add(
        builder, iree_make_string_pair(IREE_SV("hip_dylib_path"),
                                       dylib_path_list.values[i])));
  }

  // bool and int flags
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, IREE_SV("hip_use_streams"), FLAG_hip_use_streams));
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, IREE_SV("hip_allow_inline_execution"),
      FLAG_hip_allow_inline_execution));
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, IREE_SV("hip_async_allocations"), FLAG_hip_async_allocations));
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, IREE_SV("hip_async_caching"), FLAG_hip_async_caching));
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, IREE_SV("hip_tracing"), FLAG_hip_tracing));
  IREE_RETURN_IF_ERROR(iree_string_pair_builder_add_int32(
      builder, IREE_SV("hip_default_index"), FLAG_hip_default_index));

  // If there were no flag-based dylib paths, consult the environment
  // variable.
  // Read from the IREE_HIP_DYLIB_PATH env var and split by ';' (regardless
  // of platform).
  //
  // TODO: make this a single key-value pair (semicolon separated).
  // Repeated fields don't work in things like python dictionaries/JSON and we
  // want to match the environment variable formatting.
  if (dylib_path_list.count == 0) {
    char* raw_dylib_path_env = getenv("IREE_HIP_DYLIB_PATH");
    if (raw_dylib_path_env) {
      iree_string_view_t dylib_path_env =
          iree_make_cstring_view(raw_dylib_path_env);
      IREE_RETURN_IF_ERROR(
          iree_string_pair_builder_emplace_string(builder, &dylib_path_env));
      intptr_t split_index = 0;
      do {
        iree_string_view_t value;
        split_index = iree_string_view_split(dylib_path_env, ';', &value,
                                             &dylib_path_env);
        IREE_RETURN_IF_ERROR(iree_string_pair_builder_add(
            builder, iree_make_string_pair(IREE_SV("hip_dylib_path"), value)));
      } while (split_index != -1);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_hip_try_parse_int32_option(
    iree_string_view_t name, iree_string_view_t value, int32_t* out_value) {
  if (!iree_string_view_atoi_int32(value, out_value)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "option '%*.s' expected an int32 value, got: '%.*s'", (int)name.size,
        name.data, (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_try_parse_bool_option(
    iree_string_view_t name, iree_string_view_t value, bool* out_value) {
  if (iree_string_view_equal(value, IREE_SV("true"))) {
    *out_value = 1;
    return iree_ok_status();
  } else if (iree_string_view_equal(value, IREE_SV("false"))) {
    *out_value = 0;
    return iree_ok_status();
  }
  int32_t int_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_try_parse_int32_option(name, value, &int_value));
  *out_value = int_value != 0;
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
    if (iree_string_view_equal(key, IREE_SV("hip_dylib_path"))) {
      ++dylib_path_count;
    } else if (iree_string_view_equal(key, IREE_SV("hip_use_streams"))) {
      bool use_streams = false;
      IREE_RETURN_IF_ERROR(
          iree_hal_hip_try_parse_bool_option(key, value, &use_streams));
      device_params->command_buffer_mode =
          use_streams ? IREE_HAL_HIP_COMMAND_BUFFER_MODE_STREAM
                      : IREE_HAL_HIP_COMMAND_BUFFER_MODE_GRAPH;
    } else if (iree_string_view_equal(key,
                                      IREE_SV("hip_allow_inline_execution"))) {
      IREE_RETURN_IF_ERROR(iree_hal_hip_try_parse_bool_option(
          key, value, &device_params->allow_inline_execution));
    } else if (iree_string_view_equal(key, IREE_SV("hip_async_allocations"))) {
      IREE_RETURN_IF_ERROR(iree_hal_hip_try_parse_bool_option(
          key, value, &device_params->async_allocations));
    } else if (iree_string_view_equal(key, IREE_SV("hip_async_caching"))) {
      IREE_RETURN_IF_ERROR(iree_hal_hip_try_parse_bool_option(
          key, value, &device_params->async_caching));
    } else if (iree_string_view_equal(key, IREE_SV("hip_tracing"))) {
      IREE_RETURN_IF_ERROR(iree_hal_hip_try_parse_int32_option(
          key, value, &device_params->stream_tracing));
    } else if (iree_string_view_equal(key, IREE_SV("hip_default_index"))) {
      IREE_RETURN_IF_ERROR(iree_hal_hip_try_parse_int32_option(
          key, value, &driver_options->default_device_index));
    } else {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "unrecognized option: '%.*s'", (int)key.size,
                              key.data);
    }
  }

  // Populate dynamic sized values.
  //
  // TODO: make this a single key-value pair (semicolon separated).
  // Repeated fields don't work in things like python dictionaries/JSON and we
  // want to match the environment variable formatting.
  if (dylib_path_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator,
        dylib_path_count * sizeof(driver_options->hip_lib_search_paths[0]),
        (void**)&driver_options->hip_lib_search_paths));
    for (iree_host_size_t i = 0; i < pairs_size; ++i) {
      iree_string_view_t key = pairs[i].key;
      iree_string_view_t value = pairs[i].value;
      if (iree_string_view_equal(key, IREE_SV("hip_dylib_path"))) {
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
