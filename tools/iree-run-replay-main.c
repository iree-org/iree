// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/replay/execute.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/device_util.h"

IREE_FLAG_LIST(
    string, replay_file_remap,
    "Remaps captured external file path prefixes before replay opens them. "
    "Repeat as --replay_file_remap=captured_prefix=replay_prefix.");

static iree_status_t iree_tooling_parse_replay_file_remaps(
    iree_allocator_t host_allocator,
    iree_hal_replay_file_path_remap_t** out_file_path_remaps,
    iree_host_size_t* out_file_path_remap_count) {
  IREE_ASSERT_ARGUMENT(out_file_path_remaps);
  IREE_ASSERT_ARGUMENT(out_file_path_remap_count);
  *out_file_path_remaps = NULL;
  *out_file_path_remap_count = 0;

  iree_flag_string_list_t flag_list = FLAG_replay_file_remap_list();
  if (flag_list.count == 0) return iree_ok_status();

  iree_host_size_t remap_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          flag_list.count, sizeof(**out_file_path_remaps), &remap_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay file remap list is too large");
  }
  iree_hal_replay_file_path_remap_t* file_path_remaps = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, remap_size,
                                             (void**)&file_path_remaps));
  memset(file_path_remaps, 0, remap_size);
  for (iree_host_size_t i = 0; i < flag_list.count; ++i) {
    iree_string_view_t captured_prefix;
    iree_string_view_t replay_prefix;
    if (iree_string_view_split(flag_list.values[i], '=', &captured_prefix,
                               &replay_prefix) < 0 ||
        iree_string_view_is_empty(captured_prefix)) {
      iree_allocator_free(host_allocator, file_path_remaps);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--replay_file_remap values must be captured_prefix=replay_prefix");
    }
    file_path_remaps[i].captured_prefix = captured_prefix;
    file_path_remaps[i].replay_prefix = replay_prefix;
  }
  *out_file_path_remaps = file_path_remaps;
  *out_file_path_remap_count = flag_list.count;
  return iree_ok_status();
}

static iree_status_t iree_tooling_create_device_group_from_list(
    iree_hal_device_list_t* device_list, iree_allocator_t host_allocator,
    iree_hal_device_group_t** out_device_group) {
  IREE_ASSERT_ARGUMENT(device_list);
  IREE_ASSERT_ARGUMENT(out_device_group);
  *out_device_group = NULL;

  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), host_allocator,
      &frontier_tracker));

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_async_frontier_tracker_release(frontier_tracker);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < device_list->count && iree_status_is_ok(status); ++i) {
    status = iree_hal_device_group_builder_add_device(&builder,
                                                      device_list->devices[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_builder_finalize(&builder, host_allocator,
                                                    out_device_group);
  } else {
    iree_hal_device_group_builder_deinitialize(&builder);
  }
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "iree-run-replay",
      "Executes an IREE HAL replay file against one or more HAL devices.\n"
      "\n"
      "The replay stream is executed deterministically in capture order. Use\n"
      "--device= to select the target HAL device, matching iree-run-module.\n"
      "HAL-native profiling flags capture only replay execution, not tool\n"
      "setup or teardown.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();
  if (argc != 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected one replay file path argument");
  }

  iree_io_file_contents_t* file_contents = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_map(iree_make_cstring_view(argv[1]),
                                       IREE_IO_FILE_ACCESS_READ, host_allocator,
                                       &file_contents);
  }

  iree_async_proactor_pool_t* proactor_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), host_allocator,
        &proactor_pool);
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_list_t* device_list = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_devices_from_flags(
        iree_hal_available_driver_registry(), iree_hal_default_device_uri(),
        &create_params, host_allocator, &device_list);
  }

  iree_hal_device_group_t* device_group = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_device_group_from_list(
        device_list, host_allocator, &device_group);
  }

  iree_hal_profiling_from_flags_t* profiling = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        device_group, host_allocator, &profiling);
  }

  iree_hal_replay_file_path_remap_t* file_path_remaps = NULL;
  iree_host_size_t file_path_remap_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_parse_replay_file_remaps(
        host_allocator, &file_path_remaps, &file_path_remap_count);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_replay_execute_options_t options =
        iree_hal_replay_execute_options_default();
    options.file_path_remap_count = file_path_remap_count;
    options.file_path_remaps = file_path_remaps;
    status = iree_hal_replay_execute_file(
        file_contents->const_buffer, device_group, &options, host_allocator);
  }

  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  iree_allocator_free(host_allocator, file_path_remaps);
  iree_hal_device_group_release(device_group);
  iree_hal_device_list_free(device_list);
  iree_async_proactor_pool_release(proactor_pool);
  iree_io_file_contents_free(file_contents);

  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
