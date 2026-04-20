// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/replay/execute.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/device_util.h"

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

  if (iree_status_is_ok(status)) {
    iree_hal_replay_execute_options_t options =
        iree_hal_replay_execute_options_default();
    status = iree_hal_replay_execute_file(
        file_contents->const_buffer, device_group, &options, host_allocator);
  }

  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
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
