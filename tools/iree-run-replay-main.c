// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/replay/execute.h"
#include "iree/hal/replay/help.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/device_util.h"

IREE_FLAG_LIST(
    string, replay_file_remap,
    "Remaps captured external file path prefixes before replay opens them. "
    "Repeat as --replay_file_remap=captured_prefix=replay_prefix.");
IREE_FLAG_LIST(
    string, replay_executable_substitution,
    "Substitutes a captured executable payload. Repeat as "
    "--replay_executable_substitution=EXECUTABLE_ID=PATH to infer the format, "
    "or --replay_executable_substitution=EXECUTABLE_ID@FORMAT=PATH when the "
    "format must be explicit.");
IREE_FLAG(bool, agents_md, false,
          "Prints an agent-oriented Markdown guide for HAL replay capture and "
          "tooling workflows and exits.");

typedef struct iree_tooling_replay_executable_substitution_t {
  // Captured executable object id to replace.
  iree_hal_replay_object_id_t executable_id;
  // Optional replacement executable format.
  iree_string_view_t executable_format;
  // Replacement executable file path.
  iree_string_view_t source_path;
  // Mapped replacement executable file contents.
  iree_io_file_contents_t* file_contents;
} iree_tooling_replay_executable_substitution_t;

typedef struct iree_tooling_replay_executable_substitution_state_t {
  // Replacement entries parsed from --replay_executable_substitution.
  iree_tooling_replay_executable_substitution_t* entries;
  // Number of entries in |entries|.
  iree_host_size_t entry_count;
} iree_tooling_replay_executable_substitution_state_t;

static void iree_tooling_release_replay_executable_substitutions(
    iree_allocator_t host_allocator,
    iree_tooling_replay_executable_substitution_state_t* state) {
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    iree_io_file_contents_free(state->entries[i].file_contents);
  }
  iree_allocator_free(host_allocator, state->entries);
  memset(state, 0, sizeof(*state));
}

static iree_status_t iree_tooling_parse_replay_executable_substitutions(
    iree_allocator_t host_allocator,
    iree_tooling_replay_executable_substitution_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));

  iree_flag_string_list_t flag_list =
      FLAG_replay_executable_substitution_list();
  if (flag_list.count == 0) return iree_ok_status();

  iree_host_size_t entry_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          flag_list.count, sizeof(*out_state->entries), &entry_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay executable substitution list is too large");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, entry_size,
                                             (void**)&out_state->entries));
  memset(out_state->entries, 0, entry_size);
  out_state->entry_count = flag_list.count;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < flag_list.count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t selector;
    iree_string_view_t path;
    if (iree_string_view_split(flag_list.values[i], '=', &selector, &path) <
            0 ||
        iree_string_view_is_empty(selector) ||
        iree_string_view_is_empty(path)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "--replay_executable_substitution values must be "
                           "EXECUTABLE_ID=PATH or EXECUTABLE_ID@FORMAT=PATH");
      break;
    }

    iree_string_view_t id_string = selector;
    iree_string_view_t executable_format = iree_string_view_empty();
    iree_string_view_t maybe_format;
    if (iree_string_view_split(selector, '@', &id_string, &maybe_format) >= 0) {
      if (iree_string_view_is_empty(id_string) ||
          iree_string_view_is_empty(maybe_format)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--replay_executable_substitution selector must be "
            "EXECUTABLE_ID or EXECUTABLE_ID@FORMAT");
        break;
      }
      executable_format = maybe_format;
    }

    uint64_t executable_id = 0;
    if (!iree_string_view_atoi_uint64(id_string, &executable_id) ||
        executable_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--replay_executable_substitution executable id must be a non-zero "
          "integer");
      break;
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (out_state->entries[j].executable_id == executable_id) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--replay_executable_substitution repeats executable id %" PRIu64,
            executable_id);
        break;
      }
    }
    if (!iree_status_is_ok(status)) break;

    out_state->entries[i].executable_id =
        (iree_hal_replay_object_id_t)executable_id;
    out_state->entries[i].executable_format = executable_format;
    out_state->entries[i].source_path = path;
    status = iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ,
                                       host_allocator,
                                       &out_state->entries[i].file_contents);
  }
  if (!iree_status_is_ok(status)) {
    iree_tooling_release_replay_executable_substitutions(host_allocator,
                                                         out_state);
  }
  return status;
}

static iree_status_t iree_tooling_replay_executable_substitution_callback(
    void* user_data,
    const iree_hal_replay_executable_substitution_request_t* request,
    iree_hal_replay_executable_substitution_t* out_substitution) {
  iree_tooling_replay_executable_substitution_state_t* state =
      (iree_tooling_replay_executable_substitution_state_t*)user_data;
  memset(out_substitution, 0, sizeof(*out_substitution));
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    const iree_tooling_replay_executable_substitution_t* entry =
        &state->entries[i];
    if (entry->executable_id != request->executable_id) continue;
    out_substitution->substitute = true;
    out_substitution->source = entry->source_path;
    out_substitution->executable_format = entry->executable_format;
    out_substitution->executable_data = entry->file_contents->const_buffer;
    return iree_ok_status();
  }
  return iree_ok_status();
}

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

  iree_flags_set_usage("iree-run-replay", iree_hal_replay_run_usage_text());
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agents_md) {
    iree_hal_replay_print_agent_markdown(stdout);
    fflush(stdout);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(EXIT_SUCCESS);
    return EXIT_SUCCESS;
  }

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

  iree_tooling_replay_executable_substitution_state_t executable_substitutions;
  memset(&executable_substitutions, 0, sizeof(executable_substitutions));
  if (iree_status_is_ok(status)) {
    status = iree_tooling_parse_replay_executable_substitutions(
        host_allocator, &executable_substitutions);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_replay_execute_options_t options =
        iree_hal_replay_execute_options_default();
    options.file_path_remap_count = file_path_remap_count;
    options.file_path_remaps = file_path_remaps;
    if (executable_substitutions.entry_count != 0) {
      options.executable_substitution_callback.fn =
          iree_tooling_replay_executable_substitution_callback;
      options.executable_substitution_callback.user_data =
          &executable_substitutions;
    }
    status = iree_hal_replay_execute_file(
        file_contents->const_buffer, device_group, &options, host_allocator);
  }

  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  iree_tooling_release_replay_executable_substitutions(
      host_allocator, &executable_substitutions);
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
