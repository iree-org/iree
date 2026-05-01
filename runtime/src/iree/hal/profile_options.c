// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/profile_options.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/profile_sink.h"

//===----------------------------------------------------------------------===//
// iree_hal_device_profiling_options_t
//===----------------------------------------------------------------------===//

struct iree_hal_device_profiling_options_storage_t {
  // Retained sink referenced by the cloned options, or NULL.
  iree_hal_profile_sink_t* sink;
};

static iree_status_t iree_hal_device_profiling_options_add_string_storage(
    iree_host_size_t string_length, iree_host_size_t* inout_storage_length) {
  if (string_length == 0) return iree_ok_status();
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          *inout_storage_length, string_length, inout_storage_length))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profiling options string storage overflow");
  }
  return iree_ok_status();
}

static iree_string_view_t iree_hal_device_profiling_options_clone_string_view(
    iree_string_view_t source, char** inout_storage) {
  if (iree_string_view_is_empty(source)) return iree_string_view_empty();
  char* storage = *inout_storage;
  memcpy(storage, source.data, source.size);
  *inout_storage += source.size;
  return iree_make_string_view(storage, source.size);
}

IREE_API_EXPORT iree_status_t iree_hal_device_profiling_options_clone(
    const iree_hal_device_profiling_options_t* source_options,
    iree_allocator_t host_allocator,
    iree_hal_device_profiling_options_t* out_options,
    iree_hal_device_profiling_options_storage_t** out_storage) {
  IREE_ASSERT_ARGUMENT(source_options);
  IREE_ASSERT_ARGUMENT(out_options);
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_options = (iree_hal_device_profiling_options_t){0};
  *out_storage = NULL;

  iree_string_view_t executable_export_pattern = iree_string_view_empty();
  if (iree_any_bit_set(
          source_options->capture_filter.flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN)) {
    executable_export_pattern =
        source_options->capture_filter.executable_export_pattern;
    if (iree_string_view_is_empty(executable_export_pattern)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "profile capture executable export filter must "
                              "not be empty");
    }
  }

  if (source_options->counter_set_count && !source_options->counter_sets) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hardware counter set selections require a counter_sets array");
  }

  iree_host_size_t counter_name_total_count = 0;
  iree_host_size_t string_storage_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_profiling_options_add_string_storage(
      executable_export_pattern.size, &string_storage_length));
  for (iree_host_size_t i = 0; i < source_options->counter_set_count; ++i) {
    const iree_hal_profile_counter_set_selection_t* source_counter_set =
        &source_options->counter_sets[i];
    if (source_counter_set->counter_name_count &&
        !source_counter_set->counter_names) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "hardware counter set %" PRIhsz
          " has a counter_name_count but no counter_names array",
          i);
    }
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            counter_name_total_count, source_counter_set->counter_name_count,
            &counter_name_total_count))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "hardware counter name count overflow");
    }
    IREE_RETURN_IF_ERROR(iree_hal_device_profiling_options_add_string_storage(
        source_counter_set->name.size, &string_storage_length));
    for (iree_host_size_t j = 0; j < source_counter_set->counter_name_count;
         ++j) {
      IREE_RETURN_IF_ERROR(iree_hal_device_profiling_options_add_string_storage(
          source_counter_set->counter_names[j].size, &string_storage_length));
    }
  }

  iree_host_size_t counter_sets_offset = 0;
  iree_host_size_t counter_names_offset = 0;
  iree_host_size_t string_storage_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_device_profiling_options_storage_t), &total_size,
      IREE_STRUCT_FIELD(source_options->counter_set_count,
                        iree_hal_profile_counter_set_selection_t,
                        &counter_sets_offset),
      IREE_STRUCT_FIELD(counter_name_total_count, iree_string_view_t,
                        &counter_names_offset),
      IREE_STRUCT_FIELD(string_storage_length, char, &string_storage_offset)));

  iree_hal_device_profiling_options_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&storage));
  memset(storage, 0, total_size);

  iree_hal_device_profiling_options_t cloned_options = *source_options;
  if (cloned_options.sink) {
    iree_hal_profile_sink_retain(cloned_options.sink);
    storage->sink = cloned_options.sink;
  }

  char* string_storage = (char*)storage + string_storage_offset;
  cloned_options.capture_filter.executable_export_pattern =
      iree_string_view_empty();
  if (!iree_string_view_is_empty(executable_export_pattern)) {
    cloned_options.capture_filter.executable_export_pattern =
        iree_hal_device_profiling_options_clone_string_view(
            executable_export_pattern, &string_storage);
  }

  if (source_options->counter_set_count) {
    iree_hal_profile_counter_set_selection_t* counter_sets =
        (iree_hal_profile_counter_set_selection_t*)((uint8_t*)storage +
                                                    counter_sets_offset);
    iree_string_view_t* counter_names =
        (iree_string_view_t*)((uint8_t*)storage + counter_names_offset);
    for (iree_host_size_t i = 0; i < source_options->counter_set_count; ++i) {
      const iree_hal_profile_counter_set_selection_t* source_counter_set =
          &source_options->counter_sets[i];
      counter_sets[i] = *source_counter_set;
      counter_sets[i].name =
          iree_hal_device_profiling_options_clone_string_view(
              source_counter_set->name, &string_storage);
      if (source_counter_set->counter_name_count) {
        counter_sets[i].counter_names = counter_names;
        for (iree_host_size_t j = 0; j < source_counter_set->counter_name_count;
             ++j) {
          counter_names[j] =
              iree_hal_device_profiling_options_clone_string_view(
                  source_counter_set->counter_names[j], &string_storage);
        }
        counter_names += source_counter_set->counter_name_count;
      } else {
        counter_sets[i].counter_names = NULL;
      }
    }
    cloned_options.counter_sets = counter_sets;
  } else {
    cloned_options.counter_sets = NULL;
  }

  *out_options = cloned_options;
  *out_storage = storage;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_device_profiling_options_storage_free(
    iree_hal_device_profiling_options_storage_t* storage,
    iree_allocator_t host_allocator) {
  if (!storage) return;
  iree_hal_profile_sink_release(storage->sink);
  iree_allocator_free(host_allocator, storage);
}
