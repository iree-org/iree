// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/instrument_util.h"

#include <memory.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/modules/hal/types.h"

//===----------------------------------------------------------------------===//
// Instrument data management
//===----------------------------------------------------------------------===//

IREE_FLAG(string, instrument_file, "",
          "File to populate with instrument data from the program.");

static iree_status_t iree_tooling_write_iovec(iree_vm_ref_t iovec, FILE* file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  bool write_ok = false;
  if (iree_vm_buffer_isa(iovec)) {
    iree_vm_buffer_t* buffer = iree_vm_buffer_deref(iovec);
    IREE_TRACE_ZONE_APPEND_VALUE(z0, (int64_t)iree_vm_buffer_length(buffer));
    write_ok =
        fwrite(iree_vm_buffer_data(buffer), 1, iree_vm_buffer_length(buffer),
               file) == iree_vm_buffer_length(buffer);
  } else if (iree_hal_buffer_view_isa(iovec)) {
    iree_hal_buffer_view_t* buffer_view = iree_hal_buffer_view_deref(iovec);
    IREE_TRACE_ZONE_APPEND_VALUE(
        z0, (int64_t)iree_hal_buffer_view_byte_length(buffer_view));
    iree_hal_buffer_mapping_t mapping;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_buffer_map_range(iree_hal_buffer_view_buffer(buffer_view),
                                      IREE_HAL_MAPPING_MODE_SCOPED,
                                      IREE_HAL_MEMORY_ACCESS_READ, 0,
                                      IREE_WHOLE_BUFFER, &mapping));
    write_ok = fwrite(mapping.contents.data, 1, mapping.contents.data_length,
                      file) == mapping.contents.data_length;
    IREE_IGNORE_ERROR(iree_hal_buffer_unmap_range(&mapping));
  }
  IREE_TRACE_ZONE_END(z0);
  return write_ok ? iree_ok_status()
                  : iree_make_status(iree_status_code_from_errno(errno),
                                     "failed to write iovec to file");
}

iree_status_t iree_tooling_process_instrument_data(
    iree_vm_context_t* context, iree_allocator_t host_allocator) {
  // If no flag was specified we ignore instrument data.
  if (strlen(FLAG_instrument_file) == 0) return iree_ok_status();

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, FLAG_instrument_file);

  // Open the file for overwriting. We do this even if there is no instrument
  // data in the program as we'd rather have the user end up with a 0-byte file
  // when they explicitly ask for it instead of stale data from previous runs.
  FILE* file = fopen(FLAG_instrument_file, "wb");
  if (!file) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open instrument file '%s' for writing",
                            FLAG_instrument_file);
  }

  // Each query function pushes iovecs on to a list we provide; we create one
  // list and use that across all of them.
  iree_vm_list_t* iovec_list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_list_create(NULL, 8, host_allocator, &iovec_list));

  iree_vm_list_t* input_list = NULL;
  iree_status_t status =
      iree_vm_list_create(NULL, 8, host_allocator, &input_list);
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t iovec_list_ref = iree_vm_list_retain_ref(iovec_list);
    status = iree_vm_list_push_ref_move(input_list, &iovec_list_ref);
  }

  // Process instrument data from all modules in the context.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < iree_vm_context_module_count(context);
         ++i) {
      iree_vm_module_t* module = iree_vm_context_module_at(context, i);
      if (!module) continue;

      // Find the query function, if present.
      iree_vm_function_t query_func;
      iree_status_t lookup_status = iree_vm_module_lookup_function_by_name(
          module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
          IREE_SV("__query_instruments"), &query_func);
      if (!iree_status_is_ok(lookup_status)) {
        // Skip missing/invalid query function.
        iree_status_ignore(lookup_status);
        continue;
      }

      IREE_TRACE_ZONE_BEGIN(z1);
      IREE_TRACE_ZONE_APPEND_TEXT(z1, iree_vm_module_name(module).data,
                                  iree_vm_module_name(module).size);
      status = iree_vm_invoke(context, query_func, IREE_VM_INVOCATION_FLAG_NONE,
                              NULL, input_list, NULL, host_allocator);
      IREE_TRACE_ZONE_END(z1);
      if (!iree_status_is_ok(status)) break;
    }
  }

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < iree_vm_list_size(iovec_list); ++i) {
      iree_vm_ref_t iovec = iree_vm_ref_null();
      status = iree_vm_list_get_ref_assign(iovec_list, i, &iovec);
      if (!iree_status_is_ok(status)) break;
      status = iree_tooling_write_iovec(iovec, file);
      if (!iree_status_is_ok(status)) break;
    }
  }

  iree_vm_list_release(input_list);
  iree_vm_list_release(iovec_list);
  fclose(file);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
