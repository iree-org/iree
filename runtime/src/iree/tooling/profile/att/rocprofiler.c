// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/rocprofiler.h"

#include <string.h>

#include "iree/tooling/profile/att/rocm.h"

iree_status_t iree_profile_att_rocprofiler_load(
    iree_string_view_t rocm_library_path, iree_allocator_t host_allocator,
    iree_profile_att_rocprofiler_library_t* out_library) {
  memset(out_library, 0, sizeof(*out_library));
  iree_dynamic_library_t* library = NULL;
  iree_status_t status = iree_profile_att_rocm_load_dynamic_library(
      rocm_library_path, "librocprofiler-sdk.so", host_allocator, &library);

#define IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(target, symbol_name) \
  if (iree_status_is_ok(status)) {                                  \
    status = iree_profile_att_rocm_lookup_symbol(                   \
        library, symbol_name, (void**)&out_library->target);        \
  }
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(
      decoder_create, "rocprofiler_thread_trace_decoder_create");
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(
      decoder_destroy, "rocprofiler_thread_trace_decoder_destroy");
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(
      code_object_load, "rocprofiler_thread_trace_decoder_codeobj_load");
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(trace_decode,
                                         "rocprofiler_trace_decode");
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(
      info_string, "rocprofiler_thread_trace_decoder_info_string");
#undef IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN

  if (iree_status_is_ok(status)) {
    status = iree_profile_att_rocm_resolve_library_dir(
        (void*)out_library->trace_decode, rocm_library_path, host_allocator,
        &out_library->decoder_library_path);
  }

  if (iree_status_is_ok(status)) {
    out_library->library = library;
    library = NULL;
  }

  iree_dynamic_library_release(library);
  return status;
}

void iree_profile_att_rocprofiler_deinitialize(
    iree_allocator_t host_allocator,
    iree_profile_att_rocprofiler_library_t* library) {
  iree_allocator_free(host_allocator, library->decoder_library_path);
  iree_dynamic_library_release(library->library);
  memset(library, 0, sizeof(*library));
}

iree_status_t iree_profile_att_make_rocprofiler_status(
    iree_profile_att_rocprofiler_status_t status, const char* operation) {
  if (status == IREE_PROFILE_ATT_ROCPROFILER_STATUS_SUCCESS) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNKNOWN,
                          "%s failed with rocprofiler status %d", operation,
                          (int)status);
}
