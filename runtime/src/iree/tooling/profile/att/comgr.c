// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/comgr.h"

#include <string.h>

#include "iree/tooling/profile/att/rocm.h"

iree_status_t iree_profile_att_comgr_load(
    iree_string_view_t rocm_library_path, iree_allocator_t host_allocator,
    iree_profile_att_comgr_library_t* out_library) {
  memset(out_library, 0, sizeof(*out_library));
  iree_dynamic_library_t* library = NULL;
  iree_status_t status = iree_profile_att_rocm_load_dynamic_library(
      rocm_library_path, "libamd_comgr.so", host_allocator, &library);

#define IREE_PROFILE_ATT_LOOKUP_COMGR_FN(target, symbol_name) \
  if (iree_status_is_ok(status)) {                            \
    status = iree_profile_att_rocm_lookup_symbol(             \
        library, symbol_name, (void**)&out_library->target);  \
  }
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(status_string, "amd_comgr_status_string");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(create_data, "amd_comgr_create_data");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(release_data, "amd_comgr_release_data");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(set_data, "amd_comgr_set_data");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(get_data_isa_name,
                                   "amd_comgr_get_data_isa_name");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(create_disassembly_info,
                                   "amd_comgr_create_disassembly_info");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(destroy_disassembly_info,
                                   "amd_comgr_destroy_disassembly_info");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(disassemble_instruction,
                                   "amd_comgr_disassemble_instruction");
#undef IREE_PROFILE_ATT_LOOKUP_COMGR_FN

  if (iree_status_is_ok(status)) {
    out_library->library = library;
    library = NULL;
  }

  iree_dynamic_library_release(library);
  return status;
}

void iree_profile_att_comgr_deinitialize(
    iree_profile_att_comgr_library_t* library) {
  iree_dynamic_library_release(library->library);
  memset(library, 0, sizeof(*library));
}

iree_status_t iree_profile_att_make_comgr_status(
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_comgr_status_t status, const char* operation) {
  if (status == IREE_PROFILE_ATT_COMGR_STATUS_SUCCESS) return iree_ok_status();
  const char* status_string = NULL;
  if (comgr->status_string) {
    comgr->status_string(status, &status_string);
  }
  return iree_make_status(IREE_STATUS_UNKNOWN,
                          "%s failed with AMD COMGR "
                          "status %d (%s)",
                          operation, (int)status,
                          status_string ? status_string : "unknown");
}
