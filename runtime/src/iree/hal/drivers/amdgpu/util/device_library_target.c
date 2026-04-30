// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_library_target.h"

#include "iree/hal/drivers/amdgpu/util/target_id.h"

bool iree_hal_amdgpu_device_library_target_matches_file_arch(
    iree_string_view_t file_arch, iree_string_view_t target) {
  if (iree_string_view_is_empty(target)) return false;
  if (!iree_string_view_starts_with(file_arch, target)) {
    return false;
  }
  iree_string_view_t suffix =
      iree_string_view_remove_prefix(file_arch, target.size);
  return iree_string_view_is_empty(suffix) ||
         iree_string_view_starts_with(suffix, IREE_SV("."));
}

static iree_status_t
iree_hal_amdgpu_device_library_target_append_unique_candidate(
    iree_string_view_t target,
    iree_hal_amdgpu_device_library_target_candidate_list_t* candidates) {
  if (iree_string_view_is_empty(target)) return iree_ok_status();
  for (iree_host_size_t i = 0; i < candidates->count; ++i) {
    if (iree_string_view_equal(target, candidates->values[i].value)) {
      return iree_ok_status();
    }
  }
  if (candidates->count >= IREE_ARRAYSIZE(candidates->values)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU device library target candidate list capacity %" PRIhsz
        " exceeded",
        IREE_ARRAYSIZE(candidates->values));
  }
  iree_hal_amdgpu_device_library_target_candidate_t* candidate =
      &candidates->values[candidates->count];
  if (target.size >= sizeof(candidate->storage)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU device library target candidate length %" PRIhsz " exceeded",
        sizeof(candidate->storage) - 1);
  }
  memcpy(candidate->storage, target.data, target.size);
  candidate->storage[target.size] = 0;
  candidate->value = iree_make_string_view(candidate->storage, target.size);
  ++candidates->count;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_device_library_target_append_target_id_candidate(
    const iree_hal_amdgpu_target_id_t* target_id,
    iree_hal_amdgpu_device_library_target_candidate_list_t* candidates) {
  char target_id_buffer[64] = {0};
  iree_host_size_t target_id_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_target_id_format(target_id, sizeof(target_id_buffer),
                                       target_id_buffer, &target_id_length));
  return iree_hal_amdgpu_device_library_target_append_unique_candidate(
      iree_make_string_view(target_id_buffer, target_id_length), candidates);
}

iree_status_t iree_hal_amdgpu_device_library_target_candidates_from_isa(
    iree_string_view_t isa_name,
    iree_hal_amdgpu_device_library_target_candidate_list_t* out_candidates) {
  IREE_ASSERT_ARGUMENT(out_candidates);
  memset(out_candidates, 0, sizeof(*out_candidates));

  iree_hal_amdgpu_target_id_t agent_target_id;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_target_id_parse_hsa_isa_name(isa_name, &agent_target_id));

  // Try the most specific runtime binary names first. Direct arch names beat
  // code-object target fallbacks because a concrete code object is preferable
  // to a family-generic code object when both are bundled into the runtime.
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_library_target_append_target_id_candidate(
          &agent_target_id, out_candidates));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_library_target_append_unique_candidate(
          agent_target_id.processor, out_candidates));
  if (agent_target_id.kind == IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
    iree_hal_amdgpu_target_id_t code_object_target_id;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_lookup_code_object_target(
        &agent_target_id, &code_object_target_id));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_device_library_target_append_target_id_candidate(
            &code_object_target_id, out_candidates));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_device_library_target_append_unique_candidate(
            code_object_target_id.processor, out_candidates));
  }
  return iree_ok_status();
}
