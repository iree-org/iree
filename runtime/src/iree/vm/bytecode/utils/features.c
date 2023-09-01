// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/utils/features.h"

// clang-format off
static const iree_bitfield_string_mapping_t iree_vm_bytecode_feature_mappings[] = {
  {iree_vm_FeatureBits_EXT_F32, IREE_SVL("EXT_F32")},
  {iree_vm_FeatureBits_EXT_F64, IREE_SVL("EXT_F64")},
};
// clang-format on

iree_string_view_t iree_vm_bytecode_features_format(
    iree_vm_FeatureBits_enum_t value, iree_bitfield_string_temp_t* out_temp) {
  return iree_bitfield_format_inline(
      value, IREE_ARRAYSIZE(iree_vm_bytecode_feature_mappings),
      iree_vm_bytecode_feature_mappings, out_temp);
}

iree_vm_FeatureBits_enum_t iree_vm_bytecode_available_features(void) {
  iree_vm_FeatureBits_enum_t result = 0;
#if IREE_VM_EXT_F32_ENABLE
  result |= iree_vm_FeatureBits_EXT_F32;
#endif  // IREE_VM_EXT_F32_ENABLE
#if IREE_VM_EXT_F64_ENABLE
  result |= iree_vm_FeatureBits_EXT_F64;
#endif  // IREE_VM_EXT_F64_ENABLE
  return result;
}

iree_status_t iree_vm_check_feature_mismatch(
    const char* file, int line, iree_vm_FeatureBits_enum_t required_features,
    iree_vm_FeatureBits_enum_t available_features) {
  if (IREE_LIKELY(iree_all_bits_set(available_features, required_features))) {
    return iree_ok_status();
  }
#if IREE_STATUS_MODE
  const iree_vm_FeatureBits_enum_t needed_features =
      required_features & ~available_features;
  iree_bitfield_string_temp_t temp0, temp1, temp2;
  iree_string_view_t available_features_str =
      iree_vm_bytecode_features_format(available_features, &temp0);
  iree_string_view_t required_features_str =
      iree_vm_bytecode_features_format(required_features, &temp1);
  iree_string_view_t needed_features_str =
      iree_vm_bytecode_features_format(needed_features, &temp2);
  return iree_make_status_with_location(
      file, line, IREE_STATUS_INVALID_ARGUMENT,
      "required module features [%.*s] are not available in this runtime "
      "configuration; have [%.*s] while module requires [%.*s]",
      (int)needed_features_str.size, needed_features_str.data,
      (int)available_features_str.size, available_features_str.data,
      (int)required_features_str.size, required_features_str.data);
#else
  return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
#endif  // IREE_STATUS_MODE
}
