// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_UTILS_FEATURES_H_
#define IREE_VM_BYTECODE_UTILS_FEATURES_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/utils/isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Formats a buffer usage bitfield as a string.
// See iree_bitfield_format for usage.
iree_string_view_t iree_vm_bytecode_features_format(
    iree_vm_FeatureBits_enum_t value, iree_bitfield_string_temp_t* out_temp);

// Returns the features available in this build of the runtime.
iree_vm_FeatureBits_enum_t iree_vm_bytecode_available_features(void);

// Returns a pretty status reported at |file|/|line| when one or more features
// from |required_features| is missing from |available_features|.
// Returns OK if all required features are available.
iree_status_t iree_vm_check_feature_mismatch(
    const char* file, int line, iree_vm_FeatureBits_enum_t required_features,
    iree_vm_FeatureBits_enum_t available_features);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_UTILS_FEATURES_H_
