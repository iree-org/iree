// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_COMMON_H_
#define IREE_TOOLING_PROFILE_COMMON_H_

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/utils/profile_file.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the stable text name for a profile file record type.
const char* iree_profile_record_type_name(
    iree_hal_profile_file_record_type_t record_type);

// Prints |value| as a JSON string literal with control-character escaping.
void iree_profile_fprint_json_string(FILE* file, iree_string_view_t value);

// Prints a two-word 128-bit hash as lower-case hexadecimal.
void iree_profile_fprint_hash_hex(FILE* file, const uint64_t hash[2]);

// Returns true when |filter| matches |key|, treating an empty filter as
// match-all.
bool iree_profile_key_matches(iree_string_view_t key,
                              iree_string_view_t filter);

// Returns sqrt(value) without requiring this standalone tool to link libm.
double iree_profile_sqrt_f64(double value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_COMMON_H_
