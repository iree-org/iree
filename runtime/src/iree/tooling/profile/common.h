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

// Returns the stable text name for a status code used in profile records.
const char* iree_profile_status_code_name(uint32_t status_code);

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

// One entry in an iree_profile_index_t.
typedef struct iree_profile_index_entry_t {
  // Mixed key hash used to select and compare occupied slots.
  uint64_t hash;
  // Stored value plus one; zero marks an empty slot.
  iree_host_size_t value_plus_one;
} iree_profile_index_entry_t;

// Small open-addressed index from caller-defined keys to array row indexes.
//
// Profile tooling keeps rows in ordinary dynamic arrays for compact iteration
// and report rendering. Any path that asks a keyed question about those rows
// should use this index instead of scanning the array. The caller supplies the
// key hash and an equality callback that compares a candidate row index against
// the lookup key.
typedef struct iree_profile_index_t {
  // Open-addressed entry table with a power-of-two capacity.
  iree_profile_index_entry_t* entries;
  // Number of occupied entries in |entries|.
  iree_host_size_t count;
  // Allocated entry count for |entries|.
  iree_host_size_t capacity;
} iree_profile_index_t;

// Returns true when |value| at a candidate row index matches the active lookup.
typedef bool (*iree_profile_index_match_fn_t)(const void* user_data,
                                              iree_host_size_t value);

// Mixes one integer key word into a well-distributed 64-bit hash.
uint64_t iree_profile_index_mix_u64(uint64_t value);

// Combines |value| into an existing mixed hash.
uint64_t iree_profile_index_combine_u64(uint64_t hash, uint64_t value);

// Releases storage owned by |index|.
void iree_profile_index_deinitialize(iree_profile_index_t* index,
                                     iree_allocator_t host_allocator);

// Ensures |index| can hold at least |minimum_count| entries.
iree_status_t iree_profile_index_reserve(iree_profile_index_t* index,
                                         iree_allocator_t host_allocator,
                                         iree_host_size_t minimum_count);

// Looks up a row index by |hash| and caller-supplied key equality.
bool iree_profile_index_find(const iree_profile_index_t* index, uint64_t hash,
                             iree_profile_index_match_fn_t match,
                             const void* user_data,
                             iree_host_size_t* out_value);

// Inserts a new row index for |hash|.
//
// The caller must have already proven that no matching row is present. This
// keeps the append path simple: find first, grow the row array if needed, then
// insert the new row index.
iree_status_t iree_profile_index_insert(iree_profile_index_t* index,
                                        iree_allocator_t host_allocator,
                                        uint64_t hash, iree_host_size_t value);

// Replaces the row index for an existing key or inserts it when absent.
//
// This is for indexes whose key intentionally maps to the newest row with that
// key instead of the first row, such as allocation-id to latest lifecycle.
iree_status_t iree_profile_index_replace(iree_profile_index_t* index,
                                         iree_allocator_t host_allocator,
                                         uint64_t hash,
                                         iree_profile_index_match_fn_t match,
                                         const void* user_data,
                                         iree_host_size_t value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_COMMON_H_
