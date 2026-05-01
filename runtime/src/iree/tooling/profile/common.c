// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/common.h"

#include <string.h>

const char* iree_profile_record_type_name(
    iree_hal_profile_file_record_type_t record_type) {
  switch (record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      return "session_begin";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      return "chunk";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      return "session_end";
    default:
      return "unknown";
  }
}

const char* iree_profile_status_code_name(uint32_t status_code) {
  const char* name = iree_status_code_string((iree_status_code_t)status_code);
  return name[0] != '\0' ? name : "UNKNOWN_STATUS";
}

void iree_profile_fprint_json_string(FILE* file, iree_string_view_t value) {
  fputc('"', file);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    uint8_t c = (uint8_t)value.data[i];
    switch (c) {
      case '"':
        fputs("\\\"", file);
        break;
      case '\\':
        fputs("\\\\", file);
        break;
      case '\b':
        fputs("\\b", file);
        break;
      case '\f':
        fputs("\\f", file);
        break;
      case '\n':
        fputs("\\n", file);
        break;
      case '\r':
        fputs("\\r", file);
        break;
      case '\t':
        fputs("\\t", file);
        break;
      default:
        if (c < 0x20) {
          fprintf(file, "\\u%04x", c);
        } else {
          fputc(c, file);
        }
        break;
    }
  }
  fputc('"', file);
}

void iree_profile_fprint_hash_hex(FILE* file, const uint64_t hash[2]) {
  fprintf(file, "%016" PRIx64 "%016" PRIx64, hash[0], hash[1]);
}

bool iree_profile_key_matches(iree_string_view_t key,
                              iree_string_view_t filter) {
  if (iree_string_view_is_empty(filter) ||
      iree_string_view_equal(filter, IREE_SV("*"))) {
    return true;
  }
  return iree_string_view_match_pattern(key, filter);
}

double iree_profile_sqrt_f64(double value) {
  if (value <= 0.0) return 0.0;
  // Keep this standalone C tool free of libm linkage.
  double estimate = value >= 1.0 ? value : 1.0;
  for (int i = 0; i < 32; ++i) {
    estimate = 0.5 * (estimate + value / estimate);
  }
  return estimate;
}

uint64_t iree_profile_index_mix_u64(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31;
  return value;
}

uint64_t iree_profile_index_combine_u64(uint64_t hash, uint64_t value) {
  const uint64_t mixed_value = iree_profile_index_mix_u64(value);
  hash ^= mixed_value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
  return iree_profile_index_mix_u64(hash);
}

void iree_profile_index_deinitialize(iree_profile_index_t* index,
                                     iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, index->entries);
  memset(index, 0, sizeof(*index));
}

static iree_status_t iree_profile_index_reserve_capacity(
    iree_profile_index_t* index, iree_allocator_t host_allocator,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= index->capacity) return iree_ok_status();

  iree_host_size_t new_capacity = index->capacity ? index->capacity : 16;
  while (new_capacity < minimum_capacity) {
    if (IREE_UNLIKELY(new_capacity > IREE_HOST_SIZE_MAX / 2)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile index capacity overflow");
    }
    new_capacity *= 2;
  }

  iree_profile_index_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, new_capacity,
                                                   sizeof(new_entries[0]),
                                                   (void**)&new_entries));
  memset(new_entries, 0, new_capacity * sizeof(new_entries[0]));

  const iree_host_size_t new_capacity_mask = new_capacity - 1;
  for (iree_host_size_t i = 0; i < index->capacity; ++i) {
    iree_profile_index_entry_t entry = index->entries[i];
    if (entry.value_plus_one == 0) continue;
    iree_host_size_t slot = (iree_host_size_t)entry.hash & new_capacity_mask;
    while (new_entries[slot].value_plus_one != 0) {
      slot = (slot + 1) & new_capacity_mask;
    }
    new_entries[slot] = entry;
  }

  iree_allocator_free(host_allocator, index->entries);
  index->entries = new_entries;
  index->capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t iree_profile_index_reserve(iree_profile_index_t* index,
                                         iree_allocator_t host_allocator,
                                         iree_host_size_t minimum_count) {
  if (minimum_count == 0) return iree_ok_status();
  if (IREE_UNLIKELY(minimum_count > IREE_HOST_SIZE_MAX / 2)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile index count overflow");
  }
  return iree_profile_index_reserve_capacity(
      index, host_allocator, iree_max((iree_host_size_t)16, minimum_count * 2));
}

bool iree_profile_index_find(const iree_profile_index_t* index, uint64_t hash,
                             iree_profile_index_match_fn_t match,
                             const void* user_data,
                             iree_host_size_t* out_value) {
  *out_value = 0;
  if (index->capacity == 0) return false;

  const iree_host_size_t capacity_mask = index->capacity - 1;
  iree_host_size_t slot = (iree_host_size_t)hash & capacity_mask;
  for (iree_host_size_t probe_count = 0; probe_count < index->capacity;
       ++probe_count) {
    const iree_profile_index_entry_t* entry = &index->entries[slot];
    if (entry->value_plus_one == 0) return false;
    const iree_host_size_t value = entry->value_plus_one - 1;
    if (entry->hash == hash && match(user_data, value)) {
      *out_value = value;
      return true;
    }
    slot = (slot + 1) & capacity_mask;
  }
  return false;
}

static iree_status_t iree_profile_index_reserve_for_insert(
    iree_profile_index_t* index, iree_allocator_t host_allocator) {
  const iree_host_size_t load_limit = index->capacity - index->capacity / 4;
  if (index->capacity != 0 && index->count + 1 <= load_limit) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(index->count == IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile index count overflow");
  }
  return iree_profile_index_reserve(index, host_allocator, index->count + 1);
}

iree_status_t iree_profile_index_insert(iree_profile_index_t* index,
                                        iree_allocator_t host_allocator,
                                        uint64_t hash, iree_host_size_t value) {
  if (IREE_UNLIKELY(value == IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile index value overflow");
  }
  IREE_RETURN_IF_ERROR(
      iree_profile_index_reserve_for_insert(index, host_allocator));

  const iree_host_size_t capacity_mask = index->capacity - 1;
  iree_host_size_t slot = (iree_host_size_t)hash & capacity_mask;
  while (index->entries[slot].value_plus_one != 0) {
    slot = (slot + 1) & capacity_mask;
  }
  index->entries[slot].hash = hash;
  index->entries[slot].value_plus_one = value + 1;
  ++index->count;
  return iree_ok_status();
}

iree_status_t iree_profile_index_replace(iree_profile_index_t* index,
                                         iree_allocator_t host_allocator,
                                         uint64_t hash,
                                         iree_profile_index_match_fn_t match,
                                         const void* user_data,
                                         iree_host_size_t value) {
  if (IREE_UNLIKELY(value == IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile index value overflow");
  }

  if (index->capacity != 0) {
    const iree_host_size_t capacity_mask = index->capacity - 1;
    iree_host_size_t slot = (iree_host_size_t)hash & capacity_mask;
    for (iree_host_size_t probe_count = 0; probe_count < index->capacity;
         ++probe_count) {
      iree_profile_index_entry_t* entry = &index->entries[slot];
      if (entry->value_plus_one == 0) break;
      if (entry->hash == hash && match(user_data, entry->value_plus_one - 1)) {
        entry->value_plus_one = value + 1;
        return iree_ok_status();
      }
      slot = (slot + 1) & capacity_mask;
    }
  }

  return iree_profile_index_insert(index, host_allocator, hash, value);
}
