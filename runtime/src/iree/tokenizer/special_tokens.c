// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/special_tokens.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Special Tokens Lifecycle
//===----------------------------------------------------------------------===//

void iree_tokenizer_special_tokens_initialize(
    iree_tokenizer_special_tokens_t* out_special_tokens) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_special_tokens, 0, sizeof(*out_special_tokens));
  memset(out_special_tokens->first_byte_to_bucket,
         IREE_TOKENIZER_SPECIAL_TOKENS_NO_BUCKET,
         sizeof(out_special_tokens->first_byte_to_bucket));
  IREE_TRACE_ZONE_END(z0);
}

void iree_tokenizer_special_tokens_deinitialize(
    iree_tokenizer_special_tokens_t* special_tokens) {
  if (!special_tokens) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(special_tokens->allocator, special_tokens->slab);
  memset(special_tokens, 0, sizeof(*special_tokens));
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

// Ensures entry array has capacity for at least |min_capacity| entries.
static iree_status_t iree_tokenizer_special_tokens_builder_reserve_entries(
    iree_tokenizer_special_tokens_builder_t* builder,
    iree_host_size_t min_capacity) {
  if (builder->entry_capacity >= min_capacity) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_grow_array(
              builder->allocator, iree_max(16, min_capacity),
              sizeof(iree_tokenizer_special_tokens_builder_entry_t),
              &builder->entry_capacity, (void**)&builder->entries));
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Ensures string table has capacity for at least |min_capacity| bytes.
static iree_status_t iree_tokenizer_special_tokens_builder_reserve_strings(
    iree_tokenizer_special_tokens_builder_t* builder,
    iree_host_size_t min_capacity) {
  if (builder->string_capacity >= min_capacity) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_grow_array(builder->allocator, iree_max(256, min_capacity),
                                /*element_size=*/1, &builder->string_capacity,
                                (void**)&builder->string_data));
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_tokenizer_special_tokens_builder_initialize(
    iree_allocator_t allocator,
    iree_tokenizer_special_tokens_builder_t* out_builder) {
  IREE_ASSERT_ARGUMENT(out_builder);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->allocator = allocator;
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_tokenizer_special_tokens_builder_add(
    iree_tokenizer_special_tokens_builder_t* builder,
    iree_string_view_t content, iree_tokenizer_token_id_t token_id,
    iree_tokenizer_special_token_flags_t flags) {
  IREE_ASSERT_ARGUMENT(builder);

  // Validate content.
  if (content.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "special token content cannot be empty");
  }
  if (content.size > 255) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "special token content exceeds maximum length of 255 bytes");
  }

  // Check for string_size overflow before addition.
  if (builder->string_size > UINT32_MAX - content.size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "string table overflow");
  }

  // Ensure capacity for one more entry.
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_reserve_entries(
      builder, builder->entry_count + 1));

  // Ensure capacity for the string content.
  iree_host_size_t min_string_capacity = builder->string_size + content.size;
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_reserve_strings(
      builder, min_string_capacity));

  // Append string to string table.
  memcpy(builder->string_data + builder->string_size, content.data,
         content.size);

  // Add entry.
  iree_tokenizer_special_tokens_builder_entry_t* entry =
      &builder->entries[builder->entry_count];
  entry->id = token_id;
  entry->string_offset = (uint32_t)builder->string_size;
  entry->string_length = (uint16_t)content.size;
  entry->flags = flags;

  builder->string_size += content.size;
  builder->entry_count++;

  return iree_ok_status();
}

void iree_tokenizer_special_tokens_builder_deinitialize(
    iree_tokenizer_special_tokens_builder_t* builder) {
  if (!builder) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(builder->allocator, builder->entries);
  iree_allocator_free(builder->allocator, builder->string_data);
  memset(builder, 0, sizeof(*builder));
  IREE_TRACE_ZONE_END(z0);
}

// Compares two entries for sorting: first byte ascending, then length
// descending (for longest-match), then lexicographic for determinism.
// |string_data| is passed explicitly for thread safety.
static int iree_tokenizer_special_tokens_compare_entries(
    const iree_tokenizer_special_tokens_builder_entry_t* ea,
    const iree_tokenizer_special_tokens_builder_entry_t* eb,
    const uint8_t* string_data) {
  // Primary: first byte ascending.
  uint8_t first_a = string_data[ea->string_offset];
  uint8_t first_b = string_data[eb->string_offset];
  if (first_a != first_b) return (int)first_a - (int)first_b;

  // Secondary: length descending (longer first for longest-match).
  if (ea->string_length != eb->string_length) {
    return (int)eb->string_length - (int)ea->string_length;
  }

  // Tertiary: lexicographic for determinism.
  return memcmp(string_data + ea->string_offset,
                string_data + eb->string_offset, ea->string_length);
}

// Thread-safe insertion sort with explicit context. Uses O(n²) insertion sort
// which is fast for small arrays (special tokens are typically <1000 entries).
// Avoids qsort global variable thread safety issues.
static void iree_tokenizer_special_tokens_sort_entries(
    iree_tokenizer_special_tokens_builder_entry_t* entries,
    iree_host_size_t entry_count, const uint8_t* string_data) {
  for (iree_host_size_t i = 1; i < entry_count; ++i) {
    iree_tokenizer_special_tokens_builder_entry_t key = entries[i];
    iree_host_size_t j = i;
    while (j > 0 && iree_tokenizer_special_tokens_compare_entries(
                        &key, &entries[j - 1], string_data) < 0) {
      entries[j] = entries[j - 1];
      --j;
    }
    entries[j] = key;
  }
}

// Computes longest common prefix between two strings, capped at 4 bytes.
static iree_host_size_t iree_tokenizer_compute_common_prefix(
    const uint8_t* a, iree_host_size_t a_length, const uint8_t* b,
    iree_host_size_t b_length) {
  iree_host_size_t max_length = iree_min(iree_min(a_length, b_length), 4);
  for (iree_host_size_t i = 0; i < max_length; ++i) {
    if (a[i] != b[i]) return i;
  }
  return max_length;
}

// Counts distinct first bytes in sorted entries to validate bucket limit.
static iree_host_size_t iree_tokenizer_special_tokens_count_buckets(
    const iree_tokenizer_special_tokens_builder_entry_t* entries,
    iree_host_size_t entry_count, const uint8_t* string_data) {
  if (entry_count == 0) return 0;
  iree_host_size_t bucket_count = 1;
  uint8_t prev_first = string_data[entries[0].string_offset];
  for (iree_host_size_t i = 1; i < entry_count; ++i) {
    uint8_t curr_first = string_data[entries[i].string_offset];
    if (curr_first != prev_first) {
      ++bucket_count;
      prev_first = curr_first;
    }
  }
  return bucket_count;
}

iree_status_t iree_tokenizer_special_tokens_builder_build(
    iree_tokenizer_special_tokens_builder_t* builder,
    iree_allocator_t allocator,
    iree_tokenizer_special_tokens_t* out_special_tokens) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_special_tokens);

  // Always initialize output so callers get a valid zeroed struct even when
  // building from an empty builder.
  iree_tokenizer_special_tokens_initialize(out_special_tokens);

  // Empty case: nothing to allocate.
  if (builder->entry_count == 0) {
    return iree_ok_status();
  }

  // Validate entry count fits in uint16_t (bucket start/end indices).
  if (builder->entry_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many special tokens: %" PRIhsz " > %u",
                            builder->entry_count, (unsigned)UINT16_MAX);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Sort entries by first byte, then by length descending.
  // Uses thread-safe insertion sort with explicit context (no global variable).
  iree_tokenizer_special_tokens_sort_entries(
      builder->entries, builder->entry_count, builder->string_data);

  // Validate bucket count before allocating.
  iree_host_size_t bucket_count = 0;
  bucket_count = iree_tokenizer_special_tokens_count_buckets(
      builder->entries, builder->entry_count, builder->string_data);
  if (bucket_count > IREE_TOKENIZER_SPECIAL_TOKENS_MAX_BUCKETS) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(
                IREE_STATUS_RESOURCE_EXHAUSTED,
                "too many distinct special token prefixes: %" PRIhsz " > %d",
                bucket_count, IREE_TOKENIZER_SPECIAL_TOKENS_MAX_BUCKETS));
  }

  // Compute B-string data size and min/max lengths.
  iree_host_size_t bstring_data_size = 0;
  iree_host_size_t min_length = IREE_HOST_SIZE_MAX;
  iree_host_size_t max_length = 0;
  for (iree_host_size_t i = 0; i < builder->entry_count; ++i) {
    iree_host_size_t length = builder->entries[i].string_length;
    iree_host_size_t entry_size = 0;
    if (!iree_host_size_checked_add(1, length, &entry_size) ||
        !iree_host_size_checked_add(bstring_data_size, entry_size,
                                    &bstring_data_size)) {
      IREE_RETURN_AND_END_ZONE(z0,
                               iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                                "B-string data size overflow"));
    }
    if (length < min_length) min_length = length;
    if (length > max_length) max_length = length;
  }

  // Calculate slab layout: [ids][flags][bstring_offsets][bstring_data].
  iree_host_size_t slab_size = 0;
  iree_host_size_t ids_offset = 0;
  iree_host_size_t flags_offset = 0;
  iree_host_size_t offsets_offset = 0;
  iree_host_size_t data_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          /*base_size=*/0, &slab_size,
          IREE_STRUCT_FIELD(builder->entry_count, iree_tokenizer_token_id_t,
                            &ids_offset),
          IREE_STRUCT_FIELD(builder->entry_count,
                            iree_tokenizer_special_token_flags_t,
                            &flags_offset),
          IREE_STRUCT_FIELD(builder->entry_count, uint32_t, &offsets_offset),
          IREE_STRUCT_FIELD(bstring_data_size, uint8_t, &data_offset)));

  // Allocate slab.
  void* slab = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, slab_size, &slab));

  // Set up pointers into slab.
  uint8_t* base = (uint8_t*)slab;
  iree_tokenizer_token_id_t* ids =
      (iree_tokenizer_token_id_t*)(base + ids_offset);
  iree_tokenizer_special_token_flags_t* flags =
      (iree_tokenizer_special_token_flags_t*)(base + flags_offset);
  uint32_t* bstring_offsets = (uint32_t*)(base + offsets_offset);
  uint8_t* bstring_data = base + data_offset;

  // Build buckets and populate slab data.
  uint8_t built_bucket_count = 0;
  iree_host_size_t data_write_pos = 0;
  iree_host_size_t token_index = 0;

  while (token_index < builder->entry_count) {
    // Start new bucket for this first byte.
    iree_tokenizer_special_tokens_bucket_t* bucket =
        &out_special_tokens->buckets[built_bucket_count];
    iree_host_size_t bucket_start = token_index;
    uint8_t first_byte =
        builder->string_data[builder->entries[token_index].string_offset];

    // Find all entries with same first byte.
    iree_host_size_t bucket_end = token_index;
    while (bucket_end < builder->entry_count) {
      uint8_t entry_first =
          builder->string_data[builder->entries[bucket_end].string_offset];
      if (entry_first != first_byte) break;
      ++bucket_end;
    }

    // Compute common prefix for bucket (up to 4 bytes).
    const uint8_t* first_content =
        builder->string_data + builder->entries[bucket_start].string_offset;
    iree_host_size_t first_len = builder->entries[bucket_start].string_length;
    iree_host_size_t prefix_length = iree_min(first_len, 4);

    for (iree_host_size_t i = bucket_start + 1; i < bucket_end; ++i) {
      const uint8_t* content =
          builder->string_data + builder->entries[i].string_offset;
      iree_host_size_t length = builder->entries[i].string_length;
      iree_host_size_t common = iree_tokenizer_compute_common_prefix(
          first_content, prefix_length, content, length);
      if (common < prefix_length) prefix_length = common;
    }
    if (prefix_length == 0) prefix_length = 1;

    // Populate bucket metadata.
    memcpy(bucket->prefix, first_content, prefix_length);
    bucket->prefix_length = (uint8_t)prefix_length;
    bucket->start = (uint16_t)bucket_start;
    bucket->end = (uint16_t)bucket_end;

    // Set first-byte index.
    out_special_tokens->first_byte_to_bucket[first_byte] = built_bucket_count;

    // Copy tokens to slab as B-strings, recording offsets for O(1) lookup.
    for (iree_host_size_t i = bucket_start; i < bucket_end; ++i) {
      iree_tokenizer_special_tokens_builder_entry_t* entry =
          &builder->entries[i];
      ids[i] = entry->id;
      flags[i] = entry->flags;
      bstring_offsets[i] = (uint32_t)data_write_pos;  // O(1) lookup index.
      bstring_data[data_write_pos] = (uint8_t)entry->string_length;
      memcpy(&bstring_data[data_write_pos + 1],
             builder->string_data + entry->string_offset, entry->string_length);
      data_write_pos += 1 + entry->string_length;
    }

    ++built_bucket_count;
    token_index = bucket_end;
  }

  // Commit to output.
  out_special_tokens->count = builder->entry_count;
  out_special_tokens->max_length = max_length;
  out_special_tokens->min_length = min_length;
  out_special_tokens->bucket_count = built_bucket_count;
  out_special_tokens->slab = slab;
  out_special_tokens->ids = ids;
  out_special_tokens->flags = flags;
  out_special_tokens->bstring_offsets = bstring_offsets;
  out_special_tokens->data =
      iree_make_byte_span(bstring_data, bstring_data_size);
  out_special_tokens->allocator = allocator;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Matching
//===----------------------------------------------------------------------===//

// Gets pointer to the B-string at |token_index| in O(1) via precomputed
// offsets. Returns pointer to length byte; content bytes follow immediately.
static inline const uint8_t* iree_tokenizer_special_tokens_get_bstring(
    const iree_tokenizer_special_tokens_t* special_tokens,
    iree_host_size_t token_index) {
  return special_tokens->data.data +
         special_tokens->bstring_offsets[token_index];
}

// Checks if the token at |token_index| can match at the current position given
// its flags and the surrounding context. Returns true if the match is allowed.
//
// |state| provides the previous byte context (for lstrip/single_word).
// |next_byte| is the byte immediately after the match (0 if end of input).
// |at_end| is true if the match ends at the end of the available input.
static bool iree_tokenizer_special_tokens_flags_allow_match(
    const iree_tokenizer_special_tokens_t* special_tokens,
    iree_host_size_t token_index,
    const iree_tokenizer_special_tokens_encode_state_t* state,
    uint8_t next_byte, bool at_end) {
  iree_tokenizer_special_token_flags_t flags =
      special_tokens->flags[token_index];

  // No flags = unconditional match.
  if (flags == IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE) {
    return true;
  }

  // lstrip: Must be preceded by whitespace or start of input.
  if (iree_any_bit_set(flags, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP)) {
    if (!state->at_start_of_input && state->prev_byte_plus_one != 0) {
      uint8_t prev_byte = state->prev_byte_plus_one - 1;
      // lstrip specifically checks for whitespace, not general word boundaries.
      bool prev_is_whitespace =
          (prev_byte <= 0x20);  // Control chars and space.
      if (!prev_is_whitespace) {
        return false;
      }
    }
  }

  // rstrip: Must be followed by whitespace or end of input.
  if (iree_any_bit_set(flags, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP)) {
    if (!at_end) {
      // rstrip specifically checks for whitespace.
      bool next_is_whitespace = (next_byte <= 0x20);
      if (!next_is_whitespace) {
        return false;
      }
    }
  }

  // single_word: Must be a complete word (word boundaries on both sides).
  if (iree_any_bit_set(flags, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_SINGLE_WORD)) {
    // Check left boundary.
    if (!state->at_start_of_input && state->prev_byte_plus_one != 0) {
      uint8_t prev_byte = state->prev_byte_plus_one - 1;
      if (!iree_tokenizer_is_word_boundary_byte(prev_byte)) {
        return false;
      }
    }
    // Check right boundary.
    if (!at_end) {
      if (!iree_tokenizer_is_word_boundary_byte(next_byte)) {
        return false;
      }
    }
  }

  return true;
}

iree_tokenizer_special_tokens_match_result_t
iree_tokenizer_special_tokens_match(
    const iree_tokenizer_special_tokens_t* special_tokens,
    iree_string_view_t input, iree_host_size_t* out_length,
    iree_tokenizer_token_id_t* out_id,
    iree_tokenizer_special_tokens_encode_state_t* state) {
  IREE_ASSERT_ARGUMENT(state);

  if (special_tokens->count == 0 || input.size == 0) {
    return IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH;
  }

  // Continuation path: resuming a partial match from previous NEED_MORE.
  // |input| contains only NEW bytes; state->match_position tracks progress.
  //
  // Important: multiple tokens may share a prefix, so we must re-scan the
  // bucket to find which token(s) still match at the current position.
  // We can't just check the single token we happened to store - that would
  // fail if e.g. we stored <|startoftext|> but the user is typing
  // <|endoftext|>.
  if (state->match_position > 0) {
    // Get bucket from the stored token (all tokens in same bucket share
    // prefix).
    const uint8_t* stored_bstring = iree_tokenizer_special_tokens_get_bstring(
        special_tokens, state->partial_token_index);
    uint8_t first_byte = stored_bstring[1];  // First content byte after length.
    uint8_t bucket_index = special_tokens->first_byte_to_bucket[first_byte];
    const iree_tokenizer_special_tokens_bucket_t* bucket =
        &special_tokens->buckets[bucket_index];

    // Total bytes matched so far (including this input).
    iree_host_size_t total_matched = state->match_position + input.size;

    // Scan all tokens in bucket to find matches at current position.
    uint16_t need_more_token_index = 0;
    bool need_more = false;
    for (uint16_t i = bucket->start; i < bucket->end; ++i) {
      const uint8_t* bstring =
          iree_tokenizer_special_tokens_get_bstring(special_tokens, i);
      uint8_t token_length = bstring[0];
      const uint8_t* token_content = bstring + 1;

      // Skip tokens shorter than what we've already matched.
      if (token_length <= state->match_position) continue;

      // Check if new input matches token at the continuation position.
      iree_host_size_t bytes_to_check =
          iree_min(input.size, token_length - state->match_position);
      if (memcmp(input.data, token_content + state->match_position,
                 bytes_to_check) != 0) {
        continue;  // This token doesn't match.
      }

      // Input matches this token so far.
      if (total_matched >= token_length) {
        // Complete match candidate. Check flags before accepting.
        iree_host_size_t new_bytes_consumed =
            token_length - state->match_position;
        bool at_end = (new_bytes_consumed >= input.size);
        uint8_t next_byte =
            at_end ? 0 : (uint8_t)input.data[new_bytes_consumed];

        if (iree_tokenizer_special_tokens_flags_allow_match(
                special_tokens, i, state, next_byte, at_end)) {
          // Match accepted!
          *out_length = new_bytes_consumed;
          *out_id = special_tokens->ids[i];
          state->match_position = 0;
          return IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED;
        }
        // Flags rejected this match - continue scanning for other tokens.
        continue;
      }

      // Partial match - need more bytes.
      if (!need_more) {
        need_more = true;
        need_more_token_index = i;
      }
    }

    if (need_more) {
      state->match_position = (uint16_t)total_matched;
      state->partial_token_index = need_more_token_index;
      return IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE;
    }

    // No token matched - keep match_position for get_partial() recovery.
    return IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH;
  }

  // Fresh match: first byte lookup.
  uint8_t bucket_index =
      special_tokens->first_byte_to_bucket[(uint8_t)input.data[0]];
  if (bucket_index == IREE_TOKENIZER_SPECIAL_TOKENS_NO_BUCKET) {
    return IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH;
  }

  // Prefix check.
  const iree_tokenizer_special_tokens_bucket_t* bucket =
      &special_tokens->buckets[bucket_index];
  if (input.size < bucket->prefix_length) {
    if (memcmp(input.data, bucket->prefix, input.size) == 0) {
      state->match_position = (uint16_t)input.size;
      state->partial_token_index = bucket->start;
      return IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE;
    }
    return IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH;
  }
  if (memcmp(input.data, bucket->prefix, bucket->prefix_length) != 0) {
    return IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH;
  }

  // Scan bucket tokens (sorted by length descending).
  uint16_t need_more_token_index = 0;
  bool need_more = false;
  for (uint16_t i = bucket->start; i < bucket->end; ++i) {
    const uint8_t* bstring =
        iree_tokenizer_special_tokens_get_bstring(special_tokens, i);
    uint8_t token_length = bstring[0];
    const uint8_t* token_content = bstring + 1;

    if (token_length <= input.size) {
      if (memcmp(input.data, token_content, token_length) == 0) {
        // Content matches. Check flags before accepting.
        bool at_end = (token_length >= input.size);
        uint8_t next_byte = at_end ? 0 : (uint8_t)input.data[token_length];

        if (iree_tokenizer_special_tokens_flags_allow_match(
                special_tokens, i, state, next_byte, at_end)) {
          *out_length = token_length;
          *out_id = special_tokens->ids[i];
          return IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED;
        }
        // Flags rejected - continue scanning for other tokens.
      }
    } else if (memcmp(input.data, token_content, input.size) == 0) {
      if (!need_more) {
        need_more = true;
        need_more_token_index = i;
      }
    }
  }

  if (need_more) {
    state->match_position = (uint16_t)input.size;
    state->partial_token_index = need_more_token_index;
    return IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE;
  }
  return IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH;
}

//===----------------------------------------------------------------------===//
// Encode State
//===----------------------------------------------------------------------===//

iree_host_size_t iree_tokenizer_special_tokens_encode_state_get_partial(
    const iree_tokenizer_special_tokens_encode_state_t* state,
    const iree_tokenizer_special_tokens_t* special_tokens,
    uint8_t* out_buffer) {
  if (state->match_position == 0) return 0;

  // Get the specific token that was being matched. The match() function
  // tracks which token caused NEED_MORE, so we use that exact token's
  // content for reconstruction.
  const uint8_t* bstring = iree_tokenizer_special_tokens_get_bstring(
      special_tokens, state->partial_token_index);
  const uint8_t* token_content = bstring + 1;

  // Copy the matched portion of the token content.
  memcpy(out_buffer, token_content, state->match_position);

  return state->match_position;
}
