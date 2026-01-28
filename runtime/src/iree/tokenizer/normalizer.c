// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer.h"

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Normalizer Lifecycle
//===----------------------------------------------------------------------===//

void iree_tokenizer_normalizer_deinitialize(
    iree_tokenizer_normalizer_t* normalizer) {
  if (!normalizer) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (normalizer->type == IREE_TOKENIZER_NORMALIZER_SEQUENCE) {
    // Recursively deinitialize children.
    for (iree_host_size_t i = 0; i < normalizer->config.sequence.count; ++i) {
      iree_tokenizer_normalizer_deinitialize(
          &normalizer->config.sequence.children[i]);
    }
    // Free the children array.
    if (normalizer->config.sequence.children) {
      iree_allocator_free(normalizer->config.sequence.allocator,
                          normalizer->config.sequence.children);
    }
  } else if (normalizer->type == IREE_TOKENIZER_NORMALIZER_PRECOMPILED) {
    // Free the trie and normalized string.
    iree_allocator_t allocator = normalizer->config.precompiled.allocator;
    if (normalizer->config.precompiled.trie) {
      iree_allocator_free(allocator, normalizer->config.precompiled.trie);
    }
    if (normalizer->config.precompiled.normalized) {
      iree_allocator_free(allocator, normalizer->config.precompiled.normalized);
    }
  } else if (normalizer->type == IREE_TOKENIZER_NORMALIZER_REPLACE_REGEX) {
    // Free the compiled DFA data.
    if (normalizer->config.replace_regex.dfa_data) {
      iree_allocator_free(normalizer->config.replace_regex.allocator,
                          normalizer->config.replace_regex.dfa_data);
    }
  }
  memset(normalizer, 0, sizeof(*normalizer));
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Normalizer Initializers
//===----------------------------------------------------------------------===//

void iree_tokenizer_normalizer_initialize_none(
    iree_tokenizer_normalizer_t* out_normalizer) {
  memset(out_normalizer, 0, sizeof(*out_normalizer));
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_NONE;
}

void iree_tokenizer_normalizer_initialize_bert(
    iree_tokenizer_bert_normalizer_flags_t flags,
    iree_tokenizer_normalizer_t* out_normalizer) {
  memset(out_normalizer, 0, sizeof(*out_normalizer));
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_BERT;
  out_normalizer->config.bert.flags = flags;
}

void iree_tokenizer_normalizer_initialize_lowercase(
    iree_tokenizer_normalizer_t* out_normalizer) {
  memset(out_normalizer, 0, sizeof(*out_normalizer));
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_LOWERCASE;
}

void iree_tokenizer_normalizer_initialize_strip_accents(
    iree_tokenizer_normalizer_t* out_normalizer) {
  memset(out_normalizer, 0, sizeof(*out_normalizer));
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_STRIP_ACCENTS;
}

iree_status_t iree_tokenizer_normalizer_initialize_sequence(
    iree_tokenizer_normalizer_t* children, iree_host_size_t count,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t* out_normalizer) {
  IREE_ASSERT_ARGUMENT(out_normalizer);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_normalizer, 0, sizeof(*out_normalizer));

  if (count == 0) {
    out_normalizer->type = IREE_TOKENIZER_NORMALIZER_NONE;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Check for overflow in children array size.
  if (count > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_normalizer_t)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence child count overflow");
  }
  iree_host_size_t children_size = count * sizeof(iree_tokenizer_normalizer_t);

  // Allocate children array and move children into it.
  iree_tokenizer_normalizer_t* owned_children = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, children_size, (void**)&owned_children));

  // Move children: copy structs and zero originals to transfer ownership.
  memcpy(owned_children, children, children_size);
  memset(children, 0, children_size);

  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_SEQUENCE;
  out_normalizer->config.sequence.count = count;
  out_normalizer->config.sequence.children = owned_children;
  out_normalizer->config.sequence.allocator = allocator;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_tokenizer_normalizer_initialize_prepend(
    iree_string_view_t prepend, iree_tokenizer_normalizer_t* out_normalizer) {
  memset(out_normalizer, 0, sizeof(*out_normalizer));
  if (prepend.size > IREE_TOKENIZER_PREPEND_NORMALIZER_MAX_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepend string too long: %zu > %d", prepend.size,
                            IREE_TOKENIZER_PREPEND_NORMALIZER_MAX_SIZE);
  }
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_PREPEND;
  memcpy(out_normalizer->config.prepend.prepend, prepend.data, prepend.size);
  out_normalizer->config.prepend.length = (uint8_t)prepend.size;
  return iree_ok_status();
}

iree_status_t iree_tokenizer_normalizer_initialize_replace(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_tokenizer_normalizer_t* out_normalizer) {
  memset(out_normalizer, 0, sizeof(*out_normalizer));
  if (pattern.size > IREE_TOKENIZER_REPLACE_NORMALIZER_PATTERN_MAX_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pattern string too long: %zu > %d", pattern.size,
                            IREE_TOKENIZER_REPLACE_NORMALIZER_PATTERN_MAX_SIZE);
  }
  if (content.size > IREE_TOKENIZER_REPLACE_NORMALIZER_CONTENT_MAX_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "content string too long: %zu > %d", content.size,
                            IREE_TOKENIZER_REPLACE_NORMALIZER_CONTENT_MAX_SIZE);
  }
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_REPLACE;
  memcpy(out_normalizer->config.replace.pattern, pattern.data, pattern.size);
  out_normalizer->config.replace.pattern_length = (uint8_t)pattern.size;
  memcpy(out_normalizer->config.replace.content, content.data, content.size);
  out_normalizer->config.replace.content_length = (uint8_t)content.size;
  return iree_ok_status();
}

void iree_tokenizer_normalizer_initialize_nfc(
    iree_tokenizer_normalizer_t* out_normalizer) {
  memset(out_normalizer, 0, sizeof(*out_normalizer));
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_NFC;
}

void iree_tokenizer_normalizer_initialize_strip(
    bool strip_left, bool strip_right,
    iree_tokenizer_normalizer_t* out_normalizer) {
  memset(out_normalizer, 0, sizeof(*out_normalizer));
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_STRIP;
  out_normalizer->config.strip.strip_left = strip_left;
  out_normalizer->config.strip.strip_right = strip_right;
}

//===----------------------------------------------------------------------===//
// Precompiled Initializer
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_normalizer_initialize_precompiled(
    const uint8_t* data, iree_host_size_t data_length,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t* out_normalizer) {
  IREE_ASSERT_ARGUMENT(data);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_normalizer, 0, sizeof(*out_normalizer));

  // Minimum size: 4 bytes for trie_size header.
  if (data_length < 4) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "precompiled data too small: %zu bytes",
                            data_length);
  }

  // Parse the trie size (little-endian u32).
  uint32_t trie_size = (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
                       ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);

  // Validate trie size.
  if (trie_size > data_length - 4) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid trie size %u, data length %zu", trie_size,
                            data_length);
  }
  if (trie_size % 4 != 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trie size %u not divisible by 4", trie_size);
  }

  iree_host_size_t trie_count = trie_size / 4;
  iree_host_size_t normalized_length = data_length - 4 - trie_size;

  // Initialize config fields to NULL before allocation for cleanup on error.
  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_PRECOMPILED;
  out_normalizer->config.precompiled.trie = NULL;
  out_normalizer->config.precompiled.normalized = NULL;
  out_normalizer->config.precompiled.allocator = allocator;

  // Allocate and copy trie array.
  iree_status_t status =
      iree_allocator_malloc(allocator, trie_count * sizeof(uint32_t),
                            (void**)&out_normalizer->config.precompiled.trie);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_normalizer_deinitialize(out_normalizer);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Parse trie elements (little-endian u32).
  const uint8_t* trie_data = data + 4;
  for (iree_host_size_t i = 0; i < trie_count; ++i) {
    out_normalizer->config.precompiled.trie[i] =
        (uint32_t)trie_data[i * 4 + 0] | ((uint32_t)trie_data[i * 4 + 1] << 8) |
        ((uint32_t)trie_data[i * 4 + 2] << 16) |
        ((uint32_t)trie_data[i * 4 + 3] << 24);
  }
  out_normalizer->config.precompiled.trie_count = trie_count;

  // Allocate and copy normalized string.
  if (normalized_length > 0) {
    status = iree_allocator_malloc(
        allocator, normalized_length,
        (void**)&out_normalizer->config.precompiled.normalized);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_normalizer_deinitialize(out_normalizer);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    memcpy(out_normalizer->config.precompiled.normalized, data + 4 + trie_size,
           normalized_length);
  }
  out_normalizer->config.precompiled.normalized_length = normalized_length;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Replace-Regex Initializer
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_normalizer_initialize_replace_regex(
    uint8_t* dfa_data, iree_host_size_t dfa_size, iree_string_view_t content,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t* out_normalizer) {
  IREE_ASSERT_ARGUMENT(dfa_data);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_normalizer, 0, sizeof(*out_normalizer));

  if (content.size > 31) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement content too long: %zu > 31",
                            content.size);
  }

  // Load the DFA from the binary data.
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(dfa_data, dfa_size), &dfa);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  out_normalizer->type = IREE_TOKENIZER_NORMALIZER_REPLACE_REGEX;
  out_normalizer->config.replace_regex.dfa_data = dfa_data;
  out_normalizer->config.replace_regex.dfa_size = dfa_size;
  out_normalizer->config.replace_regex.dfa = dfa;
  memcpy(out_normalizer->config.replace_regex.content, content.data,
         content.size);
  out_normalizer->config.replace_regex.content_length = (uint8_t)content.size;
  out_normalizer->config.replace_regex.allocator = allocator;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// None Normalizer (passthrough)
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_apply_none(
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  if (input.size > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "output buffer too small: need %zu, have %zu",
                            input.size, capacity);
  }
  memcpy(out_buffer, input.data, input.size);
  *out_length = input.size;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lowercase Normalizer
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_apply_lowercase(
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  iree_host_size_t position = 0;
  iree_host_size_t out_pos = 0;

  while (position < input.size) {
    uint32_t cp = iree_unicode_utf8_decode(input, &position);
    cp = iree_unicode_to_lower(cp);

    int encoded_length = iree_unicode_utf8_encoded_length(cp);
    if (out_pos + encoded_length > capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    out_pos += iree_unicode_utf8_encode(cp, out_buffer + out_pos);
  }

  *out_length = out_pos;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Strip Accents Normalizer
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_apply_strip_accents(
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  iree_host_size_t position = 0;
  iree_host_size_t out_pos = 0;

  while (position < input.size) {
    uint32_t cp = iree_unicode_utf8_decode(input, &position);

    // Get NFD base character (strips combining marks).
    cp = iree_unicode_nfd_base(cp);

    // Skip combining marks (category M).
    if (iree_unicode_is_mark(cp)) {
      continue;
    }

    int encoded_length = iree_unicode_utf8_encoded_length(cp);
    if (out_pos + encoded_length > capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    out_pos += iree_unicode_utf8_encode(cp, out_buffer + out_pos);
  }

  *out_length = out_pos;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Prepend Normalizer
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_apply_prepend(
    const iree_tokenizer_prepend_normalizer_config_t* config,
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  iree_host_size_t total_size = config->length + input.size;
  if (total_size > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "output buffer too small: need %zu, have %zu",
                            total_size, capacity);
  }
  // Copy prepend string.
  memcpy(out_buffer, config->prepend, config->length);
  // Copy input after prepend.
  memcpy(out_buffer + config->length, input.data, input.size);
  *out_length = total_size;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Replace Normalizer
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_apply_replace(
    const iree_tokenizer_replace_normalizer_config_t* config,
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  // Empty pattern is a passthrough.
  if (config->pattern_length == 0) {
    if (input.size > capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    memcpy(out_buffer, input.data, input.size);
    *out_length = input.size;
    return iree_ok_status();
  }

  iree_host_size_t out_pos = 0;
  iree_host_size_t in_pos = 0;

  while (in_pos < input.size) {
    // Check if pattern matches at current position.
    bool match = false;
    if (in_pos + config->pattern_length <= input.size) {
      match = (memcmp(input.data + in_pos, config->pattern,
                      config->pattern_length) == 0);
    }

    if (match) {
      // Replace pattern with content.
      if (out_pos + config->content_length > capacity) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      memcpy(out_buffer + out_pos, config->content, config->content_length);
      out_pos += config->content_length;
      in_pos += config->pattern_length;
    } else {
      // Copy single byte.
      if (out_pos + 1 > capacity) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      out_buffer[out_pos++] = input.data[in_pos++];
    }
  }

  *out_length = out_pos;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Replace-Regex Normalizer
//===----------------------------------------------------------------------===//

// Context for collecting regex matches during replacement.
typedef struct iree_tokenizer_replace_regex_context_t {
  iree_tokenizer_regex_match_t* matches;
  iree_host_size_t count;
  iree_host_size_t capacity;
} iree_tokenizer_replace_regex_context_t;

// Callback for collecting matches.
static iree_status_t iree_tokenizer_replace_regex_collect_match(
    void* user_data, iree_tokenizer_regex_match_t match) {
  iree_tokenizer_replace_regex_context_t* context =
      (iree_tokenizer_replace_regex_context_t*)user_data;
  if (context->count >= context->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many regex matches");
  }
  context->matches[context->count++] = match;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_apply_replace_regex(
    const iree_tokenizer_normalizer_replace_regex_config_t* config,
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  // Stack-allocated match buffer (supports up to 256 matches).
  iree_tokenizer_regex_match_t matches[256];
  iree_tokenizer_replace_regex_context_t context = {
      .matches = matches,
      .count = 0,
      .capacity = 256,
  };

  // Collect all matches.
  iree_status_t status = iree_tokenizer_regex_exec(
      &config->dfa, input, iree_tokenizer_replace_regex_collect_match,
      &context);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // If no matches, copy input unchanged.
  if (context.count == 0) {
    if (input.size > capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    memcpy(out_buffer, input.data, input.size);
    *out_length = input.size;
    return iree_ok_status();
  }

  // Build output: copy non-matched regions, insert replacement for matches.
  iree_host_size_t out_position = 0;
  iree_host_size_t in_position = 0;

  for (iree_host_size_t i = 0; i < context.count; ++i) {
    iree_tokenizer_regex_match_t match = context.matches[i];

    // Copy text before this match.
    if (match.start > in_position) {
      iree_host_size_t prefix_length = match.start - in_position;
      if (out_position + prefix_length > capacity) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      memcpy(out_buffer + out_position, input.data + in_position,
             prefix_length);
      out_position += prefix_length;
    }

    // Insert replacement content.
    if (out_position + config->content_length > capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    memcpy(out_buffer + out_position, config->content, config->content_length);
    out_position += config->content_length;

    in_position = match.end;
  }

  // Copy trailing text after last match.
  if (in_position < input.size) {
    iree_host_size_t suffix_length = input.size - in_position;
    if (out_position + suffix_length > capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    memcpy(out_buffer + out_position, input.data + in_position, suffix_length);
    out_position += suffix_length;
  }

  *out_length = out_position;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Precompiled Normalizer (SentencePiece)
//===----------------------------------------------------------------------===//

// Double-array trie unit accessors.
// Unit bit layout for non-leaf: offset(bits 10-30), shift_flag(bit 9),
//                               has_leaf(bit 8), label(bits 0-7)
// Unit bit layout for leaf: is_leaf(bit 31), value(bits 0-30)

static inline bool iree_precompiled_unit_is_leaf(uint32_t unit) {
  return (unit & 0x80000000) != 0;
}

static inline uint32_t iree_precompiled_unit_value(uint32_t unit) {
  return unit & 0x7FFFFFFF;
}

static inline iree_host_size_t iree_precompiled_unit_offset(uint32_t unit) {
  // offset = (unit >> 10) << (shift * 3) where shift = (unit >> 9) & 1
  return ((iree_host_size_t)(unit >> 10)) << (((unit >> 9) & 1) * 3);
}

static inline bool iree_precompiled_unit_has_leaf(uint32_t unit) {
  return (unit & 0x100) != 0;  // Bit 8.
}

static inline uint8_t iree_precompiled_unit_label(uint32_t unit) {
  return (uint8_t)(unit & 0xFF);
}

// Performs common prefix search on the double-array trie.
// Returns the offset into the normalized string for the longest match,
// or -1 if no match is found.
static int32_t iree_precompiled_trie_lookup(
    const iree_tokenizer_precompiled_normalizer_config_t* config,
    const uint8_t* key, iree_host_size_t key_length,
    iree_host_size_t* out_matched_length) {
  if (config->trie_count == 0) {
    *out_matched_length = 0;
    return -1;
  }

  iree_host_size_t node_position = 0;
  uint32_t unit = config->trie[node_position];
  node_position ^= iree_precompiled_unit_offset(unit);

  int32_t best_value = -1;
  iree_host_size_t best_length = 0;

  for (iree_host_size_t i = 0; i < key_length; ++i) {
    uint8_t c = key[i];
    if (c == 0) break;

    node_position ^= c;
    if (node_position >= config->trie_count) break;

    unit = config->trie[node_position];
    if (iree_precompiled_unit_label(unit) != c) break;

    node_position ^= iree_precompiled_unit_offset(unit);

    if (iree_precompiled_unit_has_leaf(unit)) {
      if (node_position < config->trie_count) {
        uint32_t leaf_unit = config->trie[node_position];
        if (iree_precompiled_unit_is_leaf(leaf_unit)) {
          best_value = (int32_t)iree_precompiled_unit_value(leaf_unit);
          best_length = i + 1;
        }
      }
    }
  }

  *out_matched_length = best_length;
  return best_value;
}

// Gets the null-terminated normalized string at the given offset.
static iree_string_view_t iree_precompiled_get_normalized(
    const iree_tokenizer_precompiled_normalizer_config_t* config,
    int32_t offset) {
  if (offset < 0 || (iree_host_size_t)offset >= config->normalized_length) {
    return iree_string_view_empty();
  }
  const char* start = config->normalized + offset;
  const char* end = start;
  const char* limit = config->normalized + config->normalized_length;
  while (end < limit && *end != '\0') {
    ++end;
  }
  return iree_make_string_view(start, end - start);
}

static iree_status_t iree_tokenizer_normalizer_apply_precompiled(
    const iree_tokenizer_precompiled_normalizer_config_t* config,
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  iree_host_size_t in_position = 0;
  iree_host_size_t out_position = 0;

  while (in_position < input.size) {
    // Try to find a match in the trie starting at current position.
    iree_host_size_t matched_length = 0;
    int32_t norm_offset = iree_precompiled_trie_lookup(
        config, (const uint8_t*)(input.data + in_position),
        input.size - in_position, &matched_length);

    if (norm_offset >= 0 && matched_length > 0) {
      // Found a match - output the normalized string.
      iree_string_view_t normalized =
          iree_precompiled_get_normalized(config, norm_offset);
      if (out_position + normalized.size > capacity) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      memcpy(out_buffer + out_position, normalized.data, normalized.size);
      out_position += normalized.size;
      in_position += matched_length;
    } else {
      // No match - copy one UTF-8 character as-is.
      // Determine UTF-8 character length from the first byte.
      uint8_t first_byte = (uint8_t)input.data[in_position];
      iree_host_size_t char_length = 1;
      if ((first_byte & 0x80) == 0) {
        char_length = 1;  // ASCII.
      } else if ((first_byte & 0xE0) == 0xC0) {
        char_length = 2;  // 2-byte UTF-8.
      } else if ((first_byte & 0xF0) == 0xE0) {
        char_length = 3;  // 3-byte UTF-8.
      } else if ((first_byte & 0xF8) == 0xF0) {
        char_length = 4;  // 4-byte UTF-8.
      }
      // Clamp to remaining input.
      if (in_position + char_length > input.size) {
        char_length = input.size - in_position;
      }

      if (out_position + char_length > capacity) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      memcpy(out_buffer + out_position, input.data + in_position, char_length);
      out_position += char_length;
      in_position += char_length;
    }
  }

  *out_length = out_position;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Strip Normalizer
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_apply_strip(
    const iree_tokenizer_strip_normalizer_config_t* config,
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  iree_host_size_t start = 0;
  iree_host_size_t end = input.size;

  // Strip left: advance start past whitespace codepoints.
  if (config->strip_left) {
    while (start < end) {
      iree_host_size_t position = start;
      uint32_t codepoint = iree_unicode_utf8_decode(input, &position);
      if (!iree_unicode_is_whitespace(codepoint)) break;
      start = position;
    }
  }

  // Strip right: move end back past whitespace codepoints.
  if (config->strip_right) {
    while (end > start) {
      // Scan backward to find the start of the last codepoint.
      iree_host_size_t previous = end - 1;
      while (previous > start && (input.data[previous] & 0xC0) == 0x80) {
        --previous;
      }
      iree_host_size_t position = previous;
      uint32_t codepoint = iree_unicode_utf8_decode(input, &position);
      if (!iree_unicode_is_whitespace(codepoint)) break;
      end = previous;
    }
  }

  iree_host_size_t result_size = end - start;
  if (result_size > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "output buffer too small: need %zu, have %zu",
                            result_size, capacity);
  }

  memcpy(out_buffer, input.data + start, result_size);
  *out_length = result_size;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BERT Normalizer
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_apply_bert(
    const iree_tokenizer_bert_normalizer_config_t* config,
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  iree_host_size_t position = 0;
  iree_host_size_t out_pos = 0;

  bool clean_text =
      (config->flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT) != 0;
  bool handle_chinese =
      (config->flags &
       IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS) != 0;
  bool strip_accents =
      (config->flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS) != 0;
  bool lowercase =
      (config->flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE) != 0;

  while (position < input.size) {
    uint32_t cp = iree_unicode_utf8_decode(input, &position);

    // Clean text: skip control characters (except whitespace).
    if (clean_text) {
      if (cp == 0 || cp == 0xFFFD) {
        continue;  // Skip null and replacement chars.
      }
      if (iree_unicode_is_control(cp) && !iree_unicode_is_whitespace(cp)) {
        continue;  // Skip control chars except whitespace.
      }
    }

    // Handle Chinese characters: add spaces around CJK.
    if (handle_chinese && iree_unicode_is_cjk(cp)) {
      // Add space before.
      if (out_pos + 1 > capacity) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      out_buffer[out_pos++] = ' ';
    }

    // Strip accents.
    if (strip_accents) {
      cp = iree_unicode_nfd_base(cp);
      if (iree_unicode_is_mark(cp)) {
        continue;  // Skip combining marks.
      }
    }

    // Lowercase.
    if (lowercase) {
      cp = iree_unicode_to_lower(cp);
    }

    // Encode the codepoint.
    int encoded_length = iree_unicode_utf8_encoded_length(cp);
    if (out_pos + encoded_length > capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    out_pos += iree_unicode_utf8_encode(cp, out_buffer + out_pos);

    // Handle Chinese characters: add space after.
    if (handle_chinese && iree_unicode_is_cjk(cp)) {
      if (out_pos + 1 > capacity) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      out_buffer[out_pos++] = ' ';
    }
  }

  *out_length = out_pos;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Sequence Normalizer
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_normalizer_apply_sequence(
    const iree_tokenizer_normalizer_sequence_config_t* config,
    iree_string_view_t input, char* out_buffer, iree_host_size_t capacity,
    iree_host_size_t* out_length) {
  if (config->count == 0) {
    return iree_tokenizer_normalizer_apply_none(input, out_buffer, capacity,
                                                out_length);
  }

  // For sequences, we need to chain the normalizers.
  // We use the output buffer as a ping-pong buffer with a temp buffer.
  // For simplicity, we allocate a temp buffer on the stack.
  char temp_buffer[8192];
  char* buffers[2] = {out_buffer, temp_buffer};
  iree_host_size_t capacities[2] = {capacity, sizeof(temp_buffer)};
  int current_buf = 1;  // Start writing to temp, so final is in out_buffer.

  iree_string_view_t current_input = input;
  iree_host_size_t current_length = 0;

  for (iree_host_size_t i = 0; i < config->count; ++i) {
    int write_buf = 1 - current_buf;  // Alternate buffers.

    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_apply(
        &config->children[i], current_input, buffers[write_buf],
        capacities[write_buf], &current_length));

    current_input = iree_make_string_view(buffers[write_buf], current_length);
    current_buf = write_buf;
  }

  // If the final result is in temp_buffer, copy to out_buffer.
  if (current_buf == 1) {
    if (current_length > capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    memcpy(out_buffer, temp_buffer, current_length);
  }

  *out_length = current_length;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Per-Codepoint Normalization
//===----------------------------------------------------------------------===//

// BERT per-codepoint normalization (no CJK spacing - pre-tokenizer handles it).
static void iree_tokenizer_normalizer_normalize_codepoint_bert(
    const iree_tokenizer_bert_normalizer_config_t* config, uint32_t cp,
    uint32_t* out_codepoints, iree_host_size_t* out_count) {
  bool clean_text =
      (config->flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT) != 0;
  bool strip_accents =
      (config->flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS) != 0;
  bool lowercase =
      (config->flags & IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE) != 0;

  // Clean text: skip control characters (except whitespace).
  if (clean_text) {
    if (cp == 0 || cp == 0xFFFD) {
      *out_count = 0;
      return;
    }
    if (iree_unicode_is_control(cp) && !iree_unicode_is_whitespace(cp)) {
      *out_count = 0;
      return;
    }
  }

  // Strip accents: decompose and skip combining marks.
  if (strip_accents) {
    cp = iree_unicode_nfd_base(cp);
    if (iree_unicode_is_mark(cp)) {
      *out_count = 0;
      return;
    }
  }

  // Lowercase.
  if (lowercase) {
    cp = iree_unicode_to_lower(cp);
  }

  out_codepoints[0] = cp;
  *out_count = 1;
}

void iree_tokenizer_normalizer_normalize_codepoint(
    const iree_tokenizer_normalizer_t* normalizer, uint32_t codepoint,
    uint32_t* out_codepoints, iree_host_size_t* out_count) {
  switch (normalizer->type) {
    case IREE_TOKENIZER_NORMALIZER_NONE:
      out_codepoints[0] = codepoint;
      *out_count = 1;
      return;

    case IREE_TOKENIZER_NORMALIZER_BERT:
      iree_tokenizer_normalizer_normalize_codepoint_bert(
          &normalizer->config.bert, codepoint, out_codepoints, out_count);
      return;

    case IREE_TOKENIZER_NORMALIZER_LOWERCASE:
      out_codepoints[0] = iree_unicode_to_lower(codepoint);
      *out_count = 1;
      return;

    case IREE_TOKENIZER_NORMALIZER_STRIP_ACCENTS: {
      uint32_t cp = iree_unicode_nfd_base(codepoint);
      if (iree_unicode_is_mark(cp)) {
        *out_count = 0;
      } else {
        out_codepoints[0] = cp;
        *out_count = 1;
      }
      return;
    }

    case IREE_TOKENIZER_NORMALIZER_SEQUENCE:
      // For sequences, apply each normalizer in order.
      // Start with the input codepoint, chain through each normalizer.
      out_codepoints[0] = codepoint;
      *out_count = 1;
      for (iree_host_size_t i = 0;
           i < normalizer->config.sequence.count && *out_count > 0; ++i) {
        // Apply to first codepoint only (sequences don't expand in practice).
        uint32_t temp_out[4];
        iree_host_size_t temp_count;
        iree_tokenizer_normalizer_normalize_codepoint(
            &normalizer->config.sequence.children[i], out_codepoints[0],
            temp_out, &temp_count);
        for (iree_host_size_t j = 0; j < temp_count; ++j) {
          out_codepoints[j] = temp_out[j];
        }
        *out_count = temp_count;
      }
      return;

    case IREE_TOKENIZER_NORMALIZER_PREPEND:
      // Prepend is a string-level operation (adds prefix to entire input).
      // At the codepoint level, it's a passthrough.
      out_codepoints[0] = codepoint;
      *out_count = 1;
      return;

    case IREE_TOKENIZER_NORMALIZER_REPLACE:
      // Replace is a string-level operation (pattern can span bytes).
      // At the codepoint level, it's a passthrough.
      out_codepoints[0] = codepoint;
      *out_count = 1;
      return;

    case IREE_TOKENIZER_NORMALIZER_REPLACE_REGEX:
      // Replace-Regex is a string-level operation (pattern can span bytes).
      // At the codepoint level, it's a passthrough.
      out_codepoints[0] = codepoint;
      *out_count = 1;
      return;

    case IREE_TOKENIZER_NORMALIZER_PRECOMPILED:
      // Precompiled uses byte-level trie matching, not codepoint-level.
      // At the codepoint level, it's a passthrough.
      out_codepoints[0] = codepoint;
      *out_count = 1;
      return;

    default:
      // Unknown normalizer - passthrough.
      out_codepoints[0] = codepoint;
      *out_count = 1;
      return;
  }
}

//===----------------------------------------------------------------------===//
// Normalizer Apply Dispatch
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_normalizer_apply(
    const iree_tokenizer_normalizer_t* normalizer, iree_string_view_t input,
    char* out_buffer, iree_host_size_t capacity, iree_host_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(normalizer);
  IREE_ASSERT_ARGUMENT(out_buffer || capacity == 0);
  IREE_ASSERT_ARGUMENT(out_length);

  *out_length = 0;

  switch (normalizer->type) {
    case IREE_TOKENIZER_NORMALIZER_NONE:
      return iree_tokenizer_normalizer_apply_none(input, out_buffer, capacity,
                                                  out_length);
    case IREE_TOKENIZER_NORMALIZER_BERT:
      return iree_tokenizer_normalizer_apply_bert(
          &normalizer->config.bert, input, out_buffer, capacity, out_length);
    case IREE_TOKENIZER_NORMALIZER_LOWERCASE:
      return iree_tokenizer_normalizer_apply_lowercase(input, out_buffer,
                                                       capacity, out_length);
    case IREE_TOKENIZER_NORMALIZER_STRIP_ACCENTS:
      return iree_tokenizer_normalizer_apply_strip_accents(
          input, out_buffer, capacity, out_length);
    case IREE_TOKENIZER_NORMALIZER_SEQUENCE:
      return iree_tokenizer_normalizer_apply_sequence(
          &normalizer->config.sequence, input, out_buffer, capacity,
          out_length);
    case IREE_TOKENIZER_NORMALIZER_PREPEND:
      return iree_tokenizer_normalizer_apply_prepend(
          &normalizer->config.prepend, input, out_buffer, capacity, out_length);
    case IREE_TOKENIZER_NORMALIZER_REPLACE:
      return iree_tokenizer_normalizer_apply_replace(
          &normalizer->config.replace, input, out_buffer, capacity, out_length);
    case IREE_TOKENIZER_NORMALIZER_REPLACE_REGEX:
      return iree_tokenizer_normalizer_apply_replace_regex(
          &normalizer->config.replace_regex, input, out_buffer, capacity,
          out_length);
    case IREE_TOKENIZER_NORMALIZER_NFC:
      return iree_unicode_compose(input, out_buffer, capacity, out_length);
    case IREE_TOKENIZER_NORMALIZER_PRECOMPILED:
      return iree_tokenizer_normalizer_apply_precompiled(
          &normalizer->config.precompiled, input, out_buffer, capacity,
          out_length);
    case IREE_TOKENIZER_NORMALIZER_STRIP:
      return iree_tokenizer_normalizer_apply_strip(
          &normalizer->config.strip, input, out_buffer, capacity, out_length);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unknown normalizer type %d",
                              (int)normalizer->type);
  }
}
