// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/transforms/metaspace.h"

#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/normalizer.h"

//===----------------------------------------------------------------------===//
// Metaspace Encode - Non-Split Mode
//===----------------------------------------------------------------------===//

// Non-split mode: transform text and emit as contiguous chunks.
// Output is single logical segment, but may be emitted in multiple batches
// if input is large. Normalization is applied per-codepoint.
static iree_status_t iree_tokenizer_metaspace_encode_nosplit(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_metaspace_config_t* config, iree_string_view_t text,
    iree_host_size_t replacement_length, bool needs_prepend,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  char data[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_host_size_t data_used = 0;
  uint32_t replacement = config->replacement;

  // Prepend replacement if needed.
  if (needs_prepend) {
    data_used += iree_unicode_utf8_encode(replacement, data + data_used);
  }

  iree_host_size_t position = 0;
  while (position < text.size) {
    uint32_t cp = iree_unicode_utf8_decode(text, &position);

    // Normalize this codepoint (may produce 0-4 codepoints).
    uint32_t norm_cps[4];
    iree_host_size_t norm_count;
    iree_tokenizer_normalizer_normalize_codepoint(normalizer, cp, norm_cps,
                                                  &norm_count);

    for (iree_host_size_t i = 0; i < norm_count; ++i) {
      uint32_t ncp = norm_cps[i];
      // Only replace ASCII space (0x20) with the replacement character.
      // Other whitespace (tabs, newlines, etc.) is preserved as-is.
      // This matches HuggingFace tokenizers behavior.
      bool is_space = (ncp == ' ');
      iree_host_size_t char_length =
          is_space ? replacement_length
                   : (iree_host_size_t)iree_unicode_utf8_encoded_length(ncp);

      // Flush if this character would overflow buffer.
      if (data_used + char_length > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
        if (data_used > 0) {
          iree_string_view_t segment = iree_make_string_view(data, data_used);
          iree_string_view_list_t list = {1, &segment};
          IREE_RETURN_IF_ERROR(callback(user_data, list));
          data_used = 0;
        }
      }

      // Write transformed character.
      if (is_space) {
        data_used += iree_unicode_utf8_encode(replacement, data + data_used);
      } else {
        data_used += iree_unicode_utf8_encode(ncp, data + data_used);
      }
    }
  }

  // Flush remaining data.
  if (data_used > 0) {
    iree_string_view_t segment = iree_make_string_view(data, data_used);
    iree_string_view_list_t list = {1, &segment};
    IREE_RETURN_IF_ERROR(callback(user_data, list));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Metaspace Encode - Split Mode
//===----------------------------------------------------------------------===//

// Split mode: transform text and split at word boundaries.
// Each word becomes a segment, with leading replacement character included.
// Segments point into the stack data buffer, so we must emit all complete
// segments before flushing the data buffer. Normalization is applied
// per-codepoint.
static iree_status_t iree_tokenizer_metaspace_encode_split(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_metaspace_config_t* config, iree_string_view_t text,
    iree_host_size_t replacement_length, bool needs_prepend,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  char data[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_string_view_t segments[IREE_TOKENIZER_STRING_BATCH_CAPACITY];
  iree_host_size_t data_used = 0;
  iree_host_size_t segment_count = 0;
  iree_host_size_t segment_start = 0;
  bool in_word = false;
  bool pending_replacement = false;
  uint32_t replacement = config->replacement;

  // Prepend replacement if needed.
  if (needs_prepend) {
    data_used += iree_unicode_utf8_encode(replacement, data + data_used);
    pending_replacement = true;
  }

  iree_host_size_t position = 0;
  while (position < text.size) {
    uint32_t cp = iree_unicode_utf8_decode(text, &position);

    // Normalize this codepoint (may produce 0-4 codepoints).
    uint32_t norm_cps[4];
    iree_host_size_t norm_count;
    iree_tokenizer_normalizer_normalize_codepoint(normalizer, cp, norm_cps,
                                                  &norm_count);

    for (iree_host_size_t n = 0; n < norm_count; ++n) {
      uint32_t ncp = norm_cps[n];
      // Only replace and split on ASCII space (0x20).
      // Other whitespace (tabs, newlines, etc.) is preserved as-is.
      // This matches HuggingFace tokenizers behavior.
      bool is_space = (ncp == ' ');
      iree_host_size_t char_length =
          is_space ? replacement_length
                   : (iree_host_size_t)iree_unicode_utf8_encoded_length(ncp);

      // Check if we need to flush before processing this character.
      if (data_used + char_length > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
        // Emit all complete segments first.
        if (segment_count > 0) {
          iree_string_view_list_t list = {segment_count, segments};
          IREE_RETURN_IF_ERROR(callback(user_data, list));
          segment_count = 0;
        }

        // If in a word, copy in-progress segment data to start of buffer.
        if (in_word) {
          iree_host_size_t in_progress_length = data_used - segment_start;
          if (in_progress_length > 0) {
            memmove(data, data + segment_start, in_progress_length);
          }
          data_used = in_progress_length;
          segment_start = 0;
        } else if (pending_replacement) {
          // Not in a word but have pending replacement - preserve it at start.
          data_used = iree_unicode_utf8_encode(replacement, data);
          segment_start = 0;
        } else {
          data_used = 0;
          segment_start = 0;
        }

        // Safety check: single segment too large.
        if (data_used + char_length > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "single segment exceeds buffer capacity");
        }
      }

      if (is_space) {
        // End current word if any.
        if (in_word) {
          segments[segment_count++] = iree_make_string_view(
              data + segment_start, data_used - segment_start);
          in_word = false;

          // Check if segment array is full.
          if (segment_count == IREE_TOKENIZER_STRING_BATCH_CAPACITY) {
            iree_string_view_list_t list = {segment_count, segments};
            IREE_RETURN_IF_ERROR(callback(user_data, list));
            segment_count = 0;
            data_used = 0;
            segment_start = 0;
          }
        } else if (pending_replacement) {
          // Consecutive space: emit pending replacement as standalone segment.
          // HuggingFace emits a standalone ▁ for each consecutive space.
          segments[segment_count++] = iree_make_string_view(
              data + data_used - replacement_length, replacement_length);

          // Check if segment array is full.
          if (segment_count == IREE_TOKENIZER_STRING_BATCH_CAPACITY) {
            iree_string_view_list_t list = {segment_count, segments};
            IREE_RETURN_IF_ERROR(callback(user_data, list));
            segment_count = 0;
            data_used = 0;
            segment_start = 0;
          }
        }
        // Write replacement character.
        data_used += iree_unicode_utf8_encode(replacement, data + data_used);
        pending_replacement = true;
      } else {
        // Start new word if needed.
        if (!in_word) {
          // Include leading replacement char if there is one pending.
          if (pending_replacement) {
            segment_start = data_used - replacement_length;
          } else {
            segment_start = data_used;
          }
          in_word = true;
          pending_replacement = false;
        }
        // Write normalized character.
        data_used += iree_unicode_utf8_encode(ncp, data + data_used);
      }
    }
  }

  // Flush final segment.
  if (in_word) {
    segments[segment_count++] =
        iree_make_string_view(data + segment_start, data_used - segment_start);
  }

  // Emit remaining segments.
  if (segment_count > 0) {
    iree_string_view_list_t list = {segment_count, segments};
    IREE_RETURN_IF_ERROR(callback(user_data, list));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Metaspace Encode
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_metaspace_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_metaspace_config_t* config, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  IREE_ASSERT_ARGUMENT(config);
  if (text.size == 0) return iree_ok_status();

  // Pre-normalize the entire text before processing. Many normalizers require
  // string-level processing (Strip for whitespace trimming, NFC for canonical
  // reordering, Precompiled for character maps, etc.) that cannot be done
  // codepoint-by-codepoint.
  char normalized_stack_buffer[8192];
  iree_string_view_t normalized_text = text;
  if (normalizer != NULL &&
      normalizer->type != IREE_TOKENIZER_NORMALIZER_NONE) {
    iree_host_size_t normalized_length = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_apply(
        normalizer, text, normalized_stack_buffer,
        sizeof(normalized_stack_buffer), &normalized_length));
    normalized_text =
        iree_make_string_view(normalized_stack_buffer, normalized_length);
  }

  // If normalization produced empty text (e.g., all whitespace stripped), done.
  if (normalized_text.size == 0) return iree_ok_status();

  uint32_t replacement = config->replacement;
  iree_host_size_t replacement_length =
      (iree_host_size_t)iree_unicode_utf8_encoded_length(replacement);
  if (replacement_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid replacement codepoint");
  }

  // Determine if we need to prepend based on prepend_scheme.
  // Only ASCII space (0x20) counts as "starting with space" for prepend logic.
  // Use normalized_text since normalization may have stripped leading spaces.
  iree_host_size_t first_pos = 0;
  uint32_t first_cp =
      normalized_text.size > 0
          ? iree_unicode_utf8_decode(normalized_text, &first_pos)
          : 0;
  bool starts_with_space = (first_cp == ' ');

  bool needs_prepend = false;
  switch (config->prepend_scheme) {
    case IREE_TOKENIZER_PREPEND_ALWAYS:
      needs_prepend = true;
      break;
    case IREE_TOKENIZER_PREPEND_FIRST:
      needs_prepend = !starts_with_space;
      break;
    case IREE_TOKENIZER_PREPEND_NEVER:
      needs_prepend = false;
      break;
  }

  // Dispatch to split or non-split implementation.
  // Pass a NONE normalizer since we already applied normalization above.
  static const iree_tokenizer_normalizer_t kNoneNormalizer = {
      .type = IREE_TOKENIZER_NORMALIZER_NONE};
  if (iree_any_bit_set(config->flags, IREE_TOKENIZER_METASPACE_FLAG_SPLIT)) {
    return iree_tokenizer_metaspace_encode_split(
        &kNoneNormalizer, config, normalized_text, replacement_length,
        needs_prepend, callback, user_data);
  } else {
    return iree_tokenizer_metaspace_encode_nosplit(
        &kNoneNormalizer, config, normalized_text, replacement_length,
        needs_prepend, callback, user_data);
  }
}

//===----------------------------------------------------------------------===//
// Metaspace Decode
//===----------------------------------------------------------------------===//

// Metaspace decoding: replaces the replacement character back to spaces.
// Since replacement chars (e.g., ▁ = 3 bytes) become spaces (1 byte), output is
// always <= input size. This allows safe in-place decoding.
iree_status_t iree_tokenizer_metaspace_decode(
    const iree_tokenizer_metaspace_config_t* config, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(config);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;

  if (text.size == 0) {
    return iree_ok_status();
  }

  uint32_t replacement = config->replacement;
  int replacement_length = iree_unicode_utf8_encoded_length(replacement);
  if (replacement_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid replacement codepoint");
  }

  // Decode directly into the output buffer.
  char* write_ptr = out_buffer;
  iree_host_size_t position = 0;
  bool is_first = true;

  while (position < text.size) {
    uint32_t cp = iree_unicode_utf8_decode(text, &position);
    if (cp == replacement) {
      // First occurrence might be stripped depending on prepend_scheme.
      if (is_first && config->prepend_scheme != IREE_TOKENIZER_PREPEND_NEVER) {
        // Skip leading replacement char for ALWAYS and FIRST.
        is_first = false;
        continue;
      }
      // Check buffer capacity.
      iree_host_size_t written = (iree_host_size_t)(write_ptr - out_buffer);
      if (written >= max_size) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      *write_ptr++ = ' ';
    } else {
      // Non-replacement codepoint: copy as-is.
      int codepoint_length = iree_unicode_utf8_encoded_length(cp);
      iree_host_size_t written = (iree_host_size_t)(write_ptr - out_buffer);
      if (written + codepoint_length > max_size) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      write_ptr += iree_unicode_utf8_encode(cp, write_ptr);
    }
    is_first = false;
  }

  *out_size = (iree_host_size_t)(write_ptr - out_buffer);
  return iree_ok_status();
}
