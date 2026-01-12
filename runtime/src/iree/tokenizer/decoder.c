// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder.h"

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Decoder Lifecycle
//===----------------------------------------------------------------------===//

void iree_tokenizer_decoder_initialize_none(
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->type = IREE_TOKENIZER_DECODER_NONE;
}

void iree_tokenizer_decoder_initialize_wordpiece(
    iree_string_view_t prefix, iree_tokenizer_wordpiece_decoder_flags_t flags,
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->type = IREE_TOKENIZER_DECODER_WORDPIECE;
  out_decoder->config.wordpiece.prefix =
      prefix.data != NULL ? prefix : iree_make_cstring_view("##");
  out_decoder->config.wordpiece.flags = flags;
}

void iree_tokenizer_decoder_initialize_metaspace(
    uint32_t replacement, iree_tokenizer_metaspace_decoder_flags_t flags,
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->type = IREE_TOKENIZER_DECODER_METASPACE;
  out_decoder->config.metaspace.replacement =
      replacement ? replacement : IREE_TOKENIZER_METASPACE_REPLACEMENT;
  out_decoder->config.metaspace.flags = flags;
}

void iree_tokenizer_decoder_initialize_byte_level(
    iree_tokenizer_byte_level_decoder_flags_t flags,
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->type = IREE_TOKENIZER_DECODER_BYTE_LEVEL;
  out_decoder->config.byte_level.flags = flags;
}

void iree_tokenizer_decoder_initialize_bpe(
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->type = IREE_TOKENIZER_DECODER_BPE;
}

iree_status_t iree_tokenizer_decoder_initialize_replace(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  memset(out_decoder, 0, sizeof(*out_decoder));
  const iree_host_size_t max_pattern_size =
      sizeof(out_decoder->config.replace.pattern);
  const iree_host_size_t max_content_size =
      sizeof(out_decoder->config.replace.content);
  if (pattern.size > max_pattern_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pattern string too long: %zu > %zu", pattern.size,
                            max_pattern_size);
  }
  if (content.size > max_content_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "content string too long: %zu > %zu", content.size,
                            max_content_size);
  }
  out_decoder->type = IREE_TOKENIZER_DECODER_REPLACE;
  memcpy(out_decoder->config.replace.pattern, pattern.data, pattern.size);
  out_decoder->config.replace.pattern_length = (uint8_t)pattern.size;
  memcpy(out_decoder->config.replace.content, content.data, content.size);
  out_decoder->config.replace.content_length = (uint8_t)content.size;
  return iree_ok_status();
}

void iree_tokenizer_decoder_initialize_byte_fallback(
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->type = IREE_TOKENIZER_DECODER_BYTE_FALLBACK;
}

void iree_tokenizer_decoder_initialize_fuse(
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->type = IREE_TOKENIZER_DECODER_FUSE;
}

iree_status_t iree_tokenizer_decoder_initialize_strip(
    iree_string_view_t content, uint8_t start, uint8_t stop,
    iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  memset(out_decoder, 0, sizeof(*out_decoder));
  const iree_host_size_t max_content_size =
      sizeof(out_decoder->config.strip.content);
  if (content.size > max_content_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "content string too long: %zu > %zu", content.size,
                            max_content_size);
  }
  out_decoder->type = IREE_TOKENIZER_DECODER_STRIP;
  memcpy(out_decoder->config.strip.content, content.data, content.size);
  out_decoder->config.strip.content_length = (uint8_t)content.size;
  out_decoder->config.strip.start = start;
  out_decoder->config.strip.stop = stop;
  return iree_ok_status();
}

iree_status_t iree_tokenizer_decoder_initialize_sequence(
    iree_tokenizer_decoder_t* children, iree_host_size_t count,
    iree_allocator_t allocator, iree_tokenizer_decoder_t* out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  if (count > 0 && !children) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "children is NULL but count is %zu", count);
  }
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->type = IREE_TOKENIZER_DECODER_SEQUENCE;
  out_decoder->config.sequence.count = count;
  out_decoder->config.sequence.children = children;
  out_decoder->config.sequence.allocator = allocator;
  return iree_ok_status();
}

void iree_tokenizer_decoder_deinitialize(iree_tokenizer_decoder_t* decoder) {
  if (!decoder) return;
  switch (decoder->type) {
    case IREE_TOKENIZER_DECODER_SEQUENCE:
      // Recursively deinitialize children.
      for (iree_host_size_t i = 0; i < decoder->config.sequence.count; ++i) {
        iree_tokenizer_decoder_deinitialize(
            &decoder->config.sequence.children[i]);
      }
      if (decoder->config.sequence.children) {
        iree_allocator_free(decoder->config.sequence.allocator,
                            decoder->config.sequence.children);
      }
      break;
    default:
      // Other types have no heap allocations.
      break;
  }
  memset(decoder, 0, sizeof(*decoder));
}

//===----------------------------------------------------------------------===//
// None/BPE Decoder (concatenate tokens)
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_decoder_decode_none(
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  (void)state;
  // Passthrough: emit tokens directly as text.
  if (tokens.count == 0) return iree_ok_status();
  return callback(user_data, tokens);
}

//===----------------------------------------------------------------------===//
// WordPiece Decoder
//===----------------------------------------------------------------------===//

// Check if token should have its leading space removed during cleanup.
// This matches HuggingFace tokenizers' cleanup() function behavior.
// See:
// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/wordpiece.rs
static bool iree_tokenizer_wordpiece_cleanup_removes_space(
    iree_string_view_t token) {
  if (token.size == 0) return false;

  // Single punctuation: " ." " ?" " !" " ," → remove space.
  if (token.size == 1) {
    char c = token.data[0];
    return c == '.' || c == '?' || c == '!' || c == ',';
  }

  // Contraction patterns: " 'm" " 's" " 've" " 're" " n't" → remove space.
  if (token.data[0] == '\'') {
    if (token.size == 2) {
      char c = token.data[1];
      return c == 'm' || c == 's';  // 'm, 's
    }
    if (token.size == 3) {
      return (token.data[1] == 'v' && token.data[2] == 'e') ||  // 've
             (token.data[1] == 'r' && token.data[2] == 'e');    // 're
    }
  }
  if (token.size == 3 && token.data[0] == 'n' && token.data[1] == '\'' &&
      token.data[2] == 't') {
    return true;  // n't
  }

  return false;
}

// WordPiece decoding:
// - First token is output as-is.
// - Subsequent tokens: if they start with prefix (##), strip it and append.
// - Otherwise, add a space and append.
// - If cleanup is enabled, remove space before punctuation and contractions.
static iree_status_t iree_tokenizer_decoder_decode_wordpiece(
    const iree_tokenizer_wordpiece_decoder_config_t* config,
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  iree_string_view_t prefix = config->prefix;
  bool cleanup = iree_any_bit_set(
      config->flags, IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_CLEANUP_SPACES);

  // Stack buffer for decoded strings. Each token produces at most:
  // - 1 space + token text (without prefix)
  char data_buffer[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_string_view_t out_strings[IREE_TOKENIZER_STRING_BATCH_CAPACITY];
  iree_host_size_t out_count = 0;
  iree_host_size_t data_pos = 0;

  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    iree_string_view_t token = tokens.values[i];
    bool is_continuation = false;

    // Check if token starts with continuation prefix.
    // Guard: prefix.size > 0 prevents empty prefix from matching everything
    // (memcmp with size 0 always returns 0, making all tokens "continuations").
    if (prefix.size > 0 && token.size >= prefix.size &&
        memcmp(token.data, prefix.data, prefix.size) == 0) {
      is_continuation = true;
      token.data += prefix.size;
      token.size -= prefix.size;
    }

    // Determine if we need a space prefix.
    // Skip space for: first token, continuations, or cleanup patterns.
    bool skip_space =
        state->is_first_token || is_continuation ||
        (cleanup && iree_tokenizer_wordpiece_cleanup_removes_space(token));
    iree_host_size_t need_space = skip_space ? 0 : 1;
    iree_host_size_t total_need = need_space + token.size;

    // Flush if batch is full.
    if (out_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY ||
        data_pos + total_need > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
      if (out_count > 0) {
        iree_string_view_list_t batch = {out_count, out_strings};
        IREE_RETURN_IF_ERROR(callback(user_data, batch));
        out_count = 0;
        data_pos = 0;
      }
      // Single token exceeds buffer capacity.
      if (total_need > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "single token exceeds buffer capacity");
      }
    }

    // Build output string.
    char* out_start = data_buffer + data_pos;
    iree_host_size_t out_length = 0;

    if (need_space) {
      out_start[out_length++] = ' ';
    }
    memcpy(out_start + out_length, token.data, token.size);
    out_length += token.size;

    out_strings[out_count++] = iree_make_string_view(out_start, out_length);
    data_pos += out_length;
    state->is_first_token = false;
  }

  // Flush remaining.
  if (out_count > 0) {
    iree_string_view_list_t batch = {out_count, out_strings};
    IREE_RETURN_IF_ERROR(callback(user_data, batch));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Metaspace Decoder
//===----------------------------------------------------------------------===//

// Metaspace decoding: replace ▁ (U+2581) with space.
static iree_status_t iree_tokenizer_decoder_decode_metaspace(
    const iree_tokenizer_metaspace_decoder_config_t* config,
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  uint32_t replacement = config->replacement;

  char data_buffer[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_string_view_t out_strings[IREE_TOKENIZER_STRING_BATCH_CAPACITY];
  iree_host_size_t out_count = 0;
  iree_host_size_t data_pos = 0;

  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    iree_string_view_t token = tokens.values[i];

    // Estimate: each codepoint could expand slightly, but metaspace->space
    // is 3 bytes -> 1 byte, so output <= input.
    if (out_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY ||
        data_pos + token.size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
      if (out_count > 0) {
        iree_string_view_list_t batch = {out_count, out_strings};
        IREE_RETURN_IF_ERROR(callback(user_data, batch));
        out_count = 0;
        data_pos = 0;
      }
      // Single token exceeds buffer capacity.
      if (token.size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "single token exceeds buffer capacity");
      }
    }

    char* out_start = data_buffer + data_pos;
    iree_host_size_t out_length = 0;
    iree_host_size_t position = 0;

    while (position < token.size) {
      uint32_t cp = iree_unicode_utf8_decode(token, &position);

      if (cp == replacement) {
        // Replace metaspace with space.
        // Skip leading metaspace on first token if STRIP_LEADING flag is set
        // (i.e., add_prefix_space was used during encoding).
        bool is_leading =
            state->is_first_token && out_length == 0 && data_pos == 0;
        bool strip_leading =
            config->flags & IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING;
        if (!is_leading || !strip_leading) {
          out_start[out_length++] = ' ';
        }
      } else {
        out_length += iree_unicode_utf8_encode(cp, out_start + out_length);
      }
    }

    if (out_length > 0) {
      out_strings[out_count++] = iree_make_string_view(out_start, out_length);
      data_pos += out_length;
    }
    state->is_first_token = false;
  }

  // Flush remaining.
  if (out_count > 0) {
    iree_string_view_list_t batch = {out_count, out_strings};
    IREE_RETURN_IF_ERROR(callback(user_data, batch));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ByteLevel Decoder
//===----------------------------------------------------------------------===//

// GPT-2 uses a byte-to-Unicode mapping to handle all bytes as printable chars.
// The decoder reverses this mapping.
static uint8_t iree_tokenizer_gpt2_char_to_byte(uint32_t cp) {
  // Direct mapping for printable ASCII and Latin-1 supplement.
  if (cp >= 33 && cp <= 126) return (uint8_t)cp;
  if (cp >= 161 && cp <= 172) return (uint8_t)cp;
  if (cp >= 174 && cp <= 255) return (uint8_t)cp;

  // Reverse the offset mapping for control characters.
  if (cp >= 256) {
    static const uint8_t offset_bytes[] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  127, 128, 129, 130, 131, 132, 133, 134, 135,
        136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
        150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 173};
    iree_host_size_t idx = cp - 256;
    if (idx < sizeof(offset_bytes)) {
      return offset_bytes[idx];
    }
  }

  return (uint8_t)(cp & 0xFF);
}

static iree_status_t iree_tokenizer_decoder_decode_byte_level(
    const iree_tokenizer_byte_level_decoder_config_t* config,
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  char data_buffer[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_string_view_t out_strings[IREE_TOKENIZER_STRING_BATCH_CAPACITY];
  iree_host_size_t out_count = 0;
  iree_host_size_t data_pos = 0;

  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    iree_string_view_t token = tokens.values[i];

    // ByteLevel: each Unicode char -> 1 byte, so output <= input.
    if (out_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY ||
        data_pos + token.size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
      if (out_count > 0) {
        iree_string_view_list_t batch = {out_count, out_strings};
        IREE_RETURN_IF_ERROR(callback(user_data, batch));
        out_count = 0;
        data_pos = 0;
      }
      // Single token exceeds buffer capacity.
      if (token.size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "single token exceeds buffer capacity");
      }
    }

    char* out_start = data_buffer + data_pos;
    iree_host_size_t out_length = 0;
    iree_host_size_t position = 0;

    while (position < token.size) {
      uint32_t cp = iree_unicode_utf8_decode(token, &position);
      uint8_t byte = iree_tokenizer_gpt2_char_to_byte(cp);

      // Strip leading space if ADD_PREFIX_SPACE flag is set and this is the
      // first byte of the first token.
      if (state->is_first_token && out_length == 0 && data_pos == 0 &&
          iree_any_bit_set(
              config->flags,
              IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_ADD_PREFIX_SPACE) &&
          byte == 0x20) {
        state->is_first_token = false;
        continue;
      }

      out_start[out_length++] = (char)byte;
    }

    if (out_length > 0) {
      out_strings[out_count++] = iree_make_string_view(out_start, out_length);
      data_pos += out_length;
    }
    state->is_first_token = false;
  }

  // Flush remaining.
  if (out_count > 0) {
    iree_string_view_list_t batch = {out_count, out_strings};
    IREE_RETURN_IF_ERROR(callback(user_data, batch));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Replace Decoder
//===----------------------------------------------------------------------===//

// Replace decoding: replace all occurrences of pattern with content.
// Unlike Metaspace, does NOT strip leading replacement on first token.
static iree_status_t iree_tokenizer_decoder_decode_replace(
    const iree_tokenizer_replace_decoder_config_t* config,
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  (void)state;  // Replace doesn't track first-token state.

  iree_string_view_t pattern =
      iree_make_string_view(config->pattern, config->pattern_length);
  iree_string_view_t content =
      iree_make_string_view(config->content, config->content_length);

  // Empty pattern means no replacement.
  if (pattern.size == 0) {
    if (tokens.count == 0) return iree_ok_status();
    return callback(user_data, tokens);
  }

  char data_buffer[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_string_view_t out_strings[IREE_TOKENIZER_STRING_BATCH_CAPACITY];
  iree_host_size_t out_count = 0;
  iree_host_size_t data_pos = 0;

  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    iree_string_view_t token = tokens.values[i];

    // Worst case: every char replaced, content larger than pattern.
    // For typical ▁→space, output <= input. Be conservative.
    iree_host_size_t max_out_size =
        token.size + (token.size / pattern.size + 1) * content.size;
    if (out_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY ||
        data_pos + max_out_size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
      if (out_count > 0) {
        iree_string_view_list_t batch = {out_count, out_strings};
        IREE_RETURN_IF_ERROR(callback(user_data, batch));
        out_count = 0;
        data_pos = 0;
      }
      if (max_out_size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "single token exceeds buffer capacity");
      }
    }

    char* out_start = data_buffer + data_pos;
    iree_host_size_t out_length = 0;
    iree_host_size_t position = 0;

    while (position < token.size) {
      // Check for pattern match at current position.
      if (position + pattern.size <= token.size &&
          memcmp(token.data + position, pattern.data, pattern.size) == 0) {
        // Found pattern, emit replacement content.
        memcpy(out_start + out_length, content.data, content.size);
        out_length += content.size;
        position += pattern.size;
      } else {
        // No match, copy this byte.
        out_start[out_length++] = token.data[position++];
      }
    }

    if (out_length > 0) {
      out_strings[out_count++] = iree_make_string_view(out_start, out_length);
      data_pos += out_length;
    }
  }

  // Flush remaining.
  if (out_count > 0) {
    iree_string_view_list_t batch = {out_count, out_strings};
    IREE_RETURN_IF_ERROR(callback(user_data, batch));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ByteFallback Decoder
//===----------------------------------------------------------------------===//

// Returns true if token matches <0xHH> format exactly (6 chars).
static bool iree_tokenizer_is_byte_token(iree_string_view_t token) {
  if (token.size != 6) return false;
  if (token.data[0] != '<') return false;
  if (token.data[1] != '0') return false;
  if (token.data[2] != 'x') return false;
  if (token.data[5] != '>') return false;
  // Check hex digits at positions 3 and 4.
  char c3 = token.data[3];
  char c4 = token.data[4];
  bool valid3 = (c3 >= '0' && c3 <= '9') || (c3 >= 'A' && c3 <= 'F') ||
                (c3 >= 'a' && c3 <= 'f');
  bool valid4 = (c4 >= '0' && c4 <= '9') || (c4 >= 'A' && c4 <= 'F') ||
                (c4 >= 'a' && c4 <= 'f');
  return valid3 && valid4;
}

// Converts a hex character to its nibble value (0-15).
static uint8_t iree_tokenizer_hex_to_nibble(char c) {
  if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
  if (c >= 'A' && c <= 'F') return (uint8_t)(c - 'A' + 10);
  if (c >= 'a' && c <= 'f') return (uint8_t)(c - 'a' + 10);
  return 0;
}

// Parses <0xHH> token to byte value.
static uint8_t iree_tokenizer_parse_byte_token(iree_string_view_t token) {
  uint8_t high = iree_tokenizer_hex_to_nibble(token.data[3]);
  uint8_t low = iree_tokenizer_hex_to_nibble(token.data[4]);
  return (uint8_t)((high << 4) | low);
}

// UTF-8 replacement character (U+FFFD) encoded.
static const char kReplacementChar[] = "\xEF\xBF\xBD";
static const iree_host_size_t kReplacementCharSize = 3;

// Flushes accumulated bytes as UTF-8, emitting U+FFFD for invalid sequences.
static iree_host_size_t iree_tokenizer_flush_byte_buffer(
    const uint8_t* bytes, iree_host_size_t byte_count, char* out) {
  iree_host_size_t out_length = 0;
  iree_host_size_t i = 0;

  while (i < byte_count) {
    // Try to decode a valid UTF-8 sequence starting at i.
    uint8_t b0 = bytes[i];
    iree_host_size_t sequence_length = 0;
    bool valid = false;

    if (b0 < 0x80) {
      // ASCII: single byte, always valid.
      out[out_length++] = (char)b0;
      i++;
      continue;
    } else if ((b0 & 0xE0) == 0xC0) {
      // 2-byte sequence: 110xxxxx 10xxxxxx
      sequence_length = 2;
      if (i + 1 < byte_count && (bytes[i + 1] & 0xC0) == 0x80) {
        // Check for overlong encoding (must encode >= 0x80).
        uint32_t cp = ((b0 & 0x1F) << 6) | (bytes[i + 1] & 0x3F);
        valid = (cp >= 0x80);
      }
    } else if ((b0 & 0xF0) == 0xE0) {
      // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
      sequence_length = 3;
      if (i + 2 < byte_count && (bytes[i + 1] & 0xC0) == 0x80 &&
          (bytes[i + 2] & 0xC0) == 0x80) {
        uint32_t cp = ((b0 & 0x0F) << 12) | ((bytes[i + 1] & 0x3F) << 6) |
                      (bytes[i + 2] & 0x3F);
        // Check for overlong and surrogate.
        valid = (cp >= 0x800 && (cp < 0xD800 || cp > 0xDFFF));
      }
    } else if ((b0 & 0xF8) == 0xF0) {
      // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
      sequence_length = 4;
      if (i + 3 < byte_count && (bytes[i + 1] & 0xC0) == 0x80 &&
          (bytes[i + 2] & 0xC0) == 0x80 && (bytes[i + 3] & 0xC0) == 0x80) {
        uint32_t cp = ((b0 & 0x07) << 18) | ((bytes[i + 1] & 0x3F) << 12) |
                      ((bytes[i + 2] & 0x3F) << 6) | (bytes[i + 3] & 0x3F);
        // Check for overlong and valid range.
        valid = (cp >= 0x10000 && cp <= 0x10FFFF);
      }
    }

    if (valid) {
      // Copy valid UTF-8 sequence.
      for (iree_host_size_t j = 0; j < sequence_length; ++j) {
        out[out_length++] = (char)bytes[i + j];
      }
      i += sequence_length;
    } else {
      // Invalid byte: emit replacement character.
      memcpy(out + out_length, kReplacementChar, kReplacementCharSize);
      out_length += kReplacementCharSize;
      i++;
    }
  }

  return out_length;
}

// ByteFallback decoding: convert <0xHH> tokens to bytes with UTF-8 validation.
static iree_status_t iree_tokenizer_decoder_decode_byte_fallback(
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  (void)state;  // ByteFallback doesn't track first-token state.

  // Buffer for accumulating consecutive byte tokens.
  uint8_t byte_buffer[256];
  iree_host_size_t byte_count = 0;

  char data_buffer[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_string_view_t out_strings[IREE_TOKENIZER_STRING_BATCH_CAPACITY];
  iree_host_size_t out_count = 0;
  iree_host_size_t data_pos = 0;

  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    iree_string_view_t token = tokens.values[i];

    if (iree_tokenizer_is_byte_token(token)) {
      // Accumulate byte token.
      if (byte_count < sizeof(byte_buffer)) {
        byte_buffer[byte_count++] = iree_tokenizer_parse_byte_token(token);
      }
      // If buffer full, flush it (preserving incomplete UTF-8 sequences).
      if (byte_count >= sizeof(byte_buffer)) {
        // Detect incomplete UTF-8 tail to avoid splitting multi-byte sequences.
        iree_host_size_t incomplete_tail =
            iree_unicode_utf8_incomplete_tail_length((const char*)byte_buffer,
                                                     byte_count);
        iree_host_size_t flush_count = byte_count - incomplete_tail;

        if (flush_count > 0) {
          // Flush complete portion of byte buffer.
          if (out_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY ||
              data_pos + flush_count * 3 > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
            if (out_count > 0) {
              iree_string_view_list_t batch = {out_count, out_strings};
              IREE_RETURN_IF_ERROR(callback(user_data, batch));
              out_count = 0;
              data_pos = 0;
            }
          }
          char* out_start = data_buffer + data_pos;
          iree_host_size_t out_length = iree_tokenizer_flush_byte_buffer(
              byte_buffer, flush_count, out_start);
          if (out_length > 0) {
            out_strings[out_count++] =
                iree_make_string_view(out_start, out_length);
            data_pos += out_length;
          }
        }

        // Move incomplete tail bytes to start of buffer.
        if (incomplete_tail > 0) {
          memmove(byte_buffer, byte_buffer + flush_count, incomplete_tail);
        }
        byte_count = incomplete_tail;
      }
    } else {
      // Non-byte token: first flush any accumulated bytes.
      if (byte_count > 0) {
        if (out_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY ||
            data_pos + byte_count * 3 > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
          if (out_count > 0) {
            iree_string_view_list_t batch = {out_count, out_strings};
            IREE_RETURN_IF_ERROR(callback(user_data, batch));
            out_count = 0;
            data_pos = 0;
          }
        }
        char* out_start = data_buffer + data_pos;
        iree_host_size_t out_length = iree_tokenizer_flush_byte_buffer(
            byte_buffer, byte_count, out_start);
        if (out_length > 0) {
          out_strings[out_count++] =
              iree_make_string_view(out_start, out_length);
          data_pos += out_length;
        }
        byte_count = 0;
      }

      // Pass through non-byte token.
      if (out_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY ||
          data_pos + token.size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
        if (out_count > 0) {
          iree_string_view_list_t batch = {out_count, out_strings};
          IREE_RETURN_IF_ERROR(callback(user_data, batch));
          out_count = 0;
          data_pos = 0;
        }
        if (token.size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "single token exceeds buffer capacity");
        }
      }
      char* out_start = data_buffer + data_pos;
      memcpy(out_start, token.data, token.size);
      out_strings[out_count++] = iree_make_string_view(out_start, token.size);
      data_pos += token.size;
    }
  }

  // Flush any remaining bytes.
  if (byte_count > 0) {
    if (out_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY ||
        data_pos + byte_count * 3 > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
      if (out_count > 0) {
        iree_string_view_list_t batch = {out_count, out_strings};
        IREE_RETURN_IF_ERROR(callback(user_data, batch));
        out_count = 0;
        data_pos = 0;
      }
    }
    char* out_start = data_buffer + data_pos;
    iree_host_size_t out_length =
        iree_tokenizer_flush_byte_buffer(byte_buffer, byte_count, out_start);
    if (out_length > 0) {
      out_strings[out_count++] = iree_make_string_view(out_start, out_length);
      data_pos += out_length;
    }
  }

  // Flush remaining output.
  if (out_count > 0) {
    iree_string_view_list_t batch = {out_count, out_strings};
    IREE_RETURN_IF_ERROR(callback(user_data, batch));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Fuse Decoder
//===----------------------------------------------------------------------===//

// Fuse decoding: concatenate all tokens into a single output string.
static iree_status_t iree_tokenizer_decoder_decode_fuse(
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  (void)state;  // Fuse doesn't track first-token state.

  if (tokens.count == 0) return iree_ok_status();

  // Calculate total size needed.
  iree_host_size_t total_size = 0;
  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    total_size += tokens.values[i].size;
  }

  char data_buffer[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  if (total_size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "fused output (%zu bytes) exceeds buffer capacity",
                            total_size);
  }

  // Concatenate all tokens.
  iree_host_size_t position = 0;
  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    memcpy(data_buffer + position, tokens.values[i].data,
           tokens.values[i].size);
    position += tokens.values[i].size;
  }

  // Emit single fused string.
  iree_string_view_t fused = iree_make_string_view(data_buffer, position);
  iree_string_view_list_t result = {1, &fused};
  return callback(user_data, result);
}

//===----------------------------------------------------------------------===//
// Strip Decoder
//===----------------------------------------------------------------------===//

// Strip decoding: remove characters from token start and/or end.
static iree_status_t iree_tokenizer_decoder_decode_strip(
    const iree_tokenizer_strip_decoder_config_t* config,
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  (void)state;  // Strip doesn't track first-token state.

  iree_string_view_t strip_content =
      iree_make_string_view(config->content, config->content_length);

  // Empty content means no stripping.
  if (strip_content.size == 0 || (config->start == 0 && config->stop == 0)) {
    if (tokens.count == 0) return iree_ok_status();
    return callback(user_data, tokens);
  }

  char data_buffer[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_string_view_t out_strings[IREE_TOKENIZER_STRING_BATCH_CAPACITY];
  iree_host_size_t out_count = 0;
  iree_host_size_t data_pos = 0;

  for (iree_host_size_t i = 0; i < tokens.count; ++i) {
    iree_string_view_t token = tokens.values[i];

    // Flush if needed.
    if (out_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY ||
        data_pos + token.size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
      if (out_count > 0) {
        iree_string_view_list_t batch = {out_count, out_strings};
        IREE_RETURN_IF_ERROR(callback(user_data, batch));
        out_count = 0;
        data_pos = 0;
      }
      if (token.size > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "single token exceeds buffer capacity");
      }
    }

    // Strip from start.
    iree_host_size_t start_pos = 0;
    uint8_t start_stripped = 0;
    while (start_stripped < config->start &&
           start_pos + strip_content.size <= token.size) {
      if (memcmp(token.data + start_pos, strip_content.data,
                 strip_content.size) == 0) {
        start_pos += strip_content.size;
        start_stripped++;
      } else {
        break;
      }
    }

    // Strip from end.
    iree_host_size_t end_pos = token.size;
    uint8_t end_stripped = 0;
    while (end_stripped < config->stop &&
           end_pos >= start_pos + strip_content.size) {
      if (memcmp(token.data + end_pos - strip_content.size, strip_content.data,
                 strip_content.size) == 0) {
        end_pos -= strip_content.size;
        end_stripped++;
      } else {
        break;
      }
    }

    // Emit stripped token (may be empty).
    iree_host_size_t stripped_length = end_pos - start_pos;
    if (stripped_length > 0) {
      char* out_start = data_buffer + data_pos;
      memcpy(out_start, token.data + start_pos, stripped_length);
      out_strings[out_count++] =
          iree_make_string_view(out_start, stripped_length);
      data_pos += stripped_length;
    }
  }

  // Flush remaining.
  if (out_count > 0) {
    iree_string_view_list_t batch = {out_count, out_strings};
    IREE_RETURN_IF_ERROR(callback(user_data, batch));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Sequence Decoder
//===----------------------------------------------------------------------===//

// Maximum number of decoders in a sequence (for stack-allocated state array).
#define IREE_TOKENIZER_MAX_SEQUENCE_DEPTH 8

// Context for chaining decoder callbacks in streaming mode.
typedef struct iree_tokenizer_sequence_chain_context_t {
  const iree_tokenizer_decoder_t* decoders;  // Remaining decoders in chain.
  iree_host_size_t remaining_count;          // How many decoders left.
  iree_tokenizer_decoder_state_t* states;  // State for each remaining decoder.
  iree_tokenizer_string_callback_fn_t final_callback;
  void* final_user_data;
} iree_tokenizer_sequence_chain_context_t;

// Streaming callback that chains to the next decoder or final callback.
static iree_status_t iree_tokenizer_sequence_chain_callback(
    void* user_data, iree_string_view_list_t strings) {
  iree_tokenizer_sequence_chain_context_t* context =
      (iree_tokenizer_sequence_chain_context_t*)user_data;

  if (context->remaining_count == 0) {
    // Last decoder in chain, pass to user's callback.
    return context->final_callback(context->final_user_data, strings);
  }

  // Chain to next decoder.
  iree_tokenizer_sequence_chain_context_t next_context = {
      .decoders = context->decoders + 1,
      .remaining_count = context->remaining_count - 1,
      .states = context->states + 1,
      .final_callback = context->final_callback,
      .final_user_data = context->final_user_data,
  };

  return iree_tokenizer_decoder_decode(
      &context->decoders[0], &context->states[0], strings,
      iree_tokenizer_sequence_chain_callback, &next_context);
}

// Sequence decoding: chain multiple decoders via streaming callbacks.
// Each decoder's output flows directly to the next decoder's input without
// intermediate buffering.
static iree_status_t iree_tokenizer_decoder_decode_sequence(
    const iree_tokenizer_sequence_decoder_config_t* config,
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  if (config->count == 0) {
    // Empty sequence: passthrough.
    if (tokens.count == 0) return iree_ok_status();
    return callback(user_data, tokens);
  }

  if (config->count > IREE_TOKENIZER_MAX_SEQUENCE_DEPTH) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence decoder depth %zu exceeds maximum %d",
                            config->count, IREE_TOKENIZER_MAX_SEQUENCE_DEPTH);
  }

  // Initialize state for all decoders in the chain.
  iree_tokenizer_decoder_state_t states[IREE_TOKENIZER_MAX_SEQUENCE_DEPTH];
  for (iree_host_size_t i = 0; i < config->count; ++i) {
    iree_tokenizer_decoder_begin(&config->children[i], &states[i]);
  }
  // Propagate is_first_token from parent to first child.
  states[0].is_first_token = state->is_first_token;

  // Set up the callback chain starting from decoder[1] onwards.
  // Decoder[0] is called directly, its callback chains to decoder[1], etc.
  iree_tokenizer_sequence_chain_context_t chain_context = {
      .decoders = config->children + 1,
      .remaining_count = config->count - 1,
      .states = states + 1,
      .final_callback = callback,
      .final_user_data = user_data,
  };

  // Start the chain: decode through first decoder, which chains to the rest.
  IREE_RETURN_IF_ERROR(iree_tokenizer_decoder_decode(
      &config->children[0], &states[0], tokens,
      iree_tokenizer_sequence_chain_callback, &chain_context));

  // Update parent state.
  state->is_first_token = false;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Decoder Dispatch
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_decoder_decode(
    const iree_tokenizer_decoder_t* decoder,
    iree_tokenizer_decoder_state_t* state, iree_string_view_list_t tokens,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  IREE_ASSERT_ARGUMENT(decoder);
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(tokens.values || tokens.count == 0);
  IREE_ASSERT_ARGUMENT(callback);

  if (tokens.count == 0) {
    return iree_ok_status();
  }

  switch (decoder->type) {
    case IREE_TOKENIZER_DECODER_NONE:
    case IREE_TOKENIZER_DECODER_BPE:
      return iree_tokenizer_decoder_decode_none(state, tokens, callback,
                                                user_data);
    case IREE_TOKENIZER_DECODER_WORDPIECE:
      return iree_tokenizer_decoder_decode_wordpiece(
          &decoder->config.wordpiece, state, tokens, callback, user_data);
    case IREE_TOKENIZER_DECODER_METASPACE:
      return iree_tokenizer_decoder_decode_metaspace(
          &decoder->config.metaspace, state, tokens, callback, user_data);
    case IREE_TOKENIZER_DECODER_BYTE_LEVEL:
      return iree_tokenizer_decoder_decode_byte_level(
          &decoder->config.byte_level, state, tokens, callback, user_data);
    case IREE_TOKENIZER_DECODER_SEQUENCE:
      return iree_tokenizer_decoder_decode_sequence(
          &decoder->config.sequence, state, tokens, callback, user_data);
    case IREE_TOKENIZER_DECODER_REPLACE:
      return iree_tokenizer_decoder_decode_replace(
          &decoder->config.replace, state, tokens, callback, user_data);
    case IREE_TOKENIZER_DECODER_BYTE_FALLBACK:
      return iree_tokenizer_decoder_decode_byte_fallback(state, tokens,
                                                         callback, user_data);
    case IREE_TOKENIZER_DECODER_FUSE:
      return iree_tokenizer_decoder_decode_fuse(state, tokens, callback,
                                                user_data);
    case IREE_TOKENIZER_DECODER_STRIP:
      return iree_tokenizer_decoder_decode_strip(&decoder->config.strip, state,
                                                 tokens, callback, user_data);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unknown decoder type %d", (int)decoder->type);
  }
}
