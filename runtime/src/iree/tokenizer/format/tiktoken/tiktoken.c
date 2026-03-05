// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/tiktoken/tiktoken.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/base64.h"
#include "iree/tokenizer/byte_level_tables.h"
#include "iree/tokenizer/decoder/byte_level.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/segmenter/split.h"
#include "iree/tokenizer/special_tokens.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

// Maximum raw bytes in a single token. Observed maximum is ~60 bytes in
// standard encodings. Used consistently by the parser, vocab construction,
// and merge reconstruction to enforce the same size constraint.
#define IREE_TOKENIZER_TIKTOKEN_MAX_PARTS 256

//===----------------------------------------------------------------------===//
// Predefined Encoding Configs
//===----------------------------------------------------------------------===//

// cl100k_base regex pattern (GPT-4, GPT-3.5-turbo).
// Possessive quantifiers (++) are semantically identical to greedy quantifiers
// in a DFA engine (no backtracking), so we keep them for compatibility with the
// original tiktoken pattern.
static const char kCl100kPattern[] =
    "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}++|\\p{N}{1,3}| "
    "?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s++";

// o200k_base regex pattern (GPT-4o, GPT-4o-mini).
// Uses Unicode script categories for finer-grained word segmentation than
// cl100k_base, with separate handling for uppercase/titlecase vs lowercase.
static const char kO200kPattern[] =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "\\p{N}{1,3}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// GPT-2 regex pattern (r50k_base, p50k_base).
static const char kGPT2Pattern[] =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
    "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// cl100k_base: 5 special tokens.
static const iree_string_view_t kCl100kSpecialStrings[] = {
    IREE_SVL("<|endoftext|>"),   IREE_SVL("<|fim_prefix|>"),
    IREE_SVL("<|fim_middle|>"),  IREE_SVL("<|fim_suffix|>"),
    IREE_SVL("<|endofprompt|>"),
};
static const int32_t kCl100kSpecialIds[] = {100257, 100258, 100259, 100260,
                                            100276};

// o200k_base: 2 special tokens.
static const iree_string_view_t kO200kSpecialStrings[] = {
    IREE_SVL("<|endoftext|>"),
    IREE_SVL("<|endofprompt|>"),
};
static const int32_t kO200kSpecialIds[] = {199999, 200018};

// r50k_base: 1 special token.
static const iree_string_view_t kR50kSpecialStrings[] = {
    IREE_SVL("<|endoftext|>"),
};
static const int32_t kR50kSpecialIds[] = {50256};

// p50k_base: 1 special token.
static const iree_string_view_t kP50kSpecialStrings[] = {
    IREE_SVL("<|endoftext|>"),
};
static const int32_t kP50kSpecialIds[] = {50281};

static const iree_tokenizer_tiktoken_config_t kCl100kConfig = {
    .pattern = {kCl100kPattern, sizeof(kCl100kPattern) - 1},
    .special_token_count = IREE_ARRAYSIZE(kCl100kSpecialStrings),
    .special_token_strings = kCl100kSpecialStrings,
    .special_token_ids = kCl100kSpecialIds,
};

static const iree_tokenizer_tiktoken_config_t kO200kConfig = {
    .pattern = {kO200kPattern, sizeof(kO200kPattern) - 1},
    .special_token_count = IREE_ARRAYSIZE(kO200kSpecialStrings),
    .special_token_strings = kO200kSpecialStrings,
    .special_token_ids = kO200kSpecialIds,
};

static const iree_tokenizer_tiktoken_config_t kR50kConfig = {
    .pattern = {kGPT2Pattern, sizeof(kGPT2Pattern) - 1},
    .special_token_count = IREE_ARRAYSIZE(kR50kSpecialStrings),
    .special_token_strings = kR50kSpecialStrings,
    .special_token_ids = kR50kSpecialIds,
};

static const iree_tokenizer_tiktoken_config_t kP50kConfig = {
    .pattern = {kGPT2Pattern, sizeof(kGPT2Pattern) - 1},
    .special_token_count = IREE_ARRAYSIZE(kP50kSpecialStrings),
    .special_token_strings = kP50kSpecialStrings,
    .special_token_ids = kP50kSpecialIds,
};

const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_cl100k_base(void) {
  return &kCl100kConfig;
}

const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_o200k_base(void) {
  return &kO200kConfig;
}

const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_r50k_base(void) {
  return &kR50kConfig;
}

const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_p50k_base(void) {
  return &kP50kConfig;
}

//===----------------------------------------------------------------------===//
// Tiktoken File Parsing
//===----------------------------------------------------------------------===//

// Parsed entry from a tiktoken line.
typedef struct iree_tokenizer_tiktoken_entry_t {
  uint32_t byte_offset;  // Offset into decoded bytes buffer.
  uint16_t byte_length;  // Length of decoded token bytes.
} iree_tokenizer_tiktoken_entry_t;

// Parsed tiktoken file: array of entries + contiguous decoded byte storage.
typedef struct iree_tokenizer_tiktoken_parsed_t {
  iree_tokenizer_tiktoken_entry_t* entries;
  iree_host_size_t entry_count;
  uint8_t* bytes;
  iree_host_size_t bytes_size;
  iree_host_size_t bytes_capacity;
  iree_host_size_t entry_capacity;
  iree_allocator_t allocator;
} iree_tokenizer_tiktoken_parsed_t;

static void iree_tokenizer_tiktoken_parsed_free(
    iree_tokenizer_tiktoken_parsed_t* parsed) {
  if (!parsed) return;
  iree_allocator_free(parsed->allocator, parsed->entries);
  iree_allocator_free(parsed->allocator, parsed->bytes);
  parsed->entries = NULL;
  parsed->bytes = NULL;
  parsed->entry_count = 0;
  parsed->bytes_size = 0;
}

// Ensures capacity for at least one more entry and |additional_bytes| in the
// byte buffer.
static iree_status_t iree_tokenizer_tiktoken_parsed_ensure_capacity(
    iree_tokenizer_tiktoken_parsed_t* parsed,
    iree_host_size_t additional_bytes) {
  if (parsed->entry_count >= parsed->entry_capacity) {
    iree_host_size_t new_capacity =
        parsed->entry_capacity == 0 ? 1024 : parsed->entry_capacity * 2;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        parsed->allocator, new_capacity * sizeof(*parsed->entries),
        (void**)&parsed->entries));
    parsed->entry_capacity = new_capacity;
  }

  iree_host_size_t needed = parsed->bytes_size + additional_bytes;
  if (needed > parsed->bytes_capacity) {
    iree_host_size_t new_capacity =
        parsed->bytes_capacity == 0 ? 4096 : parsed->bytes_capacity;
    while (new_capacity < needed) new_capacity *= 2;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(parsed->allocator, new_capacity,
                                                (void**)&parsed->bytes));
    parsed->bytes_capacity = new_capacity;
  }

  return iree_ok_status();
}

// Parses a uint32 from a decimal string view.
static bool iree_tokenizer_tiktoken_parse_uint32(iree_string_view_t string,
                                                 uint32_t* out_value) {
  if (string.size == 0) return false;
  uint64_t value = 0;
  for (iree_host_size_t i = 0; i < string.size; ++i) {
    char ch = string.data[i];
    if (ch < '0' || ch > '9') return false;
    value = value * 10 + (uint64_t)(ch - '0');
    if (value > UINT32_MAX) return false;
  }
  *out_value = (uint32_t)value;
  return true;
}

// Parses a .tiktoken file into entries and decoded bytes.
//
// Format: one line per token, "<base64-encoded-bytes> <rank>\n".
// Ranks must be monotonically increasing starting from 0. Gaps are permitted
// (filled with zero-length placeholder entries) to accommodate special tokens
// that reserve IDs within the BPE rank range.
static iree_status_t iree_tokenizer_tiktoken_parse_file(
    iree_string_view_t data, iree_allocator_t allocator,
    iree_tokenizer_tiktoken_parsed_t* out_parsed) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_parsed, 0, sizeof(*out_parsed));
  out_parsed->allocator = allocator;

  if (iree_string_view_is_empty(data)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty tiktoken input");
  }

  // Stack buffer for base64 decoding of individual tokens. Uses the same
  // constant as vocab construction and merge reconstruction so the size
  // constraint is enforced consistently.
  uint8_t decode_buffer[IREE_TOKENIZER_TIKTOKEN_MAX_PARTS];

  iree_host_size_t line_number = 0;
  iree_string_view_t remaining = data;
  iree_status_t status = iree_ok_status();

  while (remaining.size > 0 && iree_status_is_ok(status)) {
    ++line_number;

    // Split off the next line.
    iree_string_view_t line;
    iree_string_view_split(remaining, '\n', &line, &remaining);

    // Strip trailing \r for Windows line endings.
    iree_string_view_consume_suffix(&line, iree_make_string_view("\r", 1));

    // Skip empty lines.
    if (iree_string_view_is_empty(line)) continue;

    // Split on space: "<base64> <rank>".
    iree_string_view_t base64_part, rank_part;
    if (iree_string_view_split(line, ' ', &base64_part, &rank_part) < 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "line %" PRIhsz
                                ": missing space between base64 token and rank",
                                line_number);
      break;
    }

    if (iree_string_view_is_empty(base64_part)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "line %" PRIhsz ": empty base64 token", line_number);
      break;
    }

    // Parse rank.
    uint32_t rank = 0;
    if (!iree_tokenizer_tiktoken_parse_uint32(rank_part, &rank)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "line %" PRIhsz ": invalid rank '%.*s'",
          line_number, (int)rank_part.size, rank_part.data);
      break;
    }

    // Validate rank ordering and handle gaps.
    // Ranks must be monotonically increasing but may have gaps where special
    // tokens reserve IDs within the BPE range (e.g., p50k_base reserves rank
    // 50256 for <|endoftext|> — the BPE file skips it and the config's
    // special_token_ids fills it later).
    if (rank < (uint32_t)out_parsed->entry_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "line %" PRIhsz ": rank %" PRIu32
                                " is out of order (expected >= %" PRIu32 ")",
                                line_number, rank,
                                (uint32_t)out_parsed->entry_count);
      break;
    }
    if (rank > (uint32_t)out_parsed->entry_count) {
      // Fill gap positions with zero-length placeholder entries. These slots
      // are reserved for special tokens and will be populated by
      // ensure_token() during vocab construction.
      for (uint32_t gap_rank = (uint32_t)out_parsed->entry_count;
           gap_rank < rank && iree_status_is_ok(status); ++gap_rank) {
        status = iree_tokenizer_tiktoken_parsed_ensure_capacity(out_parsed, 0);
        if (!iree_status_is_ok(status)) break;
        iree_tokenizer_tiktoken_entry_t* gap_entry =
            &out_parsed->entries[out_parsed->entry_count];
        gap_entry->byte_offset = 0;
        gap_entry->byte_length = 0;
        ++out_parsed->entry_count;
      }
      if (!iree_status_is_ok(status)) break;
    }

    // Validate rank fits in int32_t (token IDs are int32_t).
    if (rank > (uint32_t)INT32_MAX) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "line %" PRIhsz ": rank %" PRIu32
                                " exceeds maximum token ID",
                                line_number, rank);
      break;
    }

    // Base64 decode into a stack buffer. The decode function validates buffer
    // capacity internally, so tokens exceeding sizeof(decode_buffer) bytes
    // will fail with an appropriate error.
    iree_host_size_t decoded_length = 0;
    status = iree_base64_decode(
        base64_part, iree_make_byte_span(decode_buffer, sizeof(decode_buffer)),
        &decoded_length);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "line %" PRIhsz, line_number);
      break;
    }

    if (decoded_length == 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "line %" PRIhsz ": base64 decodes to empty token", line_number);
      break;
    }

    // Store entry.
    status = iree_tokenizer_tiktoken_parsed_ensure_capacity(out_parsed,
                                                            decoded_length);
    if (!iree_status_is_ok(status)) break;

    iree_tokenizer_tiktoken_entry_t* entry =
        &out_parsed->entries[out_parsed->entry_count];
    entry->byte_offset = (uint32_t)out_parsed->bytes_size;
    entry->byte_length = (uint16_t)decoded_length;

    memcpy(out_parsed->bytes + out_parsed->bytes_size, decode_buffer,
           decoded_length);
    out_parsed->bytes_size += decoded_length;
    ++out_parsed->entry_count;
  }

  if (!iree_status_is_ok(status)) {
    iree_tokenizer_tiktoken_parsed_free(out_parsed);
  } else if (out_parsed->entry_count == 0) {
    iree_tokenizer_tiktoken_parsed_free(out_parsed);
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tiktoken file contains no valid entries");
  }

  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)out_parsed->entry_count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// ByteLevel Encoding Helper
//===----------------------------------------------------------------------===//

// ByteLevel-encodes raw bytes into a UTF-8 string buffer.
// Returns the number of UTF-8 bytes written.
// |buffer| must have at least |raw_length * 2| bytes available.
static iree_host_size_t iree_tokenizer_tiktoken_byte_level_encode(
    const uint8_t* raw_bytes, iree_host_size_t raw_length, char* buffer) {
  iree_host_size_t position = 0;
  for (iree_host_size_t i = 0; i < raw_length; ++i) {
    const iree_tokenizer_byte_level_utf8_t* utf8 =
        &iree_tokenizer_byte_level_utf8[raw_bytes[i]];
    buffer[position] = (char)utf8->bytes[0];
    if (utf8->length == 2) {
      buffer[position + 1] = (char)utf8->bytes[1];
    }
    position += utf8->length;
  }
  return position;
}

//===----------------------------------------------------------------------===//
// BPE Merge Reconstruction
//===----------------------------------------------------------------------===//

// Reconstructs BPE merge pairs from tiktoken rank ordering.
//
// For each multi-byte token (rank >= 256), simulates BPE encoding on its raw
// bytes using only merges with rank lower than the current token's rank. When
// exactly 2 parts remain, those form the merge pair.
//
// This exploits a property of BPE training: every merge at rank R was learned
// by combining two existing tokens whose ranks are both < R. Verified against
// HuggingFace's explicit merge list with 100% accuracy across all standard
// encodings.

// Builds a lookup table mapping each byte value to its tiktoken rank.
// In real tiktoken files the mapping is NOT identity: cl100k_base has rank 0
// for byte 0x21 ('!'), not 0x00. This table inverts the first 256 entries
// (each of which decodes to a single byte) to get byte_value → rank.
static iree_status_t iree_tokenizer_tiktoken_build_byte_to_rank(
    const iree_tokenizer_tiktoken_parsed_t* parsed,
    uint32_t byte_to_rank[256]) {
  if (parsed->entry_count < 256) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tiktoken file has %" PRIhsz
                            " entries, need at least 256 single-byte tokens",
                            parsed->entry_count);
  }

  // Initialize to sentinel value to detect unmapped bytes.
  memset(byte_to_rank, 0xFF, 256 * sizeof(uint32_t));

  for (uint32_t rank = 0; rank < 256; ++rank) {
    const iree_tokenizer_tiktoken_entry_t* entry = &parsed->entries[rank];
    if (entry->byte_length != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "rank %" PRIu32 " decodes to %" PRIu16
                              " bytes (expected 1 for single-byte token)",
                              rank, entry->byte_length);
    }
    uint8_t byte_value = parsed->bytes[entry->byte_offset];
    if (byte_to_rank[byte_value] != UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "byte 0x%02X mapped by both rank %" PRIu32
                              " and rank %" PRIu32,
                              byte_value, byte_to_rank[byte_value], rank);
    }
    byte_to_rank[byte_value] = rank;
  }

  // Verify all 256 byte values are covered.
  for (uint32_t byte_value = 0; byte_value < 256; ++byte_value) {
    if (byte_to_rank[byte_value] == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "byte 0x%02X has no corresponding rank in "
                              "the first 256 tiktoken entries",
                              byte_value);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_tokenizer_tiktoken_reconstruct_merges(
    const iree_tokenizer_tiktoken_parsed_t* parsed,
    const uint32_t byte_to_rank[256], const iree_tokenizer_vocab_t* temp_vocab,
    iree_tokenizer_vocab_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  uint32_t parts[IREE_TOKENIZER_TIKTOKEN_MAX_PARTS];
  uint32_t part_byte_offsets[IREE_TOKENIZER_TIKTOKEN_MAX_PARTS];
  uint32_t part_byte_lengths[IREE_TOKENIZER_TIKTOKEN_MAX_PARTS];

  // Buffer for ByteLevel-encoding concatenated parts during vocab lookup.
  // Two adjacent parts can be at most IREE_TOKENIZER_TIKTOKEN_MAX_PARTS bytes
  // raw total, and each raw byte becomes at most 2 UTF-8 bytes.
  char utf8_buffer[IREE_TOKENIZER_TIKTOKEN_MAX_PARTS * 2];

  iree_status_t status = iree_ok_status();
  iree_host_size_t merge_count = 0;

  for (iree_host_size_t rank = 256;
       rank < parsed->entry_count && iree_status_is_ok(status); ++rank) {
    const iree_tokenizer_tiktoken_entry_t* entry = &parsed->entries[rank];
    // Skip placeholder entries at gap positions (reserved for special tokens).
    if (entry->byte_length == 0) continue;
    const uint8_t* raw_bytes = parsed->bytes + entry->byte_offset;
    iree_host_size_t raw_length = entry->byte_length;

    if (raw_length > IREE_TOKENIZER_TIKTOKEN_MAX_PARTS) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "token at rank %" PRIhsz " has %" PRIhsz
                           " bytes, exceeding maximum (%d)",
                           rank, raw_length, IREE_TOKENIZER_TIKTOKEN_MAX_PARTS);
      break;
    }

    // Initialize: each raw byte maps to its corresponding single-byte rank.
    iree_host_size_t part_count = raw_length;
    for (iree_host_size_t i = 0; i < raw_length; ++i) {
      parts[i] = byte_to_rank[raw_bytes[i]];
      part_byte_offsets[i] = entry->byte_offset + (uint32_t)i;
      part_byte_lengths[i] = 1;
    }

    // Iteratively apply the lowest-rank merge until exactly 2 parts remain.
    while (part_count > 2) {
      uint32_t best_merge_rank = (uint32_t)rank;
      iree_host_size_t best_merge_index = SIZE_MAX;

      for (iree_host_size_t i = 0; i + 1 < part_count; ++i) {
        // ByteLevel-encode the concatenation of adjacent parts and look up
        // the resulting token in the vocabulary to find its rank.
        const uint8_t* left_bytes = parsed->bytes + part_byte_offsets[i];
        const uint8_t* right_bytes = parsed->bytes + part_byte_offsets[i + 1];
        iree_host_size_t left_length = part_byte_lengths[i];
        iree_host_size_t right_length = part_byte_lengths[i + 1];

        iree_host_size_t utf8_length = 0;
        for (iree_host_size_t j = 0; j < left_length; ++j) {
          const iree_tokenizer_byte_level_utf8_t* utf8 =
              &iree_tokenizer_byte_level_utf8[left_bytes[j]];
          utf8_buffer[utf8_length] = (char)utf8->bytes[0];
          if (utf8->length == 2) {
            utf8_buffer[utf8_length + 1] = (char)utf8->bytes[1];
          }
          utf8_length += utf8->length;
        }
        for (iree_host_size_t j = 0; j < right_length; ++j) {
          const iree_tokenizer_byte_level_utf8_t* utf8 =
              &iree_tokenizer_byte_level_utf8[right_bytes[j]];
          utf8_buffer[utf8_length] = (char)utf8->bytes[0];
          if (utf8->length == 2) {
            utf8_buffer[utf8_length + 1] = (char)utf8->bytes[1];
          }
          utf8_length += utf8->length;
        }

        int32_t concat_id = iree_tokenizer_vocab_lookup(
            temp_vocab, iree_make_string_view(utf8_buffer, utf8_length));
        if (concat_id >= 0 && (uint32_t)concat_id < best_merge_rank) {
          best_merge_rank = (uint32_t)concat_id;
          best_merge_index = i;
        }
      }

      if (best_merge_index == SIZE_MAX) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "BPE merge reconstruction failed for token at rank %" PRIhsz
            ": no valid merge found with %" PRIhsz " parts remaining",
            rank, part_count);
        break;
      }

      // Apply the merge: combine adjacent parts.
      parts[best_merge_index] = best_merge_rank;
      part_byte_lengths[best_merge_index] +=
          part_byte_lengths[best_merge_index + 1];

      // Shift remaining parts down.
      for (iree_host_size_t i = best_merge_index + 1; i + 1 < part_count; ++i) {
        parts[i] = parts[i + 1];
        part_byte_offsets[i] = part_byte_offsets[i + 1];
        part_byte_lengths[i] = part_byte_lengths[i + 1];
      }
      --part_count;
    }

    if (!iree_status_is_ok(status)) break;

    if (part_count != 2) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "BPE merge reconstruction for rank %" PRIhsz
                                " ended with %" PRIhsz " parts (expected 2)",
                                rank, part_count);
      break;
    }

    status =
        iree_tokenizer_vocab_builder_add_merge(builder, parts[0], parts[1]);
    ++merge_count;
  }

  (void)merge_count;
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)merge_count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Vocab Construction
//===----------------------------------------------------------------------===//

// Adds all BPE tokens from parsed tiktoken data to a vocab builder.
// Each raw byte sequence is ByteLevel-encoded to UTF-8 for the vocabulary.
// Placeholder entries (byte_length == 0) from rank gaps are skipped — those
// slots are reserved for special tokens inserted separately.
static iree_status_t iree_tokenizer_tiktoken_add_tokens(
    const iree_tokenizer_tiktoken_parsed_t* parsed,
    iree_tokenizer_vocab_builder_t* builder) {
  // UTF-8 buffer for ByteLevel encoding.
  char utf8_buffer[IREE_TOKENIZER_TIKTOKEN_MAX_PARTS * 2];

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < parsed->entry_count && iree_status_is_ok(status); ++i) {
    const iree_tokenizer_tiktoken_entry_t* entry = &parsed->entries[i];
    // Skip placeholder entries at gap positions (reserved for special tokens).
    if (entry->byte_length == 0) continue;
    if (entry->byte_length > IREE_TOKENIZER_TIKTOKEN_MAX_PARTS) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "token at rank %" PRIhsz " has %" PRIu16
                                " bytes, exceeding maximum of %d",
                                i, entry->byte_length,
                                IREE_TOKENIZER_TIKTOKEN_MAX_PARTS);
      break;
    }
    const uint8_t* raw_bytes = parsed->bytes + entry->byte_offset;

    iree_host_size_t utf8_length = iree_tokenizer_tiktoken_byte_level_encode(
        raw_bytes, entry->byte_length, utf8_buffer);

    // Use explicit IDs so token ID == rank index, even when gaps exist.
    status = iree_tokenizer_vocab_builder_add_token_with_id(
        builder, (int32_t)i, iree_make_string_view(utf8_buffer, utf8_length),
        0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE);
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_parse_tiktoken(
    iree_string_view_t data, const iree_tokenizer_tiktoken_config_t* config,
    iree_tokenizer_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(config);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t allocator = builder->allocator;
  iree_status_t status = iree_ok_status();

  // Validate config.
  if (config->pattern.size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tiktoken config has empty regex pattern");
  }
  if (config->special_token_count > 0 &&
      (!config->special_token_strings || !config->special_token_ids)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "tiktoken config has special_token_count=%" PRIhsz
        " but NULL special_token_strings or special_token_ids",
        config->special_token_count);
  }

  // Phase 1: Parse the tiktoken file into entries + raw bytes.
  iree_tokenizer_tiktoken_parsed_t parsed = {0};
  status = iree_tokenizer_tiktoken_parse_file(data, allocator, &parsed);

  // Build byte-to-rank mapping from the first 256 entries. In real tiktoken
  // files, rank 0 is byte 0x21 ('!'), not 0x00 — the mapping is not identity.
  uint32_t byte_to_rank[256];
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_tiktoken_build_byte_to_rank(&parsed, byte_to_rank);
  }

  // Phase 2: Build temporary vocab (tokens only) for merge reconstruction.
  // The temp vocab provides text->ID lookup needed to resolve merge pairs
  // during BPE simulation.
  iree_tokenizer_vocab_builder_t* vocab_builder = NULL;
  iree_tokenizer_vocab_t* temp_vocab = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_allocate(parsed.entry_count,
                                                   allocator, &vocab_builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_tiktoken_add_tokens(&parsed, vocab_builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(vocab_builder, &temp_vocab);
    if (iree_status_is_ok(status)) {
      vocab_builder = NULL;  // Consumed by build.
    }
  }

  // Phase 3: Build final vocab (tokens + merges + special tokens).
  iree_tokenizer_vocab_t* vocab = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_allocate(parsed.entry_count,
                                                   allocator, &vocab_builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_tiktoken_add_tokens(&parsed, vocab_builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_tiktoken_reconstruct_merges(
        &parsed, byte_to_rank, temp_vocab, vocab_builder);
  }

  // Done with temp vocab.
  iree_tokenizer_vocab_free(temp_vocab);
  temp_vocab = NULL;

  // Add special tokens to the vocabulary. Special tokens have IDs above the
  // BPE vocab range (e.g., cl100k_base BPE range is 0-100255, special tokens
  // start at 100257). ensure_token inserts them with the SPECIAL attribute.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < config->special_token_count && iree_status_is_ok(status); ++i) {
      status = iree_tokenizer_vocab_builder_ensure_token(
          vocab_builder, config->special_token_ids[i],
          config->special_token_strings[i], 0.0f,
          IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
    }
  }

  // Build final vocab.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_builder_build(vocab_builder, &vocab);
    if (iree_status_is_ok(status)) {
      vocab_builder = NULL;  // Consumed by build.
    }
  }
  if (iree_status_is_ok(status)) {
    iree_tokenizer_builder_set_vocab(builder, vocab);
  }

  // Phase 4: Create BPE model from vocab.
  // BYTE_LEVEL_INPUT: input bytes are ByteLevel-encoded before lookup.
  // ENABLE_WORD_CACHE: cache BPE results per word for performance.
  iree_tokenizer_model_t* model = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_bpe_model_allocate(
        vocab,
        IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT |
            IREE_TOKENIZER_BPE_FLAG_ENABLE_WORD_CACHE,
        allocator, &model);
  }
  if (iree_status_is_ok(status)) {
    iree_tokenizer_builder_set_model(builder, model);
  }

  // Phase 5: Compile regex pattern and create segmenter.
  iree_tokenizer_segmenter_t* segmenter = NULL;
  if (iree_status_is_ok(status)) {
    iree_tokenizer_regex_dfa_t dfa;
    uint8_t* dfa_data = NULL;
    iree_tokenizer_regex_compile_error_t compile_error = {0};
    status = iree_tokenizer_regex_compile_and_load(
        config->pattern, IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, allocator,
        &dfa, &dfa_data, &compile_error);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status,
          "failed to compile tiktoken regex pattern at position %zu: %s",
          compile_error.position,
          compile_error.message ? compile_error.message : "(unknown)");
    } else {
      // ISOLATED behavior: regex matches become segments (same as GPT-2).
      status = iree_tokenizer_segmenter_split_allocate(
          dfa, dfa_data, IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED,
          /*invert=*/false, allocator, &segmenter);
      if (!iree_status_is_ok(status)) {
        iree_allocator_free(allocator, dfa_data);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_tokenizer_builder_set_segmenter(builder, segmenter);
  }

  // Phase 6: Create ByteLevel decoder.
  iree_tokenizer_decoder_t* decoder = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_decoder_byte_level_allocate(allocator, &decoder);
  }
  if (iree_status_is_ok(status)) {
    iree_tokenizer_builder_set_decoder(builder, decoder);
  }

  // Phase 7: Build special tokens collection for pre-normalization matching.
  if (iree_status_is_ok(status) && config->special_token_count > 0) {
    iree_tokenizer_special_tokens_builder_t special_builder;
    iree_tokenizer_special_tokens_builder_initialize(allocator,
                                                     &special_builder);

    for (iree_host_size_t i = 0;
         i < config->special_token_count && iree_status_is_ok(status); ++i) {
      status = iree_tokenizer_special_tokens_builder_add(
          &special_builder, config->special_token_strings[i],
          config->special_token_ids[i], IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE);
    }

    if (iree_status_is_ok(status)) {
      iree_tokenizer_special_tokens_t special_tokens;
      iree_tokenizer_special_tokens_initialize(&special_tokens);
      status = iree_tokenizer_special_tokens_builder_build(
          &special_builder, allocator, &special_tokens);
      if (iree_status_is_ok(status) &&
          !iree_tokenizer_special_tokens_is_empty(&special_tokens)) {
        iree_tokenizer_builder_set_special_tokens(builder, &special_tokens);
      }
    }

    iree_tokenizer_special_tokens_builder_deinitialize(&special_builder);
  }

  // Cleanup.
  iree_tokenizer_tiktoken_parsed_free(&parsed);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_builder_free(vocab_builder);
    iree_tokenizer_vocab_free(temp_vocab);
    // vocab, model, segmenter, decoder are cleaned up by builder deinitialize.
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_from_tiktoken(
    iree_string_view_t data, const iree_tokenizer_tiktoken_config_t* config,
    iree_allocator_t allocator, iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(config);
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator, &builder);

  iree_status_t status = iree_tokenizer_parse_tiktoken(data, config, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_builder_build(&builder, out_tokenizer);
  }

  iree_tokenizer_builder_deinitialize(&builder);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
