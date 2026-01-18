// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/wordpiece.h"

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Default configuration values.
#define IREE_TOKENIZER_WORDPIECE_DEFAULT_MAX_INPUT_CHARS_PER_WORD 200
static const iree_string_view_t
    IREE_TOKENIZER_WORDPIECE_DEFAULT_CONTINUATION_PREFIX = IREE_SVL("##");

// Scratch buffer for WordPiece encoding (prefix + word).
// This limits max_input_chars_per_word to (1024 - prefix_size) / 4 characters,
// since each UTF-8 character can be up to 4 bytes. This limit is enforced at
// tokenizer allocation time. With the default "##" prefix (2 bytes), the max
// is 255 characters.
#define IREE_TOKENIZER_WORDPIECE_SCRATCH_SIZE 1024

// Gets the effective configuration, applying defaults where needed.
static iree_tokenizer_wordpiece_config_t iree_tokenizer_wordpiece_get_config(
    const iree_tokenizer_wordpiece_config_t* config) {
  iree_tokenizer_wordpiece_config_t effective = {
      .max_input_chars_per_word =
          IREE_TOKENIZER_WORDPIECE_DEFAULT_MAX_INPUT_CHARS_PER_WORD,
      .continuing_subword_prefix =
          IREE_TOKENIZER_WORDPIECE_DEFAULT_CONTINUATION_PREFIX,
  };
  if (config) {
    if (config->max_input_chars_per_word > 0) {
      effective.max_input_chars_per_word = config->max_input_chars_per_word;
    }
    // data=NULL means use default, data!=NULL with size=0 means explicit empty.
    if (config->continuing_subword_prefix.data != NULL) {
      effective.continuing_subword_prefix = config->continuing_subword_prefix;
    }
  }
  return effective;
}

// Counts UTF-8 code points in a string view.
static iree_host_size_t iree_string_view_utf8_length(iree_string_view_t sv) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < sv.size; ++i) {
    // Count only lead bytes (0xxxxxxx or 11xxxxxx), skip continuation bytes.
    if ((sv.data[i] & 0xC0) != 0x80) {
      ++count;
    }
  }
  return count;
}

//===----------------------------------------------------------------------===//
// WordPiece Algorithm
//===----------------------------------------------------------------------===//

// WordPiece greedy longest-match algorithm:
// 1. Start with the full remaining substring
// 2. Try to find it in vocab, then try progressively shorter prefixes
// 3. When found, emit the token and continue with the remainder
// 4. For non-initial subwords, prepend the continuation prefix
// 5. If no match is found, the word is unknown

iree_status_t iree_tokenizer_wordpiece_encode_word(
    const iree_tokenizer_vocab_t* vocab,
    const iree_tokenizer_wordpiece_config_t* config, iree_string_view_t word,
    int32_t* out_ids, iree_host_size_t max_ids, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_ids);
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;

  // Handle empty input.
  if (word.size == 0) {
    return iree_ok_status();
  }

  // Get effective configuration.
  iree_tokenizer_wordpiece_config_t cfg =
      iree_tokenizer_wordpiece_get_config(config);

  // Check word length limit.
  iree_host_size_t char_count = iree_string_view_utf8_length(word);
  if (char_count > cfg.max_input_chars_per_word) {
    // Word is too long, emit [UNK].
    iree_tokenizer_special_ids_t special_ids =
        iree_tokenizer_vocab_special_ids(vocab);
    if (special_ids.unk < 0) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "word too long and no [UNK] token configured");
    }
    if (max_ids < 1) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    out_ids[0] = special_ids.unk;
    *out_count = 1;
    return iree_ok_status();
  }

  // Allocate scratch buffer for prefixed lookups on the stack.
  char scratch[IREE_TOKENIZER_WORDPIECE_SCRATCH_SIZE];
  if (cfg.continuing_subword_prefix.size + word.size > sizeof(scratch)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "word + prefix exceeds internal buffer size");
  }

  // Tokenization loop.
  iree_host_size_t token_count = 0;
  iree_host_size_t start = 0;  // Byte offset into word.
  bool is_first_subword = true;

  while (start < word.size) {
    // Find the longest matching prefix starting from |start|.
    iree_host_size_t end = word.size;
    int32_t matched_id = -1;

    // Try progressively shorter substrings until we find a match.
    while (end > start) {
      iree_string_view_t candidate;
      iree_host_size_t prefix_length = 0;

      if (is_first_subword) {
        // First subword: look up directly.
        candidate = iree_make_string_view(word.data + start, end - start);
      } else {
        // Continuation: prepend the prefix.
        prefix_length = cfg.continuing_subword_prefix.size;
        memcpy(scratch, cfg.continuing_subword_prefix.data, prefix_length);
        memcpy(scratch + prefix_length, word.data + start, end - start);
        candidate =
            iree_make_string_view(scratch, prefix_length + (end - start));
      }

      int32_t id = iree_tokenizer_vocab_lookup(vocab, candidate);
      if (id >= 0) {
        matched_id = id;
        break;
      }

      // Shorten the candidate by one UTF-8 character.
      // Walk backwards from end to find the previous code point boundary.
      iree_host_size_t new_end = end - 1;
      while (new_end > start && (word.data[new_end] & 0xC0) == 0x80) {
        --new_end;
      }
      end = new_end;
    }

    if (matched_id < 0) {
      // No match found for any prefix. The word is unknown.
      iree_tokenizer_special_ids_t special_ids =
          iree_tokenizer_vocab_special_ids(vocab);
      if (special_ids.unk < 0) {
        return iree_make_status(IREE_STATUS_NOT_FOUND,
                                "cannot tokenize word and no [UNK] configured");
      }
      if (max_ids < 1) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      out_ids[0] = special_ids.unk;
      *out_count = 1;
      return iree_ok_status();
    }

    // Output the matched token.
    if (token_count >= max_ids) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small for word subwords");
    }
    out_ids[token_count++] = matched_id;

    // Move past the matched portion.
    start = end;
    is_first_subword = false;
  }

  *out_count = token_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// WordPiece Tokenizer (unified interface)
//===----------------------------------------------------------------------===//

// Extended tokenizer structure for WordPiece.
typedef struct iree_tokenizer_wordpiece_t {
  iree_tokenizer_t base;
  iree_tokenizer_wordpiece_config_t config;
  char* prefix_storage;  // Owned copy of continuing_subword_prefix (or NULL).
} iree_tokenizer_wordpiece_t;

static void iree_tokenizer_wordpiece_destroy_impl(iree_tokenizer_t* tokenizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_wordpiece_t* wp = (iree_tokenizer_wordpiece_t*)tokenizer;
  // Destroy WordPiece-specific state. Base handles vocab and struct.
  if (wp->prefix_storage) {
    iree_allocator_free(wp->base.allocator, wp->prefix_storage);
  }
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_wordpiece_encode_word_impl(
    const iree_tokenizer_t* tokenizer, iree_string_view_t word,
    int32_t* out_ids, iree_host_size_t max_ids, iree_host_size_t* out_count) {
  const iree_tokenizer_wordpiece_t* wp =
      (const iree_tokenizer_wordpiece_t*)tokenizer;
  return iree_tokenizer_wordpiece_encode_word(wp->base.vocab, &wp->config, word,
                                              out_ids, max_ids, out_count);
}

static const iree_tokenizer_vtable_t iree_tokenizer_wordpiece_vtable = {
    .destroy = iree_tokenizer_wordpiece_destroy_impl,
    .encode_word = iree_tokenizer_wordpiece_encode_word_impl,
};

iree_status_t iree_tokenizer_wordpiece_allocate(
    iree_tokenizer_vocab_t* vocab,
    const iree_tokenizer_wordpiece_config_t* config, char* prefix_storage,
    iree_allocator_t allocator, iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get effective configuration for validation.
  iree_tokenizer_wordpiece_config_t effective_config =
      iree_tokenizer_wordpiece_get_config(config);

  // Validate that max_input_chars_per_word is compatible with the scratch
  // buffer. Each UTF-8 character can be up to 4 bytes, so we need:
  //   max_input_chars_per_word * 4 + prefix.size <= SCRATCH_SIZE
  iree_host_size_t max_bytes_needed =
      effective_config.max_input_chars_per_word * 4 +
      effective_config.continuing_subword_prefix.size;
  if (max_bytes_needed > IREE_TOKENIZER_WORDPIECE_SCRATCH_SIZE) {
    iree_host_size_t max_chars IREE_ATTRIBUTE_UNUSED =
        (IREE_TOKENIZER_WORDPIECE_SCRATCH_SIZE -
         effective_config.continuing_subword_prefix.size) /
        4;
    iree_tokenizer_vocab_free(vocab);
    if (prefix_storage) iree_allocator_free(allocator, prefix_storage);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "max_input_chars_per_word (%" PRIhsz
        ") too large; max supported is %" PRIhsz " with prefix size %" PRIhsz,
        effective_config.max_input_chars_per_word, max_chars,
        effective_config.continuing_subword_prefix.size);
  }

  // WordPiece requires an UNK token for fallback when words can't be tokenized.
  iree_tokenizer_special_ids_t special_ids =
      iree_tokenizer_vocab_special_ids(vocab);
  if (special_ids.unk < 0) {
    iree_tokenizer_vocab_free(vocab);
    if (prefix_storage) iree_allocator_free(allocator, prefix_storage);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "WordPiece vocab must have an UNK token for unknown word fallback");
  }

  // Allocate the extended tokenizer struct.
  iree_tokenizer_wordpiece_t* tokenizer = NULL;
  iree_status_t status = iree_allocator_malloc(
      allocator, sizeof(iree_tokenizer_wordpiece_t), (void**)&tokenizer);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_free(vocab);
    if (prefix_storage) iree_allocator_free(allocator, prefix_storage);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Initialize base (stores vocab - from here, iree_tokenizer_free handles it).
  iree_tokenizer_initialize(&tokenizer->base, &iree_tokenizer_wordpiece_vtable,
                            allocator, vocab, /*transform=*/NULL,
                            /*decoder=*/NULL, /*postprocessor=*/NULL);
  tokenizer->prefix_storage = prefix_storage;  // Takes ownership.

  // Store the validated config.
  tokenizer->config = effective_config;

  *out_tokenizer = &tokenizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
