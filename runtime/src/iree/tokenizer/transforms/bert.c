// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/transforms/bert.h"

#include "iree/base/internal/unicode.h"

// Maximum word size in bytes. Words exceeding this are split.
// This is generous - real words rarely exceed 100 bytes.
#define IREE_TOKENIZER_MAX_WORD_SIZE 256

//===----------------------------------------------------------------------===//
// BERT Punctuation
//===----------------------------------------------------------------------===//

// BERT treats ALL non-letter/non-number ASCII as punctuation.
// This differs from Unicode's punctuation categories.
// See: google-research/bert/tokenization.py _is_punctuation()
//
// From the original Google BERT code:
//   "Characters such as "^", "$", and "`" are not in the Unicode
//   Punctuation class but we treat them as punctuation anyways, for
//   consistency."
//
// ASCII ranges treated as punctuation:
//   33-47:   ! " # $ % & ' ( ) * + , - . /
//   58-64:   : ; < = > ? @
//   91-96:   [ \ ] ^ _ `
//   123-126: { | } ~
static inline bool iree_tokenizer_bert_is_punctuation(uint32_t codepoint) {
  if ((codepoint >= 33 && codepoint <= 47) ||
      (codepoint >= 58 && codepoint <= 64) ||
      (codepoint >= 91 && codepoint <= 96) ||
      (codepoint >= 123 && codepoint <= 126)) {
    return true;
  }
  // For non-ASCII, fall back to Unicode punctuation categories (P*).
  return iree_unicode_is_punctuation(codepoint);
}

//===----------------------------------------------------------------------===//
// BERT Encode (with inline normalization)
//===----------------------------------------------------------------------===//

// BERT encoding with fused normalization: normalizes each codepoint as we scan,
// splits on whitespace, isolates punctuation and CJK characters.
//
// This is single-pass with no heap allocation. Words are accumulated in a
// stack buffer and emitted immediately when complete.

iree_status_t iree_tokenizer_bert_encode(
    const iree_tokenizer_normalizer_t* normalizer, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  if (text.size == 0) return iree_ok_status();

  // Word accumulation buffer (stack allocated).
  char word_buffer[IREE_TOKENIZER_MAX_WORD_SIZE];
  iree_host_size_t word_length = 0;

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

      if (iree_unicode_is_whitespace(ncp)) {
        // End current word if any.
        if (word_length > 0) {
          iree_string_view_t segment =
              iree_make_string_view(word_buffer, word_length);
          iree_string_view_list_t list = {1, &segment};
          IREE_RETURN_IF_ERROR(callback(user_data, list));
          word_length = 0;
        }
      } else if (iree_tokenizer_bert_is_punctuation(ncp) ||
                 iree_unicode_is_cjk(ncp)) {
        // End current word if any.
        if (word_length > 0) {
          iree_string_view_t segment =
              iree_make_string_view(word_buffer, word_length);
          iree_string_view_list_t list = {1, &segment};
          IREE_RETURN_IF_ERROR(callback(user_data, list));
          word_length = 0;
        }
        // Emit punctuation/CJK as its own segment.
        iree_host_size_t char_length =
            iree_unicode_utf8_encode(ncp, word_buffer);
        iree_string_view_t segment =
            iree_make_string_view(word_buffer, char_length);
        iree_string_view_list_t list = {1, &segment};
        IREE_RETURN_IF_ERROR(callback(user_data, list));
      } else {
        // Accumulate normalized character into word.
        iree_host_size_t char_length =
            iree_unicode_utf8_encode(ncp, word_buffer + word_length);
        word_length += char_length;
        // If word buffer is full, emit and reset.
        if (word_length >= IREE_TOKENIZER_MAX_WORD_SIZE - 4) {
          iree_string_view_t segment =
              iree_make_string_view(word_buffer, word_length);
          iree_string_view_list_t list = {1, &segment};
          IREE_RETURN_IF_ERROR(callback(user_data, list));
          word_length = 0;
        }
      }
    }
  }

  // Flush trailing word.
  if (word_length > 0) {
    iree_string_view_t segment =
        iree_make_string_view(word_buffer, word_length);
    iree_string_view_list_t list = {1, &segment};
    IREE_RETURN_IF_ERROR(callback(user_data, list));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Whitespace Encode (with inline normalization)
//===----------------------------------------------------------------------===//

// Whitespace encoding with fused normalization: normalizes each codepoint,
// splits on whitespace only (punctuation is not isolated).

iree_status_t iree_tokenizer_whitespace_encode(
    const iree_tokenizer_normalizer_t* normalizer, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  if (text.size == 0) return iree_ok_status();

  // Word accumulation buffer (stack allocated).
  char word_buffer[IREE_TOKENIZER_MAX_WORD_SIZE];
  iree_host_size_t word_length = 0;

  iree_host_size_t position = 0;
  while (position < text.size) {
    uint32_t cp = iree_unicode_utf8_decode(text, &position);

    // Normalize this codepoint.
    uint32_t norm_cps[4];
    iree_host_size_t norm_count;
    iree_tokenizer_normalizer_normalize_codepoint(normalizer, cp, norm_cps,
                                                  &norm_count);

    for (iree_host_size_t i = 0; i < norm_count; ++i) {
      uint32_t ncp = norm_cps[i];

      if (iree_unicode_is_whitespace(ncp)) {
        // End current word if any.
        if (word_length > 0) {
          iree_string_view_t segment =
              iree_make_string_view(word_buffer, word_length);
          iree_string_view_list_t list = {1, &segment};
          IREE_RETURN_IF_ERROR(callback(user_data, list));
          word_length = 0;
        }
      } else {
        // Accumulate normalized character into word.
        iree_host_size_t char_length =
            iree_unicode_utf8_encode(ncp, word_buffer + word_length);
        word_length += char_length;
        // If word buffer is full, emit and reset.
        if (word_length >= IREE_TOKENIZER_MAX_WORD_SIZE - 4) {
          iree_string_view_t segment =
              iree_make_string_view(word_buffer, word_length);
          iree_string_view_list_t list = {1, &segment};
          IREE_RETURN_IF_ERROR(callback(user_data, list));
          word_length = 0;
        }
      }
    }
  }

  // Flush trailing word.
  if (word_length > 0) {
    iree_string_view_t segment =
        iree_make_string_view(word_buffer, word_length);
    iree_string_view_list_t list = {1, &segment};
    IREE_RETURN_IF_ERROR(callback(user_data, list));
  }

  return iree_ok_status();
}
