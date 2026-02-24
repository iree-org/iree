// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_BERT_H_
#define IREE_TOKENIZER_NORMALIZER_BERT_H_

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// BERT Normalizer Configuration
//===----------------------------------------------------------------------===//

// Configuration flags for the BERT normalizer.
// These match HuggingFace's BertNormalizer parameters.
typedef uint32_t iree_tokenizer_bert_normalizer_flags_t;

enum iree_tokenizer_bert_normalizer_flag_bits_e {
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_NONE = 0,

  // clean_text: Remove control characters and normalize whitespace.
  //   - Removes: null (U+0000), replacement char (U+FFFD), and control chars
  //   - Control chars use is_other() (Cc/Cf/Cn/Co) EXCEPT \t, \n, \r
  //   - Maps all whitespace to standard space ' '
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT = 1u << 0,

  // handle_chinese_chars: Add spaces around CJK ideographs.
  //   - Inserts space before AND after each CJK character
  //   - Uses HuggingFace's CJK ranges (see is_chinese_char in bert.rs)
  //   - Does not affect Japanese hiragana/katakana or Korean hangul
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS = 1u << 1,

  // strip_accents: Remove combining marks after NFD decomposition.
  //   - First performs NFD decomposition on each codepoint
  //   - Then removes characters with category Mn (Nonspacing Mark)
  //   - Unlike standalone StripAccents which filters Mn+Mc+Me, BERT's
  //     strip_accents only filters Mn marks
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS = 1u << 2,

  // lowercase: Convert text to lowercase.
  //   - Uses full Unicode case folding (handles İ → i etc.)
  IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE = 1u << 3,
};

// Default BERT normalizer flags matching HuggingFace's
// BertNormalizer::default(). Note: HuggingFace's default has strip_accents=None
// (defaults to lowercase value), so with lowercase=true, strip_accents is also
// true.
#define IREE_TOKENIZER_BERT_NORMALIZER_FLAG_DEFAULT           \
  (IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT |           \
   IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS | \
   IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS |        \
   IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE)

//===----------------------------------------------------------------------===//
// BERT Normalizer
//===----------------------------------------------------------------------===//

// Allocates a BERT normalizer with the specified configuration flags.
//
// The BERT normalizer combines multiple transformations in a single pass,
// matching HuggingFace's BertNormalizer behavior exactly. Transformations
// are applied in order: clean_text → handle_chinese_chars → strip_accents →
// lowercase.
//
// Example usage:
//   // Standard BERT (cased model, no lowercasing or accent stripping):
//   flags = IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT |
//           IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS;
//
//   // Standard BERT (uncased model, with lowercasing and accent stripping):
//   flags = IREE_TOKENIZER_BERT_NORMALIZER_FLAG_DEFAULT;
//
// Output expansion: The BERT normalizer can expand output due to:
//   - CJK spacing: each CJK char becomes " X " (adds 2 bytes per char)
//   - NFD decomposition: precomposed chars expand to base + marks
//   - Lowercase expansion: İ → i + combining dot above (rare)
//
// The normalizer buffers pending output to handle these expansions in a
// streaming context.
iree_status_t iree_tokenizer_normalizer_bert_allocate(
    iree_tokenizer_bert_normalizer_flags_t flags, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_BERT_H_
