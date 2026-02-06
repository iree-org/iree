// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared types for HuggingFace tokenizer.json format parsing.
//
// This header defines format-level types that are shared between section
// parsers (segmenter, model, etc.) to avoid coupling their build dependencies.

#ifndef IREE_TOKENIZER_FORMAT_HUGGINGFACE_TYPES_H_
#define IREE_TOKENIZER_FORMAT_HUGGINGFACE_TYPES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Pre-Tokenizer Flags
//===----------------------------------------------------------------------===//

// Format-level properties discovered during pre_tokenizer parsing that affect
// other tokenizer sections (model, decoder). These are cross-section signals
// that the orchestrator (tokenizer_json.c) routes to the appropriate consumers.
typedef enum iree_tokenizer_huggingface_pre_tokenizer_flag_bits_e {
  IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE = 0,
  // A ByteLevel component is present in the pre_tokenizer tree.
  // Signals that the vocabulary uses GPT-2 byte-to-unicode encoding and the
  // model must apply byte-level input transformation during encoding.
  IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_BYTE_LEVEL = 1u << 0,
  // The ByteLevel component has add_prefix_space=true.
  // A space character is prepended to the input before byte-level encoding,
  // ensuring the first token gets a leading space marker in the vocabulary.
  IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_ADD_PREFIX_SPACE = 1u << 1,
  // The Metaspace pre_tokenizer has prepend_scheme="first" or "always".
  // Signals that a prepend normalizer should be synthesized to emit the
  // replacement character (▁) before the first byte of input. This is required
  // for SentencePiece-style tokenizers where the vocabulary assumes
  // word-initial tokens start with the metaspace marker.
  IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_PREPEND = 1u << 2,
  // A Metaspace pre_tokenizer is present and requires space→replacement
  // character substitution. ALL Metaspace configs require this: the
  // pre_tokenizer replaces 0x20 (space) with the replacement character
  // (default ▁, U+2581) so that the vocabulary can distinguish word boundaries
  // from intra-word spaces.
  IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_REPLACE = 1u << 3,
  // The pre_tokenizer produces word-level segments (splits input into multiple
  // bounded-size pieces). This enables the BPE word cache optimization, which
  // memoizes tokenization results for repeated segments. Set when the
  // pre_tokenizer uses a splitting strategy: ByteLevel with use_regex=true,
  // Split (always splits), or Metaspace with split=true.
  IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT = 1u << 4,
  // The ByteLevel component has trim_offsets=true.
  // When offset tracking is enabled, token offsets should be trimmed to exclude
  // leading/trailing whitespace. This only affects offset tracking (training);
  // inference without offset tracking is unaffected.
  IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_TRIM_OFFSETS = 1u << 5,
  // A Whitespace or WhitespaceSplit pre_tokenizer is present.
  // When combined with METASPACE_REPLACE, a Strip(right) normalizer must be
  // inserted before the Replace normalizer. This ensures trailing whitespace is
  // removed before space→▁ replacement, matching HuggingFace's behavior where
  // WhitespaceSplit discards trailing whitespace before Metaspace runs.
  IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_HAS_WHITESPACE_SPLIT = 1u << 6,
} iree_tokenizer_huggingface_pre_tokenizer_flag_bits_t;
typedef uint32_t iree_tokenizer_huggingface_pre_tokenizer_flags_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_FORMAT_HUGGINGFACE_TYPES_H_
