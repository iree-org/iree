// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BPE (Byte-Pair Encoding) tokenization model.
//
// BPE is an iterative merge-based tokenization algorithm used by GPT-2,
// Llama, and many other models. Given a segment:
// 1. Break it into initial symbols (characters or bytes)
// 2. Find all adjacent pairs with merge rules
// 3. Apply the highest priority merge
// 4. Repeat until no more merges apply
//
// Example: "hello" -> ['h','e','l','l','o']
//   merge('l','l') -> ['h','e','ll','o']
//   merge('h','e') -> ['he','ll','o']
//   merge('he','ll') -> ['hell','o']
//   merge('hell','o') -> ['hello']
//
// Streaming Behavior:
// BPE uses greedy left-to-right tokenization, which is streaming-compatible.
// Each segment is tokenized independently - tokens can be emitted as they're
// determined without needing to see future segments. State is O(1): just
// tracking current segment index and position within segment for partial
// processing when output buffer fills.

#ifndef IREE_TOKENIZER_MODEL_BPE_H_
#define IREE_TOKENIZER_MODEL_BPE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/types.h"
#include "iree/tokenizer/vocab/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// BPE Configuration Flags
//===----------------------------------------------------------------------===//

// Configuration flags for BPE tokenization.
// Default (0) gives: byte fallback enabled, merges applied, no UNK fusion.
typedef enum iree_tokenizer_bpe_flag_bits_e {
  IREE_TOKENIZER_BPE_FLAG_NONE = 0,
  // Disable byte fallback (<0xXX> tokens) for unknown characters.
  // When not set (default): unknown bytes are encoded as <0xXX> tokens.
  // When set: unknown characters produce [UNK] tokens instead.
  // Models without byte fallback: DeBERTa, some BERT variants.
  IREE_TOKENIZER_BPE_FLAG_NO_BYTE_FALLBACK = 1 << 0,
  // Enable whole-word vocabulary lookup optimization.
  // When set: try matching entire segment in vocab before BPE merging.
  // If whole-segment not found, falls back to normal BPE merging.
  // Used by Llama 3.x models with comprehensive vocabularies.
  IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES = 1 << 1,
  // Fuse consecutive unknown characters into a single [UNK] token.
  // When not set (default): each unknown char produces separate [UNK].
  // When set: "xyz" (all unknown) -> single [UNK] instead of [UNK, UNK, UNK].
  // Used by Phi-3, Mistral, TinyLlama, Gemma-3.
  IREE_TOKENIZER_BPE_FLAG_FUSE_UNK = 1 << 2,
  // Enable ByteLevel input transformation (GPT-2/RoBERTa style).
  // When set: each input byte is mapped to a Unicode codepoint before vocab
  // lookup. Printable ASCII maps to itself; non-printable bytes map to
  // Unicode codepoints 0x100-0x143 (Ā-ƃ range). This allows the vocabulary
  // to represent all possible byte sequences using visible Unicode characters.
  // Used by GPT-2, RoBERTa, BART, GPT-J, GPT-Neo, and many others.
  IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT = 1 << 3,
  // Enable the word cache for repeated-segment memoization.
  // When set: a direct-mapped hash cache stores tokenization results for
  // complete segments, avoiding redundant BPE merge computation for words that
  // appear multiple times. Only effective when the pre-tokenizer produces
  // word-level segments (Split, ByteLevel+regex, Metaspace+split=true).
  // Without word-level segmentation, input flows through the streaming partial
  // path and the cache is never consulted.
  IREE_TOKENIZER_BPE_FLAG_ENABLE_WORD_CACHE = 1 << 4,
} iree_tokenizer_bpe_flag_bits_t;
typedef uint32_t iree_tokenizer_bpe_flags_t;

//===----------------------------------------------------------------------===//
// BPE Model
//===----------------------------------------------------------------------===//

// Allocates a BPE model from a vocabulary.
//
// This builds the merge lookup table needed for efficient encoding.
// The model implements the iree_tokenizer_model_vtable_t interface
// for streaming segment-to-token conversion.
//
// |vocab| is the vocabulary containing tokens, merge rules, and special IDs.
//   The model stores a reference but does not take ownership — the caller
//   must ensure vocab outlives the model.
// |flags| controls tokenization behavior (see iree_tokenizer_bpe_flag_bits_t).
// |allocator| is used for model allocation.
// |out_model| receives the allocated model on success.
iree_status_t iree_tokenizer_bpe_model_allocate(
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_bpe_flags_t flags,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model);

// Sets the end-of-word suffix for BPE encoding.
//
// When set, this suffix is appended to each segment before tokenization.
// Used by CLIP-style models where tokens like "a</w>" represent word-final
// positions. Maximum suffix length is 16 bytes.
//
// |model| must be a BPE model created by iree_tokenizer_bpe_model_allocate.
// |suffix| is the suffix to append (e.g., "</w>"). Pass empty for none.
iree_status_t iree_tokenizer_bpe_model_set_end_of_word_suffix(
    iree_tokenizer_model_t* model, iree_string_view_t suffix);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_MODEL_BPE_H_
