// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tiktoken file format loader for IREE tokenizer.
//
// Tiktoken is OpenAI's tokenizer format. A .tiktoken file is a line-oriented
// text file where each line contains a base64-encoded token and an integer
// rank:
//
//   <base64-encoded-bytes> <rank>
//
// Ranks are ordered sequentially and start from 0. Gaps are permitted (e.g.,
// p50k_base skips rank 50256, which is reserved for <|endoftext|>). Ranks
// 0-255 encode the 256 single-byte tokens. Ranks 256+ encode multi-byte BPE
// merge tokens.
//
// The .tiktoken file contains only the vocabulary. Metadata required for
// tokenization (regex pattern, special tokens) is defined externally per
// encoding and passed via iree_tokenizer_tiktoken_config_t. Predefined configs
// are provided for the standard OpenAI encodings (cl100k_base, o200k_base,
// r50k_base, p50k_base).
//
// Example usage:
//   iree_tokenizer_t* tokenizer = NULL;
//   iree_status_t status = iree_tokenizer_from_tiktoken(
//       tiktoken_data,
//       iree_tokenizer_tiktoken_config_cl100k_base(),
//       iree_allocator_system(),
//       &tokenizer);
//   if (iree_status_is_ok(status)) {
//     // Use tokenizer for encode/decode
//     iree_tokenizer_free(tokenizer);
//   }

#ifndef IREE_TOKENIZER_FORMAT_TIKTOKEN_TIKTOKEN_H_
#define IREE_TOKENIZER_FORMAT_TIKTOKEN_TIKTOKEN_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Tiktoken Configuration
//===----------------------------------------------------------------------===//

// Configuration for a tiktoken encoding.
//
// A .tiktoken file contains only the BPE vocabulary. Everything else — regex
// pattern, special tokens, encoding name — lives outside the file and varies
// per encoding. This struct carries that external metadata.
//
// All string views must remain valid for the duration of the parse/load call.
// Predefined configs (e.g., iree_tokenizer_tiktoken_config_cl100k_base())
// use static storage and are always valid.
typedef struct iree_tokenizer_tiktoken_config_t {
  // Regex pattern for pre-tokenization splitting.
  // This pattern defines how input text is segmented before BPE encoding.
  // Each encoding has a specific pattern (e.g., cl100k_base uses a pattern
  // that handles contractions, letters, numbers, and whitespace).
  // Must not be empty.
  iree_string_view_t pattern;

  // Special tokens that are matched as literal strings before tokenization.
  // These are token strings like "<|endoftext|>" that should be recognized
  // whole rather than BPE-encoded character by character.
  //
  // special_token_count may be 0 if no special tokens are defined.
  // When non-zero, special_token_strings and special_token_ids must both
  // point to arrays of special_token_count elements.
  iree_host_size_t special_token_count;
  const iree_string_view_t* special_token_strings;
  const int32_t* special_token_ids;
} iree_tokenizer_tiktoken_config_t;

//===----------------------------------------------------------------------===//
// Predefined Encoding Configs
//===----------------------------------------------------------------------===//

// Returns the config for cl100k_base (GPT-4, GPT-3.5-turbo, text-embedding).
// 100,256 BPE tokens + 5 special tokens.
const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_cl100k_base(void);

// Returns the config for o200k_base (GPT-4o, GPT-4o-mini).
// 199,998 BPE tokens + 2 special tokens.
const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_o200k_base(void);

// Returns the config for o200k_harmony (GPT-4o with extended special tokens).
// Same BPE vocabulary as o200k_base. Adds 10 named special tokens used in
// ChatGPT's message format (<|startoftext|>, <|return|>, <|constrain|>, etc.).
// The 1081 reserved-range tokens (IDs 200000-201087 minus the 10 named ones)
// are omitted from this predefined config — they reserve ID space for future
// use and do not appear in normal input text. Callers who need the full set
// can construct a custom iree_tokenizer_tiktoken_config_t.
const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_o200k_harmony(void);

// Returns the config for r50k_base (GPT-3, text-davinci-002/003).
// 50,256 BPE tokens + 1 special token.
const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_r50k_base(void);

// Returns the config for gpt2.
// Identical to r50k_base — same BPE vocabulary, pattern, and special tokens.
const iree_tokenizer_tiktoken_config_t* iree_tokenizer_tiktoken_config_gpt2(
    void);

// Returns the config for p50k_base (Codex, code-davinci-002).
// 50,280 BPE tokens + 1 special token.
const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_p50k_base(void);

// Returns the config for p50k_edit (Codex edit models).
// Same BPE vocabulary and pattern as p50k_base. Adds 3 FIM (fill-in-middle)
// special tokens: <|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>.
const iree_tokenizer_tiktoken_config_t*
iree_tokenizer_tiktoken_config_p50k_edit(void);

//===----------------------------------------------------------------------===//
// Encoding Lookup
//===----------------------------------------------------------------------===//

// Looks up a predefined tiktoken config by encoding name.
//
// Supports all 7 standard OpenAI encoding names:
//   cl100k_base, o200k_base, o200k_harmony, r50k_base, gpt2, p50k_base,
//   p50k_edit
//
// Returns NULL if the name is not recognized. Callers can construct a custom
// iree_tokenizer_tiktoken_config_t for non-standard encodings.
const iree_tokenizer_tiktoken_config_t* iree_tokenizer_tiktoken_config_by_name(
    iree_string_view_t name);

//===----------------------------------------------------------------------===//
// Top-Level Loader
//===----------------------------------------------------------------------===//

// Creates a tokenizer from a tiktoken file.
//
// Parses the tiktoken data, constructs the BPE vocabulary with merge
// reconstruction, compiles the regex pattern, and assembles a complete
// tokenizer. The resulting tokenizer is behaviorally equivalent to OpenAI's
// tiktoken library for the same encoding.
//
// |data| is the contents of a .tiktoken file.
// |config| provides the regex pattern and special tokens for this encoding.
// |allocator| is used for all allocations; stored for cleanup.
// |out_tokenizer| receives the allocated tokenizer on success.
//
// Returns:
//   IREE_STATUS_OK on success.
//   IREE_STATUS_INVALID_ARGUMENT for malformed tiktoken data, invalid base64,
//     non-contiguous ranks, or empty input.
//   IREE_STATUS_FAILED_PRECONDITION for regex compilation failure.
//   IREE_STATUS_OUT_OF_RANGE for allocation failures.
iree_status_t iree_tokenizer_from_tiktoken(
    iree_string_view_t data, const iree_tokenizer_tiktoken_config_t* config,
    iree_allocator_t allocator, iree_tokenizer_t** out_tokenizer);

//===----------------------------------------------------------------------===//
// Builder-Based Loader (Advanced)
//===----------------------------------------------------------------------===//

// Parses tiktoken data and populates a builder.
//
// This is the lower-level API that allows callers to inspect or modify the
// builder before calling iree_tokenizer_builder_build(). Use this when you
// need to customize the tokenizer after parsing.
//
// The builder must be initialized before calling this function. On success,
// the builder owns all parsed components. On failure, the builder is left in
// a valid but unspecified state (callers should deinitialize it).
//
// |data| is the contents of a .tiktoken file.
// |config| provides the regex pattern and special tokens for this encoding.
// |builder| is the pre-initialized builder to populate.
//
// Returns:
//   IREE_STATUS_OK on success.
//   IREE_STATUS_INVALID_ARGUMENT for malformed tiktoken data.
//   IREE_STATUS_FAILED_PRECONDITION for regex compilation failure.
iree_status_t iree_tokenizer_parse_tiktoken(
    iree_string_view_t data, const iree_tokenizer_tiktoken_config_t* config,
    iree_tokenizer_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_FORMAT_TIKTOKEN_TIKTOKEN_H_
