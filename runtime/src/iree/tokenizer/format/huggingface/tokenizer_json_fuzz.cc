// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for HuggingFace tokenizer.json parsing.
//
// Tests the JSON parser's robustness against:
// - Malformed JSON syntax (unclosed brackets, invalid escapes, etc.)
// - Missing required fields (model, vocab, etc.)
// - Invalid token IDs and out-of-range values
// - Invalid UTF-8 in token strings
// - Mismatched vocab sizes and merge references
// - Extreme values (max_token_length, vocab sizes, etc.)
// - Unexpected types (string where number expected, etc.)
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Treat the fuzz input as JSON content.
  iree_string_view_t json =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  // Attempt to parse the JSON as a tokenizer.
  // This exercises all component parsers:
  // - model_json (BPE/WordPiece/Unigram vocab, merges)
  // - normalizer_json (NFC, BERT, lowercase, etc.)
  // - segmenter_json (pre_tokenizer: Split, Metaspace, etc.)
  // - decoder_json (ByteLevel, Metaspace, etc.)
  // - postprocessor_json (TemplateProcessing, etc.)
  // - added_tokens_json (special tokens)
  iree_tokenizer_t* tokenizer = NULL;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json, iree_allocator_system(), &tokenizer);

  if (iree_status_is_ok(status)) {
    // Successfully parsed - clean up.
    iree_tokenizer_free(tokenizer);
  }

  // Always ignore the status - we expect most fuzzed inputs to fail.
  iree_status_ignore(status);

  return 0;
}
