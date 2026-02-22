// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for HuggingFace model JSON parsing.
//
// Tests the model parser's robustness against:
// - Malformed JSON syntax
// - Invalid vocabulary entries (empty tokens, duplicate IDs, invalid UTF-8)
// - Invalid BPE merges (unknown tokens, malformed format)
// - Out-of-range token IDs
// - Extreme vocabulary sizes
// - Missing required fields
// - Type mismatches (string where number expected, etc.)
// - Model type detection edge cases
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/format/huggingface/added_tokens_json.h"
#include "iree/tokenizer/format/huggingface/model_json.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Treat fuzz input as model JSON.
  iree_string_view_t model_json =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  // Use empty added_tokens for simplicity - the parser should handle this.
  iree_tokenizer_huggingface_added_tokens_t added_tokens;
  memset(&added_tokens, 0, sizeof(added_tokens));

  // Try parsing with default flags and tuning.
  iree_tokenizer_model_t* model = NULL;
  iree_tokenizer_vocab_t* vocab = NULL;
  iree_status_t status = iree_tokenizer_huggingface_parse_model(
      model_json, &added_tokens,
      IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE,
      iree_allocator_system(), &model, &vocab);

  if (iree_status_is_ok(status)) {
    // Successfully parsed - clean up.
    iree_tokenizer_model_free(model);
    iree_tokenizer_vocab_free(vocab);
  }

  // Always ignore status - most fuzzed inputs will fail parsing.
  iree_status_ignore(status);

  return 0;
}
