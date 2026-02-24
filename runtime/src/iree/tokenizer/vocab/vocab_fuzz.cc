// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for vocabulary construction and lookup.
//
// Tests the vocab builder and hash/trie structures against:
// - Hash collision patterns
// - Very deep trie paths (long token strings)
// - Maximum fanout trie nodes
// - Duplicate token strings
// - Invalid UTF-8 in token strings
// - Token IDs at boundaries (INT32_MAX, negative IDs)
// - Many tokens with similar prefixes
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <climits>
#include <cmath>
#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2) return 0;

  iree_tokenizer_vocab_builder_t* builder = NULL;
  iree_status_t status = iree_tokenizer_vocab_builder_allocate(
      /*capacity_hint=*/64, iree_allocator_system(), &builder);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 1: Build vocabulary from fuzzed token list
  //===--------------------------------------------------------------------===//

  // Parse tokens from input using length-prefixed format.
  // Format: [len:1][text:len][score:4]... (repeat)
  iree_host_size_t pos = 0;
  iree_host_size_t token_count = 0;
  const iree_host_size_t max_tokens = 1000;  // Limit to avoid OOM.

  while (pos < size && token_count < max_tokens) {
    // Read length byte.
    uint8_t len = data[pos++];
    if (pos + len > size) break;

    // Read token text.
    iree_string_view_t text =
        iree_make_string_view(reinterpret_cast<const char*>(data + pos), len);
    pos += len;

    // Read score if available (4 bytes), otherwise use 0.
    float score = 0.0f;
    if (pos + 4 <= size) {
      uint32_t score_bits =
          (uint32_t)data[pos] | ((uint32_t)data[pos + 1] << 8) |
          ((uint32_t)data[pos + 2] << 16) | ((uint32_t)data[pos + 3] << 24);
      // Reinterpret as float.
      // Sanitize: if NaN or Inf, use 0.
      float candidate;
      memcpy(&candidate, &score_bits, sizeof(candidate));
      if (candidate == candidate && candidate != INFINITY &&
          candidate != -INFINITY) {
        score = candidate;
      }
      pos += 4;
    }

    // Read attributes if available (1 byte), otherwise use 0.
    iree_tokenizer_token_attr_t attrs = 0;
    if (pos < size) {
      attrs = (iree_tokenizer_token_attr_t)data[pos++];
    }

    status =
        iree_tokenizer_vocab_builder_add_token(builder, text, score, attrs);
    if (!iree_status_is_ok(status)) {
      // May fail if token is duplicate - that's okay, continue.
      iree_status_ignore(status);
    }
    ++token_count;
  }

  // If we added any tokens, try to build the vocabulary.
  if (token_count == 0) {
    iree_tokenizer_vocab_builder_free(builder);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 2: Build and validate vocabulary
  //===--------------------------------------------------------------------===//

  iree_tokenizer_vocab_t* vocab = NULL;
  status = iree_tokenizer_vocab_builder_build(builder, &vocab);
  // builder is consumed by build() - do not free.

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 3: Stress test lookups
  //===--------------------------------------------------------------------===//

  // Query the vocab with various substrings from the input.
  for (iree_host_size_t i = 0; i < size && i < 100; ++i) {
    iree_host_size_t query_len = (i % 16) + 1;
    if (i + query_len > size) query_len = size - i;

    iree_string_view_t query = iree_make_string_view(
        reinterpret_cast<const char*>(data + i), query_len);

    // Try exact lookup (returns -1 if not found).
    int32_t id = iree_tokenizer_vocab_lookup(vocab, query);
    (void)id;
  }

  // Query all token IDs to exercise token-to-string path.
  iree_host_size_t vocab_capacity = iree_tokenizer_vocab_capacity(vocab);
  for (iree_host_size_t i = 0; i < vocab_capacity && i < 1000; ++i) {
    iree_string_view_t text =
        iree_tokenizer_vocab_token_text(vocab, (int32_t)i);
    (void)text;

    iree_tokenizer_token_attr_t attrs =
        iree_tokenizer_vocab_token_attrs(vocab, (int32_t)i);
    (void)attrs;
  }

  // Query with invalid IDs to test bounds checking.
  {
    (void)iree_tokenizer_vocab_token_text(vocab, -1);
    (void)iree_tokenizer_vocab_token_text(vocab, INT32_MAX);
    (void)iree_tokenizer_vocab_token_text(vocab, (int32_t)vocab_capacity);
  }

  // Clean up.
  iree_tokenizer_vocab_free(vocab);

  return 0;
}
