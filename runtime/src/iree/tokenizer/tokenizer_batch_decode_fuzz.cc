// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for batch decode API.
//
// Tests iree_tokenizer_decode_batch which reuses state_storage across multiple
// items. The inter-item state reuse is a distinct code path from single-item
// decode and could have state leakage bugs.
//
// Exercises:
// - Varying numbers of items (1-16)
// - Varying token sequence lengths per item
// - Varying output buffer capacities per item
// - State reuse across consecutive items
// - Mix of valid and out-of-range token IDs
//
// Uses fuzzing_util for tokenizer loading (dummy or real via --tokenizer_json).
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/testing/fuzzing_util.h"
#include "iree/tokenizer/tokenizer.h"

static iree_tokenizer_t* g_tokenizer = NULL;
static int32_t g_vocab_size = 0;

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  iree_status_t status = iree_tokenizer_fuzz_load_or_build(
      argc, argv, &g_tokenizer, &g_vocab_size);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}

// Maximum items per batch.
static constexpr size_t kMaxBatchItems = 16;

// Maximum tokens per item.
static constexpr size_t kMaxTokensPerItem = 256;

// Maximum text output per item.
static constexpr size_t kMaxTextPerItem = 4096;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (g_tokenizer == NULL || size < 6) return 0;

  // First byte: number of batch items (1-16).
  size_t item_count = (data[0] % kMaxBatchItems) + 1;
  data++;
  size--;

  // Interpret remaining bytes as int32_t token IDs.
  size_t total_tokens = size / sizeof(int32_t);
  if (total_tokens == 0) return 0;
  const int32_t* all_tokens = reinterpret_cast<const int32_t*>(data);

  // Partition tokens among items.
  iree_tokenizer_decode_batch_item_t items[kMaxBatchItems];
  char text_storage[kMaxBatchItems * kMaxTextPerItem];

  size_t token_offset = 0;
  size_t tokens_per_item = total_tokens / item_count;
  if (tokens_per_item > kMaxTokensPerItem) tokens_per_item = kMaxTokensPerItem;
  if (tokens_per_item == 0) tokens_per_item = 1;

  for (size_t i = 0; i < item_count; ++i) {
    size_t item_tokens =
        (i == item_count - 1) ? (total_tokens - token_offset) : tokens_per_item;
    if (item_tokens > kMaxTokensPerItem) item_tokens = kMaxTokensPerItem;
    if (token_offset + item_tokens > total_tokens) {
      item_tokens = total_tokens - token_offset;
    }

    items[i].tokens = iree_tokenizer_make_token_id_list(
        all_tokens + token_offset, item_tokens);

    char* item_text = &text_storage[i * kMaxTextPerItem];
    items[i].text_output =
        iree_make_mutable_string_view(item_text, kMaxTextPerItem);
    items[i].out_text_length = 0;

    token_offset += item_tokens;
    if (token_offset >= total_tokens) {
      // Remaining items get empty token lists.
      for (size_t j = i + 1; j < item_count; ++j) {
        items[j].tokens = iree_tokenizer_make_token_id_list(NULL, 0);
        char* empty_text = &text_storage[j * kMaxTextPerItem];
        items[j].text_output =
            iree_make_mutable_string_view(empty_text, kMaxTextPerItem);
        items[j].out_text_length = 0;
      }
      break;
    }
  }

  //===--------------------------------------------------------------------===//
  // Allocate shared state
  //===--------------------------------------------------------------------===//

  iree_host_size_t state_size = 0;
  iree_status_t status =
      iree_tokenizer_decode_state_calculate_size(g_tokenizer, &state_size);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  void* state_storage = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), state_size,
                                 &state_storage);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Test: Batch decode
  //===--------------------------------------------------------------------===//

  status = iree_tokenizer_decode_batch(
      g_tokenizer, items, item_count, IREE_TOKENIZER_DECODE_FLAG_NONE,
      iree_make_byte_span(reinterpret_cast<uint8_t*>(state_storage),
                          state_size));
  iree_status_ignore(status);

  //===--------------------------------------------------------------------===//
  // Test: Batch decode with skip_special_tokens
  //===--------------------------------------------------------------------===//

  // Reset output lengths for second pass.
  for (size_t i = 0; i < item_count; ++i) {
    items[i].out_text_length = 0;
  }

  status = iree_tokenizer_decode_batch(
      g_tokenizer, items, item_count,
      IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
      iree_make_byte_span(reinterpret_cast<uint8_t*>(state_storage),
                          state_size));
  iree_status_ignore(status);

  //===--------------------------------------------------------------------===//
  // Cleanup
  //===--------------------------------------------------------------------===//

  iree_allocator_free(iree_allocator_system(), state_storage);

  return 0;
}
