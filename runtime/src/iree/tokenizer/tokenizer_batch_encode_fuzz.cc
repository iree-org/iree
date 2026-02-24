// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for batch encode API.
//
// Tests iree_tokenizer_encode_batch which reuses state_storage across multiple
// items via encode_state_reset. The inter-item state reuse is a distinct code
// path from single-item encode and could have state leakage bugs.
//
// Exercises:
// - Varying numbers of items (1-16)
// - Varying text lengths per item
// - Varying output buffer capacities per item
// - State reuse across consecutive items
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

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  iree_status_t status =
      iree_tokenizer_fuzz_load_or_build(argc, argv, &g_tokenizer, NULL);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}

// Maximum items per batch to avoid OOM.
static constexpr size_t kMaxBatchItems = 16;

// Maximum tokens per item.
static constexpr size_t kMaxTokensPerItem = 1024;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (g_tokenizer == NULL || size < 2) return 0;

  // First byte: number of batch items (1-16).
  size_t item_count = (data[0] % kMaxBatchItems) + 1;
  data++;
  size--;

  // Partition remaining bytes among items.
  // Each item gets a roughly equal share, with remainder going to last item.
  iree_tokenizer_encode_batch_item_t items[kMaxBatchItems];
  iree_tokenizer_token_id_t token_storage[kMaxBatchItems * kMaxTokensPerItem];

  size_t offset = 0;
  size_t base_item_size = size / item_count;
  for (size_t i = 0; i < item_count; ++i) {
    size_t item_size = (i == item_count - 1) ? (size - offset) : base_item_size;

    items[i].text = iree_make_string_view(
        reinterpret_cast<const char*>(data + offset), item_size);

    // Each item gets its own slice of the token output buffer.
    iree_tokenizer_token_id_t* item_tokens =
        &token_storage[i * kMaxTokensPerItem];
    items[i].output = iree_tokenizer_make_token_output(item_tokens, NULL, NULL,
                                                       kMaxTokensPerItem);
    items[i].out_token_count = 0;

    offset += item_size;
  }

  //===--------------------------------------------------------------------===//
  // Allocate shared state and transform buffer
  //===--------------------------------------------------------------------===//

  iree_host_size_t state_size = 0;
  iree_status_t status =
      iree_tokenizer_encode_state_calculate_size(g_tokenizer, &state_size);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  void* state_storage_buf = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), state_size,
                                 &state_storage_buf);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  iree_host_size_t buffer_size =
      iree_tokenizer_transform_buffer_recommended_size(
          base_item_size > 0 ? base_item_size : 64);
  void* transform_buffer = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), buffer_size,
                                 &transform_buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), state_storage_buf);
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Test: Batch encode
  //===--------------------------------------------------------------------===//

  iree_tokenizer_encode_flags_t flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  if (iree_tokenizer_fuzz_track_offsets()) {
    flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
  }

  status = iree_tokenizer_encode_batch(
      g_tokenizer, items, item_count, flags,
      iree_make_byte_span(reinterpret_cast<uint8_t*>(state_storage_buf),
                          state_size),
      iree_make_byte_span(reinterpret_cast<uint8_t*>(transform_buffer),
                          buffer_size),
      iree_tokenizer_offset_run_list_empty());
  // RESOURCE_EXHAUSTED is expected when output buffers are too small.
  iree_status_ignore(status);

  //===--------------------------------------------------------------------===//
  // Cleanup
  //===--------------------------------------------------------------------===//

  iree_allocator_free(iree_allocator_system(), transform_buffer);
  iree_allocator_free(iree_allocator_system(), state_storage_buf);

  return 0;
}
