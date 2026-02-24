// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for precompiled normalizer binary format parsing.
//
// Tests iree_tokenizer_precompiled_normalizer_allocate against:
// - Corrupt double-array trie structures (invalid offsets, labels)
// - Invalid pool string offsets (out of bounds, truncated)
// - Malformed length fields (trie_size exceeds data, underflows)
// - Truncated data at various points
// - Extreme sizes (very large trie, very small pool)
// - Malicious format crafted to cause integer overflows
//
// This is a critical security surface: the binary format is produced by
// base64-decoding user-provided JSON, so an attacker controls the exact bytes.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer/precompiled.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Treat the fuzz input as raw binary charsmap data (post-base64-decode).
  iree_const_byte_span_t charsmap = iree_make_const_byte_span(data, size);

  // Attempt to parse the binary format.
  iree_tokenizer_normalizer_t* normalizer = NULL;
  iree_status_t status = iree_tokenizer_precompiled_normalizer_allocate(
      charsmap, iree_allocator_system(), &normalizer);

  if (iree_status_is_ok(status) && normalizer != NULL) {
    // Successfully parsed - exercise the normalizer briefly to ensure
    // internal state is consistent (not just parsed without crash).

    // Allocate state.
    iree_host_size_t state_size =
        iree_tokenizer_normalizer_state_size(normalizer);
    void* state_buffer = NULL;
    iree_status_t alloc_status = iree_allocator_malloc(
        iree_allocator_system(), state_size, &state_buffer);

    if (iree_status_is_ok(alloc_status)) {
      iree_tokenizer_normalizer_state_t* state = NULL;
      iree_status_t init_status = iree_tokenizer_normalizer_state_initialize(
          normalizer, state_buffer, &state);

      if (iree_status_is_ok(init_status)) {
        // Feed a simple test input to exercise trie lookups.
        const char* test_inputs[] = {"hello", "日本語", "\xC0\x80", "café"};
        char output[256];

        for (int i = 0; i < 4; ++i) {
          iree_string_view_t input = iree_make_cstring_view(test_inputs[i]);
          iree_mutable_string_view_t out_view =
              iree_make_mutable_string_view(output, sizeof(output));
          iree_host_size_t bytes_consumed = 0;
          iree_host_size_t bytes_written = 0;

          iree_status_t proc_status = iree_tokenizer_normalizer_state_process(
              state, input, out_view, IREE_TOKENIZER_NORMALIZER_FLAG_NONE,
              &bytes_consumed, &bytes_written);
          iree_status_ignore(proc_status);
        }

        // Finalize.
        iree_mutable_string_view_t final_view =
            iree_make_mutable_string_view(output, sizeof(output));
        iree_host_size_t final_written = 0;
        iree_status_t fin_status = iree_tokenizer_normalizer_state_finalize(
            state, final_view, &final_written);
        iree_status_ignore(fin_status);
      }
      iree_status_ignore(init_status);

      iree_allocator_free(iree_allocator_system(), state_buffer);
    }
    iree_status_ignore(alloc_status);

    iree_tokenizer_normalizer_free(normalizer);
  }

  iree_status_ignore(status);
  return 0;
}
