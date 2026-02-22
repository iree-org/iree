// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for HuggingFace decoder JSON parsing.
//
// Tests the decoder parser's robustness against:
// - Malformed JSON syntax
// - Unknown decoder types
// - Invalid parameters (empty patterns, invalid replacements)
// - Deeply nested Sequence decoders
// - Missing required fields
// - Type mismatches (string where object expected, etc.)
// - All decoder types: ByteFallback, ByteLevel, CTC, Metaspace,
//   Replace, Strip, WordPiece, Sequence
//
// When parsing succeeds, the decoder is exercised against a corpus of
// token strings to test the execution path, not just construction.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"
#include "iree/tokenizer/format/huggingface/decoder_json.h"

// Long token sequence for buffer boundary testing.
static iree_string_view_t kLongTokens[256];
static bool kLongTokensInitialized = false;

static void initialize_long_tokens(void) {
  if (kLongTokensInitialized) return;
  // Alternate between common tokens to stress decoder state.
  static const char* kPatterns[] = {"hello", " ", "world", "!", "\xE2\x96\x81"};
  for (size_t i = 0; i < 256; ++i) {
    const char* pattern = kPatterns[i % 5];
    kLongTokens[i] = iree_make_string_view(pattern, strlen(pattern));
  }
  kLongTokensInitialized = true;
}

// Test corpus of token strings for exercising parsed decoders.
// Covers: empty, ASCII, Unicode, byte fallback tokens, WordPiece continuations.
static const char* kTokenInputs[] = {
    "",                   // Empty token.
    "hello",              // Simple ASCII.
    " world",             // Leading space.
    "world",              // No prefix.
    "\xE2\x96\x81hello",  // Metaspace prefix (U+2581).
    "\xE2\x96\x81",       // Bare metaspace.
    "##ing",              // WordPiece continuation.
    "##ed",               // WordPiece continuation.
    "<0x00>",             // ByteFallback token.
    "<0xFF>",             // ByteFallback high byte.
    "<0x41>",             // ByteFallback 'A'.
    "<unk>",              // Unknown token.
    "[CLS]",              // Special BERT token.
    "[SEP]",              // Special BERT token.
    "[PAD]",              // Special BERT token.
    "\xC3\xA9",           // UTF-8: e-acute.
    "\xF0\x9F\x98\x80",   // Emoji.
    "\xC0\xC1",           // Invalid UTF-8.
};
static const size_t kNumTokenInputs =
    sizeof(kTokenInputs) / sizeof(kTokenInputs[0]);

// Exercises a decoder against test token strings using process/finalize.
// Uses a loop to handle cases where the output buffer fills up.
static void exercise_decoder(iree_tokenizer_decoder_t* decoder) {
  // Initialize long tokens on first call.
  initialize_long_tokens();

  iree_host_size_t state_size = iree_tokenizer_decoder_state_size(decoder);
  if (state_size == 0 || state_size > 64 * 1024) {
    return;  // Sanity check.
  }

  void* state_buffer = malloc(state_size);
  if (!state_buffer) return;

  // Small output buffer to force multiple process() calls.
  char output[64];
  iree_mutable_string_view_t output_view =
      iree_make_mutable_string_view(output, sizeof(output));

  // Build token string list from test inputs.
  iree_string_view_t token_views[32];
  for (size_t i = 0; i < kNumTokenInputs && i < 32; ++i) {
    token_views[i] =
        iree_make_string_view(kTokenInputs[i], strlen(kTokenInputs[i]));
  }

  // Test with short token list.
  {
    iree_tokenizer_decoder_state_t* state = NULL;
    iree_status_t status =
        iree_tokenizer_decoder_state_initialize(decoder, state_buffer, &state);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      free(state_buffer);
      return;
    }

    size_t total_consumed = 0;
    size_t token_count = kNumTokenInputs < 32 ? kNumTokenInputs : 32;
    size_t stall_count = 0;  // Track consecutive non-consuming writes.
    while (total_consumed < token_count) {
      iree_tokenizer_string_list_t tokens = iree_tokenizer_make_string_list(
          token_views + total_consumed, token_count - total_consumed);

      iree_host_size_t strings_consumed = 0;
      iree_host_size_t bytes_written = 0;
      status = iree_tokenizer_decoder_state_process(
          state, tokens, output_view, &strings_consumed, &bytes_written);

      if (iree_status_is_resource_exhausted(status)) {
        // Output buffer full - continue.
        iree_status_ignore(status);
      } else if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        break;
      }

      if (strings_consumed == 0) {
        if (bytes_written == 0) {
          break;  // No progress at all.
        }
        // Writing but not consuming - may be stuck on large expansion.
        if (++stall_count > 16) {
          break;  // Abort after too many non-consuming writes.
        }
      } else {
        stall_count = 0;  // Reset on successful consumption.
      }
      total_consumed += strings_consumed;
    }

    // Finalize with loop in case output buffer is small.
    while (iree_tokenizer_decoder_state_has_pending(state)) {
      iree_host_size_t final_written = 0;
      status = iree_tokenizer_decoder_state_finalize(state, output_view,
                                                     &final_written);
      if (!iree_status_is_ok(status) || final_written == 0) {
        iree_status_ignore(status);
        break;
      }
    }

    iree_tokenizer_decoder_state_deinitialize(state);
  }

  // Test with long token list to stress buffer boundaries.
  {
    iree_tokenizer_decoder_state_t* state = NULL;
    iree_status_t status =
        iree_tokenizer_decoder_state_initialize(decoder, state_buffer, &state);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      free(state_buffer);
      return;
    }

    size_t total_consumed = 0;
    size_t stall_count = 0;
    while (total_consumed < 256) {
      iree_tokenizer_string_list_t tokens = iree_tokenizer_make_string_list(
          kLongTokens + total_consumed, 256 - total_consumed);

      iree_host_size_t strings_consumed = 0;
      iree_host_size_t bytes_written = 0;
      status = iree_tokenizer_decoder_state_process(
          state, tokens, output_view, &strings_consumed, &bytes_written);

      if (iree_status_is_resource_exhausted(status)) {
        iree_status_ignore(status);
      } else if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        break;
      }

      if (strings_consumed == 0) {
        if (bytes_written == 0) {
          break;
        }
        if (++stall_count > 16) {
          break;
        }
      } else {
        stall_count = 0;
      }
      total_consumed += strings_consumed;
    }

    // Finalize.
    while (iree_tokenizer_decoder_state_has_pending(state)) {
      iree_host_size_t final_written = 0;
      status = iree_tokenizer_decoder_state_finalize(state, output_view,
                                                     &final_written);
      if (!iree_status_is_ok(status) || final_written == 0) {
        iree_status_ignore(status);
        break;
      }
    }

    iree_tokenizer_decoder_state_deinitialize(state);
  }

  free(state_buffer);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Treat fuzz input as decoder JSON.
  iree_string_view_t decoder_json =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  // Try parsing the JSON as a decoder.
  iree_tokenizer_decoder_t* decoder = NULL;
  iree_status_t status = iree_tokenizer_huggingface_parse_decoder(
      decoder_json, iree_allocator_system(), &decoder);

  if (iree_status_is_ok(status) && decoder != NULL) {
    // Successfully parsed - exercise the decoder with test tokens.
    exercise_decoder(decoder);
  }

  // Always ignore status - most fuzzed inputs will fail parsing.
  iree_status_ignore(status);

  // Free unconditionally - IREE free functions are null-safe.
  iree_tokenizer_decoder_free(decoder);

  return 0;
}
