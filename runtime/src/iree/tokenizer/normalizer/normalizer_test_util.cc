// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/normalizer_test_util.h"

#include <algorithm>

#include "iree/base/internal/unicode.h"

namespace iree::tokenizer::testing {

namespace {

// Maximum output buffer size for tests.
constexpr size_t kMaxOutputSize = 4096;

}  // namespace

std::string ProcessAndFinalize(iree_tokenizer_normalizer_t* normalizer,
                               const std::string& input,
                               bool expect_pending_after_process) {
  ScopedNormalizerState state(normalizer);

  char output_buffer[kMaxOutputSize];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;

  // Process the entire input.
  IREE_EXPECT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), iree_make_string_view(input.data(), input.size()),
      iree_make_mutable_string_view(output_buffer, kMaxOutputSize),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  // Note: With lazy consumption normalizers (e.g., strip with strip_right),
  // not all input may be consumed - trailing content that might need to be
  // discarded is left unconsumed. We don't assert consumed == input.size()
  // because lazy consumption is a valid semantic.

  // Verify has_pending() after process.
  bool has_pending = iree_tokenizer_normalizer_state_has_pending(state.get());
  EXPECT_EQ(has_pending, expect_pending_after_process)
      << "has_pending() after process() should be "
      << (expect_pending_after_process ? "true" : "false") << " for input: \""
      << input << "\"";

  // Collect output from process().
  std::string result(output_buffer, written);

  // Always call finalize(). Unconsumed input (e.g., trailing whitespace) is
  // implicitly discarded when the caller calls finalize without providing more.
  iree_host_size_t finalize_written = 0;
  IREE_EXPECT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(output_buffer, kMaxOutputSize),
      &finalize_written));

  // Add any output from finalize().
  result.append(output_buffer, finalize_written);

  // Verify has_pending() is false after finalize.
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";

  return result;
}

std::string ProcessChunkedAndFinalize(iree_tokenizer_normalizer_t* normalizer,
                                      const std::string& input,
                                      size_t chunk_size,
                                      bool expect_pending_after_process) {
  ScopedNormalizerState state(normalizer);

  std::string result;
  char output_buffer[kMaxOutputSize];

  // Process input in chunks, respecting UTF-8 codepoint boundaries.
  // Per the normalizer contract, callers must not split UTF-8 sequences.
  //
  // With lazy consumption, we need to handle the case where the normalizer
  // returns consumed=0 because it's waiting for more input to decide if
  // content is trailing. The real streaming contract is:
  // - Caller provides input in chunks (on codepoint boundaries)
  // - If consumed=0, caller should provide MORE input if available
  // - If no more input available, caller calls finalize
  //
  // To simulate this correctly, we track how much input has been "provided"
  // vs how much has been "consumed", keeping unconsumed bytes available.
  size_t provided_end = 0;  // How far into input we've provided to normalizer.
  size_t consumed_start = 0;  // Start of unconsumed data.

  while (provided_end < input.size() || consumed_start < provided_end) {
    // Provide more input if we can and haven't reached the end.
    if (provided_end < input.size()) {
      size_t remaining_to_provide = input.size() - provided_end;
      size_t to_provide = std::min(remaining_to_provide, chunk_size);

      // Adjust to not split UTF-8 codepoints (per normalizer contract).
      size_t incomplete_tail = iree_unicode_utf8_incomplete_tail_length(
          input.data() + provided_end, to_provide);
      to_provide -= incomplete_tail;

      // If we can't make progress, take at least one full codepoint.
      if (to_provide == 0 && remaining_to_provide > 0) {
        uint8_t first_byte = (uint8_t)input[provided_end];
        if ((first_byte & 0xE0) == 0xC0)
          to_provide = std::min(remaining_to_provide, size_t{2});
        else if ((first_byte & 0xF0) == 0xE0)
          to_provide = std::min(remaining_to_provide, size_t{3});
        else if ((first_byte & 0xF8) == 0xF0)
          to_provide = std::min(remaining_to_provide, size_t{4});
        else
          to_provide = 1;  // ASCII or invalid, take one byte.
      }

      provided_end += to_provide;
    }

    // Process from consumed_start to provided_end.
    size_t available = provided_end - consumed_start;
    if (available == 0) break;  // Nothing to process.

    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;

    IREE_EXPECT_OK(iree_tokenizer_normalizer_state_process(
        state.get(),
        iree_make_string_view(input.data() + consumed_start, available),
        iree_make_mutable_string_view(output_buffer, kMaxOutputSize),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

    result.append(output_buffer, written);
    consumed_start += consumed;

    // If no progress (consumed=0, written=0), we need more input or finalize.
    if (consumed == 0 && written == 0) {
      if (provided_end >= input.size()) {
        // No more input to provide - break and finalize.
        break;
      }
      // Continue loop to provide more input.
    }
  }

  // Verify has_pending() after all process() calls.
  bool has_pending = iree_tokenizer_normalizer_state_has_pending(state.get());
  EXPECT_EQ(has_pending, expect_pending_after_process)
      << "has_pending() after all process() calls should be "
      << (expect_pending_after_process ? "true" : "false") << " for input: \""
      << input << "\" with chunk_size=" << chunk_size;

  // Always call finalize(). Unconsumed input is implicitly discarded.
  iree_host_size_t finalize_written = 0;
  IREE_EXPECT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(output_buffer, kMaxOutputSize),
      &finalize_written));

  result.append(output_buffer, finalize_written);

  // Verify has_pending() is false after finalize.
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";

  return result;
}

void TestWithAllChunkSizes(iree_tokenizer_normalizer_t* normalizer,
                           const std::string& input,
                           const std::string& expected_output,
                           bool expect_pending_after_process) {
  // First test with no chunking (baseline).
  std::string baseline =
      ProcessAndFinalize(normalizer, input, expect_pending_after_process);
  EXPECT_EQ(baseline, expected_output)
      << "Baseline (no chunking) produced unexpected output";

  // Test with each standard chunk size.
  for (size_t chunk_size : kStandardChunkSizes) {
    SCOPED_TRACE(::testing::Message() << "chunk_size=" << chunk_size);
    std::string chunked = ProcessChunkedAndFinalize(
        normalizer, input, chunk_size, expect_pending_after_process);
    EXPECT_EQ(chunked, expected_output)
        << "Chunked processing (chunk_size=" << chunk_size
        << ") produced different output than expected";
  }
}

void TestLimitedOutputCapacity(iree_tokenizer_normalizer_t* normalizer,
                               const std::string& input,
                               const std::string& expected_output) {
  ScopedNormalizerState state(normalizer);

  std::string result;
  char output_char;

  // Process with capacity=1, one byte at a time.
  size_t position = 0;

  while (position < input.size()) {
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;

    IREE_EXPECT_OK(iree_tokenizer_normalizer_state_process(
        state.get(),
        iree_make_string_view(input.data() + position, input.size() - position),
        iree_make_mutable_string_view(&output_char, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

    if (written > 0) {
      result += output_char;
    }

    // With lazy consumption, consumed=0 && written=0 is valid when the
    // normalizer is waiting for more input (e.g., all remaining is trailing
    // whitespace). Break out and proceed to finalize.
    if (consumed == 0 && written == 0) {
      break;
    }

    position += consumed;
  }

  // Finalize with capacity=1.
  while (true) {
    iree_host_size_t finalize_written = 0;
    IREE_EXPECT_OK(iree_tokenizer_normalizer_state_finalize(
        state.get(), iree_make_mutable_string_view(&output_char, 1),
        &finalize_written));

    if (finalize_written > 0) {
      result += output_char;
    } else {
      break;  // No more finalize output.
    }
  }

  EXPECT_EQ(result, expected_output)
      << "Limited capacity (1) produced different output";

  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";
}

void TestZeroCapacityOutput(iree_tokenizer_normalizer_t* normalizer,
                            const std::string& input) {
  ScopedNormalizerState state(normalizer);

  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;

  // Process with capacity=0 should consume nothing and produce nothing.
  IREE_EXPECT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), iree_make_string_view(input.data(), input.size()),
      iree_make_mutable_string_view(nullptr, 0),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  EXPECT_EQ(consumed, 0u) << "Zero capacity should consume nothing";
  EXPECT_EQ(written, 0u) << "Zero capacity should produce no output";
}

}  // namespace iree::tokenizer::testing
