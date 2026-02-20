// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/decoder_test_util.h"

namespace iree::tokenizer::testing {

std::string ProcessAndFinalize(iree_tokenizer_decoder_t* decoder,
                               const std::vector<std::string>& tokens,
                               bool expect_pending_after_process) {
  ScopedDecoderState state(decoder);

  // Build string views for tokens.
  auto views = ToStringViews(tokens);
  auto token_list = MakeStringList(views);

  // Process all tokens at once.
  std::vector<char> output(4096);
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_CHECK_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output.data(), output.size()),
      &strings_consumed, &bytes_written));

  EXPECT_EQ(strings_consumed, tokens.size())
      << "Expected all tokens to be consumed";

  // Verify pending state matches expectation.
  EXPECT_EQ(iree_tokenizer_decoder_state_has_pending(state.get()),
            expect_pending_after_process)
      << "Unexpected has_pending() after process()";

  std::string result(output.data(), bytes_written);

  // Finalize.
  iree_host_size_t finalize_written = 0;
  IREE_CHECK_OK(iree_tokenizer_decoder_state_finalize(
      state.get(), iree_make_mutable_string_view(output.data(), output.size()),
      &finalize_written));

  result.append(output.data(), finalize_written);

  // After finalize, has_pending should be false.
  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";

  return result;
}

std::string ProcessBatchedAndFinalize(iree_tokenizer_decoder_t* decoder,
                                      const std::vector<std::string>& tokens,
                                      size_t batch_size,
                                      bool expect_pending_after_process) {
  ScopedDecoderState state(decoder);

  // Build string views for tokens.
  auto views = ToStringViews(tokens);

  std::string result;
  std::vector<char> output(4096);
  size_t position = 0;

  // Process tokens in batches.
  while (position < tokens.size()) {
    size_t remaining = tokens.size() - position;
    size_t batch = remaining < batch_size ? remaining : batch_size;

    iree_tokenizer_string_list_t token_list =
        iree_tokenizer_make_string_list(views.data() + position, batch);

    iree_host_size_t strings_consumed = 0;
    iree_host_size_t bytes_written = 0;

    IREE_CHECK_OK(iree_tokenizer_decoder_state_process(
        state.get(), token_list,
        iree_make_mutable_string_view(output.data(), output.size()),
        &strings_consumed, &bytes_written));

    // All tokens in this batch should be consumed (with sufficient output).
    EXPECT_EQ(strings_consumed, batch)
        << "Expected all tokens in batch to be consumed";

    result.append(output.data(), bytes_written);
    position += strings_consumed;
  }

  // Check pending state only after all batches processed.
  EXPECT_EQ(iree_tokenizer_decoder_state_has_pending(state.get()),
            expect_pending_after_process)
      << "Unexpected has_pending() after all batches processed";

  // Finalize.
  iree_host_size_t finalize_written = 0;
  IREE_CHECK_OK(iree_tokenizer_decoder_state_finalize(
      state.get(), iree_make_mutable_string_view(output.data(), output.size()),
      &finalize_written));

  result.append(output.data(), finalize_written);

  // After finalize, has_pending should be false.
  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";

  return result;
}

void TestWithAllBatchSizes(iree_tokenizer_decoder_t* decoder,
                           const std::vector<std::string>& tokens,
                           const std::string& expected_output,
                           bool expect_pending_after_process) {
  // First test with all tokens at once.
  std::string full_result =
      ProcessAndFinalize(decoder, tokens, expect_pending_after_process);
  EXPECT_EQ(full_result, expected_output) << "Full batch processing failed";

  // Test with various batch sizes.
  for (size_t batch_size : kStandardBatchSizes) {
    std::string result = ProcessBatchedAndFinalize(
        decoder, tokens, batch_size, expect_pending_after_process);
    EXPECT_EQ(result, expected_output)
        << "Batch size " << batch_size << " produced different output";
  }
}

void TestLimitedOutputCapacity(iree_tokenizer_decoder_t* decoder,
                               const std::vector<std::string>& tokens,
                               const std::string& expected_output) {
  ScopedDecoderState state(decoder);

  // Build string views for tokens.
  auto views = ToStringViews(tokens);

  std::string result;
  size_t position = 0;

  // Find the max token size to ensure our buffer can fit at least one token.
  size_t max_token_size = 1;
  for (const auto& token : tokens) {
    if (token.size() > max_token_size) {
      max_token_size = token.size();
    }
  }

  // Use a small buffer that can fit one token but forces multiple iterations.
  std::vector<char> output(max_token_size);

  // Process tokens one at a time with limited output capacity.
  while (position < tokens.size()) {
    iree_tokenizer_string_list_t token_list = iree_tokenizer_make_string_list(
        views.data() + position, tokens.size() - position);

    iree_host_size_t strings_consumed = 0;
    iree_host_size_t bytes_written = 0;

    IREE_CHECK_OK(iree_tokenizer_decoder_state_process(
        state.get(), token_list,
        iree_make_mutable_string_view(output.data(), output.size()),
        &strings_consumed, &bytes_written));

    // With limited capacity, we should consume at least one token per call.
    EXPECT_GE(strings_consumed, 1u)
        << "Should consume at least one token per call";

    result.append(output.data(), bytes_written);
    position += strings_consumed;
  }

  // Finalize to flush any pending data.
  while (iree_tokenizer_decoder_state_has_pending(state.get())) {
    iree_host_size_t finalize_written = 0;
    IREE_CHECK_OK(iree_tokenizer_decoder_state_finalize(
        state.get(),
        iree_make_mutable_string_view(output.data(), output.size()),
        &finalize_written));

    if (finalize_written > 0) {
      result.append(output.data(), finalize_written);
    } else {
      break;  // No more data to produce.
    }
  }

  EXPECT_EQ(result, expected_output);
}

void TestZeroCapacityOutput(iree_tokenizer_decoder_t* decoder,
                            const std::vector<std::string>& tokens) {
  ScopedDecoderState state(decoder);

  // Build string views for tokens.
  auto views = ToStringViews(tokens);
  auto token_list = MakeStringList(views);

  // Process with zero capacity.
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_CHECK_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list, iree_make_mutable_string_view(nullptr, 0),
      &strings_consumed, &bytes_written));

  // With zero capacity, nothing should be consumed or written.
  EXPECT_EQ(strings_consumed, 0u);
  EXPECT_EQ(bytes_written, 0u);
}

}  // namespace iree::tokenizer::testing
