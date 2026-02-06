// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test utilities for decoder implementations.
//
// Provides:
// - RAII wrappers for automatic resource cleanup
// - Test helpers that verify has_pending() and finalize() behavior

#ifndef IREE_TOKENIZER_DECODER_DECODER_TEST_UTIL_H_
#define IREE_TOKENIZER_DECODER_DECODER_TEST_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/decoder.h"
#include "iree/tokenizer/testing/scoped_resource.h"

namespace iree::tokenizer::testing {

//===----------------------------------------------------------------------===//
// RAII Wrappers
//===----------------------------------------------------------------------===//

using ScopedDecoder =
    ScopedResource<iree_tokenizer_decoder_t, iree_tokenizer_decoder_free>;

using ScopedDecoderState =
    ScopedState<iree_tokenizer_decoder_t, iree_tokenizer_decoder_state_t,
                iree_tokenizer_decoder_state_size,
                iree_tokenizer_decoder_state_initialize,
                iree_tokenizer_decoder_state_deinitialize>;

//===----------------------------------------------------------------------===//
// Test Constants
//===----------------------------------------------------------------------===//

// Standard batch sizes for streaming tests.
inline constexpr size_t kStandardBatchSizes[] = {1, 2, 3, 5, 8, 13, 16, 32, 64};

//===----------------------------------------------------------------------===//
// Test Helpers
//===----------------------------------------------------------------------===//

// Creates a string list from a vector of strings.
// The caller must ensure the vector outlives the returned list.
inline iree_tokenizer_string_list_t MakeStringList(
    const std::vector<iree_string_view_t>& views) {
  return iree_tokenizer_make_string_list(views.data(), views.size());
}

// Converts a vector of std::string to a vector of iree_string_view_t.
// The caller must ensure the source strings outlive the returned views.
inline std::vector<iree_string_view_t> ToStringViews(
    const std::vector<std::string>& strings) {
  std::vector<iree_string_view_t> views;
  views.reserve(strings.size());
  for (const auto& s : strings) {
    views.push_back(iree_make_string_view(s.data(), s.size()));
  }
  return views;
}

// Processes token strings through decoder and finalizes, returning decoded
// string. Verifies has_pending() matches |expect_pending_after_process| after
// process() and returns false after finalize().
std::string ProcessAndFinalize(iree_tokenizer_decoder_t* decoder,
                               const std::vector<std::string>& tokens,
                               bool expect_pending_after_process);

// Same as ProcessAndFinalize but processes tokens in batches of |batch_size|.
std::string ProcessBatchedAndFinalize(iree_tokenizer_decoder_t* decoder,
                                      const std::vector<std::string>& tokens,
                                      size_t batch_size,
                                      bool expect_pending_after_process);

// Runs ProcessAndFinalize for all kStandardBatchSizes.
// Verifies that all batch sizes produce identical output.
void TestWithAllBatchSizes(iree_tokenizer_decoder_t* decoder,
                           const std::vector<std::string>& tokens,
                           const std::string& expected_output,
                           bool expect_pending_after_process);

// Tests output buffer overflow handling with capacity=1.
void TestLimitedOutputCapacity(iree_tokenizer_decoder_t* decoder,
                               const std::vector<std::string>& tokens,
                               const std::string& expected_output);

// Tests that zero-capacity output produces no output and consumes nothing.
void TestZeroCapacityOutput(iree_tokenizer_decoder_t* decoder,
                            const std::vector<std::string>& tokens);

}  // namespace iree::tokenizer::testing

#endif  // IREE_TOKENIZER_DECODER_DECODER_TEST_UTIL_H_
