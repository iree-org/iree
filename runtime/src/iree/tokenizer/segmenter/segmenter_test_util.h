// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test utilities for segmenter implementations.
//
// Provides:
// - RAII wrappers for automatic resource cleanup
// - Test helpers that verify has_pending() and finalize() behavior

#ifndef IREE_TOKENIZER_SEGMENTER_SEGMENTER_TEST_UTIL_H_
#define IREE_TOKENIZER_SEGMENTER_SEGMENTER_TEST_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/segmenter.h"
#include "iree/tokenizer/testing/scoped_resource.h"

namespace iree::tokenizer::testing {

//===----------------------------------------------------------------------===//
// RAII Wrappers
//===----------------------------------------------------------------------===//

using ScopedSegmenter =
    ScopedResource<iree_tokenizer_segmenter_t, iree_tokenizer_segmenter_free>;

using ScopedSegmenterState =
    ScopedState<iree_tokenizer_segmenter_t, iree_tokenizer_segmenter_state_t,
                iree_tokenizer_segmenter_state_size,
                iree_tokenizer_segmenter_state_initialize,
                iree_tokenizer_segmenter_state_deinitialize>;

//===----------------------------------------------------------------------===//
// Test Constants
//===----------------------------------------------------------------------===//

// Standard chunk sizes for streaming tests.
// These values exercise boundary conditions: byte-at-a-time (1), various small
// sizes (2, 3, 5, 8, 13), and typical buffer sizes (16, 32, 64).
inline constexpr size_t kStandardChunkSizes[] = {1, 2, 3, 5, 8, 13, 16, 32, 64};

//===----------------------------------------------------------------------===//
// Test Helpers
//===----------------------------------------------------------------------===//

// Processes input through segmenter and finalizes, returning all segments.
// Verifies has_pending() matches |expect_pending_after_process| after process()
// and returns false after finalize().
std::vector<std::string> ProcessAndFinalize(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    bool expect_pending_after_process);

// Same as ProcessAndFinalize but processes input in chunks of |chunk_size|.
std::vector<std::string> ProcessChunkedAndFinalize(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    size_t chunk_size, bool expect_pending_after_process);

// Same as ProcessChunkedAndFinalize but respects UTF-8 codepoint boundaries.
// This simulates the tokenizer's behavior: chunk sizes are adjusted to avoid
// splitting multi-byte UTF-8 sequences. Use this for segmenters that require
// complete UTF-8 codepoints (e.g., regex-based segmenters using \p{L}).
std::vector<std::string> ProcessChunkedAndFinalizeUtf8(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    size_t chunk_size, bool expect_pending_after_process);

// Runs ProcessAndFinalize for all kStandardChunkSizes.
// Verifies that all chunk sizes produce identical segment lists.
void TestWithAllChunkSizes(iree_tokenizer_segmenter_t* segmenter,
                           iree_string_view_t input,
                           const std::vector<std::string>& expected_segments,
                           bool expect_pending_after_process);

// Same as TestWithAllChunkSizes but uses UTF-8-aware chunking.
// Use this for segmenters that require complete UTF-8 codepoints.
void TestWithAllChunkSizesUtf8(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    const std::vector<std::string>& expected_segments,
    bool expect_pending_after_process);

// Tests output buffer overflow handling with capacity=1.
void TestLimitedOutputCapacity(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    const std::vector<std::string>& expected_segments);

// Tests that zero-capacity output produces no segments and consumes nothing.
void TestZeroCapacityOutput(iree_tokenizer_segmenter_t* segmenter,
                            iree_string_view_t input);

}  // namespace iree::tokenizer::testing

#endif  // IREE_TOKENIZER_SEGMENTER_SEGMENTER_TEST_UTIL_H_
