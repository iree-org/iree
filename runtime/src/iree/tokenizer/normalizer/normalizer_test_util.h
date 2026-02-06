// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test utilities for normalizer implementations.
//
// Provides:
// - RAII wrappers for automatic resource cleanup
// - Test helpers that verify has_pending() and finalize() behavior

#ifndef IREE_TOKENIZER_NORMALIZER_NORMALIZER_TEST_UTIL_H_
#define IREE_TOKENIZER_NORMALIZER_NORMALIZER_TEST_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/testing/scoped_resource.h"

namespace iree::tokenizer::testing {

//===----------------------------------------------------------------------===//
// RAII Wrappers
//===----------------------------------------------------------------------===//

using ScopedNormalizer =
    ScopedResource<iree_tokenizer_normalizer_t, iree_tokenizer_normalizer_free>;

using ScopedNormalizerState =
    ScopedState<iree_tokenizer_normalizer_t, iree_tokenizer_normalizer_state_t,
                iree_tokenizer_normalizer_state_size,
                iree_tokenizer_normalizer_state_initialize,
                iree_tokenizer_normalizer_state_deinitialize>;

//===----------------------------------------------------------------------===//
// Test Constants
//===----------------------------------------------------------------------===//

// Standard chunk sizes for streaming tests.
inline constexpr size_t kStandardChunkSizes[] = {1, 2, 3, 5, 8, 13, 16, 32, 64};

//===----------------------------------------------------------------------===//
// Test Helpers
//===----------------------------------------------------------------------===//

// Processes input through normalizer and finalizes, returning normalized
// string. Verifies has_pending() matches |expect_pending_after_process| after
// process() and returns false after finalize().
std::string ProcessAndFinalize(iree_tokenizer_normalizer_t* normalizer,
                               const std::string& input,
                               bool expect_pending_after_process);

// Same as ProcessAndFinalize but processes input in chunks of |chunk_size|.
std::string ProcessChunkedAndFinalize(iree_tokenizer_normalizer_t* normalizer,
                                      const std::string& input,
                                      size_t chunk_size,
                                      bool expect_pending_after_process);

// Runs ProcessAndFinalize for all kStandardChunkSizes.
// Verifies that all chunk sizes produce identical output.
void TestWithAllChunkSizes(iree_tokenizer_normalizer_t* normalizer,
                           const std::string& input,
                           const std::string& expected_output,
                           bool expect_pending_after_process);

// Tests output buffer overflow handling with capacity=1.
void TestLimitedOutputCapacity(iree_tokenizer_normalizer_t* normalizer,
                               const std::string& input,
                               const std::string& expected_output);

// Tests that zero-capacity output produces no output and consumes nothing.
void TestZeroCapacityOutput(iree_tokenizer_normalizer_t* normalizer,
                            const std::string& input);

}  // namespace iree::tokenizer::testing

#endif  // IREE_TOKENIZER_NORMALIZER_NORMALIZER_TEST_UTIL_H_
