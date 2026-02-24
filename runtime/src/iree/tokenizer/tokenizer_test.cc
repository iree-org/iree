// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/tokenizer.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/tokenizer_test_util.h"

namespace iree {
namespace tokenizer {
namespace {

using testing::ScopedBuilder;
using testing::ScopedVocab;
using testing::ScopedVocabBuilder;

//===----------------------------------------------------------------------===//
// Builder Lifecycle
//===----------------------------------------------------------------------===//

TEST(TokenizerBuilderTest, InitializeDeinitialize) {
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(iree_allocator_system(), &builder);
  iree_tokenizer_builder_deinitialize(&builder);
}

TEST(TokenizerBuilderTest, BuildWithoutComponents) {
  ScopedBuilder builder;

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status =
      iree_tokenizer_builder_build(builder.get(), &tokenizer);

  // Should fail - missing required components (segmenter, model).
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
}

TEST(TokenizerBuilderTest, DoubleDeinitialize) {
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(iree_allocator_system(), &builder);
  iree_tokenizer_builder_deinitialize(&builder);
  // Second deinitialize should be a no-op.
  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Tokenizer Lifecycle
//===----------------------------------------------------------------------===//

TEST(TokenizerTest, FreeNull) {
  // free(nullptr) should be a no-op.
  iree_tokenizer_free(nullptr);
}

TEST(TokenizerTest, CreateAndDestroy) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       testing::CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   testing::CreateBPEModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = testing::BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  iree_tokenizer_free(tokenizer);
}

}  // namespace
}  // namespace tokenizer
}  // namespace iree
