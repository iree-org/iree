// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/wordpiece.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/decoder/decoder_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::MakeStringList;
using testing::ProcessAndFinalize;
using testing::ScopedDecoder;
using testing::ScopedDecoderState;
using testing::TestWithAllBatchSizes;
using testing::ToStringViews;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class WordPieceDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override { decoder_ = CreateDecoder("##", true); }

  ScopedDecoder CreateDecoder(const char* prefix, bool cleanup) {
    iree_tokenizer_decoder_wordpiece_config_t config =
        iree_tokenizer_make_decoder_wordpiece_config(
            iree_make_cstring_view(prefix), cleanup);
    iree_tokenizer_decoder_t* raw_decoder = nullptr;
    IREE_CHECK_OK(iree_tokenizer_decoder_wordpiece_allocate(
        config, iree_allocator_system(), &raw_decoder));
    return ScopedDecoder(raw_decoder);
  }

  iree_tokenizer_decoder_t* decoder() { return decoder_.get(); }

 private:
  ScopedDecoder decoder_;
};

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

TEST_F(WordPieceDecoderTest, CreateAndDestroy) {
  EXPECT_NE(decoder(), nullptr);
}

TEST_F(WordPieceDecoderTest, FirstTokenPassthrough) {
  TestWithAllBatchSizes(decoder(), {"Hello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, PrefixStripping) {
  // Token with ## prefix has prefix stripped, no space added.
  TestWithAllBatchSizes(decoder(), {"Hello", "##world"}, "Helloworld",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, SpacePrepending) {
  // Token without ## prefix gets space prepended.
  TestWithAllBatchSizes(decoder(), {"Hello", "world"}, "Hello world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, MixedPrefixAndSpace) {
  TestWithAllBatchSizes(decoder(), {"Hello", "##world", "!"}, "Helloworld!",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, MultipleContinuations) {
  TestWithAllBatchSizes(decoder(), {"un", "##believ", "##able"}, "unbelievable",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, EmptyInput) {
  std::string result =
      ProcessAndFinalize(decoder(), {}, /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

//===----------------------------------------------------------------------===//
// Cleanup Pattern Tests
//===----------------------------------------------------------------------===//

TEST_F(WordPieceDecoderTest, CleanupPunctuation) {
  // " ." -> "." pattern
  TestWithAllBatchSizes(decoder(), {"Hello", "."}, "Hello.",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, CleanupQuestionMark) {
  TestWithAllBatchSizes(decoder(), {"What", "?"}, "What?",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, CleanupExclamation) {
  TestWithAllBatchSizes(decoder(), {"Wow", "!"}, "Wow!",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, CleanupComma) {
  TestWithAllBatchSizes(decoder(), {"Hello", ",", "world"}, "Hello, world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, CleanupContractionNt) {
  // " n't" -> "n't"
  TestWithAllBatchSizes(decoder(), {"do", "n't"}, "don't",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, CleanupContractionM) {
  TestWithAllBatchSizes(decoder(), {"I", "'m"}, "I'm",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, CleanupContractionVe) {
  TestWithAllBatchSizes(decoder(), {"I", "'ve"}, "I've",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, CleanupContractionRe) {
  TestWithAllBatchSizes(decoder(), {"they", "'re"}, "they're",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, CleanupPossessive) {
  TestWithAllBatchSizes(decoder(), {"John", "'s"}, "John's",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, PrefixThenPunctuation) {
  // Prefix-stripped token followed by punctuation should not have space.
  TestWithAllBatchSizes(decoder(), {"go", "##ing", "."}, "going.",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Cleanup Disabled
//===----------------------------------------------------------------------===//

TEST_F(WordPieceDecoderTest, CleanupDisabledPunctuation) {
  auto decoder_no_cleanup = CreateDecoder("##", false);
  TestWithAllBatchSizes(decoder_no_cleanup.get(), {"Hello", "."}, "Hello .",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, CleanupDisabledContraction) {
  auto decoder_no_cleanup = CreateDecoder("##", false);
  TestWithAllBatchSizes(decoder_no_cleanup.get(), {"do", "n't"}, "do n't",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(WordPieceDecoderTest, SingleByteOutputBuffer) {
  // Force byte-by-byte emission to test streaming.
  ScopedDecoderState state(decoder());

  std::vector<std::string> tokens = {"Hi", "there"};
  std::string result;

  for (const auto& token : tokens) {
    // Keep string alive - views point into this vector.
    std::vector<std::string> token_vec = {token};
    auto views = ToStringViews(token_vec);
    bool consumed = false;
    while (!consumed) {
      char byte[4];
      iree_host_size_t strings_consumed = 0;
      iree_host_size_t bytes_written = 0;

      IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
          state.get(), iree_tokenizer_make_string_list(views.data(), 1),
          iree_make_mutable_string_view(byte, 1), &strings_consumed,
          &bytes_written));

      if (bytes_written > 0) {
        result += byte[0];
      }
      if (strings_consumed > 0) {
        consumed = true;
      }
      // Ensure progress to avoid infinite loop.
      ASSERT_TRUE(bytes_written > 0 || strings_consumed > 0)
          << "No progress made";
    }
  }

  EXPECT_EQ(result, "Hi there");
}

TEST_F(WordPieceDecoderTest, BufferFullMidToken) {
  // Test buffer exhaustion mid-token and correct resume.
  ScopedDecoderState state(decoder());

  // Keep string alive - views point into this vector.
  std::vector<std::string> token_vec = {"Hello"};
  auto views = ToStringViews(token_vec);

  // Process with tiny buffer (3 bytes).
  char buffer[4];
  std::string result;
  bool consumed = false;

  while (!consumed) {
    iree_host_size_t strings_consumed = 0;
    iree_host_size_t bytes_written = 0;

    IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
        state.get(), iree_tokenizer_make_string_list(views.data(), 1),
        iree_make_mutable_string_view(buffer, sizeof(buffer)),
        &strings_consumed, &bytes_written));

    result.append(buffer, bytes_written);
    if (strings_consumed > 0) {
      consumed = true;
    }
    ASSERT_TRUE(bytes_written > 0 || strings_consumed > 0);
  }

  EXPECT_EQ(result, "Hello");
}

TEST_F(WordPieceDecoderTest, BufferFullDuringCleanup) {
  // Buffer too small for cleanup replacement.
  ScopedDecoderState state(decoder());

  // First process a token to not be first anymore.
  // Keep strings alive - views point into these vectors.
  std::vector<std::string> first_vec = {"Hello"};
  auto first_views = ToStringViews(first_vec);
  char buffer[32];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(first_views.data(), 1),
      iree_make_mutable_string_view(buffer, sizeof(buffer)), &strings_consumed,
      &bytes_written));
  EXPECT_EQ(strings_consumed, 1u);

  // Now try "." with 0-byte buffer - can't write "." replacement.
  std::vector<std::string> dot_vec = {"."};
  auto dot_views = ToStringViews(dot_vec);

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(dot_views.data(), 1),
      iree_make_mutable_string_view(buffer, 0), &strings_consumed,
      &bytes_written));
  EXPECT_EQ(strings_consumed, 0u);
  EXPECT_EQ(bytes_written, 0u);

  // Resume with adequate buffer.
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(dot_views.data(), 1),
      iree_make_mutable_string_view(buffer, sizeof(buffer)), &strings_consumed,
      &bytes_written));
  EXPECT_EQ(strings_consumed, 1u);
  EXPECT_EQ(bytes_written, 1u);
  EXPECT_EQ(buffer[0], '.');
}

TEST_F(WordPieceDecoderTest, NoDoubleSpaceOnResume) {
  // Ensure no double space when buffer fills during space-prepend phase.
  ScopedDecoderState state(decoder());

  // Process first token.
  // Keep strings alive - views point into these vectors.
  std::vector<std::string> first_vec = {"Hi"};
  auto first_views = ToStringViews(first_vec);
  char buffer[32];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(first_views.data(), 1),
      iree_make_mutable_string_view(buffer, sizeof(buffer)), &strings_consumed,
      &bytes_written));
  EXPECT_EQ(strings_consumed, 1u);

  // Process second token with single-byte buffer (just fits space).
  std::vector<std::string> second_vec = {"there"};
  auto second_views = ToStringViews(second_vec);
  char tiny[4];

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(second_views.data(), 1),
      iree_make_mutable_string_view(tiny, 1), &strings_consumed,
      &bytes_written));

  // Should have written space but not consumed token.
  EXPECT_EQ(bytes_written, 1u);
  EXPECT_EQ(tiny[0], ' ');
  EXPECT_EQ(strings_consumed, 0u);

  // Resume - should NOT add another space.
  std::string rest;
  while (strings_consumed == 0) {
    IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
        state.get(), iree_tokenizer_make_string_list(second_views.data(), 1),
        iree_make_mutable_string_view(buffer, sizeof(buffer)),
        &strings_consumed, &bytes_written));
    rest.append(buffer, bytes_written);
  }

  // Should be "there" not " there" (no double space).
  EXPECT_EQ(rest, "there");
}

TEST_F(WordPieceDecoderTest, CleanupAtExactBufferBoundary) {
  // Buffer size exactly matches cleanup replacement length.
  // Pattern " ." (2 chars) -> "." (1 char) - buffer exactly 1 byte.
  ScopedDecoderState state(decoder());

  // First token to avoid is_first_token.
  std::vector<std::string> first_vec = {"Hello"};
  auto first_views = ToStringViews(first_vec);
  char buffer[32];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(first_views.data(), 1),
      iree_make_mutable_string_view(buffer, sizeof(buffer)), &strings_consumed,
      &bytes_written));
  EXPECT_EQ(strings_consumed, 1u);
  EXPECT_EQ(std::string(buffer, bytes_written), "Hello");

  // Process "." with buffer exactly 1 byte (exact fit for replacement ".").
  std::vector<std::string> dot_vec = {"."};
  auto dot_views = ToStringViews(dot_vec);
  char exact_fit[4];

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(dot_views.data(), 1),
      iree_make_mutable_string_view(exact_fit, 1), &strings_consumed,
      &bytes_written));

  // Should fit exactly: replacement "." is 1 byte, buffer is 1 byte.
  EXPECT_EQ(strings_consumed, 1u);
  EXPECT_EQ(bytes_written, 1u);
  EXPECT_EQ(exact_fit[0], '.');
}

TEST_F(WordPieceDecoderTest, CleanupContractionAtExactBoundary) {
  // Pattern " n't" (4 chars) -> "n't" (3 chars) - buffer exactly 3 bytes.
  ScopedDecoderState state(decoder());

  // First token.
  std::vector<std::string> first_vec = {"do"};
  auto first_views = ToStringViews(first_vec);
  char buffer[32];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(first_views.data(), 1),
      iree_make_mutable_string_view(buffer, sizeof(buffer)), &strings_consumed,
      &bytes_written));
  EXPECT_EQ(strings_consumed, 1u);

  // Process "n't" with buffer exactly 3 bytes (exact fit for replacement).
  std::vector<std::string> nt_vec = {"n't"};
  auto nt_views = ToStringViews(nt_vec);
  char exact_fit[4];

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(nt_views.data(), 1),
      iree_make_mutable_string_view(exact_fit, 3), &strings_consumed,
      &bytes_written));

  // Replacement "n't" is exactly 3 bytes.
  EXPECT_EQ(strings_consumed, 1u);
  EXPECT_EQ(bytes_written, 3u);
  EXPECT_EQ(std::string(exact_fit, 3), "n't");
}

TEST_F(WordPieceDecoderTest, SplitTokenWithCleanupActive) {
  // Token that matches cleanup pattern, but has remaining bytes that must be
  // split across multiple process() calls.
  // Pattern " 's" (3 chars) -> "'s" (2 chars), then continue with remaining.
  ScopedDecoderState state(decoder());

  // First token.
  std::vector<std::string> first_vec = {"John"};
  auto first_views = ToStringViews(first_vec);
  char buffer[32];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(first_views.data(), 1),
      iree_make_mutable_string_view(buffer, sizeof(buffer)), &strings_consumed,
      &bytes_written));
  EXPECT_EQ(strings_consumed, 1u);

  // Process "'s_extra" - matches 's cleanup but has extra bytes.
  // Buffer only fits the replacement "'s" (2 bytes), not the "_extra" (6).
  std::vector<std::string> possessive_vec = {"'s_extra"};
  auto possessive_views = ToStringViews(possessive_vec);
  char small_buffer[4];

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(possessive_views.data(), 1),
      iree_make_mutable_string_view(small_buffer, 2), &strings_consumed,
      &bytes_written));

  // Should have written "'s" but not consumed token (more bytes remain).
  EXPECT_EQ(bytes_written, 2u);
  EXPECT_EQ(std::string(small_buffer, 2), "'s");
  EXPECT_EQ(strings_consumed, 0u);
  EXPECT_TRUE(iree_tokenizer_decoder_state_has_pending(state.get()));

  // Resume with larger buffer - should get "_extra" without repeating "'s".
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(possessive_views.data(), 1),
      iree_make_mutable_string_view(buffer, sizeof(buffer)), &strings_consumed,
      &bytes_written));

  EXPECT_EQ(strings_consumed, 1u);
  EXPECT_EQ(bytes_written, 6u);  // "_extra" = 6 bytes
  EXPECT_EQ(std::string(buffer, bytes_written), "_extra");
  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()));
}

TEST_F(WordPieceDecoderTest, CleanupMatchThenByteByByteCopy) {
  // Cleanup pattern matches, then remaining token bytes are copied one at a
  // time (single-byte buffer after cleanup).
  ScopedDecoderState state(decoder());

  // First token.
  std::vector<std::string> first_vec = {"Hi"};
  auto first_views = ToStringViews(first_vec);
  char buffer[32];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(first_views.data(), 1),
      iree_make_mutable_string_view(buffer, sizeof(buffer)), &strings_consumed,
      &bytes_written));
  EXPECT_EQ(strings_consumed, 1u);

  // Process ",world" - comma triggers cleanup " ," -> ",", then "world"
  // remains.
  std::vector<std::string> comma_vec = {",world"};
  auto comma_views = ToStringViews(comma_vec);
  std::string result;

  // Byte by byte extraction - reset consumed counter before loop.
  strings_consumed = 0;
  while (strings_consumed == 0) {
    char byte[4];
    IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
        state.get(), iree_tokenizer_make_string_list(comma_views.data(), 1),
        iree_make_mutable_string_view(byte, 1), &strings_consumed,
        &bytes_written));

    if (bytes_written > 0) {
      result += byte[0];
    }
    ASSERT_TRUE(bytes_written > 0 || strings_consumed > 0) << "No progress";
  }

  // Should be ",world" (cleanup removed the virtual space).
  EXPECT_EQ(result, ",world");
}

//===----------------------------------------------------------------------===//
// HuggingFace Parity
//===----------------------------------------------------------------------===//

TEST_F(WordPieceDecoderTest, HF_FirstTokenWithPrefix) {
  // HuggingFace test case (cleanup=false).
  // First token with prefix stays unchanged.
  auto decoder_no_cleanup = CreateDecoder("##", false);
  TestWithAllBatchSizes(decoder_no_cleanup.get(),
                        {"##uelo", "Ara", "##új", "##o", "No", "##guera"},
                        "##uelo Araújo Noguera",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, HF_BertContraction) {
  // BERT tokenizes "don't" as ["don", "'", "t"].
  // Cleanup pattern " ' " needs trailing space, so this doesn't match.
  // Result has spaces around apostrophe.
  TestWithAllBatchSizes(decoder(), {"don", "'", "t"}, "don ' t",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, HF_SimpleSentence) {
  // Simple sentence with BERT-style tokenization.
  TestWithAllBatchSizes(decoder(), {"Hello", ",", "world", "!"},
                        "Hello, world!",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(WordPieceDecoderTest, EmptyPrefix) {
  // Decoder with empty prefix - all non-first tokens get space.
  auto decoder_empty_prefix = CreateDecoder("", true);
  TestWithAllBatchSizes(decoder_empty_prefix.get(), {"Hello", "world"},
                        "Hello world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, TokenIsJustPrefix) {
  // Token is exactly "##" - gets stripped, leaving empty string.
  TestWithAllBatchSizes(decoder(), {"Hello", "##"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, LongToken) {
  // Long token should work correctly.
  std::string long_token(1000, 'x');
  TestWithAllBatchSizes(decoder(), {long_token}, long_token,
                        /*expect_pending_after_process=*/false);
}

TEST_F(WordPieceDecoderTest, UnicodeTokens) {
  // Unicode should pass through correctly.
  TestWithAllBatchSizes(decoder(), {"日本", "##語"}, "日本語",
                        /*expect_pending_after_process=*/false);
}

}  // namespace
}  // namespace iree::tokenizer
