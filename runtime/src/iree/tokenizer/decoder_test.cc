// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Callback context for collecting decoded strings.
struct DecodeContext {
  std::string* output;
};

// Callback that appends decoded strings to the output.
static iree_status_t DecodeCallback(void* user_data,
                                    iree_string_view_list_t strings) {
  DecodeContext* ctx = static_cast<DecodeContext*>(user_data);
  for (iree_host_size_t i = 0; i < strings.count; ++i) {
    ctx->output->append(strings.values[i].data, strings.values[i].size);
  }
  return iree_ok_status();
}

class DecoderTest : public ::testing::Test {
 protected:
  std::string Decode(const iree_tokenizer_decoder_t* decoder,
                     const std::vector<std::string>& tokens) {
    std::vector<iree_string_view_t> views;
    views.reserve(tokens.size());
    for (const auto& token : tokens) {
      views.push_back(
          iree_make_string_view(token.data(), (iree_host_size_t)token.size()));
    }

    std::string output;
    DecodeContext ctx = {&output};

    iree_string_view_list_t token_list = {
        (iree_host_size_t)views.size(),
        views.data(),
    };

    iree_tokenizer_decoder_state_t state;
    iree_tokenizer_decoder_begin(decoder, &state);

    iree_status_t status = iree_tokenizer_decoder_decode(
        decoder, &state, token_list, DecodeCallback, &ctx);
    IREE_EXPECT_OK(status);
    return output;
  }
};

//===----------------------------------------------------------------------===//
// None Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, NoneConcatenates) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_none(&dec);

  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "HelloWorld");
  EXPECT_EQ(Decode(&dec, {"a", "b", "c"}), "abc");
  EXPECT_EQ(Decode(&dec, {}), "");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// WordPiece Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, WordPieceBasic) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_wordpiece(
      iree_string_view_empty(), IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_DEFAULT,
      &dec);

  // First token as-is, continuation tokens (##) joined without space.
  EXPECT_EQ(Decode(&dec, {"Hello", "##World"}), "HelloWorld");
  EXPECT_EQ(Decode(&dec, {"un", "##believ", "##able"}), "unbelievable");

  // Non-continuation tokens get spaces.
  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "Hello World");
  EXPECT_EQ(Decode(&dec, {"The", "quick", "brown", "fox"}),
            "The quick brown fox");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, WordPieceCustomPrefix) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_wordpiece(
      IREE_SVL("@@"), IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_DEFAULT, &dec);

  EXPECT_EQ(Decode(&dec, {"Hello", "@@World"}), "HelloWorld");
  EXPECT_EQ(Decode(&dec, {"Hello", "##World"}), "Hello ##World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

// Regression test: empty prefix should NOT treat all tokens as continuations.
// Bug: memcmp with size 0 always returns 0, making all tokens "continuations"
// and concatenating without spaces. Fix: guard with prefix.size > 0.
TEST_F(DecoderTest, WordPieceEmptyPrefixAddsSpaces) {
  iree_tokenizer_decoder_t dec;
  // Empty prefix string - no continuation detection should occur.
  iree_tokenizer_decoder_initialize_wordpiece(
      iree_string_view_empty(), IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_DEFAULT,
      &dec);

  // With empty prefix, NO token is a continuation, so spaces added between all.
  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "Hello World");
  EXPECT_EQ(Decode(&dec, {"The", "quick", "brown", "fox"}),
            "The quick brown fox");

  // First token never gets a prefix space.
  EXPECT_EQ(Decode(&dec, {"Single"}), "Single");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, WordPieceCleanupSpaces) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_wordpiece(
      iree_string_view_empty(),
      IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_CLEANUP_SPACES, &dec);

  // With cleanup_tokenization_spaces, no space before punctuation.
  // These test cases match HuggingFace tokenizers:
  // https://github.com/huggingface/tokenizers/blob/main/bindings/python/tests/bindings/test_decoders.py
  EXPECT_EQ(Decode(&dec, {"Hello", ",", "World", "!"}), "Hello, World!");

  // Contractions: tokens like 'm, 's come as single tokens, not split.
  EXPECT_EQ(Decode(&dec, {"I", "'m", "Jo", "##hn"}), "I'm John");
  EXPECT_EQ(Decode(&dec, {"She", "'s", "here", "."}), "She's here.");
  EXPECT_EQ(Decode(&dec, {"We", "'ve", "been", "there", "."}),
            "We've been there.");
  EXPECT_EQ(Decode(&dec, {"They", "'re", "coming", "!"}), "They're coming!");
  EXPECT_EQ(Decode(&dec, {"I", "do", "n't", "know", "."}), "I don't know.");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Metaspace Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, MetaspaceBasic) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING,
      &dec);  // 0 = default ▁

  // ▁ (U+2581) becomes space.
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), "Hello World");
  EXPECT_EQ(Decode(&dec, {"▁The", "▁quick", "▁brown", "▁fox"}),
            "The quick brown fox");

  // Leading metaspace on first token is stripped.
  EXPECT_EQ(Decode(&dec, {"▁Hello"}), "Hello");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, MetaspaceMidWord) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &dec);

  // Tokens without ▁ are continuation pieces.
  EXPECT_EQ(Decode(&dec, {"▁un", "believ", "able"}), "unbelievable");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, MetaspaceEmptyInput) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &dec);

  // Empty token list.
  EXPECT_EQ(Decode(&dec, {}), "");

  // Empty string token (should produce nothing).
  EXPECT_EQ(Decode(&dec, {""}), "");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, MetaspaceFirstTokenNoSeparator) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &dec);

  // First token without ▁ - no space added.
  EXPECT_EQ(Decode(&dec, {"Hello", "▁World"}), "Hello World");

  // First token starts with ▁ - stripped.
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), "Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, MetaspaceNoStripLeading) {
  // Test add_prefix_space: false behavior - don't strip leading metaspace.
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_DEFAULT, &dec);

  // First token's leading ▁ should become a space (not stripped).
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), " Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, MetaspaceNoStripLeadingMidWord) {
  // Test add_prefix_space: false with subword tokens.
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_DEFAULT, &dec);

  // First token's leading ▁ becomes space.
  EXPECT_EQ(Decode(&dec, {"▁un", "believ", "able"}), " unbelievable");

  // No leading ▁ on first token - no space added.
  EXPECT_EQ(Decode(&dec, {"Hello", "▁World"}), "Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Metaspace Streaming Tests (multi-batch)
//===----------------------------------------------------------------------===//

class MetaspaceStreamingTest : public ::testing::Test {
 protected:
  std::string DecodeStreaming(
      const iree_tokenizer_decoder_t* decoder,
      const std::vector<std::vector<std::string>>& batches) {
    std::string output;
    DecodeContext ctx = {&output};

    iree_tokenizer_decoder_state_t state;
    iree_tokenizer_decoder_begin(decoder, &state);

    for (const auto& batch : batches) {
      std::vector<iree_string_view_t> views;
      views.reserve(batch.size());
      for (const auto& token : batch) {
        views.push_back(iree_make_string_view(token.data(),
                                              (iree_host_size_t)token.size()));
      }

      iree_string_view_list_t token_list = {
          (iree_host_size_t)views.size(),
          views.data(),
      };

      iree_status_t status = iree_tokenizer_decoder_decode(
          decoder, &state, token_list, DecodeCallback, &ctx);
      IREE_EXPECT_OK(status);
    }
    return output;
  }
};

TEST_F(MetaspaceStreamingTest, MultipleBatchesMaintainState) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &dec);

  // First batch has first token (▁ stripped), second batch continues.
  // State should track that we're past the first token.
  std::string result =
      DecodeStreaming(&dec, {{"▁Hello"}, {"▁World"}, {"▁again"}});
  EXPECT_EQ(result, "Hello World again");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(MetaspaceStreamingTest, EmptyBatchInMiddle) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &dec);

  // Empty batch in the middle should not affect state.
  std::string result = DecodeStreaming(&dec, {{"▁Hello"}, {}, {"▁World"}});
  EXPECT_EQ(result, "Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(MetaspaceStreamingTest, SingleTokenPerBatch) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &dec);

  // One token per batch, streaming mode.
  std::string result =
      DecodeStreaming(&dec, {{"▁The"}, {"▁quick"}, {"▁brown"}, {"▁fox"}});
  EXPECT_EQ(result, "The quick brown fox");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(MetaspaceStreamingTest, ContinuationTokensAcrossBatches) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &dec);

  // Subword split across batches: "un" + "believ" + "able".
  std::string result = DecodeStreaming(&dec, {{"▁un"}, {"believ"}, {"able"}});
  EXPECT_EQ(result, "unbelievable");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// ByteLevel Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, ByteLevelBasic) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_DEFAULT, &dec);

  // Printable ASCII passes through.
  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "HelloWorld");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteLevelSpace) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_DEFAULT, &dec);

  // GPT-2 encodes space (byte 32) as Ġ (U+0120 = 288 = 256 + 32).
  // The character Ġ is UTF-8 encoded as C4 A0.
  EXPECT_EQ(Decode(&dec, {"Hello", "\xC4\xA0", "World"}), "Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteLevelNewline) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_DEFAULT, &dec);

  // GPT-2 encodes newline (byte 10) as Ċ (U+010A = 266 = 256 + 10).
  // The character Ċ is UTF-8 encoded as C4 8A.
  EXPECT_EQ(Decode(&dec, {"Line1", "\xC4\x8A", "Line2"}), "Line1\nLine2");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteLevelMultiplePrefixes) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_DEFAULT, &dec);

  // Multiple Ġ (space markers) in a row.
  // Ġ is UTF-8 C4 A0, each becomes a space.
  EXPECT_EQ(Decode(&dec, {"\xC4\xA0\xC4\xA0\xC4\xA0"}), "   ");

  // Token with multiple spaces mixed with text.
  EXPECT_EQ(Decode(&dec, {"Hello\xC4\xA0\xC4\xA0World"}), "Hello  World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteLevelMixedControlChars) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_DEFAULT, &dec);

  // Mix of normal text, space (Ġ), and newline (Ċ).
  // "Hello\nWorld test" with GPT-2 encoding.
  EXPECT_EQ(Decode(&dec, {"Hello", "\xC4\x8A", "World", "\xC4\xA0", "test"}),
            "Hello\nWorld test");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteLevelEmptyInput) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_DEFAULT, &dec);

  // Empty token list.
  EXPECT_EQ(Decode(&dec, {}), "");

  // Empty string token.
  EXPECT_EQ(Decode(&dec, {""}), "");

  // Multiple empty tokens.
  EXPECT_EQ(Decode(&dec, {"", "", ""}), "");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteLevelAddPrefixSpaceFlag) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_ADD_PREFIX_SPACE, &dec);

  // When ADD_PREFIX_SPACE is set, a leading space (Ġ) on the first token
  // should be stripped (it was added during encoding to mark word start).
  // Ġ = C4 A0 in UTF-8.
  EXPECT_EQ(Decode(&dec, {"\xC4\xA0Hello"}), "Hello");

  // Second token's space is preserved.
  EXPECT_EQ(Decode(&dec, {"\xC4\xA0Hello", "\xC4\xA0World"}), "Hello World");

  // Without leading Ġ, nothing is stripped.
  EXPECT_EQ(Decode(&dec, {"Hello", "\xC4\xA0World"}), "Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// BPE Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, BPEConcatenates) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_bpe(&dec);

  // BPE decoder is same as None - just concatenates.
  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "HelloWorld");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Buffer Capacity Tests
//===----------------------------------------------------------------------===//

// Test that single tokens exceeding buffer capacity are rejected.
// This tests the fix for the buffer overflow bug where oversized tokens
// could write past the internal data_buffer.
class DecoderBufferCapacityTest : public ::testing::Test {
 protected:
  // Decodes and expects a specific status.
  iree_status_t DecodeWithStatus(const iree_tokenizer_decoder_t* decoder,
                                 const std::vector<std::string>& tokens) {
    std::vector<iree_string_view_t> views;
    views.reserve(tokens.size());
    for (const auto& token : tokens) {
      views.push_back(
          iree_make_string_view(token.data(), (iree_host_size_t)token.size()));
    }

    iree_string_view_list_t token_list = {
        (iree_host_size_t)views.size(),
        views.data(),
    };

    iree_tokenizer_decoder_state_t state;
    iree_tokenizer_decoder_begin(decoder, &state);

    return iree_tokenizer_decoder_decode(
        decoder, &state, token_list,
        [](void*, iree_string_view_list_t) { return iree_ok_status(); },
        nullptr);
  }
};

TEST_F(DecoderBufferCapacityTest, WordPieceRejectsOversizedToken) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_wordpiece(
      iree_string_view_empty(), IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_DEFAULT,
      &dec);

  // Create a token larger than 4KB (IREE_TOKENIZER_DATA_BATCH_CAPACITY).
  // The decoder needs space for: prefix_len + token + space = 0 + 5000 + 1 =
  // 5001 bytes. This exceeds the 4096 byte buffer.
  std::string huge_token(5000, 'x');

  iree_status_t status = DecodeWithStatus(&dec, {huge_token});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderBufferCapacityTest, MetaspaceRejectsOversizedToken) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &dec);

  // Create a token larger than 4KB.
  std::string huge_token(5000, 'x');

  iree_status_t status = DecodeWithStatus(&dec, {huge_token});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderBufferCapacityTest, ByteLevelRejectsOversizedToken) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_DEFAULT, &dec);

  // Create a token larger than 4KB.
  std::string huge_token(5000, 'x');

  iree_status_t status = DecodeWithStatus(&dec, {huge_token});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderBufferCapacityTest, NormalSizedTokensWork) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_wordpiece(
      iree_string_view_empty(), IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_DEFAULT,
      &dec);

  // Create tokens that fit within the buffer.
  std::string normal_token(1000, 'x');

  iree_status_t status = DecodeWithStatus(&dec, {normal_token, normal_token});
  IREE_EXPECT_OK(status);

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Replace Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, ReplaceBasic) {
  iree_tokenizer_decoder_t dec;
  // Replace ▁ (U+2581) with space.
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_replace(
      IREE_SVL("\xE2\x96\x81"),  // ▁ in UTF-8
      IREE_SVL(" "), &dec));

  // ▁Hello → " Hello", ▁World → " World", concatenated = " Hello World".
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), " Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ReplaceMultipleOccurrences) {
  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_replace(
      IREE_SVL("x"), IREE_SVL("_"), &dec));

  // Multiple x's in single token should all be replaced.
  EXPECT_EQ(Decode(&dec, {"axxbxxc"}), "a__b__c");
  EXPECT_EQ(Decode(&dec, {"xxx"}), "___");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ReplaceEmptyPattern) {
  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_replace(
      iree_string_view_empty(), IREE_SVL("_"), &dec));

  // Empty pattern - no replacement occurs.
  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "HelloWorld");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ReplaceNoLeadingStrip) {
  iree_tokenizer_decoder_t dec;
  // Replace ▁ with space (like Metaspace but without special first-token
  // handling).
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_replace(
      IREE_SVL("\xE2\x96\x81"), IREE_SVL(" "), &dec));

  // Unlike Metaspace, Replace does NOT strip leading replacement on first
  // token.
  EXPECT_EQ(Decode(&dec, {"▁Hello"}), " Hello");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ReplaceContentLargerThanPattern) {
  iree_tokenizer_decoder_t dec;
  // Replace single char with longer string.
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_replace(
      IREE_SVL("x"), IREE_SVL("XYZ"), &dec));

  EXPECT_EQ(Decode(&dec, {"axb"}), "aXYZb");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ReplacePatternLargerThanContent) {
  iree_tokenizer_decoder_t dec;
  // Replace longer pattern with single char.
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_replace(
      IREE_SVL("abc"), IREE_SVL("_"), &dec));

  EXPECT_EQ(Decode(&dec, {"xabcy"}), "x_y");
  EXPECT_EQ(Decode(&dec, {"abcabc"}), "__");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// ByteFallback Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, ByteFallbackSimpleASCII) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // <0xHH> tokens are converted to bytes.
  EXPECT_EQ(Decode(&dec, {"<0x48>", "<0x69>"}), "Hi");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackPassthrough) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // Non-byte tokens pass through unchanged.
  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "HelloWorld");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackMixed) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // Mix of byte tokens and regular tokens.
  EXPECT_EQ(Decode(&dec, {"Hello", "<0x20>", "World"}), "Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackMultiByteUTF8) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // Multi-byte UTF-8 sequence for 叫 (U+53EB): E5 8F AB
  EXPECT_EQ(Decode(&dec, {"<0xE5>", "<0x8F>", "<0xAB>"}), "叫");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackInvalidUTF8) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // Invalid UTF-8 byte 0xFF - should produce replacement character U+FFFD.
  std::string result = Decode(&dec, {"<0xFF>"});
  EXPECT_EQ(result, "\xEF\xBF\xBD");  // U+FFFD in UTF-8

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackIncompleteUTF8) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // Incomplete UTF-8 sequence: E5 followed by non-continuation, then
  // passthrough. E5 is start of 3-byte sequence but "X" interrupts it.
  std::string result = Decode(&dec, {"<0xE5>", "X"});
  // E5 alone is invalid, should be replaced with U+FFFD, then "X" passes
  // through.
  EXPECT_EQ(result, "\xEF\xBF\xBDX");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackLowercase) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // Lowercase hex should also work.
  EXPECT_EQ(Decode(&dec, {"<0x48>", "<0x69>"}), "Hi");
  EXPECT_EQ(Decode(&dec, {"<0x4a>", "<0x6f>"}), "Jo");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackNotByteTokens) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // These look similar but are not valid byte tokens.
  EXPECT_EQ(Decode(&dec, {"<0xGG>"}), "<0xGG>");    // Invalid hex.
  EXPECT_EQ(Decode(&dec, {"<0x4>"}), "<0x4>");      // Too short.
  EXPECT_EQ(Decode(&dec, {"0x48"}), "0x48");        // Missing < >.
  EXPECT_EQ(Decode(&dec, {"<0x123>"}), "<0x123>");  // Too long.

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackBufferBoundaryUTF8) {
  // This test verifies that UTF-8 sequences spanning the 256-byte internal
  // buffer boundary are handled correctly. Without the fix, a multi-byte UTF-8
  // sequence could be split across two buffer flushes, causing corruption.

  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // Generate tokens that will fill the buffer to exactly 255 bytes, then add
  // a 2-byte UTF-8 sequence. If the fix is working, the UTF-8 sequence will
  // be preserved intact across the buffer boundary.

  // Create 255 ASCII byte tokens (each becomes 1 byte).
  std::vector<std::string> tokens;
  std::string expected;
  for (int i = 0; i < 255; ++i) {
    // Use byte 0x61 = 'a' for all filler bytes.
    tokens.push_back("<0x61>");
    expected += 'a';
  }

  // Now add a 2-byte UTF-8 sequence "é" (U+00E9 = 0xC3 0xA9).
  // The first byte (0xC3) will be byte 256, triggering a buffer flush.
  // The fix ensures 0xC3 is kept for the next buffer since it's incomplete.
  tokens.push_back("<0xC3>");
  tokens.push_back("<0xA9>");
  expected += "\xC3\xA9";  // é in UTF-8

  EXPECT_EQ(Decode(&dec, tokens), expected);

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackBufferBoundary3ByteUTF8) {
  // Test a 3-byte UTF-8 sequence at the buffer boundary.

  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // Fill to 254 bytes, then add a 3-byte sequence "中" (U+4E2D = 0xE4 0xB8
  // 0xAD). When we hit byte 256 (0xB8), the flush should preserve [0xE4]
  // incomplete. When we hit byte 257 (0xAD), the flush should preserve [0xE4,
  // 0xB8].
  std::vector<std::string> tokens;
  std::string expected;
  for (int i = 0; i < 254; ++i) {
    tokens.push_back("<0x62>");  // 'b'
    expected += 'b';
  }

  tokens.push_back("<0xE4>");
  tokens.push_back("<0xB8>");
  tokens.push_back("<0xAD>");
  expected += "\xE4\xB8\xAD";  // 中 in UTF-8

  EXPECT_EQ(Decode(&dec, tokens), expected);

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, ByteFallbackBufferBoundary4ByteUTF8) {
  // Test a 4-byte UTF-8 sequence at the buffer boundary.

  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_byte_fallback(&dec);

  // Fill to 253 bytes, then add a 4-byte sequence U+1F600 (grinning face
  // emoji). 0xF0 0x9F 0x98 0x80
  std::vector<std::string> tokens;
  std::string expected;
  for (int i = 0; i < 253; ++i) {
    tokens.push_back("<0x63>");  // 'c'
    expected += 'c';
  }

  tokens.push_back("<0xF0>");
  tokens.push_back("<0x9F>");
  tokens.push_back("<0x98>");
  tokens.push_back("<0x80>");
  expected += "\xF0\x9F\x98\x80";  // 😀 in UTF-8

  EXPECT_EQ(Decode(&dec, tokens), expected);

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Fuse Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, FuseBasic) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_fuse(&dec);

  // Fuse concatenates all tokens into a single output string.
  EXPECT_EQ(Decode(&dec, {"Hello", " ", "World"}), "Hello World");
  EXPECT_EQ(Decode(&dec, {"a", "b", "c"}), "abc");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, FuseEmpty) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_fuse(&dec);

  EXPECT_EQ(Decode(&dec, {}), "");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, FuseSingleToken) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_fuse(&dec);

  EXPECT_EQ(Decode(&dec, {"SingleToken"}), "SingleToken");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Strip Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, StripLeadingOnly) {
  iree_tokenizer_decoder_t dec;
  // Strip 1 space from start, 0 from end.
  IREE_EXPECT_OK(
      iree_tokenizer_decoder_initialize_strip(IREE_SVL(" "), 1, 0, &dec));

  EXPECT_EQ(Decode(&dec, {" Hello"}), "Hello");
  EXPECT_EQ(Decode(&dec, {"  Hello"}), " Hello");  // Only strips 1.
  EXPECT_EQ(Decode(&dec, {"Hello"}), "Hello");     // Nothing to strip.

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, StripTrailingOnly) {
  iree_tokenizer_decoder_t dec;
  // Strip 0 from start, 1 from end.
  IREE_EXPECT_OK(
      iree_tokenizer_decoder_initialize_strip(IREE_SVL(" "), 0, 1, &dec));

  EXPECT_EQ(Decode(&dec, {"Hello "}), "Hello");
  EXPECT_EQ(Decode(&dec, {"Hello  "}), "Hello ");  // Only strips 1.
  EXPECT_EQ(Decode(&dec, {"Hello"}), "Hello");     // Nothing to strip.

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, StripBothEnds) {
  iree_tokenizer_decoder_t dec;
  // Strip 2 from start, 2 from end.
  IREE_EXPECT_OK(
      iree_tokenizer_decoder_initialize_strip(IREE_SVL(" "), 2, 2, &dec));

  EXPECT_EQ(Decode(&dec, {"  Hello  "}), "Hello");
  EXPECT_EQ(Decode(&dec, {"   Hello   "}), " Hello ");  // Only strips 2 each.

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, StripMultipleTokens) {
  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(
      iree_tokenizer_decoder_initialize_strip(IREE_SVL(" "), 1, 0, &dec));

  // Each token gets stripped independently.
  EXPECT_EQ(Decode(&dec, {" Hello", " World"}), "HelloWorld");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, StripNonSpaceCharacter) {
  iree_tokenizer_decoder_t dec;
  // Strip underscores.
  IREE_EXPECT_OK(
      iree_tokenizer_decoder_initialize_strip(IREE_SVL("_"), 1, 1, &dec));

  EXPECT_EQ(Decode(&dec, {"_Hello_"}), "Hello");
  EXPECT_EQ(Decode(&dec, {"__Hello__"}), "_Hello_");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Sequence Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderTest, SequenceSingleDecoder) {
  // Single decoder in sequence.
  iree_tokenizer_decoder_t* children = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       sizeof(iree_tokenizer_decoder_t),
                                       (void**)&children));
  iree_tokenizer_decoder_initialize_fuse(&children[0]);

  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_sequence(
      children, 1, iree_allocator_system(), &dec));

  EXPECT_EQ(Decode(&dec, {"a", "b", "c"}), "abc");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, SequenceEmpty) {
  // Empty sequence - acts as passthrough.
  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_sequence(
      nullptr, 0, iree_allocator_system(), &dec));

  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "HelloWorld");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, SequenceTinyLlamaChain) {
  // Full TinyLlama decoder chain: Replace → ByteFallback → Fuse → Strip.
  iree_tokenizer_decoder_t* children = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       4 * sizeof(iree_tokenizer_decoder_t),
                                       (void**)&children));

  // 1. Replace ▁ with space.
  IREE_EXPECT_OK(
      iree_tokenizer_decoder_initialize_replace(IREE_SVL("\xE2\x96\x81"),  // ▁
                                                IREE_SVL(" "), &children[0]));

  // 2. ByteFallback for <0xHH> tokens.
  iree_tokenizer_decoder_initialize_byte_fallback(&children[1]);

  // 3. Fuse all tokens into one.
  iree_tokenizer_decoder_initialize_fuse(&children[2]);

  // 4. Strip 1 leading space.
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_strip(IREE_SVL(" "), 1, 0,
                                                         &children[3]));

  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_sequence(
      children, 4, iree_allocator_system(), &dec));

  // Test: ▁Hello▁World should become "Hello World" (leading space stripped).
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), "Hello World");

  // Test with byte tokens.
  EXPECT_EQ(Decode(&dec, {"▁Hi", "<0x21>"}), "Hi!");  // 0x21 = '!'

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, SequenceStatePropagatesToFirstToken) {
  // Test that is_first_token state propagates correctly through sequence.
  iree_tokenizer_decoder_t* children = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       2 * sizeof(iree_tokenizer_decoder_t),
                                       (void**)&children));

  // Metaspace strips leading ▁ on first token only.
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &children[0]);
  iree_tokenizer_decoder_initialize_fuse(&children[1]);

  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_sequence(
      children, 2, iree_allocator_system(), &dec));

  // First token should have leading ▁ stripped (is_first_token = true).
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), "Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, SequenceReplaceAndStrip) {
  // Replace + Strip combination.
  iree_tokenizer_decoder_t* children = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       2 * sizeof(iree_tokenizer_decoder_t),
                                       (void**)&children));

  // Replace ▁ with space.
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_replace(
      IREE_SVL("\xE2\x96\x81"), IREE_SVL(" "), &children[0]));

  // Strip 1 leading space.
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_strip(IREE_SVL(" "), 1, 0,
                                                         &children[1]));

  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_sequence(
      children, 2, iree_allocator_system(), &dec));

  // ▁ at start of each token becomes space, then 1 leading space is stripped.
  // Token "▁Hello" → " Hello" → "Hello" (strip 1)
  // Token "▁World" → " World" → "World" (strip 1)
  // Result: "HelloWorld"
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), "HelloWorld");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderBufferCapacityTest, SequenceDecoderErrorPropagation) {
  // Test that errors from a decoder in the middle of a sequence propagate.
  iree_tokenizer_decoder_t* children = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       3 * sizeof(iree_tokenizer_decoder_t),
                                       (void**)&children));

  // First: Fuse (should succeed).
  iree_tokenizer_decoder_initialize_fuse(&children[0]);

  // Second: Metaspace (will fail on oversized input).
  iree_tokenizer_decoder_initialize_metaspace(
      0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING, &children[1]);

  // Third: Strip (never reached).
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_strip(IREE_SVL(" "), 1, 0,
                                                         &children[2]));

  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_sequence(
      children, 3, iree_allocator_system(), &dec));

  // Oversized token should fail in the Metaspace decoder.
  std::string huge_token(5000, 'x');
  iree_status_t status = DecodeWithStatus(&dec, {huge_token});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderTest, SequenceDecoderDeepChain) {
  // Test a chain of 5+ decoders.
  iree_tokenizer_decoder_t* children = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       5 * sizeof(iree_tokenizer_decoder_t),
                                       (void**)&children));

  // Chain: Replace(▁→space) → ByteFallback → Fuse → Strip(1 leading) →
  // Strip(1 trailing).
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_replace(
      IREE_SVL("\xE2\x96\x81"), IREE_SVL(" "), &children[0]));
  iree_tokenizer_decoder_initialize_byte_fallback(&children[1]);
  iree_tokenizer_decoder_initialize_fuse(&children[2]);
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_strip(IREE_SVL(" "), 1, 0,
                                                         &children[3]));
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_strip(IREE_SVL(" "), 0, 1,
                                                         &children[4]));

  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_sequence(
      children, 5, iree_allocator_system(), &dec));

  // Input: "▁Hello▁" → " Hello " → "Hello " (strip leading) → "Hello" (strip
  // trailing).
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁"}), "Hello");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Callback Error Propagation Tests
//===----------------------------------------------------------------------===//

static iree_status_t FailingCallback(void* user_data,
                                     iree_string_view_list_t strings) {
  (void)user_data;
  (void)strings;
  return iree_make_status(IREE_STATUS_ABORTED, "callback error");
}

TEST(DecoderCallbackTest, CallbackErrorPropagates) {
  iree_tokenizer_decoder_t dec;
  iree_tokenizer_decoder_initialize_none(&dec);

  iree_string_view_t tokens[] = {IREE_SV("Hello")};
  iree_string_view_list_t token_list = {1, tokens};

  iree_tokenizer_decoder_state_t state;
  iree_tokenizer_decoder_begin(&dec, &state);

  iree_status_t status = iree_tokenizer_decoder_decode(
      &dec, &state, token_list, FailingCallback, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
  iree_status_free(status);

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST(DecoderCallbackTest, SequenceCallbackErrorPropagates) {
  // Error from callback should propagate through Sequence decoder.
  iree_tokenizer_decoder_t* children = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       sizeof(iree_tokenizer_decoder_t),
                                       (void**)&children));
  iree_tokenizer_decoder_initialize_fuse(&children[0]);

  iree_tokenizer_decoder_t dec;
  IREE_EXPECT_OK(iree_tokenizer_decoder_initialize_sequence(
      children, 1, iree_allocator_system(), &dec));

  iree_string_view_t tokens[] = {IREE_SV("Hello")};
  iree_string_view_list_t token_list = {1, tokens};

  iree_tokenizer_decoder_state_t state;
  iree_tokenizer_decoder_begin(&dec, &state);

  iree_status_t status = iree_tokenizer_decoder_decode(
      &dec, &state, token_list, FailingCallback, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
  iree_status_free(status);

  iree_tokenizer_decoder_deinitialize(&dec);
}

}  // namespace
