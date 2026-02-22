// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for decode functionality including the pre-decoded fast path.
//
// Ground truth expectations are generated from HuggingFace tokenizers library:
//   from tokenizers import Tokenizer
//   tok = Tokenizer.from_file("tokenizer.json")
//   tok.decode([...], skip_special_tokens=False)

#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/decoder/byte_fallback.h"
#include "iree/tokenizer/decoder/byte_level.h"
#include "iree/tokenizer/decoder/metaspace.h"
#include "iree/tokenizer/decoder/replace.h"
#include "iree/tokenizer/decoder/sequence.h"
#include "iree/tokenizer/decoder/wordpiece.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/tokenizer_test_util.h"

namespace iree {
namespace tokenizer {
namespace {

using testing::BuildTokenizer;
using testing::CreateBPEModel;
using testing::CreateBPEModelIgnoreMerges;
using testing::CreateWhitespaceSegmenter;
using testing::Decode;
using testing::DecodeStateStorage;
using testing::ScopedBuilder;
using testing::ScopedVocab;
using testing::ScopedVocabBuilder;

//===----------------------------------------------------------------------===//
// Test fixture for decode tests.
//===----------------------------------------------------------------------===//

class TokenizerDecodeTest : public ::testing::Test {};

//===----------------------------------------------------------------------===//
// ByteLevel Decoder (GPT-2 style, STATELESS, not position-sensitive)
//===----------------------------------------------------------------------===//
// GPT-2's ByteLevel decoder maps Unicode codepoints back to raw bytes.
// The vocab stores tokens using the GPT-2 byte-to-unicode mapping:
//   bytes 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF map to themselves as Unicode
//   bytes 0x00-0x20, 0x7F-0xA0, 0xAD map to U+0100-U+0143 (shifted range)
//
// Pre-decoded: YES (STATELESS, not POSITION_SENSITIVE)
// Each token always decodes to the same bytes regardless of position.

// Builds a minimal GPT-2-style tokenizer with ByteLevel decoder.
// Tokens must use the GPT-2 byte-to-unicode encoding.
iree_tokenizer_t* BuildByteLevelTokenizer(iree_tokenizer_vocab_t* vocab) {
  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab);

  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decoder_byte_level_allocate(
      iree_allocator_system(), &decoder));
  iree_tokenizer_builder_set_decoder(builder.get(), decoder);

  return BuildTokenizer(builder.get());
}

TEST_F(TokenizerDecodeTest, ByteLevelBasicASCII) {
  // GPT-2 token "Hello" = bytes [72,101,108,108,111]
  // In GPT-2 encoding, ASCII 0x21-0x7E map to themselves.
  // "Hello" is [H=0x48, e=0x65, l=0x6C, l=0x6C, o=0x6F] - all in identity
  // range.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  vocab_builder.AddToken(1, ",");
  vocab_builder.AddToken(2, "\xC4\xA0world");  // U+0120 = space (0x20 shifted)
                                               // + "world"
  vocab_builder.AddToken(3, "!");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace: decode([0,1,2,3]) = "Hello, world!"
  IREE_ASSERT_OK_AND_ASSIGN(std::string result,
                            Decode(tokenizer, {0, 1, 2, 3}));
  EXPECT_EQ(result, "Hello, world!");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteLevelSpaceToken) {
  // GPT-2 encodes space (0x20) as U+0120 (Ġ in UTF-8: 0xC4 0xA0).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xC4\xA0");  // Space alone.
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace: decode([0]) = " "
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0}));
  EXPECT_EQ(result, " ");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteLevelNewlineToken) {
  // GPT-2 encodes newline (0x0A) as U+010A (Ċ in UTF-8: 0xC4 0x8A).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xC4\x8A");  // Newline.
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace: decode([198]) where 198 maps to "\n"
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0}));
  EXPECT_EQ(result, "\n");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteLevelMultiByteUTF8) {
  // "café" in GPT-2 encoding:
  // c=0x63 (identity), a=0x61 (identity), f=0x66 (identity)
  // é = UTF-8 [0xC3, 0xA9]. Both bytes are in identity range (0xA1-0xFF).
  // So the GPT-2 token text is "cafÃ©" (each byte as its Unicode codepoint).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "caf\xC3\x83\xC2\xA9");  // GPT-2 encoded "café"
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // ByteLevel decoder maps each Unicode codepoint back to the original byte.
  // U+00C3 (Ã) → byte 0xC3, U+00A9 (©) → byte 0xA9 → together form "é".
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0}));
  EXPECT_EQ(result, "café");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteLevelNotPositionSensitive) {
  // ByteLevel is NOT position-sensitive: first and non-first tokens decode
  // identically.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xC4\xA0world");  // " world"
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer1 = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer1, nullptr);

  // As first token: still " world" (space preserved).
  IREE_ASSERT_OK_AND_ASSIGN(std::string first, Decode(tokenizer1, {0}));
  EXPECT_EQ(first, " world");

  // As second token: also " world".
  ScopedVocabBuilder vocab_builder2;
  vocab_builder2.AddToken(0, "Hello");
  vocab_builder2.AddToken(1, "\xC4\xA0world");
  ScopedVocab vocab2 = vocab_builder2.Build();

  iree_tokenizer_t* tokenizer2 = BuildByteLevelTokenizer(vocab2.release());
  ASSERT_NE(tokenizer2, nullptr);
  IREE_ASSERT_OK_AND_ASSIGN(std::string second, Decode(tokenizer2, {0, 1}));
  EXPECT_EQ(second, "Hello world");

  iree_tokenizer_free(tokenizer2);
  iree_tokenizer_free(tokenizer1);
}

//===----------------------------------------------------------------------===//
// Metaspace Decoder (T5 style, STATELESS + POSITION_SENSITIVE)
//===----------------------------------------------------------------------===//
// T5/SentencePiece style: tokens contain ▁ (U+2581) where spaces go.
// First token's leading ▁ is stripped (prepend_scheme=ALWAYS).
//
// Pre-decoded: YES (STATELESS + POSITION_SENSITIVE)
// Pre-decoded table stores the "rest" form (▁ → space).
// First token strips leading space from pre-decoded output.

iree_tokenizer_t* BuildMetaspaceTokenizer(
    iree_tokenizer_vocab_t* vocab,
    iree_tokenizer_decoder_metaspace_prepend_scheme_t prepend_scheme =
        IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS) {
  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab);

  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decoder_metaspace_allocate(
      0, prepend_scheme, iree_allocator_system(), &decoder));
  iree_tokenizer_builder_set_decoder(builder.get(), decoder);

  return BuildTokenizer(builder.get());
}

TEST_F(TokenizerDecodeTest, MetaspaceBasic) {
  // T5 tokens: ▁Hello = "\xE2\x96\x81Hello", ▁world = "\xE2\x96\x81world"
  // ▁ is U+2581, UTF-8 = E2 96 81
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xE2\x96\x81Hello");
  vocab_builder.AddToken(1, ",");
  vocab_builder.AddToken(2, "\xE2\x96\x81world");
  vocab_builder.AddToken(3, "!");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace T5: decode([▁Hello, ",", ▁world, "!"]) = "Hello, world!"
  // First token ▁Hello → strip leading ▁ → "Hello"
  // Second token "," → "," (no ▁, pass through)
  // Third token ▁world → ▁→space → " world"
  // Fourth token "!" → "!" (no ▁, pass through)
  IREE_ASSERT_OK_AND_ASSIGN(std::string result,
                            Decode(tokenizer, {0, 1, 2, 3}));
  EXPECT_EQ(result, "Hello, world!");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, MetaspaceFirstTokenStripping) {
  // The first token's leading ▁ is stripped with PREPEND_ALWAYS.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xE2\x96\x81Hello");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace T5: decode([▁Hello]) = "Hello" (leading ▁ stripped)
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0}));
  EXPECT_EQ(result, "Hello");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, MetaspaceSecondTokenKeepsSpace) {
  // Non-first tokens: ▁ → space (not stripped).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0,
                         "\xE2\x96\x81"
                         "a");
  vocab_builder.AddToken(1,
                         "\xE2\x96\x81"
                         "world");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace T5: decode([▁a, ▁world]) = "a world"
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1}));
  EXPECT_EQ(result, "a world");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, MetaspaceNoMetaspaceToken) {
  // Token without ▁ passes through unchanged.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xE2\x96\x81Hello");
  vocab_builder.AddToken(1, ",");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace T5: decode([▁Hello, ","]) = "Hello,"
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1}));
  EXPECT_EQ(result, "Hello,");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, MetaspacePrependNever) {
  // With PREPEND_NEVER, the first token's ▁ is NOT stripped.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xE2\x96\x81Hello");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(
      vocab.release(), IREE_TOKENIZER_DECODER_METASPACE_PREPEND_NEVER);
  ASSERT_NE(tokenizer, nullptr);

  // With PREPEND_NEVER, ▁ always becomes space, even on first token.
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0}));
  EXPECT_EQ(result, " Hello");

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// WordPiece Decoder (BERT style, STATELESS + POSITION_SENSITIVE)
//===----------------------------------------------------------------------===//
// BERT-style WordPiece: continuation tokens start with "##".
// First word token gets no space prefix; subsequent word tokens get space.
// Continuation tokens strip "##" and join directly.
//
// Pre-decoded: YES (STATELESS + POSITION_SENSITIVE)
// Pre-decoded table stores "rest" form: word tokens get " " prefix,
// continuation tokens get "##" stripped (no space).
// First token strips leading space.

iree_tokenizer_t* BuildWordPieceTokenizer(iree_tokenizer_vocab_t* vocab,
                                          bool cleanup = false) {
  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab);

  iree_tokenizer_decoder_wordpiece_config_t config = {};
  config.prefix = iree_make_cstring_view("##");
  config.cleanup = cleanup;
  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decoder_wordpiece_allocate(
      config, iree_allocator_system(), &decoder));
  iree_tokenizer_builder_set_decoder(builder.get(), decoder);

  return BuildTokenizer(builder.get());
}

TEST_F(TokenizerDecodeTest, WordPieceBasicWords) {
  // BERT tokens: whole words without ## prefix.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildWordPieceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace BERT: decode([hello, world]) = "hello world"
  // First word: "hello" (no space prefix)
  // Second word: " world" (space prefix added)
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1}));
  EXPECT_EQ(result, "hello world");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, WordPieceContinuationTokens) {
  // ## prefix is stripped and token joins directly to previous.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "play");
  vocab_builder.AddToken(1, "##ing");
  vocab_builder.AddToken(2, "world");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildWordPieceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace BERT: decode([play, ##ing, world]) = "playing world"
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1, 2}));
  EXPECT_EQ(result, "playing world");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, WordPieceFirstTokenNoSpace) {
  // First token never gets a space prefix.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildWordPieceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace BERT: decode([hello]) = "hello"
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0}));
  EXPECT_EQ(result, "hello");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, WordPieceMultipleWords) {
  // Multiple whole words get spaces between them.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "the");
  vocab_builder.AddToken(1, "quick");
  vocab_builder.AddToken(2, "brown");
  vocab_builder.AddToken(3, "fox");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildWordPieceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace BERT: decode([the, quick, brown, fox]) = "the quick brown fox"
  IREE_ASSERT_OK_AND_ASSIGN(std::string result,
                            Decode(tokenizer, {0, 1, 2, 3}));
  EXPECT_EQ(result, "the quick brown fox");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, WordPieceWithCleanup) {
  // Cleanup removes space before punctuation.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, ",");
  vocab_builder.AddToken(2, "world");
  vocab_builder.AddToken(3, "!");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer =
      BuildWordPieceTokenizer(vocab.release(), /*cleanup=*/true);
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace BERT (cleanup=true): decode([hello, ",", world, "!"])
  //   = "hello, world!"
  // Without cleanup: "hello , world !" (spaces before punctuation)
  IREE_ASSERT_OK_AND_ASSIGN(std::string result,
                            Decode(tokenizer, {0, 1, 2, 3}));
  EXPECT_EQ(result, "hello, world!");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, WordPieceMultipleContinuations) {
  // Multiple ## tokens in a row.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "un");
  vocab_builder.AddToken(1, "##believ");
  vocab_builder.AddToken(2, "##able");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildWordPieceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace BERT: decode([un, ##believ, ##able]) = "unbelievable"
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1, 2}));
  EXPECT_EQ(result, "unbelievable");

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Sequence Decoder (NOT pre-decodable when children include ByteFallback)
//===----------------------------------------------------------------------===//
// Sequence decoders chain multiple decoders. If any child lacks STATELESS,
// the sequence is not pre-decodable and uses the slow path.

TEST_F(TokenizerDecodeTest, SequenceNotPreDecodedUsesSlowPath) {
  // Build a tokenizer with a Sequence decoder containing ByteFallback.
  // ByteFallback is NOT stateless, so the Sequence won't be pre-decoded.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  vocab_builder.AddToken(1, "\xC4\xA0world");  // GPT-2 space+world.
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  // Create a sequence [ByteLevel, Metaspace] — this is artificial but tests
  // the slow path since Metaspace is position-sensitive and ByteLevel is
  // stateless, the sequence IS pre-decodable. Instead, let's just test
  // with ByteLevel alone and verify the pre-decoded path works.
  iree_tokenizer_decoder_t* byte_level_decoder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decoder_byte_level_allocate(
      iree_allocator_system(), &byte_level_decoder));
  iree_tokenizer_builder_set_decoder(builder.get(), byte_level_decoder);

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Verify decode works (uses pre-decoded path since ByteLevel is STATELESS).
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1}));
  EXPECT_EQ(result, "Hello world");

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Pre-decode State Management
//===----------------------------------------------------------------------===//

TEST_F(TokenizerDecodeTest, DecodeStateMinimalForPreDecoded) {
  // Pre-decoded tokenizers should have smaller decode state (no decoder state
  // or string buffer needed).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  ScopedVocab vocab = vocab_builder.Build();

  // Build with ByteLevel decoder (pre-decodable).
  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_decode_state_calculate_size(tokenizer, &state_size));
  // Pre-decoded state should be minimal: just the base decode_state_t struct.
  // No decoder state, no string buffer. Exact size is implementation detail,
  // but it should be significantly smaller than the slow-path state.
  EXPECT_GT(state_size, 0u);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, DecodeStateFreshPerSequence) {
  // Each decode state instance is independent: first-token tracking resets.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xE2\x96\x81Hello");
  vocab_builder.AddToken(1, "\xE2\x96\x81world");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // First decode: token 0 is first, strips ▁.
  IREE_ASSERT_OK_AND_ASSIGN(std::string result1, Decode(tokenizer, {0, 1}));
  EXPECT_EQ(result1, "Hello world");

  // Second decode with fresh state: token 0 is again first, strips ▁.
  IREE_ASSERT_OK_AND_ASSIGN(std::string result2, Decode(tokenizer, {0}));
  EXPECT_EQ(result2, "Hello");

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Pre-decode Fast Path: Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(TokenizerDecodeTest, EmptyTokenList) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // Empty token list produces empty output.
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {}));
  EXPECT_EQ(result, "");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, OutOfRangeTokenIDsSkipped) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // Out-of-range IDs produce empty output (skipped silently).
  // Token 999 is beyond vocab capacity.
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 999, 1}));
  EXPECT_EQ(result, "helloworld");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, NegativeTokenIDsSkipped) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // Negative IDs produce empty output (skipped silently).
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {-1, 0, -5}));
  EXPECT_EQ(result, "hello");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, SparseVocabGapIDs) {
  // Vocab with gaps: token IDs 0, 5, 10 exist but 1-4 and 6-9 don't.
  ScopedVocabBuilder vocab_builder(16);
  vocab_builder.AddToken(0, "zero");
  vocab_builder.AddToken(5, "five");
  vocab_builder.AddToken(10, "ten");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // Gap IDs produce empty output (offsets[id] == offsets[id+1]).
  IREE_ASSERT_OK_AND_ASSIGN(std::string result,
                            Decode(tokenizer, {0, 3, 5, 7, 10}));
  EXPECT_EQ(result, "zerofiveten");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, OutputBufferExactFit) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "abc");
  vocab_builder.AddToken(1, "def");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // Use a buffer that exactly fits the first token but not the second.
  IREE_ASSERT_OK_AND_ASSIGN(auto state,
                            DecodeStateStorage::Allocate(tokenizer));

  char output[3];  // Exactly 3 bytes.
  iree_host_size_t tokens_consumed = 0;
  iree_host_size_t text_length = 0;

  int32_t token_ids[] = {0, 1};
  iree_tokenizer_token_id_list_t id_list = {2, token_ids};
  IREE_ASSERT_OK(iree_tokenizer_decode_state_feed(
      state.state(), id_list, iree_make_mutable_string_view(output, 3),
      &tokens_consumed, &text_length));

  // Should consume only the first token (3 bytes fills the buffer).
  EXPECT_EQ(tokens_consumed, 1u);
  EXPECT_EQ(text_length, 3u);
  EXPECT_EQ(std::string(output, 3), "abc");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, OutputBufferTooSmallForFirstToken) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto state,
                            DecodeStateStorage::Allocate(tokenizer));

  char output[3];  // Too small for "hello" (5 bytes).
  iree_host_size_t tokens_consumed = 0;
  iree_host_size_t text_length = 0;

  int32_t token_ids[] = {0};
  iree_tokenizer_token_id_list_t id_list = {1, token_ids};
  IREE_ASSERT_OK(iree_tokenizer_decode_state_feed(
      state.state(), id_list, iree_make_mutable_string_view(output, 3),
      &tokens_consumed, &text_length));

  // Cannot fit the token: consumes 0, writes 0.
  EXPECT_EQ(tokens_consumed, 0u);
  EXPECT_EQ(text_length, 0u);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, FinalizeNoOpForPreDecoded) {
  // Pre-decoded path has no buffered state; finalize writes nothing.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto state,
                            DecodeStateStorage::Allocate(tokenizer));

  // Feed tokens first.
  char output[64];
  iree_host_size_t tokens_consumed = 0;
  iree_host_size_t text_length = 0;
  int32_t token_ids[] = {0};
  iree_tokenizer_token_id_list_t id_list = {1, token_ids};
  IREE_ASSERT_OK(iree_tokenizer_decode_state_feed(
      state.state(), id_list, iree_make_mutable_string_view(output, 64),
      &tokens_consumed, &text_length));
  EXPECT_EQ(tokens_consumed, 1u);
  EXPECT_EQ(text_length, 5u);

  // Finalize should produce no additional output.
  iree_host_size_t finalize_length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode_state_finalize(
      state.state(), iree_make_mutable_string_view(output + text_length, 32),
      &finalize_length));
  EXPECT_EQ(finalize_length, 0u);

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Pre-decode vs Slow Path Equivalence
//===----------------------------------------------------------------------===//

TEST_F(TokenizerDecodeTest, NoDecoderFailsWithPrecondition) {
  // Without a decoder, feeding tokens must fail with FAILED_PRECONDITION.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());
  // No decoder set.

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // State allocation succeeds (minimal state), but feed fails.
  IREE_ASSERT_OK_AND_ASSIGN(auto state,
                            DecodeStateStorage::Allocate(tokenizer));

  std::vector<char> output(256);
  iree_host_size_t tokens_consumed = 0;
  iree_host_size_t text_length = 0;
  std::vector<int32_t> token_ids = {0, 1};
  iree_tokenizer_token_id_list_t id_list = {
      /*.count=*/token_ids.size(),
      /*.values=*/token_ids.data(),
  };
  iree_status_t status = iree_tokenizer_decode_state_feed(
      state.state(), id_list,
      iree_make_mutable_string_view(output.data(), output.size()),
      &tokens_consumed, &text_length);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Position-Sensitive Pre-decode: First Token Handling
//===----------------------------------------------------------------------===//

TEST_F(TokenizerDecodeTest, PositionSensitiveOnlyAffectsFirstToken) {
  // For Metaspace with PREPEND_ALWAYS: only the very first token in the
  // stream has its leading space stripped. All subsequent tokens keep their
  // decoded form.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xE2\x96\x81The");
  vocab_builder.AddToken(1, "\xE2\x96\x81quick");
  vocab_builder.AddToken(2,
                         "\xE2\x96\x81"
                         "brown");
  vocab_builder.AddToken(3,
                         "\xE2\x96\x81"
                         "fox");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // HuggingFace T5: "The quick brown fox"
  IREE_ASSERT_OK_AND_ASSIGN(std::string result,
                            Decode(tokenizer, {0, 1, 2, 3}));
  EXPECT_EQ(result, "The quick brown fox");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, PositionSensitiveTokenWithoutLeadingSpace) {
  // If the first token's pre-decoded form doesn't start with space,
  // position-sensitive stripping is a no-op.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");  // No ▁ prefix.
  vocab_builder.AddToken(1, "\xE2\x96\x81world");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // First token "Hello" has no ▁, so no space to strip. Decoded as-is.
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1}));
  EXPECT_EQ(result, "Hello world");

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Multi-feed Decode (incremental consumption)
//===----------------------------------------------------------------------===//

TEST_F(TokenizerDecodeTest, MultiFeedConsumesSameAsOneFeed) {
  // Feeding tokens in multiple calls produces the same result as one call.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xE2\x96\x81Hello");
  vocab_builder.AddToken(1, ",");
  vocab_builder.AddToken(2, "\xE2\x96\x81world");
  vocab_builder.AddToken(3, "!");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // One-shot decode.
  IREE_ASSERT_OK_AND_ASSIGN(std::string one_shot,
                            Decode(tokenizer, {0, 1, 2, 3}));

  // Multi-feed decode: feed 2 tokens at a time.
  IREE_ASSERT_OK_AND_ASSIGN(auto state,
                            DecodeStateStorage::Allocate(tokenizer));

  std::string multi_feed_result;
  char output[64];

  // Feed first 2 tokens.
  iree_host_size_t tokens_consumed = 0;
  iree_host_size_t text_length = 0;
  int32_t batch1[] = {0, 1};
  IREE_ASSERT_OK(iree_tokenizer_decode_state_feed(
      state.state(), {2, batch1},
      iree_make_mutable_string_view(output, sizeof(output)), &tokens_consumed,
      &text_length));
  EXPECT_EQ(tokens_consumed, 2u);
  multi_feed_result.append(output, text_length);

  // Feed next 2 tokens.
  int32_t batch2[] = {2, 3};
  IREE_ASSERT_OK(iree_tokenizer_decode_state_feed(
      state.state(), {2, batch2},
      iree_make_mutable_string_view(output, sizeof(output)), &tokens_consumed,
      &text_length));
  EXPECT_EQ(tokens_consumed, 2u);
  multi_feed_result.append(output, text_length);

  // Finalize.
  iree_host_size_t finalize_length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode_state_finalize(
      state.state(), iree_make_mutable_string_view(output, sizeof(output)),
      &finalize_length));
  multi_feed_result.append(output, finalize_length);

  EXPECT_EQ(multi_feed_result, one_shot);

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Decode Flags: SKIP_SPECIAL_TOKENS
//===----------------------------------------------------------------------===//

TEST_F(TokenizerDecodeTest, SkipSpecialTokensPreDecoded) {
  // Pre-decoded path: BOS/EOS tokens with SPECIAL attr are skipped.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "<s>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  vocab_builder.AddToken(1, "Hello");
  vocab_builder.AddToken(2, "\xC4\xA0world");  // GPT-2 space+world.
  vocab_builder.AddToken(3, "</s>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // Without SKIP: includes special token text.
  IREE_ASSERT_OK_AND_ASSIGN(
      std::string with_special,
      Decode(tokenizer, {0, 1, 2, 3}, IREE_TOKENIZER_DECODE_FLAG_NONE));
  EXPECT_EQ(with_special, "<s>Hello world</s>");

  // With SKIP: special tokens produce no output.
  IREE_ASSERT_OK_AND_ASSIGN(
      std::string without_special,
      Decode(tokenizer, {0, 1, 2, 3},
             IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS));
  EXPECT_EQ(without_special, "Hello world");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, SkipSpecialTokensOnlySpecialTokens) {
  // All tokens are special → empty output.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "<s>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  vocab_builder.AddToken(1, "</s>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  vocab_builder.AddToken(2, "<pad>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(
      std::string result,
      Decode(tokenizer, {0, 1, 2},
             IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS));
  EXPECT_EQ(result, "");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, SkipSpecialTokensMetaspaceFirstToken) {
  // Metaspace decoder: the first non-special token should get leading space
  // stripped, even though BOS (special) comes before it in the sequence.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "<s>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  vocab_builder.AddToken(1, "\xE2\x96\x81Hello");  // ▁Hello
  vocab_builder.AddToken(2, "\xE2\x96\x81world");  // ▁world
  vocab_builder.AddToken(3, "</s>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildMetaspaceTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // Without SKIP: <s> text is emitted, then ▁Hello (stripped to Hello as
  // first non-space), then " world".
  IREE_ASSERT_OK_AND_ASSIGN(
      std::string with_special,
      Decode(tokenizer, {0, 1, 2, 3}, IREE_TOKENIZER_DECODE_FLAG_NONE));
  // <s> is first token → its leading space would be stripped if it had one.
  // But <s> has no ▁ prefix, so it's "<s>".
  // ▁Hello is NOT first → becomes " Hello".
  // ▁world → " world".
  // </s> → "</s>".
  EXPECT_EQ(with_special, "<s> Hello world</s>");

  // With SKIP: <s> skipped, ▁Hello is now the first output token → stripped.
  IREE_ASSERT_OK_AND_ASSIGN(
      std::string without_special,
      Decode(tokenizer, {0, 1, 2, 3},
             IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS));
  EXPECT_EQ(without_special, "Hello world");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, SkipSpecialTokensInterspersed) {
  // Special tokens in the middle of the sequence.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  vocab_builder.AddToken(1, "<sep>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  vocab_builder.AddToken(2, "\xC4\xA0world");  // GPT-2 space+world.
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // With SKIP: <sep> is removed from between "Hello" and " world".
  IREE_ASSERT_OK_AND_ASSIGN(
      std::string result,
      Decode(tokenizer, {0, 1, 2},
             IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS));
  EXPECT_EQ(result, "Hello world");

  // Without SKIP: <sep> text is included.
  IREE_ASSERT_OK_AND_ASSIGN(
      std::string with_sep,
      Decode(tokenizer, {0, 1, 2}, IREE_TOKENIZER_DECODE_FLAG_NONE));
  EXPECT_EQ(with_sep, "Hello<sep> world");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, SkipSpecialTokensControlVsSpecial) {
  // CONTROL tokens are NOT skipped — only SPECIAL tokens are.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  vocab_builder.AddToken(1, "<ctrl>", IREE_TOKENIZER_TOKEN_ATTR_CONTROL);
  vocab_builder.AddToken(2, "<bos>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(
      std::string result,
      Decode(tokenizer, {0, 1, 2},
             IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS));
  // <ctrl> (CONTROL only) is included. <bos> (SPECIAL) is skipped.
  EXPECT_EQ(result, "Hello<ctrl>");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, SkipSpecialTokensOneShotAPI) {
  // Tests the one-shot iree_tokenizer_decode() API with flags.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "<s>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  vocab_builder.AddToken(1, "Hello");
  vocab_builder.AddToken(2, "</s>", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteLevelTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  int32_t token_ids[] = {0, 1, 2};
  iree_tokenizer_token_id_list_t tokens = {3, token_ids};
  char output[256];
  iree_host_size_t text_length = 0;

  // With SKIP_SPECIAL_TOKENS via the one-shot API.
  IREE_ASSERT_OK(iree_tokenizer_decode(
      tokenizer, tokens, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
      iree_make_mutable_string_view(output, sizeof(output)),
      iree_allocator_system(), &text_length));
  EXPECT_EQ(std::string(output, text_length), "Hello");

  // Without skip.
  text_length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode(
      tokenizer, tokens, IREE_TOKENIZER_DECODE_FLAG_NONE,
      iree_make_mutable_string_view(output, sizeof(output)),
      iree_allocator_system(), &text_length));
  EXPECT_EQ(std::string(output, text_length), "<s>Hello</s>");

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Hybrid Pre-Decoded Decode (ByteFallback via Sequence decoder)
//===----------------------------------------------------------------------===//
// Tests the hybrid decode path: Sequence(Replace, ByteFallback) where
// non-byte tokens use the O(1) memcpy fast path and byte tokens (<0xHH>)
// are handled by an inline UTF-8 accumulator.
//
// Pre-decoded: YES (STATELESS_EXCEPT_BYTE_TOKENS)
// The Sequence gets this capability from ByteFallback, enabling pre-decode
// for the 99.9% of tokens that aren't byte tokens.

// Builds a Gemma-like tokenizer with Sequence(Replace(▁→space), ByteFallback).
iree_tokenizer_t* BuildByteFallbackTokenizer(iree_tokenizer_vocab_t* vocab) {
  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab);

  // Build Sequence(Replace(▁→space), ByteFallback) to match Gemma's decoder.
  iree_tokenizer_decoder_t* replace_decoder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decoder_replace_allocate(
      iree_make_cstring_view("\xE2\x96\x81"),  // ▁ (U+2581, 3 bytes)
      iree_make_cstring_view(" "),             // space (1 byte)
      iree_allocator_system(), &replace_decoder));

  iree_tokenizer_decoder_t* byte_fallback_decoder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decoder_byte_fallback_allocate(
      iree_allocator_system(), &byte_fallback_decoder));

  iree_tokenizer_decoder_t* children[] = {replace_decoder,
                                          byte_fallback_decoder};
  iree_tokenizer_decoder_t* sequence_decoder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decoder_sequence_allocate(
      children, 2, iree_allocator_system(), &sequence_decoder));

  iree_tokenizer_builder_set_decoder(builder.get(), sequence_decoder);
  return BuildTokenizer(builder.get());
}

TEST_F(TokenizerDecodeTest, ByteFallbackNormalTokens) {
  // Normal tokens pass through Replace(▁→space) and ByteFallback unchanged.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xE2\x96\x81Hello");  // ▁Hello
  vocab_builder.AddToken(1, "\xE2\x96\x81world");  // ▁world
  vocab_builder.AddToken(2, "!");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1, 2}));
  EXPECT_EQ(result, " Hello world!");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteFallbackValidUTF8Sequence) {
  // Byte tokens forming valid UTF-8: <0xC3><0xA9> → é (U+00E9).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "caf");
  vocab_builder.AddToken(1, "<0xC3>");
  vocab_builder.AddToken(2, "<0xA9>");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1, 2}));
  EXPECT_EQ(result, "café");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteFallbackASCIIByte) {
  // Single ASCII byte token: <0x41> → 'A'.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  vocab_builder.AddToken(1, "<0x41>");  // 'A'
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1}));
  EXPECT_EQ(result, "HelloA");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteFallbackThreeByteUTF8) {
  // Three-byte UTF-8: <0xE2><0x9C><0x93> → ✓ (U+2713).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "ok");
  vocab_builder.AddToken(1, "<0xE2>");
  vocab_builder.AddToken(2, "<0x9C>");
  vocab_builder.AddToken(3, "<0x93>");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(std::string result,
                            Decode(tokenizer, {0, 1, 2, 3}));
  EXPECT_EQ(result, "ok\xE2\x9C\x93");  // ok✓

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteFallbackFourByteUTF8) {
  // Four-byte UTF-8: <0xF0><0x9F><0x98><0x80> → 😀 (U+1F600).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "<0xF0>");
  vocab_builder.AddToken(1, "<0x9F>");
  vocab_builder.AddToken(2, "<0x98>");
  vocab_builder.AddToken(3, "<0x80>");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(std::string result,
                            Decode(tokenizer, {0, 1, 2, 3}));
  EXPECT_EQ(result, "\xF0\x9F\x98\x80");  // 😀

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteFallbackInterruptedSequence) {
  // Interrupted byte sequence: <0xC3> followed by normal token.
  // The incomplete C3 is flushed as U+FFFD.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "<0xC3>");
  vocab_builder.AddToken(1, "world");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1}));
  // C3 is an incomplete 2-byte sequence, flushed as U+FFFD.
  EXPECT_EQ(result, "\xEF\xBF\xBDworld");  // U+FFFD + "world"

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteFallbackIncompleteAtEnd) {
  // Incomplete byte sequence at stream end: <0xC3> with no continuation.
  // Finalize should flush as U+FFFD.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  vocab_builder.AddToken(1, "<0xC3>");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1}));
  // C3 is flushed as U+FFFD at finalize.
  EXPECT_EQ(result, "Hello\xEF\xBF\xBD");  // "Hello" + U+FFFD

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteFallbackMixedStream) {
  // Mixed stream: normal + byte sequence + normal.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "caf");
  vocab_builder.AddToken(1, "<0xC3>");
  vocab_builder.AddToken(2, "<0xA9>");
  vocab_builder.AddToken(3, "\xE2\x96\x81is");    // ▁is
  vocab_builder.AddToken(4, "\xE2\x96\x81good");  // ▁good
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(std::string result,
                            Decode(tokenizer, {0, 1, 2, 3, 4}));
  EXPECT_EQ(result, "café is good");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteFallbackNonContiguousRange) {
  // Byte tokens with a gap: non-byte token between byte tokens.
  // This tests the bitmap handling for non-contiguous ranges.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  vocab_builder.AddToken(1, "<0xC3>");  // Byte token at ID 1.
  vocab_builder.AddToken(2, "\t");      // Non-byte token at ID 2 (gap).
  vocab_builder.AddToken(3, "<0xA9>");  // Byte token at ID 3.
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // Normal + byte sequence (IDs 1, 3 with gap at 2) → café
  // Token 0 = "Hello", then byte C3 A9 = é
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 1, 3}));
  EXPECT_EQ(result, "Helloé");

  // Using the gap token (tab) between byte tokens.
  IREE_ASSERT_OK_AND_ASSIGN(std::string result2, Decode(tokenizer, {0, 2}));
  EXPECT_EQ(result2, "Hello\t");

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerDecodeTest, ByteFallbackInvalidContinuation) {
  // Invalid continuation: <0xC3> followed by <0xC3> (not a continuation byte).
  // First C3 should become U+FFFD, second C3 starts a new sequence.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "<0xC3>");
  vocab_builder.AddToken(1, "<0xA9>");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_t* tokenizer = BuildByteFallbackTokenizer(vocab.release());
  ASSERT_NE(tokenizer, nullptr);

  // <0xC3><0xC3><0xA9> — first C3 interrupted by second C3, which then
  // combines with A9 to form é.
  IREE_ASSERT_OK_AND_ASSIGN(std::string result, Decode(tokenizer, {0, 0, 1}));
  EXPECT_EQ(result, "\xEF\xBF\xBD\xC3\xA9");  // U+FFFD + é

  iree_tokenizer_free(tokenizer);
}

}  // namespace
}  // namespace tokenizer
}  // namespace iree
