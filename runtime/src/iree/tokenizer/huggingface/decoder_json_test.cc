// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/decoder_json.h"

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

class DecoderJsonTest : public ::testing::Test {
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
// WordPiece JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, WordPieceDefault) {
  const char* json = R"({
    "decoder": {
      "type": "WordPiece",
      "prefix": "##",
      "cleanup": true
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_WORDPIECE);
  EXPECT_EQ(Decode(&dec, {"Hello", "##World"}), "HelloWorld");
  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderJsonTest, WordPieceNoCleanup) {
  const char* json = R"({
    "decoder": {
      "type": "WordPiece",
      "cleanup": false
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  // Without cleanup, space before punctuation is preserved.
  EXPECT_EQ(Decode(&dec, {"Hello", ",", "World"}), "Hello , World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Metaspace JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, MetaspaceDefault) {
  const char* json = R"({
    "decoder": {
      "type": "Metaspace",
      "replacement": "▁"
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_METASPACE);
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), "Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// ByteLevel JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, ByteLevel) {
  const char* json = R"({
    "decoder": {
      "type": "ByteLevel"
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_BYTE_LEVEL);

  iree_tokenizer_decoder_deinitialize(&dec);
}

// Real-world config from BART, Llama 3.x, CLIP, etc.
// use_regex is an encoding-only parameter (we support it in the pre-tokenizer).
// The decoder should accept it since we fully support it in encoding.
TEST_F(DecoderJsonTest, ByteLevelWithUseRegex) {
  const char* json = R"({
    "decoder": {
      "type": "ByteLevel",
      "add_prefix_space": true,
      "trim_offsets": true,
      "use_regex": true
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_BYTE_LEVEL);

  iree_tokenizer_decoder_deinitialize(&dec);
}

// use_regex=false should also be accepted.
TEST_F(DecoderJsonTest, ByteLevelWithUseRegexFalse) {
  const char* json = R"({
    "decoder": {
      "type": "ByteLevel",
      "use_regex": false
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_BYTE_LEVEL);

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// BPE JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, BPE) {
  const char* json = R"({
    "decoder": {
      "type": "BPE"
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_BPE);

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Null/Missing Decoder
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, NullDecoder) {
  const char* json = R"({
    "decoder": null
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_NONE);

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderJsonTest, MissingDecoder) {
  const char* json = R"({
    "model": {}
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_NONE);

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Error Cases
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, UnsupportedType) {
  const char* json = R"({
    "decoder": {
      "type": "SomeUnknownDecoder"
    }
  })";

  iree_tokenizer_decoder_t dec;
  iree_status_t status = iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec);

  EXPECT_TRUE(iree_status_is_unimplemented(status));
  iree_status_free(status);
}

//===----------------------------------------------------------------------===//
// Replace JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, ReplaceWithStringPattern) {
  // Replace decoder with String pattern (common case).
  const char* json = R"({
    "decoder": {
      "type": "Replace",
      "pattern": {"String": "▁"},
      "content": " "
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_REPLACE);
  // ▁ replaced with space; Replace does NOT strip leading space on first token.
  // ▁Hello → " Hello", ▁World → " World", concatenated = " Hello World".
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), " Hello World");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderJsonTest, ReplaceWithRegexPatternUnimplemented) {
  // Regex patterns are not yet supported - should return UNIMPLEMENTED.
  const char* json = R"({
    "decoder": {
      "type": "Replace",
      "pattern": {"Regex": "\\s+"},
      "content": " "
    }
  })";

  iree_tokenizer_decoder_t dec;
  iree_status_t status = iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec);

  EXPECT_TRUE(iree_status_is_unimplemented(status));
  iree_status_free(status);
}

//===----------------------------------------------------------------------===//
// ByteFallback JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, ByteFallback) {
  const char* json = R"({
    "decoder": {
      "type": "ByteFallback"
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_BYTE_FALLBACK);
  EXPECT_EQ(Decode(&dec, {"<0x48>", "<0x69>"}), "Hi");
  EXPECT_EQ(Decode(&dec, {"Hello", "<0x21>"}), "Hello!");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Fuse JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, Fuse) {
  const char* json = R"({
    "decoder": {
      "type": "Fuse"
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_FUSE);
  EXPECT_EQ(Decode(&dec, {"a", "b", "c"}), "abc");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Strip JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, Strip) {
  const char* json = R"({
    "decoder": {
      "type": "Strip",
      "content": " ",
      "start": 1,
      "stop": 0
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_STRIP);
  EXPECT_EQ(Decode(&dec, {" Hello"}), "Hello");
  EXPECT_EQ(Decode(&dec, {"  Hello"}), " Hello");  // Only strips 1.

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderJsonTest, StripBothEnds) {
  const char* json = R"({
    "decoder": {
      "type": "Strip",
      "content": "_",
      "start": 2,
      "stop": 2
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(Decode(&dec, {"__Hello__"}), "Hello");

  iree_tokenizer_decoder_deinitialize(&dec);
}

//===----------------------------------------------------------------------===//
// Sequence JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(DecoderJsonTest, SequenceTinyLlamaStyle) {
  // TinyLlama uses: Replace → ByteFallback → Fuse → Strip.
  const char* json = R"({
    "decoder": {
      "type": "Sequence",
      "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0}
      ]
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_SEQUENCE);
  // ▁Hello▁World → " Hello World" → fused → strip leading space → "Hello World"
  EXPECT_EQ(Decode(&dec, {"▁Hello", "▁World"}), "Hello World");
  // Test byte tokens in sequence.
  EXPECT_EQ(Decode(&dec, {"▁Hi", "<0x21>"}), "Hi!");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderJsonTest, SequenceEmpty) {
  const char* json = R"({
    "decoder": {
      "type": "Sequence",
      "decoders": []
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  // Empty sequence is optimized to NONE type (passthrough).
  EXPECT_EQ(dec.type, IREE_TOKENIZER_DECODER_NONE);
  EXPECT_EQ(Decode(&dec, {"Hello", "World"}), "HelloWorld");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderJsonTest, SequenceSingleDecoder) {
  const char* json = R"({
    "decoder": {
      "type": "Sequence",
      "decoders": [
        {"type": "Fuse"}
      ]
    }
  })";

  iree_tokenizer_decoder_t dec;
  IREE_ASSERT_OK(iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec));

  EXPECT_EQ(Decode(&dec, {"a", "b", "c"}), "abc");

  iree_tokenizer_decoder_deinitialize(&dec);
}

TEST_F(DecoderJsonTest, SequenceWithNestedError) {
  // If a child decoder fails parsing, the sequence should fail.
  const char* json = R"({
    "decoder": {
      "type": "Sequence",
      "decoders": [
        {"type": "Replace", "pattern": {"Regex": "invalid"}, "content": " "}
      ]
    }
  })";

  iree_tokenizer_decoder_t dec;
  iree_status_t status = iree_tokenizer_decoder_parse_json(IREE_SV(json), &dec);

  // Should fail because Regex is unimplemented.
  EXPECT_TRUE(iree_status_is_unimplemented(status));
  iree_status_free(status);
}

}  // namespace
