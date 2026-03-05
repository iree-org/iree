// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/tiktoken/tiktoken.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"

namespace {

// A hand-crafted minimal tiktoken file.
// Contains 256 single-byte tokens (ranks 0-255) and a few merges (256+).
//
// The first 256 entries are the single-byte tokens. For simplicity, we use
// the actual base64 encoding of each raw byte value. Then we add a few
// multi-byte merge tokens that would appear in a real tiktoken file.
//
// This is a subset — enough to test parsing, ByteLevel encoding, merge
// reconstruction, and basic tokenization.

// Helper: generates a minimal tiktoken file with 256 single-byte tokens
// plus a few hand-crafted merge tokens.
static std::string GenerateMinimalTiktokenData() {
  // Base64 encoding of single bytes 0x00-0xFF.
  // Pre-computed: base64(byte) for each byte value.
  static const char* kSingleByteBase64[256] = {
      "AA", "AQ", "Ag", "Aw", "BA", "BQ", "Bg", "Bw",  // 0x00-0x07
      "CA", "CQ", "Cg", "Cw", "DA", "DQ", "Dg", "Dw",  // 0x08-0x0F
      "EA", "EQ", "Eg", "Ew", "FA", "FQ", "Fg", "Fw",  // 0x10-0x17
      "GA", "GQ", "Gg", "Gw", "HA", "HQ", "Hg", "Hw",  // 0x18-0x1F
      "IA", "IQ", "Ig", "Iw", "JA", "JQ", "Jg", "Jw",  // 0x20-0x27
      "KA", "KQ", "Kg", "Kw", "LA", "LQ", "Lg", "Lw",  // 0x28-0x2F
      "MA", "MQ", "Mg", "Mw", "NA", "NQ", "Ng", "Nw",  // 0x30-0x37
      "OA", "OQ", "Og", "Ow", "PA", "PQ", "Pg", "Pw",  // 0x38-0x3F
      "QA", "QQ", "Qg", "Qw", "RA", "RQ", "Rg", "Rw",  // 0x40-0x47
      "SA", "SQ", "Sg", "Sw", "TA", "TQ", "Tg", "Tw",  // 0x48-0x4F
      "UA", "UQ", "Ug", "Uw", "VA", "VQ", "Vg", "Vw",  // 0x50-0x57
      "WA", "WQ", "Wg", "Ww", "XA", "XQ", "Xg", "Xw",  // 0x58-0x5F
      "YA", "YQ", "Yg", "Yw", "ZA", "ZQ", "Zg", "Zw",  // 0x60-0x67
      "aA", "aQ", "ag", "aw", "bA", "bQ", "bg", "bw",  // 0x68-0x6F
      "cA", "cQ", "cg", "cw", "dA", "dQ", "dg", "dw",  // 0x70-0x77
      "eA", "eQ", "eg", "ew", "fA", "fQ", "fg", "fw",  // 0x78-0x7F
      "gA", "gQ", "gg", "gw", "hA", "hQ", "hg", "hw",  // 0x80-0x87
      "iA", "iQ", "ig", "iw", "jA", "jQ", "jg", "jw",  // 0x88-0x8F
      "kA", "kQ", "kg", "kw", "lA", "lQ", "lg", "lw",  // 0x90-0x97
      "mA", "mQ", "mg", "mw", "nA", "nQ", "ng", "nw",  // 0x98-0x9F
      "oA", "oQ", "og", "ow", "pA", "pQ", "pg", "pw",  // 0xA0-0xA7
      "qA", "qQ", "qg", "qw", "rA", "rQ", "rg", "rw",  // 0xA8-0xAF
      "sA", "sQ", "sg", "sw", "tA", "tQ", "tg", "tw",  // 0xB0-0xB7
      "uA", "uQ", "ug", "uw", "vA", "vQ", "vg", "vw",  // 0xB8-0xBF
      "wA", "wQ", "wg", "ww", "xA", "xQ", "xg", "xw",  // 0xC0-0xC7
      "yA", "yQ", "yg", "yw", "zA", "zQ", "zg", "zw",  // 0xC8-0xCF
      "0A", "0Q", "0g", "0w", "1A", "1Q", "1g", "1w",  // 0xD0-0xD7
      "2A", "2Q", "2g", "2w", "3A", "3Q", "3g", "3w",  // 0xD8-0xDF
      "4A", "4Q", "4g", "4w", "5A", "5Q", "5g", "5w",  // 0xE0-0xE7
      "6A", "6Q", "6g", "6w", "7A", "7Q", "7g", "7w",  // 0xE8-0xEF
      "8A", "8Q", "8g", "8w", "9A", "9Q", "9g", "9w",  // 0xF0-0xF7
      "+A", "+Q", "+g", "+w", "/A", "/Q", "/g", "/w",  // 0xF8-0xFF
  };

  std::string result;
  result.reserve(8192);

  // Ranks 0-255: single-byte tokens.
  for (int i = 0; i < 256; ++i) {
    result += kSingleByteBase64[i];
    result += " ";
    result += std::to_string(i);
    result += "\n";
  }

  // Rank 256: "in" (bytes 0x69, 0x6E) -> base64("in") = "aW4="
  // This merges byte 0x69 ('i', rank 105) + byte 0x6E ('n', rank 110).
  result += "aW4= 256\n";

  // Rank 257: " t" (bytes 0x20, 0x74) -> base64(" t") = "IHQ="
  // Merges byte 0x20 (space, rank 32) + byte 0x74 ('t', rank 116).
  result += "IHQ= 257\n";

  // Rank 258: "th" (bytes 0x74, 0x68) -> base64("th") = "dGg="
  // Merges byte 0x74 ('t', rank 116) + byte 0x68 ('h', rank 104).
  result += "dGg= 258\n";

  // Rank 259: " th" (bytes 0x20, 0x74, 0x68) -> base64(" th") = "IHRo"
  // Merges " t" (rank 257) + "h" (rank 104).
  // Or: space (32) + "th" (258). BPE simulation determines which.
  result += "IHRo 259\n";

  return result;
}

// Generates tiktoken data with non-identity byte-to-rank mapping.
// Rank 0 maps to byte 0x21 ('!'), rank 1 to byte 0x22 ('"'), etc., wrapping
// around so that the 68 "non-printable" bytes get high ranks. This mimics the
// real cl100k_base/GPT-2 byte ordering where printable bytes come first.
//
// Merge tokens use the same raw bytes as GenerateMinimalTiktokenData, but the
// rank assignments for the constituent bytes differ because of the permutation.
static std::string GenerateShuffledTiktokenData() {
  // The permutation: rank r maps to byte ((r + 0x21) & 0xFF).
  // Inverse: byte b maps to rank ((b - 0x21) & 0xFF) = ((b + 0xDF) & 0xFF).
  static const char* kSingleByteBase64[256] = {
      "AA", "AQ", "Ag", "Aw", "BA", "BQ", "Bg", "Bw",  // 0x00-0x07
      "CA", "CQ", "Cg", "Cw", "DA", "DQ", "Dg", "Dw",  // 0x08-0x0F
      "EA", "EQ", "Eg", "Ew", "FA", "FQ", "Fg", "Fw",  // 0x10-0x17
      "GA", "GQ", "Gg", "Gw", "HA", "HQ", "Hg", "Hw",  // 0x18-0x1F
      "IA", "IQ", "Ig", "Iw", "JA", "JQ", "Jg", "Jw",  // 0x20-0x27
      "KA", "KQ", "Kg", "Kw", "LA", "LQ", "Lg", "Lw",  // 0x28-0x2F
      "MA", "MQ", "Mg", "Mw", "NA", "NQ", "Ng", "Nw",  // 0x30-0x37
      "OA", "OQ", "Og", "Ow", "PA", "PQ", "Pg", "Pw",  // 0x38-0x3F
      "QA", "QQ", "Qg", "Qw", "RA", "RQ", "Rg", "Rw",  // 0x40-0x47
      "SA", "SQ", "Sg", "Sw", "TA", "TQ", "Tg", "Tw",  // 0x48-0x4F
      "UA", "UQ", "Ug", "Uw", "VA", "VQ", "Vg", "Vw",  // 0x50-0x57
      "WA", "WQ", "Wg", "Ww", "XA", "XQ", "Xg", "Xw",  // 0x58-0x5F
      "YA", "YQ", "Yg", "Yw", "ZA", "ZQ", "Zg", "Zw",  // 0x60-0x67
      "aA", "aQ", "ag", "aw", "bA", "bQ", "bg", "bw",  // 0x68-0x6F
      "cA", "cQ", "cg", "cw", "dA", "dQ", "dg", "dw",  // 0x70-0x77
      "eA", "eQ", "eg", "ew", "fA", "fQ", "fg", "fw",  // 0x78-0x7F
      "gA", "gQ", "gg", "gw", "hA", "hQ", "hg", "hw",  // 0x80-0x87
      "iA", "iQ", "ig", "iw", "jA", "jQ", "jg", "jw",  // 0x88-0x8F
      "kA", "kQ", "kg", "kw", "lA", "lQ", "lg", "lw",  // 0x90-0x97
      "mA", "mQ", "mg", "mw", "nA", "nQ", "ng", "nw",  // 0x98-0x9F
      "oA", "oQ", "og", "ow", "pA", "pQ", "pg", "pw",  // 0xA0-0xA7
      "qA", "qQ", "qg", "qw", "rA", "rQ", "rg", "rw",  // 0xA8-0xAF
      "sA", "sQ", "sg", "sw", "tA", "tQ", "tg", "tw",  // 0xB0-0xB7
      "uA", "uQ", "ug", "uw", "vA", "vQ", "vg", "vw",  // 0xB8-0xBF
      "wA", "wQ", "wg", "ww", "xA", "xQ", "xg", "xw",  // 0xC0-0xC7
      "yA", "yQ", "yg", "yw", "zA", "zQ", "zg", "zw",  // 0xC8-0xCF
      "0A", "0Q", "0g", "0w", "1A", "1Q", "1g", "1w",  // 0xD0-0xD7
      "2A", "2Q", "2g", "2w", "3A", "3Q", "3g", "3w",  // 0xD8-0xDF
      "4A", "4Q", "4g", "4w", "5A", "5Q", "5g", "5w",  // 0xE0-0xE7
      "6A", "6Q", "6g", "6w", "7A", "7Q", "7g", "7w",  // 0xE8-0xEF
      "8A", "8Q", "8g", "8w", "9A", "9Q", "9g", "9w",  // 0xF0-0xF7
      "+A", "+Q", "+g", "+w", "/A", "/Q", "/g", "/w",  // 0xF8-0xFF
  };

  std::string result;
  result.reserve(8192);

  // Ranks 0-255: single-byte tokens in rotated order.
  // Rank r maps to byte ((r + 0x21) & 0xFF).
  for (int rank = 0; rank < 256; ++rank) {
    uint8_t byte_value = (uint8_t)((rank + 0x21) & 0xFF);
    result += kSingleByteBase64[byte_value];
    result += " ";
    result += std::to_string(rank);
    result += "\n";
  }

  // Merges use the same raw byte sequences as the identity-mapped version.
  // The byte-to-rank mapping is different, but merge reconstruction handles it.

  // Rank 256: "in" (bytes 0x69, 0x6E) -> base64("in") = "aW4="
  result += "aW4= 256\n";

  // Rank 257: " t" (bytes 0x20, 0x74) -> base64(" t") = "IHQ="
  result += "IHQ= 257\n";

  // Rank 258: "th" (bytes 0x74, 0x68) -> base64("th") = "dGg="
  result += "dGg= 258\n";

  // Rank 259: " th" (bytes 0x20, 0x74, 0x68) -> base64(" th") = "IHRo"
  result += "IHRo 259\n";

  return result;
}

// Simple regex that matches words and whitespace (GPT-2 style, simplified).
static const char kTestPattern[] =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
    "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

class TiktokenTest : public ::testing::Test {
 protected:
  void SetUp() override { allocator_ = iree_allocator_system(); }

  void TearDown() override {}

  iree_allocator_t allocator_;
};

//===----------------------------------------------------------------------===//
// Parsing Tests
//===----------------------------------------------------------------------===//

TEST_F(TiktokenTest, ParseEmptyInput) {
  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/{kTestPattern, strlen(kTestPattern)},
      /*.special_token_count=*/0,
      /*.special_token_strings=*/nullptr,
      /*.special_token_ids=*/nullptr,
  };

  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_parse_tiktoken(iree_string_view_empty(),
                                                      &config, &builder));

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TiktokenTest, ParseInvalidBase64) {
  // Invalid base64 character in token.
  const char* data = "Z!== 0\n";

  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/{kTestPattern, strlen(kTestPattern)},
      /*.special_token_count=*/0,
      /*.special_token_strings=*/nullptr,
      /*.special_token_ids=*/nullptr,
  };

  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_parse_tiktoken(
                            iree_make_cstring_view(data), &config, &builder));

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TiktokenTest, ParseNonContiguousRanks) {
  // Ranks 0, 2 (missing 1).
  const char* data = "AA 0\nAg 2\n";

  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/{kTestPattern, strlen(kTestPattern)},
      /*.special_token_count=*/0,
      /*.special_token_strings=*/nullptr,
      /*.special_token_ids=*/nullptr,
  };

  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_parse_tiktoken(
                            iree_make_cstring_view(data), &config, &builder));

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TiktokenTest, ParseMissingRank) {
  const char* data = "AA\n";

  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/{kTestPattern, strlen(kTestPattern)},
      /*.special_token_count=*/0,
      /*.special_token_strings=*/nullptr,
      /*.special_token_ids=*/nullptr,
  };

  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_parse_tiktoken(
                            iree_make_cstring_view(data), &config, &builder));

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TiktokenTest, ParseEmptyPattern) {
  const char* data = "AA 0\n";

  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/iree_string_view_empty(),
      /*.special_token_count=*/0,
      /*.special_token_strings=*/nullptr,
      /*.special_token_ids=*/nullptr,
  };

  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_parse_tiktoken(
                            iree_make_cstring_view(data), &config, &builder));

  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Full Tokenizer Construction
//===----------------------------------------------------------------------===//

TEST_F(TiktokenTest, ConstructFromMinimalData) {
  std::string data = GenerateMinimalTiktokenData();

  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/{kTestPattern, strlen(kTestPattern)},
      /*.special_token_count=*/0,
      /*.special_token_strings=*/nullptr,
      /*.special_token_ids=*/nullptr,
  };

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_tiktoken(
      iree_make_string_view(data.data(), data.size()), &config, allocator_,
      &tokenizer));

  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TiktokenTest, ConstructWithShuffledByteOrder) {
  std::string data = GenerateShuffledTiktokenData();

  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/{kTestPattern, strlen(kTestPattern)},
      /*.special_token_count=*/0,
      /*.special_token_strings=*/nullptr,
      /*.special_token_ids=*/nullptr,
  };

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_tiktoken(
      iree_make_string_view(data.data(), data.size()), &config, allocator_,
      &tokenizer));

  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TiktokenTest, ConstructWithSpecialTokens) {
  std::string data = GenerateMinimalTiktokenData();

  static const iree_string_view_t special_strings[] = {
      IREE_SVL("<|endoftext|>"),
  };
  static const int32_t special_ids[] = {260};

  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/{kTestPattern, strlen(kTestPattern)},
      /*.special_token_count=*/1,
      /*.special_token_strings=*/special_strings,
      /*.special_token_ids=*/special_ids,
  };

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_tiktoken(
      iree_make_string_view(data.data(), data.size()), &config, allocator_,
      &tokenizer));

  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Predefined Configs
//===----------------------------------------------------------------------===//

TEST(TiktokenConfig, Cl100kBaseConfig) {
  const iree_tokenizer_tiktoken_config_t* config =
      iree_tokenizer_tiktoken_config_cl100k_base();
  EXPECT_GT(config->pattern.size, 0u);
  EXPECT_EQ(config->special_token_count, 5u);
  EXPECT_EQ(config->special_token_ids[0], 100257);
}

TEST(TiktokenConfig, O200kBaseConfig) {
  const iree_tokenizer_tiktoken_config_t* config =
      iree_tokenizer_tiktoken_config_o200k_base();
  EXPECT_GT(config->pattern.size, 0u);
  EXPECT_EQ(config->special_token_count, 2u);
  EXPECT_EQ(config->special_token_ids[0], 199999);
}

TEST(TiktokenConfig, R50kBaseConfig) {
  const iree_tokenizer_tiktoken_config_t* config =
      iree_tokenizer_tiktoken_config_r50k_base();
  EXPECT_GT(config->pattern.size, 0u);
  EXPECT_EQ(config->special_token_count, 1u);
  EXPECT_EQ(config->special_token_ids[0], 50256);
}

TEST(TiktokenConfig, P50kBaseConfig) {
  const iree_tokenizer_tiktoken_config_t* config =
      iree_tokenizer_tiktoken_config_p50k_base();
  EXPECT_GT(config->pattern.size, 0u);
  EXPECT_EQ(config->special_token_count, 1u);
  EXPECT_EQ(config->special_token_ids[0], 50281);
}

//===----------------------------------------------------------------------===//
// Graceful Input Handling
//===----------------------------------------------------------------------===//

TEST_F(TiktokenTest, TrailingNewlineHandled) {
  // Tiktoken files typically end with a trailing newline.
  std::string data = GenerateMinimalTiktokenData();
  // Data already ends with newline from the generator. Verify it works.
  ASSERT_EQ(data.back(), '\n');

  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/{kTestPattern, strlen(kTestPattern)},
      /*.special_token_count=*/0,
      /*.special_token_strings=*/nullptr,
      /*.special_token_ids=*/nullptr,
  };

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_tiktoken(
      iree_make_string_view(data.data(), data.size()), &config, allocator_,
      &tokenizer));
  iree_tokenizer_free(tokenizer);
}

TEST_F(TiktokenTest, EmptyLinesSkipped) {
  std::string data = GenerateMinimalTiktokenData();
  // Insert some empty lines.
  data = "\n\n" + data;

  iree_tokenizer_tiktoken_config_t config = {
      /*.pattern=*/{kTestPattern, strlen(kTestPattern)},
      /*.special_token_count=*/0,
      /*.special_token_strings=*/nullptr,
      /*.special_token_ids=*/nullptr,
  };

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_tiktoken(
      iree_make_string_view(data.data(), data.size()), &config, allocator_,
      &tokenizer));
  iree_tokenizer_free(tokenizer);
}

}  // namespace
