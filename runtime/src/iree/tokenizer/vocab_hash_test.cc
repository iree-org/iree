// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab_hash.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Helper to build a string table from a list of strings.
// Returns the string table and populates tokens with offsets/lengths.
static std::vector<uint8_t> BuildStringTable(
    const std::vector<std::string>& strings,
    std::vector<iree_tokenizer_token_t>* tokens) {
  std::vector<uint8_t> table;
  tokens->clear();
  tokens->reserve(strings.size());
  for (const auto& s : strings) {
    iree_tokenizer_token_t token = {};
    token.string_offset = static_cast<uint32_t>(table.size());
    token.string_length = static_cast<uint16_t>(s.size());
    token.attributes = IREE_TOKENIZER_TOKEN_ATTR_NONE;
    tokens->push_back(token);
    table.insert(table.end(), s.begin(), s.end());
  }
  return table;
}

TEST(VocabHashTest, EmptyVocab) {
  iree_tokenizer_vocab_hash_t* hash = nullptr;
  iree_const_byte_span_t empty_table = {nullptr, 0};
  IREE_ASSERT_OK(iree_tokenizer_vocab_hash_build(
      nullptr, 0, empty_table, IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT,
      iree_allocator_system(), &hash));
  ASSERT_NE(hash, nullptr);

  // Lookups should return -1.
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("")), -1);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("hello")), -1);

  iree_tokenizer_vocab_hash_free(hash);
}

TEST(VocabHashTest, SingleToken) {
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({"hello"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_hash_build(
      tokens.data(), tokens.size(), table_span,
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));
  ASSERT_NE(hash, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("hello")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("world")), -1);

  iree_tokenizer_vocab_hash_free(hash);
}

TEST(VocabHashTest, MultipleTokens) {
  std::vector<iree_tokenizer_token_t> tokens;
  auto table =
      BuildStringTable({"the", "quick", "brown", "fox", "jumps"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_hash_build(
      tokens.data(), tokens.size(), table_span,
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));
  ASSERT_NE(hash, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("the")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("quick")), 1);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("brown")), 2);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("fox")), 3);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("jumps")), 4);

  // Non-existent tokens.
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("lazy")), -1);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("dog")), -1);

  iree_tokenizer_vocab_hash_free(hash);
}

TEST(VocabHashTest, EmptyStringToken) {
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({"", "nonempty"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_hash_build(
      tokens.data(), tokens.size(), table_span,
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));
  ASSERT_NE(hash, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("nonempty")), 1);

  iree_tokenizer_vocab_hash_free(hash);
}

TEST(VocabHashTest, LongStringToken) {
  std::string long_string(1000, 'x');
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({long_string, "short"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_hash_build(
      tokens.data(), tokens.size(), table_span,
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));
  ASSERT_NE(hash, nullptr);

  EXPECT_EQ(
      iree_tokenizer_vocab_hash_lookup(
          hash, iree_make_string_view(long_string.data(), long_string.size())),
      0);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("short")), 1);

  iree_tokenizer_vocab_hash_free(hash);
}

TEST(VocabHashTest, CollisionHandling) {
  // Create many tokens to force collisions in the hash table.
  std::vector<std::string> strings;
  for (int i = 0; i < 100; ++i) {
    strings.push_back("token_" + std::to_string(i));
  }

  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable(strings, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_hash_build(
      tokens.data(), tokens.size(), table_span,
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));
  ASSERT_NE(hash, nullptr);

  // Verify all tokens can be found.
  for (int i = 0; i < 100; ++i) {
    std::string expected = "token_" + std::to_string(i);
    EXPECT_EQ(
        iree_tokenizer_vocab_hash_lookup(
            hash, iree_make_string_view(expected.data(), expected.size())),
        i)
        << "Failed to find token: " << expected;
  }

  // Verify non-existent tokens return -1.
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("token_100")), -1);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("not_a_token")), -1);

  iree_tokenizer_vocab_hash_free(hash);
}

TEST(VocabHashTest, NullHashLookup) {
  // Lookup on null hash should return -1 (not crash).
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(nullptr, IREE_SV("test")), -1);
}

TEST(VocabHashTest, HighLoadFactorClamped) {
  // Test that 100% load factor is clamped to 90% to prevent infinite loops
  // when searching for non-existent keys in a full table.
  std::vector<std::string> strings;
  for (int i = 0; i < 10; ++i) {
    strings.push_back("token_" + std::to_string(i));
  }

  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable(strings, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  // Request 100% load factor - should be internally clamped to 90%.
  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_hash_build(tokens.data(), tokens.size(), table_span,
                                      100, iree_allocator_system(), &hash));
  ASSERT_NE(hash, nullptr);

  // All existing tokens should be found.
  for (int i = 0; i < 10; ++i) {
    std::string expected = "token_" + std::to_string(i);
    EXPECT_EQ(
        iree_tokenizer_vocab_hash_lookup(
            hash, iree_make_string_view(expected.data(), expected.size())),
        i);
  }

  // CRITICAL: Search for non-existent key must terminate (not infinite loop).
  // Before the fix, 100% load would cause infinite probing here.
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("not_found")), -1);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("token_999")), -1);

  iree_tokenizer_vocab_hash_free(hash);
}

TEST(VocabHashTest, ManyTokensWithHighLoad) {
  // Stress test with many tokens at high load to verify no infinite loops.
  std::vector<std::string> strings;
  for (int i = 0; i < 1000; ++i) {
    strings.push_back("tok" + std::to_string(i));
  }

  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable(strings, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  // Use 90% load (the maximum allowed).
  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_hash_build(tokens.data(), tokens.size(), table_span,
                                      90, iree_allocator_system(), &hash));
  ASSERT_NE(hash, nullptr);

  // Spot check some tokens.
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("tok0")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("tok500")), 500);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("tok999")), 999);

  // Search for many non-existent keys - must all terminate.
  for (int i = 1000; i < 1100; ++i) {
    std::string missing = "tok" + std::to_string(i);
    EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(
                  hash, iree_make_string_view(missing.data(), missing.size())),
              -1);
  }

  iree_tokenizer_vocab_hash_free(hash);
}

TEST(VocabHashTest, ZeroLoadFactorClamped) {
  // Test that 0% load factor is clamped to 1% (minimum valid).
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({"test"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_hash_build(tokens.data(), tokens.size(), table_span,
                                      0, iree_allocator_system(), &hash));
  ASSERT_NE(hash, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("test")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_hash_lookup(hash, IREE_SV("missing")), -1);

  iree_tokenizer_vocab_hash_free(hash);
}

}  // namespace
