// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab/vocab_trie.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Helper to build a string table from a list of strings.
// Returns the string table and populates tokens with offsets/lengths.
std::vector<uint8_t> BuildStringTable(
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

// Helper to traverse a trie with a cursor and collect all matching token IDs.
std::vector<std::pair<int32_t, size_t>> TraverseTrie(
    const iree_tokenizer_vocab_trie_t* trie, const std::string& input) {
  std::vector<std::pair<int32_t, size_t>> matches;
  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, trie);

  for (size_t i = 0; i < input.size(); ++i) {
    if (!iree_tokenizer_trie_cursor_advance(&cursor,
                                            static_cast<uint8_t>(input[i]))) {
      break;
    }
    int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
    if (token_id >= 0) {
      matches.push_back({token_id, i + 1});  // length = i + 1
    }
  }
  return matches;
}

TEST(VocabTrieTest, EmptyVocab) {
  iree_tokenizer_vocab_trie_t* trie = nullptr;
  iree_const_byte_span_t empty_table = {nullptr, 0};
  IREE_ASSERT_OK(iree_tokenizer_vocab_trie_build(
      nullptr, 0, empty_table, iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_trie_max_depth(trie), 0u);
  EXPECT_GE(iree_tokenizer_vocab_trie_node_count(trie), 1u);  // At least root.

  // Traversal should find nothing.
  auto matches = TraverseTrie(trie, "hello");
  EXPECT_TRUE(matches.empty());

  iree_tokenizer_vocab_trie_free(trie);
}

TEST(VocabTrieTest, SingleToken) {
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({"hello"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_trie_build(tokens.data(), tokens.size(), table_span,
                                      iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_trie_max_depth(trie), 5u);

  // Should find "hello" at position 5.
  auto matches = TraverseTrie(trie, "hello");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 0);    // token ID
  EXPECT_EQ(matches[0].second, 5u);  // length

  // Should find "hello" as prefix of longer string.
  matches = TraverseTrie(trie, "helloworld");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 0);
  EXPECT_EQ(matches[0].second, 5u);

  // Should not find "hell" (not a complete token).
  matches = TraverseTrie(trie, "hell");
  EXPECT_TRUE(matches.empty());

  // Should not find "world".
  matches = TraverseTrie(trie, "world");
  EXPECT_TRUE(matches.empty());

  iree_tokenizer_vocab_trie_free(trie);
}

TEST(VocabTrieTest, MultipleTokensNoSharedPrefix) {
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({"the", "quick", "brown", "fox"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_trie_build(tokens.data(), tokens.size(), table_span,
                                      iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_trie_max_depth(trie),
            5u);  // "quick" or "brown"

  auto matches = TraverseTrie(trie, "the");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 0);

  matches = TraverseTrie(trie, "quick");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 1);

  matches = TraverseTrie(trie, "brown");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 2);

  matches = TraverseTrie(trie, "fox");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 3);

  iree_tokenizer_vocab_trie_free(trie);
}

TEST(VocabTrieTest, SharedPrefixes) {
  // Tokens that share prefixes: "a", "ab", "abc", "abcd"
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({"a", "ab", "abc", "abcd"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_trie_build(tokens.data(), tokens.size(), table_span,
                                      iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_trie_max_depth(trie), 4u);

  // Traversing "abcd" should find all four tokens.
  auto matches = TraverseTrie(trie, "abcd");
  ASSERT_EQ(matches.size(), 4u);
  EXPECT_EQ(matches[0].first, 0);  // "a" at length 1
  EXPECT_EQ(matches[0].second, 1u);
  EXPECT_EQ(matches[1].first, 1);  // "ab" at length 2
  EXPECT_EQ(matches[1].second, 2u);
  EXPECT_EQ(matches[2].first, 2);  // "abc" at length 3
  EXPECT_EQ(matches[2].second, 3u);
  EXPECT_EQ(matches[3].first, 3);  // "abcd" at length 4
  EXPECT_EQ(matches[3].second, 4u);

  // Traversing "abc" should find three tokens.
  matches = TraverseTrie(trie, "abc");
  ASSERT_EQ(matches.size(), 3u);

  // Traversing "ab" should find two tokens.
  matches = TraverseTrie(trie, "ab");
  ASSERT_EQ(matches.size(), 2u);

  // Traversing "a" should find one token.
  matches = TraverseTrie(trie, "a");
  ASSERT_EQ(matches.size(), 1u);

  iree_tokenizer_vocab_trie_free(trie);
}

TEST(VocabTrieTest, BranchingTokens) {
  // Tokens that branch: "cat", "car", "cab", "can"
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({"cat", "car", "cab", "can"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_trie_build(tokens.data(), tokens.size(), table_span,
                                      iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  // All tokens have same length.
  EXPECT_EQ(iree_tokenizer_vocab_trie_max_depth(trie), 3u);

  auto matches = TraverseTrie(trie, "cat");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 0);

  matches = TraverseTrie(trie, "car");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 1);

  matches = TraverseTrie(trie, "cab");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 2);

  matches = TraverseTrie(trie, "can");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 3);

  // "ca" is not a token.
  matches = TraverseTrie(trie, "ca");
  EXPECT_TRUE(matches.empty());

  // "cup" doesn't exist.
  matches = TraverseTrie(trie, "cup");
  EXPECT_TRUE(matches.empty());

  iree_tokenizer_vocab_trie_free(trie);
}

TEST(VocabTrieTest, LongToken) {
  std::string long_token(100, 'x');
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({long_token, "short"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_trie_build(tokens.data(), tokens.size(), table_span,
                                      iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_trie_max_depth(trie), 100u);

  auto matches = TraverseTrie(trie, long_token);
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 0);
  EXPECT_EQ(matches[0].second, 100u);

  matches = TraverseTrie(trie, "short");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 1);

  iree_tokenizer_vocab_trie_free(trie);
}

TEST(VocabTrieTest, BinaryData) {
  // Test with binary data (non-ASCII bytes).
  std::string binary_token;
  binary_token.push_back('\x00');
  binary_token.push_back('\xFF');
  binary_token.push_back('\x80');

  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({binary_token, "ascii"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_trie_build(tokens.data(), tokens.size(), table_span,
                                      iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  // Manual traversal for binary data.
  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, trie);
  EXPECT_TRUE(iree_tokenizer_trie_cursor_advance(&cursor, 0x00));
  EXPECT_TRUE(iree_tokenizer_trie_cursor_advance(&cursor, 0xFF));
  EXPECT_TRUE(iree_tokenizer_trie_cursor_advance(&cursor, 0x80));
  EXPECT_EQ(iree_tokenizer_trie_cursor_token_id(&cursor), 0);
  EXPECT_EQ(iree_tokenizer_trie_cursor_depth(&cursor), 3u);

  iree_tokenizer_vocab_trie_free(trie);
}

TEST(VocabTrieTest, ManyTokens) {
  // Create many tokens to stress test the trie.
  std::vector<std::string> strings;
  for (int i = 0; i < 1000; ++i) {
    strings.push_back("token_" + std::to_string(i));
  }

  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable(strings, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_trie_build(tokens.data(), tokens.size(), table_span,
                                      iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  // Verify we can find all tokens.
  // Note: Some tokens are prefixes of others (e.g., "token_1" is prefix of
  // "token_10"), so we may find multiple matches. We just verify the exact
  // token is found as the LAST match (longest match).
  for (int i = 0; i < 1000; ++i) {
    std::string expected = "token_" + std::to_string(i);
    auto matches = TraverseTrie(trie, expected);
    ASSERT_GE(matches.size(), 1u) << "Failed to find: " << expected;
    // The last match should be the exact token.
    EXPECT_EQ(matches.back().first, i) << "Token ID mismatch for: " << expected;
    EXPECT_EQ(matches.back().second, expected.size())
        << "Token length mismatch for: " << expected;
  }

  // Verify non-existent tokens aren't found as exact matches.
  // "token_1000" will match "token_1" and "token_100" as prefixes.
  auto matches = TraverseTrie(trie, "token_1000");
  // Should find token_1 (7 bytes), token_10 (8 bytes), token_100 (9 bytes)
  // but NOT token_1000 (10 bytes).
  for (const auto& match : matches) {
    EXPECT_NE(match.first, 1000) << "Should not find token_1000";
  }

  iree_tokenizer_vocab_trie_free(trie);
}

TEST(VocabTrieTest, CursorResetMidTraversal) {
  std::vector<iree_tokenizer_token_t> tokens;
  auto table = BuildStringTable({"abc", "xyz"}, &tokens);
  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_trie_build(tokens.data(), tokens.size(), table_span,
                                      iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, trie);

  // Start traversing "abc".
  EXPECT_TRUE(iree_tokenizer_trie_cursor_advance(&cursor, 'a'));
  EXPECT_EQ(iree_tokenizer_trie_cursor_depth(&cursor), 1u);

  // Reset mid-traversal.
  iree_tokenizer_trie_cursor_reset(&cursor, trie);
  EXPECT_EQ(iree_tokenizer_trie_cursor_depth(&cursor), 0u);

  // Now traverse "xyz".
  EXPECT_TRUE(iree_tokenizer_trie_cursor_advance(&cursor, 'x'));
  EXPECT_TRUE(iree_tokenizer_trie_cursor_advance(&cursor, 'y'));
  EXPECT_TRUE(iree_tokenizer_trie_cursor_advance(&cursor, 'z'));
  EXPECT_EQ(iree_tokenizer_trie_cursor_token_id(&cursor), 1);

  iree_tokenizer_vocab_trie_free(trie);
}

TEST(VocabTrieTest, NullTrieHandling) {
  // Operations on null trie should be safe.
  EXPECT_EQ(iree_tokenizer_vocab_trie_max_depth(nullptr), 0u);
  EXPECT_EQ(iree_tokenizer_vocab_trie_node_count(nullptr), 0u);

  // Cursor with null trie should not crash.
  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, nullptr);
  EXPECT_FALSE(iree_tokenizer_trie_cursor_advance(&cursor, 'a'));
  EXPECT_EQ(iree_tokenizer_trie_cursor_token_id(&cursor), -1);

  // Free null should not crash.
  iree_tokenizer_vocab_trie_free(nullptr);
}

TEST(VocabTrieTest, UnusedTokensSkipped) {
  // Create tokens where some are marked as UNUSED (gap tokens).
  std::vector<uint8_t> table;
  std::vector<iree_tokenizer_token_t> tokens;

  // Token 0: "hello" (normal)
  {
    iree_tokenizer_token_t token = {};
    token.string_offset = static_cast<uint32_t>(table.size());
    token.string_length = 5;
    token.attributes = IREE_TOKENIZER_TOKEN_ATTR_NONE;
    tokens.push_back(token);
    table.insert(table.end(), {'h', 'e', 'l', 'l', 'o'});
  }

  // Token 1: "gap" (UNUSED - should be skipped)
  {
    iree_tokenizer_token_t token = {};
    token.string_offset = static_cast<uint32_t>(table.size());
    token.string_length = 3;
    token.attributes = IREE_TOKENIZER_TOKEN_ATTR_UNUSED;
    tokens.push_back(token);
    table.insert(table.end(), {'g', 'a', 'p'});
  }

  // Token 2: "world" (normal)
  {
    iree_tokenizer_token_t token = {};
    token.string_offset = static_cast<uint32_t>(table.size());
    token.string_length = 5;
    token.attributes = IREE_TOKENIZER_TOKEN_ATTR_NONE;
    tokens.push_back(token);
    table.insert(table.end(), {'w', 'o', 'r', 'l', 'd'});
  }

  iree_const_byte_span_t table_span = {table.data(), table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_trie_build(tokens.data(), tokens.size(), table_span,
                                      iree_allocator_system(), &trie));
  ASSERT_NE(trie, nullptr);

  // "hello" should be found with ID 0.
  auto matches = TraverseTrie(trie, "hello");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 0);

  // "gap" should NOT be found (it's UNUSED).
  matches = TraverseTrie(trie, "gap");
  EXPECT_TRUE(matches.empty());

  // "world" should be found with ID 2.
  matches = TraverseTrie(trie, "world");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].first, 2);

  iree_tokenizer_vocab_trie_free(trie);
}

}  // namespace
