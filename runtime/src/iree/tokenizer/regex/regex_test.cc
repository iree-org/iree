// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// End-to-end regex tests that verify compiled patterns match expected strings.
// These tests exercise the complete pipeline: pattern -> compile -> execute.
//
// For compiler-only tests (error cases, DFA structure), see compile_test.cc.
// For executor-only tests (hand-built DFAs), see exec_test.cc.

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/internal/unicode.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/regex/compile.h"

namespace {

//===----------------------------------------------------------------------===//
// Test Utilities
//===----------------------------------------------------------------------===//

// Helper to compile a pattern and verify it succeeds.
class CompiledPattern {
 public:
  explicit CompiledPattern(const char* pattern,
                           iree_tokenizer_regex_compile_flags_t flags = 0) {
    iree_tokenizer_regex_compile_error_t error = {0};
    status_ = iree_tokenizer_regex_compile_and_load(IREE_SV(pattern), flags,
                                                    iree_allocator_system(),
                                                    &dfa_, &storage_, &error);
    if (!iree_status_is_ok(status_) && error.message) {
      error_message_ = error.message;
      error_position_ = error.position;
    }
  }

  ~CompiledPattern() {
    iree_status_ignore(status_);
    if (storage_) {
      iree_allocator_free(iree_allocator_system(), storage_);
    }
  }

  bool ok() const { return iree_status_is_ok(status_); }
  iree_status_code_t status_code() const { return iree_status_code(status_); }
  const char* error_message() const { return error_message_.c_str(); }
  iree_host_size_t error_position() const { return error_position_; }

  const iree_tokenizer_regex_dfa_t* dfa() const {
    return ok() ? &dfa_ : nullptr;
  }

  // Match collection for easy testing.
  std::vector<std::pair<size_t, size_t>> FindMatches(const char* text) {
    matches_.clear();
    if (!ok()) return {};
    IREE_EXPECT_OK(
        iree_tokenizer_regex_exec(&dfa_, IREE_SV(text), MatchCallback, this));
    return matches_;
  }

  // Match texts for easy assertions.
  std::vector<std::string> FindMatchTexts(const char* text) {
    auto matches = FindMatches(text);
    std::vector<std::string> result;
    for (auto& m : matches) {
      result.push_back(std::string(text + m.first, m.second - m.first));
    }
    return result;
  }

 private:
  static iree_status_t MatchCallback(void* user_data,
                                     iree_tokenizer_regex_match_t match) {
    auto* self = static_cast<CompiledPattern*>(user_data);
    self->matches_.push_back({match.start, match.end});
    return iree_ok_status();
  }

  iree_status_t status_;
  iree_tokenizer_regex_dfa_t dfa_ = {};
  uint8_t* storage_ = nullptr;
  std::string error_message_;
  iree_host_size_t error_position_ = 0;
  std::vector<std::pair<size_t, size_t>> matches_;
};

//===----------------------------------------------------------------------===//
// Basic Matching Tests
//===----------------------------------------------------------------------===//

TEST(Regex, SingleLiteral) {
  CompiledPattern pat("a");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");
}

TEST(Regex, MultipleLiterals) {
  CompiledPattern pat("abc");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xabcyabcz");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "abc");
  EXPECT_EQ(matches[1], "abc");
}

TEST(Regex, NoMatch) {
  CompiledPattern pat("xyz");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc");
  EXPECT_EQ(matches.size(), 0);
}

//===----------------------------------------------------------------------===//
// Character Class Tests
//===----------------------------------------------------------------------===//

TEST(Regex, CharacterClass) {
  CompiledPattern pat("[abc]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xabbcay");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abbca");
}

TEST(Regex, CharacterClassRange) {
  CompiledPattern pat("[a-z]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("123abc456def789");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "abc");
  EXPECT_EQ(matches[1], "def");
}

TEST(Regex, CharacterClassNegated) {
  CompiledPattern pat("[^0-9]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc123def456");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "abc");
  EXPECT_EQ(matches[1], "def");
}

TEST(Regex, CharacterClassWithSpecials) {
  // Test character class with special chars that need escaping.
  CompiledPattern pat("[a\\-z]+");  // Literal hyphen via escape.
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a-z-a");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a-z-a");
}

TEST(Regex, CharacterClassHyphenAtStart) {
  // Hyphen at start of character class is literal.
  CompiledPattern pat("[-az]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a-z-a");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a-z-a");
}

TEST(Regex, CharacterClassHyphenAtEnd) {
  // Hyphen at end of character class is literal.
  CompiledPattern pat("[az-]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a-z-a");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a-z-a");
}

TEST(Regex, CharacterClassEscapedPunctuation) {
  // Escaped punctuation in character class should be treated as literal.
  // Many regex patterns (e.g., SigLIP) over-escape punctuation unnecessarily.
  CompiledPattern pat("[\\!\\~\\#]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a!~#b");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "!~#");
}

TEST(Regex, CharacterClassEscapedPunctuationExtended) {
  // Test more escaped punctuation characters.
  CompiledPattern pat("[\\$\\%\\&\\'\\(\\)\\*\\+]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("x$%&'()*+y");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "$%&'()*+");
}

TEST(Regex, CharacterClassSigLIPStyle) {
  // Pattern similar to SigLIP's normalizer: punctuation removal character class
  // with many escaped characters that don't technically need escaping.
  CompiledPattern pat(
      "[!\"\\#\\$%\\&'\\(\\)\\*\\+,\\-\\.:;=\\?@\\[\\\\\\]\\^_`\\{\\|\\}\\~]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello!world");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "!");

  matches = pat.FindMatchTexts("a~b");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "~");

  matches = pat.FindMatchTexts("test@example.com");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "@");
  EXPECT_EQ(matches[1], ".");
}

//===----------------------------------------------------------------------===//
// Nested Character Class Tests
//===----------------------------------------------------------------------===//

TEST(Regex, NestedCharClassSimple) {
  // [a[bc]] is equivalent to [abc].
  CompiledPattern pat("[a[bc]]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xabcyx");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc");
}

TEST(Regex, NestedCharClassEquivalent) {
  // Verify [a[bc]]+ and [abc]+ produce identical matches.
  CompiledPattern pat1("[a[bc]]+");
  CompiledPattern pat2("[abc]+");
  ASSERT_TRUE(pat1.ok()) << pat1.error_message();
  ASSERT_TRUE(pat2.ok()) << pat2.error_message();

  const char* test_strings[] = {"abc", "xabcyx", "a", "b", "c", "cab", "bac"};
  for (const char* str : test_strings) {
    auto matches1 = pat1.FindMatchTexts(str);
    auto matches2 = pat2.FindMatchTexts(str);
    EXPECT_EQ(matches1, matches2) << "Mismatch for input: " << str;
  }
}

TEST(Regex, NestedCharClassAdjacent) {
  // [a[bc][de]] is equivalent to [abcde].
  CompiledPattern pat("[a[bc][de]]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xabcdey");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abcde");
}

TEST(Regex, NestedCharClassDeep) {
  // [a[b[c]]] is equivalent to [abc].
  CompiledPattern pat("[a[b[c]]]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xabcyx");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc");
}

TEST(Regex, NestedCharClassWithRanges) {
  // [a[0-9]] is equivalent to [a0-9].
  CompiledPattern pat("[a[0-9]]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xa1b2c");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "a1");
  EXPECT_EQ(matches[1], "2");
}

TEST(Regex, NestedCharClassOuterNegated) {
  // [^a[bc]] is NOT {a, b, c}.
  CompiledPattern pat("[^a[bc]]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xyzabc123");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "xyz");
  EXPECT_EQ(matches[1], "123");
}

TEST(Regex, NestedCharClassInnerNegatedError) {
  // [a[^bc]] - nested negation is not supported.
  CompiledPattern pat("[a[^bc]]");
  EXPECT_FALSE(pat.ok());
  EXPECT_NE(std::string(pat.error_message()).find("negated"), std::string::npos)
      << "Error message should mention negation: " << pat.error_message();
}

TEST(Regex, NestedCharClassUnclosed) {
  // [a[bc - missing closing brackets.
  CompiledPattern pat("[a[bc");
  EXPECT_FALSE(pat.ok());
  EXPECT_NE(std::string(pat.error_message()).find("unterminated"),
            std::string::npos)
      << "Error message should mention unterminated: " << pat.error_message();
}

TEST(Regex, NestedCharClassEmpty) {
  // [[]] - inner class is empty, but outer has content from inner closing.
  // Actually, [] is empty which would be an error, so [a[]] has 'a' and
  // the inner [] tries to be empty. Let's test this edge case.
  CompiledPattern pat("[a[]]+");
  // The ']' after '[' becomes content of the inner class due to parsing rules.
  // This is a bit tricky - [] at the start has special handling.
  // For now, just verify it either parses or gives a clear error.
  if (pat.ok()) {
    // If it parses, the inner [] should be empty but the outer has 'a'.
    auto matches = pat.FindMatchTexts("xaaax");
    ASSERT_GE(matches.size(), 1);
  }
  // Either way is acceptable for this edge case.
}

TEST(Regex, NestedCharClassWithEscapes) {
  // [a[\n\t]] - nested class with escape sequences.
  CompiledPattern pat("[a[\\n\\t]]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xa\n\ta");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a\n\ta");
}

TEST(Regex, NestedCharClassWithShorthand) {
  // [a[\d]] - nested class with shorthand.
  CompiledPattern pat("[a[\\d]]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xa123ax");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a123a");
}

TEST(Regex, BloomStylePattern) {
  // BLOOM tokenizer uses patterns like [^(\s|[.,!?])]+
  // This tests the core nested class functionality.
  CompiledPattern pat("[^(\\s|[.,!?])]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("Hello, world!");
  // Should match "Hello" and "world" (excluding punctuation and spaces).
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "Hello");
  EXPECT_EQ(matches[1], "world");
}

//===----------------------------------------------------------------------===//
// Shorthand Class Tests
//===----------------------------------------------------------------------===//

TEST(Regex, ShorthandDigit) {
  CompiledPattern pat("\\d+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc123def456");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "123");
  EXPECT_EQ(matches[1], "456");
}

TEST(Regex, ShorthandNonDigit) {
  CompiledPattern pat("\\D+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc123def");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "abc");
  EXPECT_EQ(matches[1], "def");
}

TEST(Regex, ShorthandWord) {
  CompiledPattern pat("\\w+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello_world! foo_bar123");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "hello_world");
  EXPECT_EQ(matches[1], "foo_bar123");
}

TEST(Regex, ShorthandWhitespace) {
  CompiledPattern pat("\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello  world\t\nfoo");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "  ");
  EXPECT_EQ(matches[1], "\t\n");
}

TEST(Regex, ShorthandNonWhitespace) {
  CompiledPattern pat("\\S+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("  hello  world  ");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "hello");
  EXPECT_EQ(matches[1], "world");
}

//===----------------------------------------------------------------------===//
// Quantifier Tests
//===----------------------------------------------------------------------===//

TEST(Regex, QuantifierStar) {
  CompiledPattern pat("ab*c");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("ac abc abbc abbbc");
  ASSERT_EQ(matches.size(), 4);
  EXPECT_EQ(matches[0], "ac");
  EXPECT_EQ(matches[1], "abc");
  EXPECT_EQ(matches[2], "abbc");
  EXPECT_EQ(matches[3], "abbbc");
}

TEST(Regex, QuantifierPlus) {
  CompiledPattern pat("ab+c");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("ac abc abbc");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "abc");
  EXPECT_EQ(matches[1], "abbc");
}

TEST(Regex, QuantifierOptional) {
  CompiledPattern pat("ab?c");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("ac abc abbc");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "ac");
  EXPECT_EQ(matches[1], "abc");
}

TEST(Regex, QuantifierExact) {
  CompiledPattern pat("a{3}");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a aa aaa aaaa");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "aaa");
  EXPECT_EQ(matches[1], "aaa");  // From aaaa.
}

TEST(Regex, QuantifierRange) {
  CompiledPattern pat("a{2,4}");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a aa aaa aaaa aaaaa");
  ASSERT_EQ(matches.size(), 4);
  EXPECT_EQ(matches[0], "aa");
  EXPECT_EQ(matches[1], "aaa");
  EXPECT_EQ(matches[2], "aaaa");
  EXPECT_EQ(matches[3], "aaaa");  // Greedy, takes first 4 of 5.
}

TEST(Regex, QuantifierMinOnly) {
  CompiledPattern pat("a{2,}");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a aa aaa");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "aa");
  EXPECT_EQ(matches[1], "aaa");
}

//===----------------------------------------------------------------------===//
// Possessive Quantifier Tests
//===----------------------------------------------------------------------===//
// Possessive quantifiers (++, *+, ?+, {n,m}+) are identical to greedy
// quantifiers in DFA-based engines since DFAs don't backtrack. We accept
// the syntax for compatibility with tiktoken/PCRE patterns.

TEST(Regex, PossessivePlus) {
  CompiledPattern pat("a++");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("aaa b aa");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "aaa");
  EXPECT_EQ(matches[1], "aa");
}

TEST(Regex, PossessiveStar) {
  CompiledPattern pat("ba*+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("b ba baa baaa");
  ASSERT_EQ(matches.size(), 4);
  EXPECT_EQ(matches[0], "b");
  EXPECT_EQ(matches[1], "ba");
  EXPECT_EQ(matches[2], "baa");
  EXPECT_EQ(matches[3], "baaa");
}

TEST(Regex, PossessiveOptional) {
  CompiledPattern pat("colou?+r");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("color colour");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "color");
  EXPECT_EQ(matches[1], "colour");
}

TEST(Regex, PossessiveQuantifierBraces) {
  CompiledPattern pat("a{2,4}+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a aa aaa aaaa aaaaa");
  ASSERT_EQ(matches.size(), 4);
  EXPECT_EQ(matches[0], "aa");
  EXPECT_EQ(matches[1], "aaa");
  EXPECT_EQ(matches[2], "aaaa");
  EXPECT_EQ(matches[3], "aaaa");  // Greedy, takes first 4 of 5.
}

TEST(Regex, PossessiveWithUnicode) {
  // Test possessive with Unicode property (like tiktoken patterns).
  CompiledPattern pat("\\p{L}++");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("Hello 123 World");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "Hello");
  EXPECT_EQ(matches[1], "World");
}

TEST(Regex, PossessiveInCharClass) {
  // Test possessive with character class.
  CompiledPattern pat("[a-z]++");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc 123 def");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "abc");
  EXPECT_EQ(matches[1], "def");
}

//===----------------------------------------------------------------------===//
// Alternation Tests
//===----------------------------------------------------------------------===//

TEST(Regex, Alternation) {
  CompiledPattern pat("cat|dog");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("I have a cat and a dog");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "cat");
  EXPECT_EQ(matches[1], "dog");
}

TEST(Regex, AlternationMultiple) {
  CompiledPattern pat("a|bb|ccc");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xcccybbza");
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], "ccc");
  EXPECT_EQ(matches[1], "bb");
  EXPECT_EQ(matches[2], "a");
}

TEST(Regex, AlternationWithQuantifier) {
  CompiledPattern pat("(cat|dog)+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("catdogcat");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "catdogcat");
}

//===----------------------------------------------------------------------===//
// Group Tests
//===----------------------------------------------------------------------===//

TEST(Regex, GroupNonCapturing) {
  CompiledPattern pat("(?:ab)+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xabababy");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "ababab");
}

TEST(Regex, GroupCaseInsensitive) {
  CompiledPattern pat("(?i:abc)");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("ABC abc AbC");
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], "ABC");
  EXPECT_EQ(matches[1], "abc");
  EXPECT_EQ(matches[2], "AbC");
}

TEST(Regex, GlobalCaseInsensitiveLiteral) {
  CompiledPattern pat("hello",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello HELLO Hello hElLo HeLLO");
  ASSERT_EQ(matches.size(), 5);
  EXPECT_EQ(matches[0], "hello");
  EXPECT_EQ(matches[1], "HELLO");
  EXPECT_EQ(matches[2], "Hello");
  EXPECT_EQ(matches[3], "hElLo");
  EXPECT_EQ(matches[4], "HeLLO");
}

TEST(Regex, GlobalCaseInsensitiveConcat) {
  CompiledPattern pat("abc",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc ABC Abc aBc ABc aBC abC AbC");
  ASSERT_EQ(matches.size(), 8);
}

TEST(Regex, GlobalCaseInsensitiveAlternation) {
  CompiledPattern pat("cat|dog",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("CAT Dog cat DOG Cat dOg");
  ASSERT_EQ(matches.size(), 6);
}

TEST(Regex, GlobalCaseInsensitiveQuantifier) {
  CompiledPattern pat("a+",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("aaa AAA aAa AaA");
  ASSERT_EQ(matches.size(), 4);
  EXPECT_EQ(matches[0], "aaa");
  EXPECT_EQ(matches[1], "AAA");
  EXPECT_EQ(matches[2], "aAa");
  EXPECT_EQ(matches[3], "AaA");
}

TEST(Regex, GlobalCaseInsensitiveCharClassRange) {
  CompiledPattern pat("[a-c]+",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc ABC aBc");
  ASSERT_EQ(matches.size(), 3);
}

TEST(Regex, GlobalCaseInsensitiveCharClassExplicit) {
  CompiledPattern pat("[abc]+",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc ABC BCA");
  ASSERT_EQ(matches.size(), 3);
}

TEST(Regex, GlobalCaseInsensitiveGroup) {
  CompiledPattern pat("(?:abc)+",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abcABC ABCabc");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "abcABC");
  EXPECT_EQ(matches[1], "ABCabc");
}

TEST(Regex, GlobalCaseInsensitiveNestedGroups) {
  CompiledPattern pat("(?:a(?:bc))+",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abcABC");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abcABC");
}

TEST(Regex, GlobalCaseInsensitiveNumbersUnaffected) {
  CompiledPattern pat("a1b2c3",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a1b2c3 A1B2C3 a1B2c3");
  ASSERT_EQ(matches.size(), 3);
}

TEST(Regex, GlobalCaseInsensitiveSpecialsUnaffected) {
  CompiledPattern pat("a\\.b",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a.b A.B a.B A.b");
  ASSERT_EQ(matches.size(), 4);
}

TEST(Regex, InlineCaseInsensitiveStillWorks) {
  CompiledPattern pat("(?i:abc)");  // No global flag.
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc ABC AbC");
  ASSERT_EQ(matches.size(), 3);
}

TEST(Regex, GlobalFlagPlusInlineGroup) {
  CompiledPattern pat("(?i:abc)",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc ABC AbC");
  ASSERT_EQ(matches.size(), 3);
}

TEST(Regex, GlobalCaseInsensitiveContractions) {
  CompiledPattern pat("'s|'t|'re|'ve|'m|'ll|'d",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("I'M he's CAN'T I'LL");
  ASSERT_EQ(matches.size(), 4);
  EXPECT_EQ(matches[0], "'M");
  EXPECT_EQ(matches[1], "'s");
  EXPECT_EQ(matches[2], "'T");
  EXPECT_EQ(matches[3], "'LL");
}

TEST(Regex, GlobalCaseInsensitiveUnicodeUnaffected) {
  CompiledPattern pat("\\p{L}+",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Unicode property \p{L} already matches all letter cases.
  auto matches = pat.FindMatchTexts("hello HELLO");
  ASSERT_EQ(matches.size(), 2);
}

TEST(Regex, GlobalCaseInsensitiveWordShorthand) {
  CompiledPattern pat("\\w+",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // \w = [a-zA-Z0-9_], already includes both cases.
  auto matches = pat.FindMatchTexts("hello HELLO Hello123");
  ASSERT_EQ(matches.size(), 3);
}

TEST(Regex, GlobalCaseInsensitiveDotUnaffected) {
  CompiledPattern pat("a.c",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc ABC axc AXC a1c A1C");
  ASSERT_EQ(matches.size(), 6);
}

//===----------------------------------------------------------------------===//
// Dot Tests
//===----------------------------------------------------------------------===//

TEST(Regex, DotMatchesAnything) {
  CompiledPattern pat("a.c");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc a1c a c a\tc");
  ASSERT_EQ(matches.size(), 4);
  EXPECT_EQ(matches[0], "abc");
  EXPECT_EQ(matches[1], "a1c");
  EXPECT_EQ(matches[2], "a c");
  EXPECT_EQ(matches[3], "a\tc");
}

TEST(Regex, DotDoesNotMatchNewline) {
  CompiledPattern pat("a.c");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a\nc");
  EXPECT_EQ(matches.size(), 0);
}

//===----------------------------------------------------------------------===//
// Escape Tests
//===----------------------------------------------------------------------===//

TEST(Regex, EscapeNewline) {
  CompiledPattern pat("\\n");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a\nb\nc");
  ASSERT_EQ(matches.size(), 2);
}

TEST(Regex, EscapeTab) {
  CompiledPattern pat("\\t");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a\tb\tc");
  ASSERT_EQ(matches.size(), 2);
}

TEST(Regex, EscapeSpecialChars) {
  CompiledPattern pat("\\[\\]\\(\\)\\{\\}");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("x[](){}y");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "[](){}");
}

TEST(Regex, EscapeBackslash) {
  CompiledPattern pat("\\\\");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a\\b\\c");
  ASSERT_EQ(matches.size(), 2);
}

//===----------------------------------------------------------------------===//
// Negative Lookahead Tests
//===----------------------------------------------------------------------===//

TEST(Regex, NegativeLookaheadChar) {
  CompiledPattern pat("a(?!b)");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("ab ac ad");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "a");  // "ac" - 'a' at position 3.
  EXPECT_EQ(matches[1], "a");  // "ad" - 'a' at position 6.
}

TEST(Regex, NegativeLookaheadShorthand) {
  // Pattern: \s+(?!\S) - whitespace NOT immediately followed by non-whitespace.
  //
  // With correct regex backtracking semantics:
  // - Greedy \s+ consumes all whitespace
  // - Lookahead checks if next char is \S (non-whitespace)
  // - If lookahead fails, backtrack the quantifier and retry
  // - This means we match the longest run where lookahead passes at each step
  CompiledPattern pat("\\s+(?!\\S)");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "hello  world   "
  // Position 5-6: "  " - first space followed by space (passes), second by 'w'
  // (fails)
  //   -> Match " " at [5,6) due to backtracking
  // Position 12-14: "   " - at EOS, all pass
  //   -> Match "   " at [12,15)
  auto matches = pat.FindMatchTexts("hello  world   ");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], " ");    // Single space between words.
  EXPECT_EQ(matches[1], "   ");  // Trailing whitespace.
}

//===----------------------------------------------------------------------===//
// Anchor Tests (^ and $)
//===----------------------------------------------------------------------===//

TEST(Regex, StartAnchorBasic) {
  // ^abc should only match "abc" at the start of input.
  CompiledPattern pat("^abc");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Should match when "abc" is at start.
  auto matches = pat.FindMatchTexts("abc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc");

  // Should NOT match when "abc" is not at start.
  matches = pat.FindMatchTexts("xabc");
  EXPECT_EQ(matches.size(), 0);

  // Should NOT match in the middle.
  matches = pat.FindMatchTexts("xyzabc");
  EXPECT_EQ(matches.size(), 0);
}

TEST(Regex, StartAnchorWithTrailingText) {
  // ^abc should match "abc" at start even with trailing text.
  CompiledPattern pat("^abc");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abcdef");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc");
}

TEST(Regex, EndAnchorBasic) {
  // abc$ should only match "abc" at the end of input.
  CompiledPattern pat("abc$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Should match when "abc" is at end.
  auto matches = pat.FindMatchTexts("abc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc");

  // Should match "abc" at end with prefix.
  matches = pat.FindMatchTexts("xyzabc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc");

  // Should NOT match when there's text after "abc".
  matches = pat.FindMatchTexts("abcx");
  EXPECT_EQ(matches.size(), 0);
}

TEST(Regex, BothAnchorsFullMatch) {
  // ^abc$ should only match if the entire input is "abc".
  CompiledPattern pat("^abc$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Exact match.
  auto matches = pat.FindMatchTexts("abc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc");

  // Should NOT match with trailing text.
  matches = pat.FindMatchTexts("abcd");
  EXPECT_EQ(matches.size(), 0);

  // Should NOT match with leading text.
  matches = pat.FindMatchTexts("xabc");
  EXPECT_EQ(matches.size(), 0);

  // Should NOT match with both.
  matches = pat.FindMatchTexts("xabcy");
  EXPECT_EQ(matches.size(), 0);
}

TEST(Regex, StartAnchorWithQuantifier) {
  // ^a+ should match one or more 'a' at the start.
  CompiledPattern pat("^a+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("aaab");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "aaa");

  // Should NOT match if 'a' is not at start.
  matches = pat.FindMatchTexts("baaa");
  EXPECT_EQ(matches.size(), 0);
}

TEST(Regex, EndAnchorWithQuantifier) {
  // a+$ should match one or more 'a' at the end.
  CompiledPattern pat("a+$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("baaa");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "aaa");

  // Should NOT match if 'a' is not at end.
  matches = pat.FindMatchTexts("aaab");
  EXPECT_EQ(matches.size(), 0);
}

TEST(Regex, AnchorInAlternation) {
  // ^a|b$ - match 'a' at start OR 'b' at end.
  CompiledPattern pat("^a|b$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Match 'a' at start.
  auto matches = pat.FindMatchTexts("axxb");
  // 'a' at start should match, but 'b' at end also... depends on
  // implementation. In leftmost-longest semantics, we should get 'a' first,
  // then 'b'.
  ASSERT_GE(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");
}

//===----------------------------------------------------------------------===//
// Anchor Edge Cases (suggested by Gemini 3.0 Pro)
//===----------------------------------------------------------------------===//

TEST(Regex, AnchorPrecedenceInAlternation) {
  // ^a|b - the ^ only binds to 'a', so 'b' can match anywhere.
  CompiledPattern pat("^a|b");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Input "cba" - 'a' is not at start so ^a fails, but 'b' matches at index 1.
  auto matches = pat.FindMatchTexts("cba");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "b");

  // Input "abc" - 'a' is at start, so ^a matches.
  matches = pat.FindMatchTexts("abc");
  ASSERT_GE(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");
}

TEST(Regex, EndAnchorPrecedenceInAlternation) {
  // a|b$ - the $ only binds to 'b', so 'a' can match anywhere.
  CompiledPattern pat("a|b$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Input "axb" - 'b' is at end, 'a' also matches.
  auto matches = pat.FindMatchTexts("axb");
  ASSERT_GE(matches.size(), 2);
  EXPECT_EQ(matches[0], "a");
  EXPECT_EQ(matches[1], "b");

  // Input "bxa" - 'a' at end but b$ fails (b not at end), 'a' matches.
  matches = pat.FindMatchTexts("bxa");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");
}

TEST(Regex, NewlineAsLiteralWithStartAnchor) {
  // ^abc should NOT match if there's a newline at position 0.
  // Our engine uses string anchors, not line anchors.
  CompiledPattern pat("^abc");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Input starts with newline - should not match.
  auto matches = pat.FindMatchTexts("\nabc");
  EXPECT_EQ(matches.size(), 0);

  // Input "abc\n" - should match "abc" at start (newline after is irrelevant).
  matches = pat.FindMatchTexts("abc\n");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc");
}

TEST(Regex, NewlineAsLiteralWithEndAnchor) {
  // abc$ should NOT match if there's a newline after "abc".
  CompiledPattern pat("abc$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Input "abc\n" - 'abc' is NOT at end (newline follows).
  auto matches = pat.FindMatchTexts("abc\n");
  EXPECT_EQ(matches.size(), 0);

  // Input "\nabc" - 'abc' IS at end.
  matches = pat.FindMatchTexts("\nabc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc");
}

TEST(Regex, ImpossibleAnchorInMiddle) {
  // Pattern "a^b" has start anchor in middle - syntactically valid but
  // impossible to match (there's no position that is both after 'a' and at
  // start of string).
  CompiledPattern pat("a^b");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Should never match.
  auto matches = pat.FindMatchTexts("ab");
  EXPECT_EQ(matches.size(), 0);

  matches = pat.FindMatchTexts("a^b");  // Literal interpretation won't work.
  EXPECT_EQ(matches.size(), 0);
}

TEST(Regex, StartAnchorWithUnicodeChar) {
  // ^. with a multi-byte UTF-8 character at start.
  CompiledPattern pat("^.");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Euro sign (€ = 0xE2 0x82 0xAC, 3 bytes) at start.
  auto matches = pat.FindMatchTexts("€abc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "€");

  // 4-byte emoji at start.
  matches = pat.FindMatchTexts("😀abc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "😀");
}

TEST(Regex, EndAnchorWithUnicodeChar) {
  // .$ with a multi-byte UTF-8 character at end.
  CompiledPattern pat(".$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Euro sign at end.
  auto matches = pat.FindMatchTexts("abc€");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "€");

  // 4-byte emoji at end.
  matches = pat.FindMatchTexts("abc😀");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "😀");
}

TEST(Regex, GreedyPatternWithBothAnchors) {
  // ^.*$ matches the entire input.
  CompiledPattern pat("^.*$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello world");
  ASSERT_EQ(matches.size(), 1) << "Expected 1 match for ^.*$ on 'hello world'";
  EXPECT_EQ(matches[0], "hello world");

  // With newlines — . does not match newline by default.
  // So ^.*$ on "hello\nworld" should not match because .* stops at \n
  // and we're not at end of string.
  matches = pat.FindMatchTexts("hello\nworld");
  ASSERT_EQ(matches.size(), 0) << "Expected 0 matches: . doesn't match newline";
}

TEST(Regex, StartAnchorNoMatchInMiddle) {
  // Ensure ^ patterns don't accidentally match in middle of input.
  CompiledPattern pat("^foo");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "barfoo" has "foo" but not at start.
  auto matches = pat.FindMatchTexts("barfoo");
  EXPECT_EQ(matches.size(), 0);

  // "foobarfoo" has "foo" at start AND at end - only start should match.
  matches = pat.FindMatchTexts("foobarfoo");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "foo");
}

TEST(Regex, EndAnchorNoMatchAtStart) {
  // Ensure $ patterns don't match at start.
  CompiledPattern pat("foo$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "foobar" has "foo" but not at end.
  auto matches = pat.FindMatchTexts("foobar");
  EXPECT_EQ(matches.size(), 0);

  // "foobarfoo" has "foo" at start AND at end - only end should match.
  matches = pat.FindMatchTexts("foobarfoo");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "foo");
}

TEST(Regex, AnchorWithLookahead) {
  // ^a(?!b) - start anchor with negative lookahead.
  CompiledPattern pat("^a(?!b)");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "ac" - 'a' at start, followed by 'c' (not 'b'), should match.
  auto matches = pat.FindMatchTexts("ac");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");

  // "ab" - 'a' at start, followed by 'b', lookahead fails.
  matches = pat.FindMatchTexts("ab");
  EXPECT_EQ(matches.size(), 0);

  // "xac" - 'a' not at start, should not match.
  matches = pat.FindMatchTexts("xac");
  EXPECT_EQ(matches.size(), 0);
}

TEST(Regex, EndAnchorWithLookahead) {
  // This is an unusual pattern: a(?!b)$ - 'a' followed by nothing (end).
  // The lookahead (?!b) should pass because there's no 'b' at end.
  CompiledPattern pat("a(?!b)$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "xa" - 'a' at end, no 'b' follows, should match.
  auto matches = pat.FindMatchTexts("xa");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");

  // Note: "xab" would have 'b' at end, but "a$" wouldn't match anyway
  // because 'a' is not at end.
  matches = pat.FindMatchTexts("xab");
  EXPECT_EQ(matches.size(), 0);
}

//===----------------------------------------------------------------------===//
// Unicode Escape Tests (\uXXXX)
//===----------------------------------------------------------------------===//

TEST(Regex, UnicodeEscapeASCII) {
  // \u0041 = 'A', should work as literal.
  CompiledPattern pat("\\u0041+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("AAA BBB");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "AAA");
}

TEST(Regex, UnicodeEscapeInCharClassASCII) {
  // [\u0041-\u005A] = [A-Z]
  CompiledPattern pat("[\\u0041-\\u005A]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("HELLO world");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "HELLO");
}

TEST(Regex, UnicodeEscapeNonASCII) {
  // \u4E00 = '一' (CJK character "one"). Non-ASCII Unicode escapes now use
  // exact matching (single-element codepoint range) for precision.
  CompiledPattern pat("\\u4E00+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Should match only the exact character 一, not other CJK characters.
  auto matches = pat.FindMatchTexts("一一一 世界");
  ASSERT_GE(matches.size(), 1);
  EXPECT_EQ(matches[0], "一一一");  // Exact match only.
}

TEST(Regex, UnicodeEscapeInCharClassNonASCII) {
  // [\u4E00\u4E16\u754C] - character class with exact CJK codepoints.
  // Non-ASCII in character classes now use exact codepoint ranges.
  CompiledPattern pat("[\\u4E00\\u4E16\\u754C]+");  // 一, 世, 界
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Should match the exact characters specified, not all CJK.
  auto matches = pat.FindMatchTexts("一世界 中文");
  ASSERT_GE(matches.size(), 1);
  EXPECT_EQ(matches[0], "一世界");  // Only the specified characters.
}

TEST(Regex, UnicodeEscapeInvalid) {
  // Too few hex digits.
  CompiledPattern pat1("\\u00");
  EXPECT_FALSE(pat1.ok());
  EXPECT_TRUE(strstr(pat1.error_message(), "4 hex digits") != nullptr);

  // Invalid hex character.
  CompiledPattern pat2("\\u00GG");
  EXPECT_FALSE(pat2.ok());
}

TEST(Regex, UnicodeEscapeLowercase) {
  // Lowercase hex digits should work.
  CompiledPattern pat("\\u0061+");  // 'a'
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("aaa BBB");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "aaa");
}

TEST(Regex, UnicodeEscapeExactMatchNoFalsePositives) {
  // \u4E2D = 中. Should match only 中, not other CJK characters.
  CompiledPattern pat("\\u4E2D+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Should NOT match 国 or 文 (different CJK characters).
  auto no_matches = pat.FindMatchTexts("国文");
  EXPECT_EQ(no_matches.size(), 0);

  // Should match only 中.
  auto matches = pat.FindMatchTexts("中中中");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "中中中");
}

TEST(Regex, UnicodeEscapeSymbolExactMatch) {
  // \u263A = ☺ (white smiling face). Should match only ☺, not other symbols.
  CompiledPattern pat("\\u263A");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("Hello ☺ World");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "☺");

  // Should NOT match ☻ (black smiling face, U+263B).
  auto no_match = pat.FindMatchTexts("Hello ☻ World");
  EXPECT_EQ(no_match.size(), 0);
}

TEST(Regex, SingleNonASCIILiteralExactMatch) {
  // Single literal 中 in character class should match only 中.
  CompiledPattern pat("[中]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("中中中");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "中中中");

  // Should NOT match other CJK characters.
  auto no_match = pat.FindMatchTexts("国文");
  EXPECT_EQ(no_match.size(), 0);
}

TEST(Regex, FourSingleCodepointsWithinLimit) {
  // Exactly 4 single non-ASCII codepoints should work (at the limit).
  CompiledPattern pat("[一二三四]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("一二三四五");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "一二三四");  // 五 not matched.
}

TEST(Regex, ExceedingRangeLimitError) {
  // 11 single non-ASCII codepoints exceed the range limit.
  CompiledPattern pat("[一二三四五六七八九十百]+");
  EXPECT_FALSE(pat.ok());
  EXPECT_TRUE(strstr(pat.error_message(), "too many Unicode ranges") !=
              nullptr);
}

TEST(Regex, MixedRangesAndSingleCodepoints) {
  // Mix of Unicode range and single codepoints share the range pool.
  // [一-丁中] = range (4E00-4E01, 2 chars) + single (4E2D) = 2 ranges used.
  CompiledPattern pat("[一-丁中]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // 一 (4E00) and 丁 (4E01) are in range, 中 (4E2D) is single.
  auto matches = pat.FindMatchTexts("一丁中");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "一丁中");

  // 三 (4E09) is not in range and not a single codepoint.
  auto no_match = pat.FindMatchTexts("三");
  EXPECT_EQ(no_match.size(), 0);
}

TEST(Regex, MixedASCIIAndUnicodeInCharClass) {
  // Mixed ASCII and non-ASCII codepoints in same character class.
  // ASCII goes to bitmap, non-ASCII uses ranges.
  CompiledPattern pat("[abc中]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc中def");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc中");
}

//===----------------------------------------------------------------------===//
// Unicode Category Tests
//===----------------------------------------------------------------------===//

TEST(Regex, UnicodeLetter) {
  CompiledPattern pat("\\p{L}+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // ASCII letters work, plus we should see Unicode categories used.
  auto matches = pat.FindMatchTexts("hello 123 world");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "hello");
  EXPECT_EQ(matches[1], "world");
}

TEST(Regex, UnicodeLetterCJK) {
  // CJK ideographs are letters, so \p{L}+ should match them.
  CompiledPattern pat("\\p{L}+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Japanese text: Hiragana + Kanji (all are \p{L}).
  auto matches = pat.FindMatchTexts("こんにちは世界");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "こんにちは世界");

  // Mixed ASCII and CJK.
  matches = pat.FindMatchTexts("Hello こんにちは world 世界");
  ASSERT_EQ(matches.size(), 4);
  EXPECT_EQ(matches[0], "Hello");
  EXPECT_EQ(matches[1], "こんにちは");
  EXPECT_EQ(matches[2], "world");
  EXPECT_EQ(matches[3], "世界");
}

TEST(Regex, UnicodeNumber) {
  CompiledPattern pat("\\p{N}+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc 123 def 456");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "123");
  EXPECT_EQ(matches[1], "456");
}

TEST(Regex, UnicodeNotLetterOrNumber) {
  CompiledPattern pat("[^\\p{L}\\p{N}]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello! world?");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "! ");  // Includes space.
  EXPECT_EQ(matches[1], "?");
}

//===----------------------------------------------------------------------===//
// GPT-2 Pattern Tests
//===----------------------------------------------------------------------===//

// GPT-2 pattern (simplified - we test components):
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

TEST(Regex, GPT2Contractions) {
  CompiledPattern pat("'s|'t|'re|'ve|'m|'ll|'d");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("I'm can't you're we've he'll they'd");
  ASSERT_EQ(matches.size(), 6);
  EXPECT_EQ(matches[0], "'m");
  EXPECT_EQ(matches[1], "'t");
  EXPECT_EQ(matches[2], "'re");
  EXPECT_EQ(matches[3], "'ve");
  EXPECT_EQ(matches[4], "'ll");
  EXPECT_EQ(matches[5], "'d");
}

TEST(Regex, GPT2OptionalSpaceLetters) {
  CompiledPattern pat(" ?\\p{L}+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("Hello World");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "Hello");
  EXPECT_EQ(matches[1], " World");  // Space included.
}

TEST(Regex, GPT2OptionalSpaceNumbers) {
  CompiledPattern pat(" ?\\p{N}+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("test 123 456");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], " 123");  // Space included.
  EXPECT_EQ(matches[1], " 456");
}

TEST(Regex, GPT2TrailingWhitespace) {
  CompiledPattern pat("\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello world   ");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], " ");    // Between hello and world.
  EXPECT_EQ(matches[1], "   ");  // Trailing (matched by \s+(?!\S)).
}

//===----------------------------------------------------------------------===//
// Llama 3 Pattern Tests
//===----------------------------------------------------------------------===//

TEST(Regex, Llama3CaseInsensitiveContractions) {
  CompiledPattern pat("(?i:'s|'t|'re|'ve|'m|'ll|'d)");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("I'M CAN'T");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "'M");
  EXPECT_EQ(matches[1], "'T");
}

TEST(Regex, Llama3NumberRange) {
  CompiledPattern pat("\\p{N}{1,3}");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("1 12 123 1234");
  ASSERT_EQ(matches.size(), 5);
  EXPECT_EQ(matches[0], "1");
  EXPECT_EQ(matches[1], "12");
  EXPECT_EQ(matches[2], "123");
  EXPECT_EQ(matches[3], "123");  // First 3 of 1234.
  EXPECT_EQ(matches[4], "4");    // Remaining.
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST(Regex, SingleCharPattern) {
  CompiledPattern pat("x");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xxx");
  ASSERT_EQ(matches.size(), 3);
}

TEST(Regex, LongAlternation) {
  // Many alternatives.
  CompiledPattern pat("a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abcdefghijklmnop");
  EXPECT_EQ(matches.size(), 16);
}

TEST(Regex, NestedGroups) {
  CompiledPattern pat("((ab)+c)+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abcababc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abcababc");
}

TEST(Regex, ComplexCharacterClass) {
  CompiledPattern pat("[a-zA-Z0-9_]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("Hello_World_123!");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "Hello_World_123");
}

//===----------------------------------------------------------------------===//
// Real-World Pattern Smoke Tests
//===----------------------------------------------------------------------===//

// These test that the patterns compile without verifying exact behavior
// (since that depends on UTF-8 input handling).

TEST(Regex, GPT2FullPatternCompiles) {
  // GPT-2 pattern (ASCII portions testable).
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Basic smoke test.
  auto matches = pat.FindMatchTexts("Hello world!");
  EXPECT_GE(matches.size(), 1);
}

TEST(Regex, Llama3PatternCompiles) {
  CompiledPattern pat(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Regex, Qwen2PatternCompiles) {
  CompiledPattern pat(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();
}

//===----------------------------------------------------------------------===//
// PCRE Compatibility: Lookahead vs Fallback Semantics
//===----------------------------------------------------------------------===//
//
// These tests verify PCRE-compatible behavior for patterns with mixed
// lookahead branches. The key insight is that PCRE uses left-to-right
// alternation priority with backtracking.
//
// In PCRE:
//   \s+(?!\S)|\s+  with "   x" -> matches "  " (2 spaces)
//     - First branch \s+(?!\S) is tried
//     - It greedily matches 3 spaces, but lookahead (?!\S) fails (sees 'x')
//     - Backtracks to 2 spaces, lookahead passes (next char is space, not \S)
//     - Match ends at position 2
//
// For DFA with PCRE compatibility:
//   - When fallback branch comes AFTER lookahead: prefer lookahead-passed
//   - When non-lookahead branch comes BEFORE lookahead: prefer longer match
//   - The key is computing this per-DFA-state based on which branches accept

TEST(Regex, LookaheadFallback_SimpleTwoSpaces) {
  // Pattern: \s+(?!\S)|\s+ - lookahead at branch 0, fallback at branch 1.
  // Since fallback (1) > lookahead (0), prefer lookahead-passed position.
  CompiledPattern pat("\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "  x" - two spaces followed by 'x'.
  // Lookahead passes at 1 space (next is space), fails at 2 spaces (next is x).
  auto matches = pat.FindMatchTexts("  x");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], " ");  // 1 space (lookahead passed).
  EXPECT_EQ(matches[1], " ");  // 1 space (fallback before x).
}

TEST(Regex, LookaheadFallback_TrailingSpaces) {
  // Trailing spaces at end of string - lookahead (?!\S) passes at EOF.
  CompiledPattern pat("\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("end   ");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0],
            "   ");  // All 3 trailing spaces (lookahead passes at EOF).
}

TEST(Regex, LookaheadFallback_MixedContent) {
  // "hello  world   " - mixed spaces and words.
  CompiledPattern pat("\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello  world   ");
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], " ");    // 1 space (lookahead passed, next is space).
  EXPECT_EQ(matches[1], " ");    // 1 space before 'w' (fallback).
  EXPECT_EQ(matches[2], "   ");  // Trailing 3 spaces (lookahead passes at EOF).
}

TEST(Regex, IndependentBranch_NewlineOnly) {
  // Pattern: \s*[\r\n]+|\s+
  // For "\n\n" (only newlines), first branch matches all.
  CompiledPattern pat("\\s*[\\r\\n]+|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("\n\n");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "\n\n");
}

TEST(Regex, IndependentBranch_SpacesThenNewline) {
  // Pattern: \s*[\r\n]+|\s+
  // For "  \n", first branch matches all (spaces via \s*, newline via [\r\n]+).
  CompiledPattern pat("\\s*[\\r\\n]+|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("  \n");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "  \n");
}

TEST(Regex, Qwen2Pattern_ConsecutiveSpacesBeforeWord) {
  // Qwen2 pattern (simplified to relevant branches):
  // \s*[\r\n]+|\s+(?!\S)|\s+
  //
  // Input: "    print" (4 spaces before word)
  //
  // First branch (\s*[\r\n]+) cannot match - no newline.
  // So only branches 1 and 2 compete in the DFA state.
  // Branch 1 (lookahead) is at position 1, Branch 2 (fallback) at position 2.
  // Since min_no_lookahead (2) > min_lookahead (1): prefer lookahead-passed.
  CompiledPattern pat("\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("    print");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "   ");  // 3 spaces (lookahead passed, next is space).
  EXPECT_EQ(matches[1], " ");    // 1 space before 'p' (fallback).
}

TEST(Regex, Qwen2Pattern_NewlineThenIndent) {
  // \s*[\r\n]+|\s+(?!\S)|\s+ with "\n    x"
  //
  // First match: "\n" via branch 0 (newline-matching branch).
  // Second match: At position 1, only branches 1 and 2 compete (no newline).
  // Prefer lookahead-passed: "   " (3 spaces where lookahead passes).
  // Third match: " " before 'x' (fallback).
  CompiledPattern pat("\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("\n    x");
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], "\n");   // Newline alone.
  EXPECT_EQ(matches[1], "   ");  // 3 spaces (lookahead passed).
  EXPECT_EQ(matches[2], " ");    // 1 space before x (fallback).
}

TEST(Regex, Qwen2Pattern_DoubleNewlineInText) {
  // \s*[\r\n]+|\s+(?!\S)|\s+ with "a\n\nb"
  // Both newlines matched together by first branch.
  CompiledPattern pat("\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a\n\nb");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "\n\n");
}

TEST(Regex, Qwen2Pattern_TrailingWhitespace) {
  // \s*[\r\n]+|\s+(?!\S)|\s+ with "end   "
  // Trailing spaces - lookahead passes at EOF.
  CompiledPattern pat("\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("end   ");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "   ");
}

//===----------------------------------------------------------------------===//
// Additional Coverage Tests
//===----------------------------------------------------------------------===//

TEST(Regex, EscapedDot) {
  // Escaped dot should match literal '.' only.
  CompiledPattern pat("\\.");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a.b.c");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], ".");
  EXPECT_EQ(matches[1], ".");
}

TEST(Regex, EscapedPipe) {
  // Escaped pipe should match literal '|' not alternation.
  CompiledPattern pat("a\\|b");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a|b ab");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a|b");
}

TEST(Regex, EscapedStar) {
  // Escaped star should match literal '*'.
  CompiledPattern pat("a\\*b");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a*b ab aab");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a*b");
}

TEST(Regex, EscapedPlus) {
  // Escaped plus should match literal '+'.
  CompiledPattern pat("a\\+b");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a+b ab aab");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a+b");
}

TEST(Regex, EscapedQuestion) {
  // Escaped question should match literal '?'.
  CompiledPattern pat("a\\?b");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a?b ab");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a?b");
}

TEST(Regex, EscapedCaret) {
  // Escaped caret should match literal '^'.
  CompiledPattern pat("\\^abc");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("^abc abc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "^abc");
}

TEST(Regex, EscapedDollar) {
  // Escaped dollar should match literal '$'.
  CompiledPattern pat("abc\\$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc$ abc");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc$");
}

TEST(Regex, CarriageReturnEscape) {
  // \r should match carriage return.
  CompiledPattern pat("\\r\\n");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a\r\nb");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "\r\n");
}

TEST(Regex, CharacterClassWithBracket) {
  // ] at start of character class should be literal.
  CompiledPattern pat("[]a]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a]]]a");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a]]]a");
}

TEST(Regex, CharacterClassWithEscapedBracket) {
  // \] inside character class should be literal ].
  CompiledPattern pat("[a\\]]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a]a]");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a]a]");
}

TEST(Regex, CharacterClassWithBackslash) {
  // Escaped backslash inside character class.
  CompiledPattern pat("[a\\\\]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a\\a\\");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a\\a\\");
}

TEST(Regex, QuantifierZero) {
  // {0} should match empty string (zero repetitions).
  CompiledPattern pat("a{0}b");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "a{0}b" is equivalent to "b" - matches b.
  auto matches = pat.FindMatchTexts("b ab");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "b");
  EXPECT_EQ(matches[1], "b");
}

TEST(Regex, QuantifierZeroToN) {
  // {0,2} means 0 to 2 repetitions.
  CompiledPattern pat("xa{0,2}y");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xy xay xaay xaaay");
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], "xy");
  EXPECT_EQ(matches[1], "xay");
  EXPECT_EQ(matches[2], "xaay");
}

TEST(Regex, DeepNestedGroups) {
  // Multiple levels of nesting.
  CompiledPattern pat("((((a))))");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("aaa");
  ASSERT_EQ(matches.size(), 3);
}

TEST(Regex, AlternationInGroup) {
  // Alternation within a group.
  CompiledPattern pat("(a|b|c)+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abcabc xyz cba");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "abcabc");
  EXPECT_EQ(matches[1], "cba");
}

TEST(Regex, MultipleConsecutiveGroups) {
  // Groups one after another.
  CompiledPattern pat("(ab)(cd)(ef)");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("xabcdefy");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abcdef");
}

TEST(Regex, GroupWithQuantifierZeroOrMore) {
  // Group with * quantifier.
  CompiledPattern pat("(ab)*c");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("c abc ababc");
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], "c");
  EXPECT_EQ(matches[1], "abc");
  EXPECT_EQ(matches[2], "ababc");
}

TEST(Regex, DotPlusQuantifier) {
  // .+ matches one or more of anything (except newline).
  CompiledPattern pat(".+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello\nworld");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "hello");
  EXPECT_EQ(matches[1], "world");
}

TEST(Regex, CaseInsensitiveDoesNotAffectNumbers) {
  // Case insensitivity only affects letters.
  CompiledPattern pat("(?i:abc123)");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("ABC123 abc123 AbC123");
  ASSERT_EQ(matches.size(), 3);
}

TEST(Regex, UnicodePropertyPunctuation) {
  CompiledPattern pat("\\p{P}+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("Hello, world! How are you?");
  ASSERT_GE(matches.size(), 2);  // At least comma and ! and ?.
}

//===----------------------------------------------------------------------===//
// Stress Tests
//===----------------------------------------------------------------------===//

TEST(Regex, ManyAlternatives) {
  // 26 alternatives (all lowercase letters).
  CompiledPattern pat("a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("the quick brown fox");
  EXPECT_EQ(matches.size(), 16);  // "thequickbrownfox" - 16 letters.
}

TEST(Regex, LongLiteral) {
  // Very long literal string.
  CompiledPattern pat("abcdefghijklmnopqrstuvwxyz");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abcdefghijklmnopqrstuvwxyz");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abcdefghijklmnopqrstuvwxyz");
}

TEST(Regex, RepeatedComplexGroup) {
  // Complex group repeated multiple times.
  CompiledPattern pat("([a-z]+[0-9]+)+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("abc123def456");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abc123def456");
}

//===----------------------------------------------------------------------===//
// Real-World Tokenizer Patterns from HuggingFace Models
//===----------------------------------------------------------------------===//
// These patterns are extracted from actual tokenizer.json files to ensure
// our regex engine handles production tokenizer patterns correctly.

// Falcon-7B (tiiuae/falcon-7b) - Sequence pretokenizer with digit splitting.
TEST(Regex, HF_Falcon_DigitPattern) {
  // Falcon splits 3-digit number sequences.
  CompiledPattern pat("[0-9][0-9][0-9]");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("123456789");
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], "123");
  EXPECT_EQ(matches[1], "456");
  EXPECT_EQ(matches[2], "789");
}

// GPT-4 / tiktoken cl100k_base pattern.
// Uses possessive quantifiers (++), which are semantically identical to greedy
// quantifiers in DFA-based engines since DFAs don't backtrack.
TEST(Regex, Tiktoken_cl100k_base) {
  // Original tiktoken pattern from the tiktoken library.
  // Simplified to remove $ anchor (not applicable to find_all semantics).
  CompiledPattern pat(
      "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}++|\\p{N}{1,3}| "
      "?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s++");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Test basic tokenization.
  auto matches = pat.FindMatchTexts("Hello, world! I'm testing 123.");
  EXPECT_GE(matches.size(), 5);
}

// GPT-2 / RoBERTa / BART / Whisper (ByteLevel pretokenizer with use_regex=true)
// Pattern is hardcoded in tokenizers library, not in JSON.
TEST(Regex, HF_GPT2_FullPattern) {
  // The classic GPT-2 pattern used by ~18 models.
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Test basic tokenization.
  auto matches = pat.FindMatchTexts("Hello, world! I'm testing 123.");
  EXPECT_GE(matches.size(), 5);
}

// Llama 3 / Llama 3.1 / Llama 3.2 (Split pretokenizer)
// Used by: NousResearch/Meta-Llama-3.1-8B, unsloth/Llama-3.2-1B-Instruct
TEST(Regex, HF_Llama3_FullPattern) {
  CompiledPattern pat(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Llama 3 splits numbers into max 3 digits.
  auto matches = pat.FindMatchTexts("123456");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "123");
  EXPECT_EQ(matches[1], "456");
}

TEST(Regex, HF_Llama3_Contractions) {
  CompiledPattern pat(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Case-insensitive contractions: "I'M CAN'T YOU'RE" splits as:
  // "I", "'M", " CAN", "'T", " YOU", "'RE"
  // Note: The pattern [^\r\n\p{L}\p{N}]?\p{L}+ captures leading space with
  // word.
  auto matches = pat.FindMatchTexts("I'M CAN'T YOU'RE");
  ASSERT_EQ(matches.size(), 6);
  EXPECT_EQ(matches[0], "I");
  EXPECT_EQ(matches[1], "'M");
  EXPECT_EQ(matches[2], " CAN");  // Space captured with word.
  EXPECT_EQ(matches[3], "'T");
  EXPECT_EQ(matches[4], " YOU");  // Space captured with word.
  EXPECT_EQ(matches[5], "'RE");
}

// Qwen2 / Qwen2.5 / Qwen3 (Split pretokenizer)
// Used by: Qwen/Qwen2-0.5B, Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen3-0.6B
TEST(Regex, HF_Qwen2_FullPattern) {
  CompiledPattern pat(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Qwen splits each digit individually (not {1,3} like Llama).
  auto matches = pat.FindMatchTexts("123");
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], "1");
  EXPECT_EQ(matches[1], "2");
  EXPECT_EQ(matches[2], "3");
}

TEST(Regex, HF_Qwen2_MixedContent) {
  CompiledPattern pat(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("Hello world! 42 is the answer.");
  EXPECT_GE(matches.size(), 8);
}

// StableLM 2 (Split pretokenizer)
// Used by: stabilityai/stablelm-2-1_6b
TEST(Regex, HF_StableLM2_FullPattern) {
  // Same pattern as Qwen2 but with explicit \r\n.
  CompiledPattern pat(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Test newline handling.
  auto matches = pat.FindMatchTexts("line1\nline2");
  EXPECT_GE(matches.size(), 3);
}

// CLIP ViT Base (Split pretokenizer)
// Used by: openai/clip-vit-base-patch32
TEST(Regex, HF_CLIP_Base_FullPattern) {
  CompiledPattern pat(
      "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+"
      "|[\\p{N}]|[^\\s\\p{L}\\p{N}]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // CLIP special tokens.
  auto matches = pat.FindMatchTexts("<|startoftext|>hello<|endoftext|>");
  EXPECT_GE(matches.size(), 3);
}

// CLIP ViT Large (Split pretokenizer - without special tokens)
// Used by: openai/clip-vit-large-patch14
TEST(Regex, HF_CLIP_Large_FullPattern) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("A photo of a cat.");
  EXPECT_GE(matches.size(), 5);
}

// DeepSeek V3 patterns (complex multi-part pretokenizer)
// Used by: deepseek-ai/DeepSeek-V3
TEST(Regex, HF_DeepSeek_NumberPattern) {
  // Numbers in 1-3 digit chunks.
  CompiledPattern pat("\\p{N}{1,3}");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("12345678");
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], "123");
  EXPECT_EQ(matches[1], "456");
  EXPECT_EQ(matches[2], "78");
}

TEST(Regex, HF_DeepSeek_CJKPattern) {
  // CJK character range using script-specific pseudo-bytes.
  // The pattern [一-龥] matches CJK Unified Ideographs (U+4E00-U+9FFF).
  CompiledPattern pat("[一-龥]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Test with CJK text.
  auto matches = pat.FindMatchTexts("世界你好");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "世界你好");

  // Mixed content - CJK range should only match CJK, not Hiragana.
  matches = pat.FindMatchTexts("Hello 世界 こんにちは");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "世界");
}

TEST(Regex, UnicodeCharRange_Hiragana) {
  // Hiragana character range (U+3040-U+309F).
  CompiledPattern pat("[ぁ-ゟ]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("こんにちは世界");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "こんにちは");  // Only Hiragana, not the Kanji.
}

TEST(Regex, UnicodeCharRange_Katakana) {
  // Katakana character range (U+30A0-U+30FF).
  CompiledPattern pat("[ァ-ヿ]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("カタカナ ひらがな");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "カタカナ");
}

TEST(Regex, UnicodeCharRange_MultipleScripts) {
  // Combined pattern matching multiple script ranges.
  // This matches CJK OR Hiragana OR Katakana.
  CompiledPattern pat("[一-龥ぁ-ゟァ-ヿ]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // All Japanese scripts should match as one unit.
  auto matches = pat.FindMatchTexts("漢字とひらがなとカタカナ");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "漢字とひらがなとカタカナ");
}

TEST(Regex, UnicodeCharRange_CrossScript) {
  // Cross-script ranges work with exact codepoint matching.
  // U+3041 (Hiragana start) to U+9FA5 (CJK end) is a wide range.
  CompiledPattern pat("[ぁ-龥]+");  // Hiragana start to CJK end.
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Matches Hiragana, Katakana, and CJK within the range.
  auto matches = pat.FindMatchTexts("あアア漢字");  // All within range.
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "あアア漢字");
}

TEST(Regex, UnicodeCharRange_MixedASCIIUnicodeError) {
  // Range between ASCII and Unicode should error.
  CompiledPattern pat("[a-龥]+");
  EXPECT_FALSE(pat.ok());
  EXPECT_TRUE(strstr(pat.error_message(), "ASCII and Unicode") != nullptr);
}

TEST(Regex, UnicodeCharRange_LatinExtended) {
  // Latin Extended-A range uses exact codepoint matching.
  // U+0100-U+017F contains accented Latin characters.
  CompiledPattern pat("[Ā-ſ]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Matches Latin Extended characters.
  auto matches = pat.FindMatchTexts("hello Āēīōū world");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "Āēīōū");
}

TEST(Regex, HF_DeepSeek_ComplexPattern) {
  // Full DeepSeek pattern with punctuation handling.
  CompiledPattern pat(
      "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|"
      "[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| "
      "?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Punctuation followed by letters (like "#include").
  auto matches = pat.FindMatchTexts("#include <stdio.h>");
  EXPECT_GE(matches.size(), 3);
}

//===----------------------------------------------------------------------===//
// Real-World Pattern Edge Cases
//===----------------------------------------------------------------------===//

TEST(Regex, HF_TrailingWhitespaceHandling) {
  // All LLM patterns handle trailing whitespace specially via \s+(?!\S).
  CompiledPattern pat("\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Trailing spaces at EOF should be captured as one match.
  auto matches = pat.FindMatchTexts("text   ");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "   ");
}

TEST(Regex, HF_NewlinePreservation) {
  // Patterns like \s*[\r\n]+ preserve newlines separately.
  CompiledPattern pat("\\s*[\\r\\n]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("a\n\nb");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "\n\n");
}

TEST(Regex, HF_OptionalLeadingPunctuation) {
  // Pattern like [^\r\n\p{L}\p{N}]?\p{L}+ for words with optional prefix.
  CompiledPattern pat("[^\\r\\n\\p{L}\\p{N}]?\\p{L}+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Should match "hello" and ".world" (with leading dot).
  auto matches = pat.FindMatchTexts("hello.world");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "hello");
  EXPECT_EQ(matches[1], ".world");
}

TEST(Regex, HF_PunctuationWithNewlines) {
  // Pattern like [^\s\p{L}\p{N}]+[\r\n]* captures punctuation with trailing
  // newlines.
  CompiledPattern pat("[^\\s\\p{L}\\p{N}]+[\\r\\n]*");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("hello!\n");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "!\n");
}

//===----------------------------------------------------------------------===//
// Optional End Anchor Tests
//
// These tests verify that when an accept state is reachable via BOTH anchored
// (through $) and unanchored paths, the unanchored path correctly wins.
// The bug was a first-visit-wins race in epsilon closure anchor tracking.
//===----------------------------------------------------------------------===//

TEST(Regex, EndAnchorOptionalInAlternation) {
  // Pattern: a$|a - 'a' at end OR 'a' anywhere.
  // The same accept state is reachable via anchored path (a$) and unanchored
  // path (a). Input "ab" should match "a" because the unanchored branch allows
  // matching anywhere, not just at end.
  CompiledPattern pat("a$|a");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "ab": 'a' is NOT at end, but unanchored branch should match.
  auto matches = pat.FindMatchTexts("ab");
  ASSERT_EQ(matches.size(), 1) << "Should match 'a' via unanchored branch";
  EXPECT_EQ(matches[0], "a");

  // "ba": 'a' IS at end, either branch can match.
  matches = pat.FindMatchTexts("ba");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");
}

TEST(Regex, EndAnchorOptionalReversedAlternation) {
  // Pattern: a|a$ - order reversed to test both traversal orders.
  // Should behave identically to a$|a.
  CompiledPattern pat("a|a$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("ab");
  ASSERT_EQ(matches.size(), 1) << "Should match 'a' via unanchored branch";
  EXPECT_EQ(matches[0], "a");

  matches = pat.FindMatchTexts("ba");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");
}

TEST(Regex, EndAnchorTransitiveInAlternation) {
  // Pattern: ab$|ab - same accept reachable via anchored and unanchored paths.
  // Tests that anchor status propagates correctly through intermediate states.
  CompiledPattern pat("ab$|ab");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "abx": "ab" is NOT at end, unanchored branch should match.
  auto matches = pat.FindMatchTexts("abx");
  ASSERT_EQ(matches.size(), 1) << "Should match 'ab' via unanchored branch";
  EXPECT_EQ(matches[0], "ab");

  // "xab": "ab" IS at end, either branch matches.
  matches = pat.FindMatchTexts("xab");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "ab");
}

TEST(Regex, EndAnchorNestedAlternation) {
  // Pattern: (?:a$|a) - non-capturing group with optional anchor.
  CompiledPattern pat("(?:a$|a)");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("ab");
  ASSERT_EQ(matches.size(), 1) << "Should match 'a' via unanchored branch";
  EXPECT_EQ(matches[0], "a");
}

TEST(Regex, EndAnchorWithQuantifierInAlternation) {
  // Pattern: (?:a$|a)+ - repeated alternation with optional anchor.
  CompiledPattern pat("(?:a$|a)+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "aab": Multiple 'a's, none at end. Should match "aa".
  auto matches = pat.FindMatchTexts("aab");
  ASSERT_EQ(matches.size(), 1) << "Should match 'aa' mid-string";
  EXPECT_EQ(matches[0], "aa");

  // "baa": Multiple 'a's at end. Should match "aa".
  matches = pat.FindMatchTexts("baa");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "aa");
}

TEST(Regex, StartAnchorWithOptionalEndAnchor) {
  // Pattern: ^a$|^a - start required, end optional.
  // Both branches require start anchor, but only first requires end.
  CompiledPattern pat("^a$|^a");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "ab": 'a' at start, not at end. Second branch should match.
  auto matches = pat.FindMatchTexts("ab");
  ASSERT_EQ(matches.size(), 1) << "Start anchor ok, end not required";
  EXPECT_EQ(matches[0], "a");

  // "ba": 'a' not at start. Neither branch matches.
  matches = pat.FindMatchTexts("ba");
  EXPECT_EQ(matches.size(), 0) << "Start anchor required";

  // "a": Both anchors satisfied.
  matches = pat.FindMatchTexts("a");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");
}

TEST(Regex, AllPathsEndAnchored) {
  // Pattern: a$|b$ - both branches require end anchor.
  // Control test: when ALL paths go through $, end anchor IS required.
  CompiledPattern pat("a$|b$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // "ab": Only 'b' is at end.
  auto matches = pat.FindMatchTexts("ab");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "b");

  // "ba": Only 'a' is at end.
  matches = pat.FindMatchTexts("ba");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "a");

  // "abc": Neither 'a' nor 'b' at end.
  matches = pat.FindMatchTexts("abc");
  EXPECT_EQ(matches.size(), 0) << "End anchor required for both branches";
}

//===----------------------------------------------------------------------===//
// Consecutive Newline Tests
//===----------------------------------------------------------------------===//

// These tests verify that consecutive newlines are matched SEPARATELY when
// followed by non-whitespace, as required by patterns like \s+(?!\S)|\s+.

TEST(Regex, ConsecutiveNewlines_TwoNewlinesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\n\nA"
  //
  // First '\n' passes lookahead (followed by whitespace).
  // Second '\n' fails lookahead (followed by 'A'), uses fallback.
  // Result: two separate matches.
  CompiledPattern pat("\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("\n\nA");
  ASSERT_EQ(matches.size(), 2)
      << "Consecutive newlines before text should produce separate matches";
  EXPECT_EQ(matches[0], "\n");  // First newline.
  EXPECT_EQ(matches[1], "\n");  // Second newline.
}

TEST(Regex, ConsecutiveNewlines_TwoNewlinesAtEndOfInput) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "A\n\n"
  //
  // At EOS, negative lookahead (?!\S) passes (no character = not \S).
  // So both newlines should be combined into one match.
  CompiledPattern pat("\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("A\n\n");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "\n\n");  // Both newlines combined at EOS.
}

TEST(Regex, ConsecutiveNewlines_ThreeNewlinesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\n\n\nA"
  //
  // First two '\n': lookahead passes (followed by whitespace).
  // Third '\n': lookahead fails (followed by 'A'), uses fallback.
  // Result: Match "\n\n" for first two, then "\n" for third.
  CompiledPattern pat("\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("\n\n\nA");
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], "\n\n");  // First two newlines.
  EXPECT_EQ(matches[1], "\n");    // Third newline.
}

TEST(Regex, GPT2Pattern_ConsecutiveNewlinesBeforeWord) {
  // Full GPT-2 pattern with "\n\nAll" (typical paragraph break in text).
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  auto matches = pat.FindMatchTexts("\n\nAll:");
  ASSERT_GE(matches.size(), 3);
  // First two matches should be separate newlines.
  EXPECT_EQ(matches[0], "\n");
  EXPECT_EQ(matches[1], "\n");
}

// Streaming test with compiled GPT-2 pattern.
// This tests the actual code path used by the tokenizer.
TEST(Regex, GPT2Pattern_ConsecutiveNewlinesBeforeWord_Streaming) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  std::vector<std::pair<size_t, size_t>> matches;
  auto callback = [](void* user_data,
                     iree_tokenizer_regex_match_t match) -> iree_status_t {
    auto* m = static_cast<std::vector<std::pair<size_t, size_t>>*>(user_data);
    m->push_back({match.start, match.end});
    return iree_ok_status();
  };

  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, pat.dfa());

  // Feed entire input as single chunk (simulating tokenizer behavior).
  const char* text = "\n\nAll:";
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      pat.dfa(), &state, iree_make_cstring_view(text), 0, /*stride=*/NULL,
      callback, &matches));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      pat.dfa(), &state, strlen(text), callback, &matches));

  // First two matches should be separate newlines.
  ASSERT_GE(matches.size(), 3) << "Expected at least 3 matches";
  EXPECT_EQ(matches[0].first, 0);
  EXPECT_EQ(matches[0].second, 1);  // First '\n'
  EXPECT_EQ(matches[1].first, 1);
  EXPECT_EQ(matches[1].second, 2);  // Second '\n'
}

//===----------------------------------------------------------------------===//
// CJK Text with Punctuation
//===----------------------------------------------------------------------===//

// Verifies that CJK punctuation without letters matches via [^\s\p{L}\p{N}]+.
TEST(GPT2, OnlyPunctuation) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // CJK punctuation only - no letters at all.
  const char* text = "。、《》「」？！";

  std::vector<std::pair<size_t, size_t>> matches;
  auto callback = [](void* user_data,
                     iree_tokenizer_regex_match_t match) -> iree_status_t {
    auto* m = static_cast<std::vector<std::pair<size_t, size_t>>*>(user_data);
    m->push_back({match.start, match.end});
    return iree_ok_status();
  };

  IREE_EXPECT_OK(iree_tokenizer_regex_exec(
      pat.dfa(), iree_make_cstring_view(text), callback, &matches));

  // Should produce at least one match - punctuation run.
  // The [^\s\p{L}\p{N}]+ branch matches non-space, non-letter, non-digit chars.
  EXPECT_GT(matches.size(), 0u)
      << "Expected punctuation to match [^\\s\\p{L}\\p{N}]+";
}

// Ruby annotation brackets 《》 (U+300A, U+300B) should terminate letter runs
// and match as punctuation via [^\s\p{L}\p{N}]+.
TEST(GPT2, RubyAnnotationBrackets) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Sample from actual Japanese test file with ruby annotations.
  // Format: kanji《furigana》 where 《》 are CJK angle brackets.
  const char* text = "物狂《ものぐる》ほしさ";

  std::vector<std::pair<size_t, size_t>> matches;
  auto callback = [](void* user_data,
                     iree_tokenizer_regex_match_t match) -> iree_status_t {
    auto* m = static_cast<std::vector<std::pair<size_t, size_t>>*>(user_data);
    m->push_back({match.start, match.end});
    return iree_ok_status();
  };

  IREE_EXPECT_OK(iree_tokenizer_regex_exec(
      pat.dfa(), iree_make_cstring_view(text), callback, &matches));

  // Expected matches:
  // - "物狂" (letters before 《)
  // - "《" (punctuation)
  // - "ものぐる" (letters inside brackets)
  // - "》" (punctuation)
  // - "ほしさ" (letters after brackets)
  ASSERT_GE(matches.size(), 5u)
      << "Expected at least 5 matches (letters + brackets), got "
      << matches.size();

  // Verify match boundaries.
  std::vector<std::string> segments;
  for (const auto& m : matches) {
    segments.push_back(std::string(text + m.first, m.second - m.first));
  }

  // Verify we got the expected segmentation.
  EXPECT_EQ(segments[0], "物狂");      // Letters before 《
  EXPECT_EQ(segments[1], "《");        // Opening bracket
  EXPECT_EQ(segments[2], "ものぐる");  // Letters inside
  EXPECT_EQ(segments[3], "》");        // Closing bracket
  EXPECT_EQ(segments[4], "ほしさ");    // Letters after
}

// Streaming version of RubyAnnotationBrackets - feed text in small chunks.
TEST(GPT2, RubyAnnotationBrackets_Streaming) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  const char* text = "物狂《ものぐる》ほしさ";
  size_t text_len = strlen(text);

  // Single-chunk ground truth.
  std::vector<std::pair<size_t, size_t>> single_chunk_matches;
  auto callback = [](void* user_data,
                     iree_tokenizer_regex_match_t match) -> iree_status_t {
    auto* m = static_cast<std::vector<std::pair<size_t, size_t>>*>(user_data);
    m->push_back({match.start, match.end});
    return iree_ok_status();
  };

  IREE_EXPECT_OK(iree_tokenizer_regex_exec(pat.dfa(),
                                           iree_make_cstring_view(text),
                                           callback, &single_chunk_matches));

  // Streaming: feed 6 bytes at a time (2 CJK chars or 2 brackets).
  constexpr size_t kChunkSize = 6;
  std::vector<std::pair<size_t, size_t>> streaming_matches;

  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, pat.dfa());

  size_t offset = 0;
  while (offset < text_len) {
    size_t chunk_size = std::min(kChunkSize, text_len - offset);
    size_t incomplete =
        iree_unicode_utf8_incomplete_tail_length(text + offset, chunk_size);
    chunk_size -= incomplete;
    if (chunk_size == 0) {
      chunk_size = std::min(text_len - offset, (size_t)4);
    }

    iree_string_view_t chunk = {text + offset, chunk_size};
    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
        pat.dfa(), &state, chunk, offset, /*stride=*/NULL, callback,
        &streaming_matches));
    offset += chunk_size;
  }

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      pat.dfa(), &state, text_len, callback, &streaming_matches));

  EXPECT_EQ(streaming_matches, single_chunk_matches)
      << "Streaming produced " << streaming_matches.size()
      << " matches vs single-chunk " << single_chunk_matches.size();
}

// (。？！) should terminate matches and produce separate segments.
TEST(GPT2, CJKTextWithPunctuation_SingleChunk) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // CJK text with punctuation that should break \p{L}+ matches.
  // Japanese/Chinese punctuation: 。(U+3002), ？(U+FF1F), ！(U+FF01)
  const char* text =
      "今日は天気がいいですね。"  // 36 bytes, ends with 。
      "明天会更好吗？"            // 21 bytes, ends with ？
      "こんにちは！";             // 18 bytes, ends with ！

  std::vector<std::pair<size_t, size_t>> matches;
  auto callback = [](void* user_data,
                     iree_tokenizer_regex_match_t match) -> iree_status_t {
    auto* m = static_cast<std::vector<std::pair<size_t, size_t>>*>(user_data);
    m->push_back({match.start, match.end});
    return iree_ok_status();
  };

  IREE_EXPECT_OK(iree_tokenizer_regex_exec(
      pat.dfa(), iree_make_cstring_view(text), callback, &matches));

  // Should produce multiple matches: letter runs + punctuation marks.
  EXPECT_GE(matches.size(), 3u)
      << "Expected multiple matches from CJK text with punctuation, got "
      << matches.size();

  // No single match should span the entire text - punctuation breaks matches.
  size_t max_match_size = 0;
  for (const auto& m : matches) {
    max_match_size = std::max(max_match_size, m.second - m.first);
  }

  // Each CJK character is 3 bytes. 36 bytes = 12 chars is reasonable for one
  // sentence. 75 bytes covering all text would indicate punctuation isn't
  // breaking matches.
  EXPECT_LT(max_match_size, 50u)
      << "Match too large - punctuation should break matches. Max: "
      << max_match_size;
}

// Verifies streaming produces identical results to single-chunk processing.
TEST(GPT2, CJKTextWithPunctuation_Streaming) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  const char* text =
      "今日は天気がいいですね。"
      "明天会更好吗？"
      "こんにちは！";

  size_t text_len = strlen(text);

  // Single-chunk as ground truth.
  std::vector<std::pair<size_t, size_t>> single_chunk_matches;
  auto callback = [](void* user_data,
                     iree_tokenizer_regex_match_t match) -> iree_status_t {
    auto* m = static_cast<std::vector<std::pair<size_t, size_t>>*>(user_data);
    m->push_back({match.start, match.end});
    return iree_ok_status();
  };

  IREE_EXPECT_OK(iree_tokenizer_regex_exec(pat.dfa(),
                                           iree_make_cstring_view(text),
                                           callback, &single_chunk_matches));

  // Streaming: feed entire text then finalize.
  std::vector<std::pair<size_t, size_t>> streaming_matches;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, pat.dfa());

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      pat.dfa(), &state, iree_make_cstring_view(text), 0, /*stride=*/NULL,
      callback, &streaming_matches));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      pat.dfa(), &state, text_len, callback, &streaming_matches));

  EXPECT_EQ(streaming_matches, single_chunk_matches)
      << "Streaming produced " << streaming_matches.size()
      << " matches vs single-chunk " << single_chunk_matches.size();
}

// Verifies small-chunk streaming produces identical results to single-chunk.
// Uses ~16-byte chunks (adjusted to UTF-8 codepoint boundaries).
TEST(GPT2, CJKTextWithPunctuation_SmallChunks) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  const char* text =
      "今日は天気がいいですね。"
      "明天会更好吗？"
      "こんにちは！";

  size_t text_len = strlen(text);

  // Single-chunk as ground truth.
  std::vector<std::pair<size_t, size_t>> single_chunk_matches;
  auto callback = [](void* user_data,
                     iree_tokenizer_regex_match_t match) -> iree_status_t {
    auto* m = static_cast<std::vector<std::pair<size_t, size_t>>*>(user_data);
    m->push_back({match.start, match.end});
    return iree_ok_status();
  };

  IREE_EXPECT_OK(iree_tokenizer_regex_exec(pat.dfa(),
                                           iree_make_cstring_view(text),
                                           callback, &single_chunk_matches));

  // Feed in ~16-byte chunks, adjusted to not split UTF-8 codepoints.
  // The regex executor requires complete codepoints at chunk boundaries.
  constexpr size_t kTargetChunkSize = 16;
  std::vector<std::pair<size_t, size_t>> chunked_matches;

  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, pat.dfa());

  size_t offset = 0;
  while (offset < text_len) {
    size_t remaining = text_len - offset;
    size_t chunk_size = std::min(kTargetChunkSize, remaining);

    // Adjust chunk_size to end at a UTF-8 codepoint boundary.
    size_t incomplete =
        iree_unicode_utf8_incomplete_tail_length(text + offset, chunk_size);
    chunk_size -= incomplete;

    // If chunk_size is 0, we need more bytes to complete the codepoint.
    // This shouldn't happen with kTargetChunkSize >= 4.
    ASSERT_GT(chunk_size, 0u)
        << "Chunk too small to contain complete codepoint";

    iree_string_view_t chunk = {text + offset, chunk_size};

    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(pat.dfa(), &state, chunk,
                                                  offset, /*stride=*/NULL,
                                                  callback, &chunked_matches))
        << "Feed failed at offset " << offset;

    offset += chunk_size;
  }

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      pat.dfa(), &state, text_len, callback, &chunked_matches));

  EXPECT_EQ(chunked_matches, single_chunk_matches)
      << "Chunked produced " << chunked_matches.size()
      << " matches vs single-chunk " << single_chunk_matches.size();
}

// Verifies behavior when processing a long run of CJK letters with NO
// punctuation. The \p{L}+ pattern will accumulate state without emitting
// until finalize(). This tests the streaming accumulation behavior.
TEST(GPT2, CJKLetterRunWithoutPunctuation) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Build a long string of only CJK letters (no punctuation).
  // Each CJK character is 3 bytes, so 100 chars = 300 bytes.
  std::string text;
  const char* cjk_chars[] = {
      "日", "本", "語", "中", "文", "字", "漢", "字", "言", "語",
      "今", "天", "明", "天", "昨", "天", "時", "間", "空", "間",
  };
  for (int i = 0; i < 100; ++i) {
    text += cjk_chars[i % 20];
  }

  size_t text_len = text.length();
  ASSERT_EQ(text_len, 300u);  // 100 chars * 3 bytes each

  // Single-chunk should produce exactly one match covering entire text.
  std::vector<std::pair<size_t, size_t>> single_chunk_matches;
  auto callback = [](void* user_data,
                     iree_tokenizer_regex_match_t match) -> iree_status_t {
    auto* m = static_cast<std::vector<std::pair<size_t, size_t>>*>(user_data);
    m->push_back({match.start, match.end});
    return iree_ok_status();
  };

  IREE_EXPECT_OK(iree_tokenizer_regex_exec(
      pat.dfa(), iree_make_string_view(text.data(), text.length()), callback,
      &single_chunk_matches));

  ASSERT_EQ(single_chunk_matches.size(), 1u)
      << "Expected exactly one match for continuous letter run";
  EXPECT_EQ(single_chunk_matches[0].first, 0u);
  EXPECT_EQ(single_chunk_matches[0].second, text_len);

  // Streaming: feed in ~50-byte chunks, respecting UTF-8 boundaries.
  // Key observation: NO matches should be emitted until finalize().
  constexpr size_t kTargetChunkSize = 50;
  std::vector<std::pair<size_t, size_t>> streaming_matches;

  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, pat.dfa());

  size_t offset = 0;
  while (offset < text_len) {
    size_t remaining = text_len - offset;
    size_t chunk_size = std::min(kTargetChunkSize, remaining);

    // Adjust to UTF-8 boundary (all CJK chars are 3 bytes, so align to 3).
    // For a pure CJK string, chunk_size should already be aligned, but
    // use the general-purpose function for correctness.
    size_t incomplete = iree_unicode_utf8_incomplete_tail_length(
        text.data() + offset, chunk_size);
    chunk_size -= incomplete;
    ASSERT_GT(chunk_size, 0u);

    iree_string_view_t chunk = {text.data() + offset, chunk_size};
    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(pat.dfa(), &state, chunk,
                                                  offset, /*stride=*/NULL,
                                                  callback, &streaming_matches))
        << "Feed failed at offset " << offset;

    // No intermediate matches expected - entire text is one \p{L}+ run.
    EXPECT_EQ(streaming_matches.size(), 0u)
        << "Unexpected intermediate match at offset " << offset;

    offset += chunk_size;
  }

  // Only finalize should produce the match.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      pat.dfa(), &state, text_len, callback, &streaming_matches));

  EXPECT_EQ(streaming_matches.size(), 1u)
      << "Expected exactly one match after finalize";
  if (!streaming_matches.empty()) {
    EXPECT_EQ(streaming_matches[0].first, 0u);
    EXPECT_EQ(streaming_matches[0].second, text_len);
  }
}

//===----------------------------------------------------------------------===//
// Backtracking Edge Cases
//===----------------------------------------------------------------------===//

// Verifies that patterns requiring more bytes than the input don't crash when
// backtracking. This tests the case where a partial match fails and the
// backtrack position is past the end of input (next_start >= chunk.size).
TEST(Regex, BacktrackPastEndOfInput) {
  // Pattern: requires at least 23 bytes (2 + 21 dots after .*)
  // Input: only 10 bytes
  // The pattern will start matching, but fail when it can't match 21 more dots.
  CompiledPattern pat("..*....................");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Short input that can't possibly match (needs 23+ bytes).
  auto matches = pat.FindMatchTexts("abcdefghij");  // 10 bytes
  EXPECT_EQ(matches.size(), 0) << "Pattern requires 23+ bytes, input has 10";

  // Very short input - single byte.
  matches = pat.FindMatchTexts("a");
  EXPECT_EQ(matches.size(), 0);

  // Empty input.
  matches = pat.FindMatchTexts("");
  EXPECT_EQ(matches.size(), 0);
}

// Pattern with many dots that requires exactly N bytes. Tests the boundary
// condition where backtracking position equals chunk size.
TEST(Regex, BacktrackExactBoundary) {
  // Pattern: exactly 5 bytes
  CompiledPattern pat(".....");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Input of exactly 4 bytes - one short. Partial match fails at position 4.
  auto matches = pat.FindMatchTexts("abcd");
  EXPECT_EQ(matches.size(), 0);

  // Input of exactly 5 bytes - matches.
  matches = pat.FindMatchTexts("abcde");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abcde");

  // Input of 6 bytes - matches first 5.
  matches = pat.FindMatchTexts("abcdef");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], "abcde");
}

// Tests that binary input (non-ASCII bytes) doesn't cause crashes during
// backtracking. This exercises the multi-byte UTF-8 sequence length handling.
TEST(Regex, BacktrackWithBinaryInput) {
  // Pattern: 10 dots
  CompiledPattern pat("..........");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Binary input with multi-byte UTF-8 characters (5 bytes total).
  // Each character is 3 bytes: "一二" = 6 bytes, but we want less than 10.
  const char* input = "\xE4\xB8\x80";  // Single CJK character "一" (3 bytes)
  auto matches = pat.FindMatchTexts(input);
  EXPECT_EQ(matches.size(), 0);

  // Input with invalid UTF-8 start byte (0xFF) followed by ASCII.
  // This tests that we handle replacement characters correctly.
  const char binary_input[] = {'\xFF', 'a', 'b', 'c', '\0'};
  matches = pat.FindMatchTexts(binary_input);
  EXPECT_EQ(matches.size(), 0);
}

}  // namespace
