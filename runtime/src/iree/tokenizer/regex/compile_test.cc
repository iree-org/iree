// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Regex compiler tests focused on compilation behavior:
// - Error detection and reporting (syntax errors, unsupported features)
// - DFA structure verification (state count, flags, validation)
//
// For end-to-end matching tests, see regex_test.cc.
// For executor-only tests with hand-built DFAs, see exec_test.cc.

#include "iree/tokenizer/regex/compile.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/regex/internal/parser.h"

namespace {

//===----------------------------------------------------------------------===//
// Test Utilities
//===----------------------------------------------------------------------===//

// Helper to compile a pattern and check for errors.
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

 private:
  iree_status_t status_;
  iree_tokenizer_regex_dfa_t dfa_ = {};
  uint8_t* storage_ = nullptr;
  std::string error_message_;
  iree_host_size_t error_position_ = 0;
};

//===----------------------------------------------------------------------===//
// Compilation Error Tests
//===----------------------------------------------------------------------===//

TEST(Compile, EmptyPatternFails) {
  CompiledPattern pat("");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_INVALID_ARGUMENT);
}

TEST(Compile, UnbalancedParenthesis) {
  CompiledPattern pat("(abc");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_INVALID_ARGUMENT);
}

TEST(Compile, UnbalancedBracket) {
  CompiledPattern pat("[abc");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_INVALID_ARGUMENT);
}

TEST(Compile, InvalidEscape) {
  CompiledPattern pat("\\q");  // \q is not a valid escape.
  EXPECT_FALSE(pat.ok());
}

TEST(Compile, InvalidEscapeInCharClass) {
  CompiledPattern pat("[\\q]");  // \q is not valid inside char class either.
  EXPECT_FALSE(pat.ok());
}

TEST(Compile, InvalidUnicodeProperty) {
  CompiledPattern pat("\\p{X}");  // X is not a valid property.
  EXPECT_FALSE(pat.ok());
}

TEST(Compile, UnicodeSubcategoriesNotSupported) {
  // Unicode subcategories (Lu, Ll, Nd, etc.) are not supported.
  // Only general categories (L, N, P, M, S, Z, C) are available.
  CompiledPattern lu("\\p{Lu}");  // Uppercase Letter - not supported.
  EXPECT_FALSE(lu.ok());
  EXPECT_EQ(lu.status_code(), IREE_STATUS_INVALID_ARGUMENT);

  CompiledPattern nd("\\p{Nd}");  // Decimal Number - not supported.
  EXPECT_FALSE(nd.ok());

  // General categories should work.
  CompiledPattern l("\\p{L}");  // All Letters - supported.
  EXPECT_TRUE(l.ok()) << l.error_message();

  CompiledPattern n("\\p{N}");  // All Numbers - supported.
  EXPECT_TRUE(n.ok()) << n.error_message();
}

TEST(Compile, SurrogateCodepointsRejected) {
  // Surrogate codepoints (U+D800-U+DFFF) are reserved for UTF-16 encoding
  // and are not valid Unicode scalar values.
  CompiledPattern low("\\uD800");  // Low surrogate start.
  EXPECT_FALSE(low.ok());
  EXPECT_EQ(low.status_code(), IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(strstr(low.error_message(), "surrogate"), nullptr)
      << "Expected 'surrogate' in error, got: " << low.error_message();

  CompiledPattern high("\\uDFFF");  // High surrogate end.
  EXPECT_FALSE(high.ok());
  EXPECT_EQ(high.status_code(), IREE_STATUS_INVALID_ARGUMENT);

  CompiledPattern mid("\\uD9AB");  // Middle of surrogate range.
  EXPECT_FALSE(mid.ok());
  EXPECT_EQ(mid.status_code(), IREE_STATUS_INVALID_ARGUMENT);

  // Valid codepoints just outside the surrogate range should work.
  CompiledPattern before("\\uD7FF");  // Just before surrogates.
  EXPECT_TRUE(before.ok()) << before.error_message();

  CompiledPattern after("\\uE000");  // Just after surrogates.
  EXPECT_TRUE(after.ok()) << after.error_message();
}

TEST(Compile, QuantifierWithoutPrecedingElement) {
  CompiledPattern pat("*abc");
  EXPECT_FALSE(pat.ok());
}

TEST(Compile, InvalidQuantifierRange) {
  CompiledPattern pat("a{5,2}");  // min > max.
  EXPECT_FALSE(pat.ok());
}

TEST(Compile, InvalidQuantifierEmpty) {
  CompiledPattern pat("a{}");  // Empty braces.
  EXPECT_FALSE(pat.ok());
}

TEST(Compile, InvalidQuantifierMissingMin) {
  CompiledPattern pat("a{,3}");  // Missing min before comma.
  EXPECT_FALSE(pat.ok());
}

TEST(Compile, UnterminatedEscape) {
  // Pattern ending with backslash.
  CompiledPattern pat("abc\\");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_INVALID_ARGUMENT);
}

TEST(Compile, ExcessiveNestingDepthFails) {
  // Build a pattern with 101 nested groups: (((((...(a)...)))))
  // The parser has a limit of 100 levels to prevent stack overflow.
  std::string pattern;
  for (int i = 0; i < 101; ++i) {
    pattern += "(";
  }
  pattern += "a";
  for (int i = 0; i < 101; ++i) {
    pattern += ")";
  }

  CompiledPattern pat(pattern.c_str());
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(strstr(pat.error_message(), "nesting depth"), nullptr)
      << "Expected error message to mention nesting depth, got: "
      << pat.error_message();
}

TEST(Compile, ExcessiveAlternationBranchesFails) {
  // Build a pattern with 256 alternation branches: a0|a1|a2|...|a255.
  // The NFA has a limit of 255 branches because branch_index is uint8_t.
  std::string pattern;
  for (int i = 0; i < 256; ++i) {
    if (i > 0) pattern += "|";
    pattern += "a" + std::to_string(i);
  }

  CompiledPattern pat(pattern.c_str());
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_RESOURCE_EXHAUSTED);
}

TEST(Compile, MaxAlternationBranchesSucceeds) {
  // Build a pattern with exactly 255 alternation branches - should succeed.
  std::string pattern;
  for (int i = 0; i < 255; ++i) {
    if (i > 0) pattern += "|";
    pattern += "a" + std::to_string(i);
  }

  CompiledPattern pat(pattern.c_str());
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, MaxNestingDepthSucceeds) {
  // Build a pattern with exactly 100 nested groups - should succeed.
  std::string pattern;
  for (int i = 0; i < 100; ++i) {
    pattern += "(";
  }
  pattern += "a";
  for (int i = 0; i < 100; ++i) {
    pattern += ")";
  }

  CompiledPattern pat(pattern.c_str());
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, InvalidUnicodePropertyEmpty) {
  // Empty unicode property.
  CompiledPattern pat("\\p{}");
  EXPECT_FALSE(pat.ok());
}

TEST(Compile, InvalidUnicodePropertyUnclosed) {
  // Unclosed unicode property.
  CompiledPattern pat("\\p{L");
  EXPECT_FALSE(pat.ok());
}

TEST(Compile, PossessivePlusValid) {
  // a++ is a valid possessive quantifier (one or more, possessive).
  CompiledPattern pat("a++");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, PossessiveStarValid) {
  // a*+ is a valid possessive quantifier (zero or more, possessive).
  // Prefix with 'b' to avoid empty-match validation error.
  CompiledPattern pat("ba*+");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, LazyStarValid) {
  // a*? is a lazy quantifier (zero or more, non-greedy).
  // DFA engines are inherently greedy, so lazy modifiers are accepted but
  // have no effect on matching behavior.
  // Prefix with 'b' to avoid empty-match validation error.
  CompiledPattern pat("ba*?c");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, LazyPlusValid) {
  // a+? is a lazy quantifier (one or more, non-greedy).
  CompiledPattern pat("a+?");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, LazyQuestionValid) {
  // a?? is a lazy quantifier (zero or one, non-greedy).
  // Prefix with 'b' to avoid empty-match validation error.
  CompiledPattern pat("ba??c");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, LazyBraceQuantifierValid) {
  // a{1,3}? is a lazy quantifier with explicit bounds.
  CompiledPattern pat("a{1,3}?");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, EmptyAlternativeBranch) {
  // Empty branch in alternation - "|b" means "" or "b".
  CompiledPattern pat("|b");
  // This might be valid (matching empty string or b).
  // If valid, it should compile.
  if (pat.ok()) {
    // Just ensure it compiles without crashing.
    EXPECT_NE(pat.dfa(), nullptr);
  }
}

TEST(Compile, TrailingAlternation) {
  // "a|" means "a" or "".
  CompiledPattern pat("a|");
  // This might be valid depending on implementation.
  // Just ensure it doesn't crash.
  (void)pat.ok();
}

//===----------------------------------------------------------------------===//
// Negative Lookahead Error Tests
//===----------------------------------------------------------------------===//

// Character class lookahead is not implemented - verify compile-time rejection.
TEST(Compile, NegativeLookaheadCharClassRejectsAtCompileTime) {
  // Explicit character class in lookahead is not yet supported.
  CompiledPattern pat("a(?![abc])");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_UNIMPLEMENTED);
}

TEST(Compile, NegativeLookaheadCharRangeRejectsAtCompileTime) {
  // Character range in lookahead should fail at compile time.
  CompiledPattern pat("a(?![a-z])");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_UNIMPLEMENTED);
}

TEST(Compile, NegativeLookaheadNegatedClassRejectsAtCompileTime) {
  // Negated character class in lookahead should also fail at compile time.
  CompiledPattern pat("a(?![^abc])");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_UNIMPLEMENTED);
}

TEST(Compile, NegativeLookaheadDotRejectsAtCompileTime) {
  // Dot lookahead is not supported - only makes sense at end-of-string.
  CompiledPattern pat("a(?!.)");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_UNIMPLEMENTED);
}

TEST(Compile, NegativeLookaheadUnicodePropRejectsAtCompileTime) {
  // Unicode property lookahead is not supported.
  CompiledPattern pat(R"(a(?!\p{L}))");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_UNIMPLEMENTED);
}

// Mid-pattern lookahead is not supported - must be at end of branch.
TEST(Compile, NegativeLookaheadMidPatternRejects) {
  // Lookahead followed by more pattern elements is invalid.
  CompiledPattern pat("a(?!b)c");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_INVALID_ARGUMENT);
}

TEST(Compile, NegativeLookaheadAtStartRejects) {
  // Lookahead at start followed by more pattern is invalid.
  CompiledPattern pat("(?!a)b");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_INVALID_ARGUMENT);
}

TEST(Compile, NegativeLookaheadMultipleRejects) {
  // Multiple lookaheads in sequence - first one is mid-pattern.
  CompiledPattern pat("x(?!y)(?!z)");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_INVALID_ARGUMENT);
}

// Verify that supported lookahead types still compile.
TEST(Compile, NegativeLookaheadSupportedTypesCompile) {
  // Single character lookahead.
  CompiledPattern pat1("a(?!b)");
  EXPECT_TRUE(pat1.ok()) << pat1.error_message();

  // Shorthand lookaheads.
  CompiledPattern pat2("a(?!\\d)");
  EXPECT_TRUE(pat2.ok()) << pat2.error_message();

  CompiledPattern pat3("a(?!\\s)");
  EXPECT_TRUE(pat3.ok()) << pat3.error_message();

  CompiledPattern pat4("a(?!\\S)");
  EXPECT_TRUE(pat4.ok()) << pat4.error_message();

  CompiledPattern pat5("a(?!\\w)");
  EXPECT_TRUE(pat5.ok()) << pat5.error_message();
}

TEST(Compile, NegativeLookaheadInAlternationBranchValid) {
  // Lookahead at end of each alternation branch is valid.
  CompiledPattern pat("a(?!b)|c(?!d)");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, NegativeLookaheadInGroupEndValid) {
  // Lookahead at end of group is valid.
  CompiledPattern pat("(ab(?!c))+");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, ConflictingLookaheadsReject) {
  // Same prefix with different lookahead constraints is not supported.
  // a(?!b)|a(?!c) - both branches match 'a' but have conflicting lookahead.
  CompiledPattern pat("a(?!b)|a(?!c)");
  EXPECT_FALSE(pat.ok());
  EXPECT_EQ(pat.status_code(), IREE_STATUS_INVALID_ARGUMENT);
}

//===----------------------------------------------------------------------===//
// Anchor Error Tests
//===----------------------------------------------------------------------===//

TEST(Compile, BothAnchorsEmptyPattern) {
  // ^$ would match empty string, but our engine rejects empty-string matches.
  // This should fail validation.
  CompiledPattern pat("^$");
  EXPECT_FALSE(pat.ok());
}

//===----------------------------------------------------------------------===//
// DFA Structure Verification Tests
//===----------------------------------------------------------------------===//

TEST(Compile, DfaHasCorrectStateCount) {
  CompiledPattern pat("abc");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // abc needs at least 4 states: start, a, ab, abc(accept).
  EXPECT_GE(iree_tokenizer_regex_dfa_state_count(pat.dfa()), 4);
}

TEST(Compile, DfaValidates) {
  CompiledPattern pat("\\w+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  iree_status_t validation = iree_tokenizer_regex_dfa_validate(pat.dfa());
  IREE_EXPECT_OK(validation);
}

TEST(Compile, DfaUnicodeFlag) {
  CompiledPattern pat("\\p{L}+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  EXPECT_TRUE(iree_tokenizer_regex_dfa_uses_unicode(pat.dfa()));
}

TEST(Compile, DfaNoUnicodeFlag) {
  CompiledPattern pat("[a-z]+");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  EXPECT_FALSE(iree_tokenizer_regex_dfa_uses_unicode(pat.dfa()));
}

TEST(Compile, CaseInsensitiveCompiles) {
  // Case insensitive flag should be accepted by the compiler.
  // Note: The DFA flag is informational only - the compiler handles case
  // expansion in the transitions, not at runtime.
  CompiledPattern pat("abc",
                      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_CASE_INSENSITIVE);
  ASSERT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, StartAnchorSetsDfaFlag) {
  // ^abc should set the anchor flag.
  CompiledPattern pat("^abc");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // Verify the DFA has the anchor flag set.
  EXPECT_TRUE((pat.dfa()->header->flags &
               IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS) != 0)
      << "DFA should have HAS_ANCHORS flag set for ^abc";
  EXPECT_NE(pat.dfa()->start_anchor_bitmap, nullptr)
      << "start_anchor_bitmap should be non-null";

  // Check that start state requires start anchor.
  uint16_t start_state = pat.dfa()->header->start_state;
  bool start_requires_anchor =
      (pat.dfa()->start_anchor_bitmap[start_state / 64] &
       (1ULL << (start_state % 64))) != 0;
  EXPECT_TRUE(start_requires_anchor)
      << "Start state should require start anchor for ^abc";
}

TEST(Compile, EndAnchorSetsDfaFlag) {
  CompiledPattern pat("abc$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  EXPECT_TRUE((pat.dfa()->header->flags &
               IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS) != 0)
      << "DFA should have HAS_ANCHORS flag set for abc$";
  EXPECT_NE(pat.dfa()->end_anchor_bitmap, nullptr)
      << "end_anchor_bitmap should be non-null";
}

TEST(Compile, LookaheadSetsDfaFlag) {
  CompiledPattern pat("\\s+(?!\\S)");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  EXPECT_TRUE((pat.dfa()->header->flags &
               IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD) != 0)
      << "DFA should have HAS_LOOKAHEAD flag set for \\s+(?!\\S)";
  EXPECT_NE(pat.dfa()->lookahead, nullptr)
      << "lookahead table should be non-null";
}

//===----------------------------------------------------------------------===//
// Compilation Success Tests (patterns that should compile)
//===----------------------------------------------------------------------===//

TEST(Compile, BasicPatternsCompile) {
  // Single literal.
  EXPECT_TRUE(CompiledPattern("a").ok());

  // Multiple literals.
  EXPECT_TRUE(CompiledPattern("abc").ok());

  // Character class.
  EXPECT_TRUE(CompiledPattern("[abc]").ok());

  // Character range.
  EXPECT_TRUE(CompiledPattern("[a-z]").ok());

  // Negated class.
  EXPECT_TRUE(CompiledPattern("[^abc]").ok());
}

TEST(Compile, ShorthandClassesCompile) {
  EXPECT_TRUE(CompiledPattern("\\d").ok());
  EXPECT_TRUE(CompiledPattern("\\D").ok());
  EXPECT_TRUE(CompiledPattern("\\w").ok());
  EXPECT_TRUE(CompiledPattern("\\W").ok());
  EXPECT_TRUE(CompiledPattern("\\s").ok());
  EXPECT_TRUE(CompiledPattern("\\S").ok());
}

TEST(Compile, QuantifiersCompile) {
  // Quantifiers that require at least one match.
  EXPECT_TRUE(CompiledPattern("a+").ok());
  EXPECT_TRUE(CompiledPattern("a{1}").ok());
  EXPECT_TRUE(CompiledPattern("a{2,4}").ok());
  EXPECT_TRUE(CompiledPattern("a{2,}").ok());

  // Quantifiers with context (can't match empty at start).
  EXPECT_TRUE(CompiledPattern("ba*c").ok());
  EXPECT_TRUE(CompiledPattern("ba?c").ok());
  EXPECT_TRUE(CompiledPattern("ba{0,2}c").ok());

  // Bounded quantifier.
  EXPECT_TRUE(CompiledPattern("a{3}").ok());
}

TEST(Compile, GroupsCompile) {
  EXPECT_TRUE(CompiledPattern("(abc)").ok());
  EXPECT_TRUE(CompiledPattern("(?:abc)").ok());
  EXPECT_TRUE(CompiledPattern("(?i:abc)").ok());
  EXPECT_TRUE(CompiledPattern("((ab)+c)+").ok());
}

TEST(Compile, AlternationCompiles) {
  EXPECT_TRUE(CompiledPattern("a|b").ok());
  EXPECT_TRUE(CompiledPattern("cat|dog").ok());
  EXPECT_TRUE(CompiledPattern("a|b|c|d|e").ok());
}

TEST(Compile, AnchorsCompile) {
  EXPECT_TRUE(CompiledPattern("^abc").ok());
  EXPECT_TRUE(CompiledPattern("abc$").ok());
  EXPECT_TRUE(CompiledPattern("^abc$").ok());
  EXPECT_TRUE(CompiledPattern("^a|b$").ok());
}

TEST(Compile, UnicodeCategoriesCompile) {
  EXPECT_TRUE(CompiledPattern("\\p{L}").ok());
  EXPECT_TRUE(CompiledPattern("\\p{N}").ok());
  EXPECT_TRUE(CompiledPattern("\\p{P}").ok());
  EXPECT_TRUE(CompiledPattern("\\p{M}").ok());
  EXPECT_TRUE(CompiledPattern("\\p{S}").ok());
  EXPECT_TRUE(CompiledPattern("\\p{Z}").ok());
  EXPECT_TRUE(CompiledPattern("\\p{C}").ok());
}

TEST(Compile, EscapeSequencesCompile) {
  EXPECT_TRUE(CompiledPattern("\\n").ok());
  EXPECT_TRUE(CompiledPattern("\\r").ok());
  EXPECT_TRUE(CompiledPattern("\\t").ok());
  EXPECT_TRUE(CompiledPattern("\\\\").ok());
  EXPECT_TRUE(CompiledPattern("\\.").ok());
  EXPECT_TRUE(CompiledPattern("\\[").ok());
  EXPECT_TRUE(CompiledPattern("\\]").ok());
  EXPECT_TRUE(CompiledPattern("\\(").ok());
  EXPECT_TRUE(CompiledPattern("\\)").ok());
  EXPECT_TRUE(CompiledPattern("\\{").ok());
  EXPECT_TRUE(CompiledPattern("\\}").ok());
  EXPECT_TRUE(CompiledPattern("\\|").ok());
  EXPECT_TRUE(CompiledPattern("\\*").ok());
  EXPECT_TRUE(CompiledPattern("\\+").ok());
  EXPECT_TRUE(CompiledPattern("\\?").ok());
  EXPECT_TRUE(CompiledPattern("\\^").ok());
  EXPECT_TRUE(CompiledPattern("\\$").ok());
}

//===----------------------------------------------------------------------===//
// Real-World Patterns Compile Test
//===----------------------------------------------------------------------===//

TEST(Compile, GPT2PatternCompiles) {
  CompiledPattern pat(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, Llama3PatternCompiles) {
  CompiledPattern pat(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

TEST(Compile, Qwen2PatternCompiles) {
  CompiledPattern pat(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
  EXPECT_TRUE(pat.ok()) << pat.error_message();
}

//===----------------------------------------------------------------------===//
// Parser API Tests
//===----------------------------------------------------------------------===//

TEST(Parse, NullArenaReturnsError) {
  iree_tokenizer_regex_ast_node_t* ast = nullptr;
  iree_tokenizer_regex_parse_error_t error = {};
  iree_status_t status =
      iree_tokenizer_regex_parse(IREE_SV("abc"), nullptr, &ast, &error);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(ast, nullptr);
}

TEST(Parse, NullOutAstReturnsError) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  iree_tokenizer_regex_parse_error_t error = {};
  iree_status_t status =
      iree_tokenizer_regex_parse(IREE_SV("abc"), &arena, nullptr, &error);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

//===----------------------------------------------------------------------===//
// End Anchor Bitmap Tests
//
// These tests verify that end_anchor_bitmap is correctly computed when
// an accept state is reachable via both anchored ($) and unanchored paths.
//===----------------------------------------------------------------------===//

// Helper to check if a state is in the end_anchor_bitmap.
bool IsInEndAnchorBitmap(const iree_tokenizer_regex_dfa_t* dfa,
                         uint16_t state_id) {
  if (!dfa->end_anchor_bitmap) return false;
  return (dfa->end_anchor_bitmap[state_id / 64] & (1ULL << (state_id % 64))) !=
         0;
}

// Helper to check if a state is accepting.
bool IsAccepting(const iree_tokenizer_regex_dfa_t* dfa, uint16_t state_id) {
  return (dfa->accepting_bitmap[state_id / 64] & (1ULL << (state_id % 64))) !=
         0;
}

TEST(Compile, EndAnchorBitmapOptionalAnchor) {
  // Pattern: a$|a - accept state reachable via anchored and unanchored paths.
  // The accept state should NOT require end anchor (unanchored path exists).
  // Since no state requires anchors, HAS_ANCHORS flag should NOT be set.
  CompiledPattern pat("a$|a");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  // HAS_ANCHORS flag should NOT be set because no state requires anchors.
  // The pattern contains `$`, but the accept is reachable without it.
  EXPECT_FALSE((pat.dfa()->header->flags &
                IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS) != 0)
      << "DFA should NOT have HAS_ANCHORS flag (no state requires anchor)";

  // If HAS_ANCHORS is set for some reason, verify no accepts are in bitmap.
  if (pat.dfa()->end_anchor_bitmap) {
    for (uint16_t i = 0; i < pat.dfa()->header->num_states; ++i) {
      if (IsAccepting(pat.dfa(), i)) {
        EXPECT_FALSE(IsInEndAnchorBitmap(pat.dfa(), i))
            << "Accepting state " << i
            << " should NOT require end anchor for pattern 'a$|a' "
            << "(reachable via unanchored path)";
      }
    }
  }
}

TEST(Compile, EndAnchorBitmapOptionalAnchorReversed) {
  // Pattern: a|a$ - order reversed, should behave the same as a$|a.
  // Since no state requires anchors, HAS_ANCHORS flag should NOT be set.
  CompiledPattern pat("a|a$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  EXPECT_FALSE((pat.dfa()->header->flags &
                IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS) != 0)
      << "DFA should NOT have HAS_ANCHORS flag (no state requires anchor)";

  if (pat.dfa()->end_anchor_bitmap) {
    for (uint16_t i = 0; i < pat.dfa()->header->num_states; ++i) {
      if (IsAccepting(pat.dfa(), i)) {
        EXPECT_FALSE(IsInEndAnchorBitmap(pat.dfa(), i))
            << "Accepting state " << i
            << " should NOT require end anchor for pattern 'a|a$'";
      }
    }
  }
}

TEST(Compile, EndAnchorBitmapAllAnchored) {
  // Pattern: a$|b$ - both branches require end anchor.
  // All accepting states SHOULD be in end_anchor_bitmap.
  CompiledPattern pat("a$|b$");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  EXPECT_TRUE((pat.dfa()->header->flags &
               IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS) != 0);

  // All accepting states should require end anchor.
  bool found_accepting = false;
  for (uint16_t i = 0; i < pat.dfa()->header->num_states; ++i) {
    if (IsAccepting(pat.dfa(), i)) {
      found_accepting = true;
      EXPECT_TRUE(IsInEndAnchorBitmap(pat.dfa(), i))
          << "Accepting state " << i
          << " SHOULD require end anchor for pattern 'a$|b$' "
          << "(all paths go through $)";
    }
  }
  EXPECT_TRUE(found_accepting) << "Should have at least one accepting state";
}

TEST(Compile, EndAnchorBitmapTransitive) {
  // Pattern: ab$|ab - tests transitive propagation through intermediate states.
  // Since no state requires anchors (unanchored path exists), HAS_ANCHORS
  // should NOT be set.
  CompiledPattern pat("ab$|ab");
  ASSERT_TRUE(pat.ok()) << pat.error_message();

  EXPECT_FALSE((pat.dfa()->header->flags &
                IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS) != 0)
      << "DFA should NOT have HAS_ANCHORS flag (no state requires anchor)";

  if (pat.dfa()->end_anchor_bitmap) {
    for (uint16_t i = 0; i < pat.dfa()->header->num_states; ++i) {
      if (IsAccepting(pat.dfa(), i)) {
        EXPECT_FALSE(IsInEndAnchorBitmap(pat.dfa(), i))
            << "Accepting state " << i
            << " should NOT require end anchor for pattern 'ab$|ab'";
      }
    }
  }
}

}  // namespace
