// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/regex/exec.h"

#include <cstring>
#include <map>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Test DFA Builder
//===----------------------------------------------------------------------===//

// Helper class to build hand-crafted DFA binaries for testing.
// This will be replaced by the actual compiler once implemented.
class TestDfaBuilder {
 public:
  TestDfaBuilder() = default;

  void SetNumStates(uint16_t num_states) { num_states_ = num_states; }
  void SetStartState(uint16_t start) { start_state_ = start; }
  // Set unanchored start state (for position > 0, excludes ^ paths).
  // If not called, defaults to same as start_state.
  void SetUnanchoredStartState(uint16_t start) {
    unanchored_start_state_ = start;
    has_unanchored_start_ = true;
  }
  void SetFlags(uint16_t flags) { flags_ = flags; }

  // Add a transition: state --byte--> next_state.
  void AddTransition(uint16_t state, uint8_t byte, uint16_t next_state) {
    transitions_.push_back({state, byte, next_state});
  }

  // Add range of transitions: state --[lo,hi]--> next_state.
  void AddTransitionRange(uint16_t state, uint8_t lo, uint8_t hi,
                          uint16_t next_state) {
    for (int b = lo; b <= hi; ++b) {
      AddTransition(state, (uint8_t)b, next_state);
    }
  }

  void SetAccepting(uint16_t state) { accepting_.insert(state); }

  // Set lookahead for a state (negative character lookahead).
  void SetLookaheadNegChar(uint16_t state, uint8_t reject_char) {
    lookahead_[state] = {IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CHAR,
                         reject_char};
    flags_ |= IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD;
  }

  // Set lookahead for a state (negative shorthand class).
  void SetLookaheadNegShorthand(uint16_t state,
                                iree_tokenizer_regex_shorthand_t shorthand) {
    lookahead_[state] = {IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_SHORTHAND,
                         (uint8_t)shorthand};
    flags_ |= IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD;
  }

  // Set lookahead for a state with fallback (for patterns like a+(?!b)|a+).
  void SetLookaheadNegCharWithFallback(uint16_t state, uint8_t reject_char) {
    lookahead_[state] = {
        IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CHAR_WITH_FALLBACK,
        reject_char};
    flags_ |= IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD;
  }

  // Set lookahead for a state with fallback (shorthand class).
  void SetLookaheadNegShorthandWithFallback(
      uint16_t state, iree_tokenizer_regex_shorthand_t shorthand) {
    lookahead_[state] = {
        IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_SHORTHAND_WITH_FALLBACK,
        (uint8_t)shorthand};
    flags_ |= IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD;
  }

  std::vector<uint8_t> Build() {
    bool has_lookahead =
        (flags_ & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD) != 0;

    // Calculate sizes.
    size_t header_size = sizeof(iree_tokenizer_regex_dfa_header_t);
    size_t trans_size = (size_t)num_states_ * 256 * sizeof(uint16_t);
    size_t bitmap_words = (num_states_ + 63) / 64;
    size_t bitmap_size = bitmap_words * sizeof(uint64_t);
    size_t lookahead_size =
        has_lookahead ? num_states_ * sizeof(iree_tokenizer_regex_lookahead_t)
                      : 0;
    size_t total_size = header_size + trans_size + bitmap_size + lookahead_size;

    std::vector<uint8_t> data(total_size, 0);

    // Fill header.
    auto* header = (iree_tokenizer_regex_dfa_header_t*)data.data();
    header->magic = IREE_TOKENIZER_UTIL_REGEX_DFA_MAGIC;
    header->version = IREE_TOKENIZER_UTIL_REGEX_DFA_VERSION;
    header->flags = flags_;
    header->num_states = num_states_;
    header->num_accepting = (uint16_t)accepting_.size();
    header->start_state = start_state_;
    header->unanchored_start_state =
        has_unanchored_start_ ? unanchored_start_state_ : start_state_;

    // Fill transition table (default to NO_TRANSITION).
    auto* trans = (uint16_t*)(data.data() + header_size);
    for (size_t i = 0; i < (size_t)num_states_ * 256; ++i) {
      trans[i] = IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION;
    }
    for (const auto& t : transitions_) {
      trans[(size_t)t.state * 256 + t.byte] = t.next_state;
    }

    // Fill accepting bitmap.
    auto* bitmap = (uint64_t*)(data.data() + header_size + trans_size);
    for (uint16_t state : accepting_) {
      bitmap[state / 64] |= (1ULL << (state % 64));
    }

    // Fill lookahead table if present.
    if (has_lookahead) {
      auto* la = (iree_tokenizer_regex_lookahead_t*)(data.data() + header_size +
                                                     trans_size + bitmap_size);
      for (uint16_t i = 0; i < num_states_; ++i) {
        auto it = lookahead_.find(i);
        if (it != lookahead_.end()) {
          la[i] = it->second;
        } else {
          la[i] = {IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NONE, 0};
        }
      }
    }

    return data;
  }

 private:
  struct Transition {
    uint16_t state;
    uint8_t byte;
    uint16_t next_state;
  };

  uint16_t num_states_ = 0;
  uint16_t start_state_ = 0;
  uint16_t unanchored_start_state_ = 0;
  bool has_unanchored_start_ = false;
  uint16_t flags_ = 0;
  std::vector<Transition> transitions_;
  std::set<uint16_t> accepting_;
  std::map<uint16_t, iree_tokenizer_regex_lookahead_t> lookahead_;
};

// Build a DFA that matches exactly "abc".
// States: 0 (start) --'a'--> 1 --'b'--> 2 --'c'--> 3 (accept)
std::vector<uint8_t> BuildAbcDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(4);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(1, 'b', 2);
  builder.AddTransition(2, 'c', 3);
  builder.SetAccepting(3);
  return builder.Build();
}

// Build a DFA that matches [a-z]+ (one or more lowercase letters).
// States: 0 (start) --[a-z]--> 1 (accept, self-loop)
std::vector<uint8_t> BuildLowerAlphaDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransitionRange(0, 'a', 'z', 1);
  builder.AddTransitionRange(1, 'a', 'z', 1);
  builder.SetAccepting(1);
  return builder.Build();
}

// Build a DFA that matches [a-z]+ in Unicode mode.
// This enables UTF-8 decoding so we can test UTF-8 validation.
std::vector<uint8_t> BuildLowerAlphaUnicodeDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_UNICODE);
  builder.AddTransitionRange(0, 'a', 'z', 1);
  builder.AddTransitionRange(1, 'a', 'z', 1);
  builder.SetAccepting(1);
  return builder.Build();
}

// Build a DFA that matches [0-9]+ (one or more digits).
std::vector<uint8_t> BuildDigitsDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransitionRange(0, '0', '9', 1);
  builder.AddTransitionRange(1, '0', '9', 1);
  builder.SetAccepting(1);
  return builder.Build();
}

// Build a DFA that matches "a|b|c" (single char: a OR b OR c).
std::vector<uint8_t> BuildAltDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(0, 'b', 1);
  builder.AddTransition(0, 'c', 1);
  builder.SetAccepting(1);
  return builder.Build();
}

// Build a DFA that matches "a|abc|b" to test match resumption.
// When matching "abd":
//   - Matches "a" at position 0, continues trying "abc"
//   - Fails at 'd' (position 2) with last_accept = 1
//   - Should resume at position 1 to find "b"
// States:
//   0 (start) --'a'--> 1 (accept), --'b'--> 4 (accept)
//   1 --'b'--> 2
//   2 --'c'--> 3 (accept)
std::vector<uint8_t> BuildAOrAbcOrBDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(5);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(0, 'b', 4);
  builder.AddTransition(1, 'b', 2);
  builder.AddTransition(2, 'c', 3);
  builder.SetAccepting(1);  // "a"
  builder.SetAccepting(3);  // "abc"
  builder.SetAccepting(4);  // "b"
  return builder.Build();
}

//===----------------------------------------------------------------------===//
// Match Collection Helper
//===----------------------------------------------------------------------===//

struct MatchCollector {
  std::vector<std::pair<size_t, size_t>> matches;

  static iree_status_t Callback(void* user_data,
                                iree_tokenizer_regex_match_t match) {
    auto* self = (MatchCollector*)user_data;
    self->matches.push_back({match.start, match.end});
    return iree_ok_status();
  }
};

//===----------------------------------------------------------------------===//
// Loading Tests
//===----------------------------------------------------------------------===//

TEST(DfaLoad, ValidDfa) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_EXPECT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  EXPECT_EQ(iree_tokenizer_regex_dfa_state_count(&dfa), 4);
  EXPECT_EQ(iree_tokenizer_regex_dfa_start_state(&dfa), 0);
  EXPECT_FALSE(iree_tokenizer_regex_dfa_is_accepting(&dfa, 0));
  EXPECT_FALSE(iree_tokenizer_regex_dfa_is_accepting(&dfa, 1));
  EXPECT_FALSE(iree_tokenizer_regex_dfa_is_accepting(&dfa, 2));
  EXPECT_TRUE(iree_tokenizer_regex_dfa_is_accepting(&dfa, 3));
}

TEST(DfaLoad, TooSmall) {
  uint8_t data[4] = {0};
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data, sizeof(data)), &dfa);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(DfaLoad, InvalidMagic) {
  auto data = BuildAbcDfa();
  // Corrupt magic.
  data[0] = 'X';
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);
}

TEST(DfaLoad, UnsupportedVersion) {
  auto data = BuildAbcDfa();
  // Set version to 99.
  auto* header = (iree_tokenizer_regex_dfa_header_t*)data.data();
  header->version = 99;
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
}

TEST(DfaLoad, TruncatedData) {
  auto data = BuildAbcDfa();
  // Only provide half the data.
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size() / 2), &dfa);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(DfaLoad, UnalignedDataRejected) {
  auto data = BuildAbcDfa();
  // Create a buffer with extra byte at start to make data unaligned.
  std::vector<uint8_t> unaligned(data.size() + 16);
  // Find an offset that makes the data pointer non-8-byte-aligned.
  size_t offset = 1;
  while (((uintptr_t)(unaligned.data() + offset) % 8) == 0) {
    offset++;
  }
  memcpy(unaligned.data() + offset, data.data(), data.size());

  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(unaligned.data() + offset, data.size()), &dfa);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

//===----------------------------------------------------------------------===//
// Validation Tests
//===----------------------------------------------------------------------===//

TEST(DfaValidate, ValidDfa) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_EXPECT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));
}

TEST(DfaValidate, AcceptingCountMismatch) {
  auto data = BuildAbcDfa();
  // Corrupt accepting count in header.
  auto* header = (iree_tokenizer_regex_dfa_header_t*)data.data();
  header->num_accepting = 5;  // Wrong - only 1 accepting state.

  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, status);
}

TEST(DfaValidate, LookaheadNegClassRejected) {
  // LOOKAHEAD_NEG_CLASS is defined but not implemented.
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  builder.SetFlags(IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD);
  auto data = builder.Build();

  // Manually patch the lookahead type to NEG_CLASS.
  size_t header_size = sizeof(iree_tokenizer_regex_dfa_header_t);
  size_t trans_size = 2 * 256 * sizeof(uint16_t);
  size_t bitmap_size = sizeof(uint64_t);
  auto* lookahead =
      (iree_tokenizer_regex_lookahead_t*)(data.data() + header_size +
                                          trans_size + bitmap_size);
  lookahead[1].type = IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CLASS;
  lookahead[1].data = 0;

  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
}

TEST(DfaValidate, EmptyStringMatchRejected) {
  // Start state as accepting without end anchor = empty-string match.
  TestDfaBuilder builder;
  builder.SetNumStates(1);
  builder.SetStartState(0);
  builder.SetAccepting(0);
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(DfaLoad, SentinelCollisionRejected) {
  // num_states = 0xFFFF would collide with NO_TRANSITION sentinel.
  // We can't easily build such a large DFA, so we patch the header.
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  auto data = builder.Build();

  // Patch header to claim 0xFFFF states.
  auto* header = (iree_tokenizer_regex_dfa_header_t*)data.data();
  header->num_states = 0xFFFF;

  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

//===----------------------------------------------------------------------===//
// Step Function Tests
//===----------------------------------------------------------------------===//

TEST(DfaStep, BasicTransitions) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // Follow the "abc" path.
  // For ASCII bytes, codepoint == byte.
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 0, 'a', 'a'), 1);
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 1, 'b', 'b'), 2);
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 2, 'c', 'c'), 3);

  // Non-matching transitions.
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 0, 'x', 'x'),
            IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION);
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 1, 'a', 'a'),
            IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION);
}

//===----------------------------------------------------------------------===//
// Execution Tests
//===----------------------------------------------------------------------===//

TEST(DfaExec, ExactMatch) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("abc");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 3);
}

TEST(DfaExec, MultipleMatches) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("abcXabc");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 3);
  EXPECT_EQ(collector.matches[1].first, 4);
  EXPECT_EQ(collector.matches[1].second, 7);
}

TEST(DfaExec, Digits) {
  auto data = BuildDigitsDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("abc123xyz456");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 3);
  EXPECT_EQ(collector.matches[0].second, 6);
  EXPECT_EQ(collector.matches[1].first, 9);
  EXPECT_EQ(collector.matches[1].second, 12);
}

TEST(DfaExec, NoMatch) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("xyz");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  EXPECT_EQ(collector.matches.size(), 0);
}

TEST(DfaExec, EmptyInput) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  EXPECT_EQ(collector.matches.size(), 0);
}

TEST(DfaExec, PartialMatch) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  // "ab" is a partial match for "abc" - should not match.
  iree_string_view_t text = IREE_SVL("ab");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  EXPECT_EQ(collector.matches.size(), 0);
}

TEST(DfaExec, LeftmostLongestMatching) {
  auto data = BuildLowerAlphaDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("hello world");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  // Should match "hello" and "world", not individual letters.
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 5);
  EXPECT_EQ(collector.matches[1].first, 6);
  EXPECT_EQ(collector.matches[1].second, 11);
}

TEST(DfaExec, Alternation) {
  auto data = BuildAltDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("aXbYc");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 3);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
  EXPECT_EQ(collector.matches[2].first, 4);
  EXPECT_EQ(collector.matches[2].second, 5);
}

TEST(DfaExec, ResumeAfterEmitAtCorrectPosition) {
  // After emitting a match due to a failed longer path, resume scanning from
  // the end of the emitted match, not from the failure position. This ensures
  // characters between last_accept and the failure point are checked for new
  // matches.
  //
  // Pattern: a|abc|b
  // Input: "abd"
  // Execution:
  //   - 'a' at 0: match state 1 (accept), last_accept = 1
  //   - 'b' at 1: state 2 (not accepting)
  //   - 'd' at 2: NO_TRANSITION, emit [0,1)="a"
  //   - Resume at position 1 (not 2!)
  //   - 'b' at 1: match state 4 (accept), last_accept = 2
  //   - 'd' at 2: NO_TRANSITION, emit [1,2)="b"
  //   - Resume at position 2
  //   - 'd' at 2: no match
  // Expected: matches "a" at [0,1) and "b" at [1,2)
  auto data = BuildAOrAbcOrBDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("abd");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  // Without the fix, only "a" is found (skips "b" at position 1).
  // With the fix, both "a" and "b" are found.
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);  // "a"
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);  // "b"
}

//===----------------------------------------------------------------------===//
// Streaming Tests
//===----------------------------------------------------------------------===//

TEST(DfaStreaming, ChunkedInput) {
  auto data = BuildLowerAlphaDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // Feed "hello world" in two chunks: "hello " and "world".
  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  const char* chunk1 = "hello ";
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(chunk1, strlen(chunk1)), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));

  // "hello" should be emitted after processing the space.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 5);

  const char* chunk2 = "world";
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(chunk2, strlen(chunk2)),
      strlen(chunk1), /*stride=*/NULL, MatchCollector::Callback, &collector));

  // "world" hasn't been emitted yet (no terminator seen).
  EXPECT_EQ(collector.matches.size(), 1);

  // Finalize to flush pending match.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, strlen(chunk1) + strlen(chunk2), MatchCollector::Callback,
      &collector));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[1].first, 6);
  EXPECT_EQ(collector.matches[1].second, 11);
}

TEST(DfaStreaming, MatchAcrossChunks) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // Feed "abc" as "a", "b", "c".
  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("a", 1), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));
  EXPECT_EQ(collector.matches.size(), 0);

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("b", 1), 1, /*stride=*/NULL,
      MatchCollector::Callback, &collector));
  EXPECT_EQ(collector.matches.size(), 0);

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("c", 1), 2, /*stride=*/NULL,
      MatchCollector::Callback, &collector));
  // Still no match emitted - we're in accepting state but waiting for more.
  EXPECT_EQ(collector.matches.size(), 0);

  // Finalize emits the match.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 3);
}

//===----------------------------------------------------------------------===//
// Convenience Wrapper Tests
//===----------------------------------------------------------------------===//

TEST(DfaCountMatches, Basic) {
  auto data = BuildLowerAlphaDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  iree_host_size_t count = 0;
  iree_string_view_t text = IREE_SVL("one two three");
  IREE_EXPECT_OK(iree_tokenizer_regex_count_matches(&dfa, text, &count));
  EXPECT_EQ(count, 3);
}

TEST(DfaHasMatch, Found) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  bool has_match = false;
  iree_string_view_t text = IREE_SVL("XXXabcYYY");
  IREE_EXPECT_OK(iree_tokenizer_regex_has_match(&dfa, text, &has_match));
  EXPECT_TRUE(has_match);
}

TEST(DfaHasMatch, NotFound) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  bool has_match = false;
  iree_string_view_t text = IREE_SVL("XXXYYY");
  IREE_EXPECT_OK(iree_tokenizer_regex_has_match(&dfa, text, &has_match));
  EXPECT_FALSE(has_match);
}

//===----------------------------------------------------------------------===//
// Lookahead at Chunk Boundaries Tests
//===----------------------------------------------------------------------===//

// Build a DFA that matches "a(?!b)" - 'a' NOT followed by 'b'.
// States: 0 (start) --'a'--> 1 (accept with neg lookahead for 'b')
std::vector<uint8_t> BuildANotFollowedByBDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  builder.SetLookaheadNegChar(1, 'b');
  return builder.Build();
}

TEST(Streaming, LookaheadAcrossChunk_Reject) {
  // Pattern: a(?!b) - match 'a' NOT followed by 'b'
  // Input: "ab" split as "a" + "b"
  // Expected: NO match (b follows a)
  auto data = BuildANotFollowedByBDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "a".
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("a", 1), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Feed "b".
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("b", 1), 1, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Finalize.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector));

  // Should NOT match - 'b' follows 'a'.
  EXPECT_EQ(collector.matches.size(), 0);
}

TEST(Streaming, LookaheadAcrossChunk_Accept) {
  // Pattern: a(?!b) - match 'a' NOT followed by 'b'
  // Input: "ac" split as "a" + "c"
  // Expected: match "a" at [0,1)
  auto data = BuildANotFollowedByBDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "a".
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("a", 1), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Feed "c".
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("c", 1), 1, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Finalize.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector));

  // Should match "a" - 'c' doesn't reject the lookahead.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

TEST(Streaming, LookaheadAtRealEOS) {
  // Pattern: a(?!b) - match 'a' NOT followed by 'b'
  // Input: "a" (nothing follows)
  // Expected: match "a" at [0,1) - EOS means no 'b' follows.
  auto data = BuildANotFollowedByBDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "a".
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("a", 1), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Finalize (no more input).
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 1, MatchCollector::Callback, &collector));

  // Should match "a" - at EOS, negative lookahead passes.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

// Build a DFA for GPT-2 style pattern: \s+(?!\S) - whitespace NOT followed by
// non-whitespace. States: 0 (start) --\s--> 1 (accept with neg lookahead
// for \S)
//                1 --\s--> 1 (self-loop)
std::vector<uint8_t> BuildWhitespaceNotFollowedByNonWhitespaceDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  // \s matches space, tab, newline, etc.
  builder.AddTransition(0, ' ', 1);
  builder.AddTransition(0, '\t', 1);
  builder.AddTransition(0, '\n', 1);
  builder.AddTransition(0, '\r', 1);
  builder.AddTransition(1, ' ', 1);
  builder.AddTransition(1, '\t', 1);
  builder.AddTransition(1, '\n', 1);
  builder.AddTransition(1, '\r', 1);
  builder.SetAccepting(1);
  builder.SetLookaheadNegShorthand(1, IREE_TOKENIZER_UTIL_REGEX_SHORTHAND_S);
  return builder.Build();
}

TEST(Streaming, WhitespaceNegLookahead_Reject) {
  // Pattern: \s+(?!\S)
  // Input: "  x" split as "  " + "x"
  //
  // DFA semantics: At each accepting state, we check if the next character
  // satisfies the lookahead. With leftmost-longest matching:
  // - First space: lookahead checks next char (second space). Space is not
  //   non-whitespace, so lookahead passes. Record accept at position 1.
  // - Second space: lookahead checks next char (x). x IS non-whitespace,
  //   so lookahead fails. Don't update accept.
  // - When 'x' comes, no transition - emit match for first space.
  //
  // Expected: match " " at [0,1) - the first space is followed by whitespace.
  auto data = BuildWhitespaceNotFollowedByNonWhitespaceDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "  " (two spaces).
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("  ", 2), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Feed "x".
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("x", 1), 2, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Finalize.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector));

  // DFA matches the first space (followed by whitespace, not non-whitespace).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

TEST(Streaming, WhitespaceNegLookahead_AtEOS) {
  // Pattern: \s+(?!\S)
  // Input: "  " (two spaces at end)
  // Expected: match "  " at [0,2) - EOS means no non-whitespace follows.
  auto data = BuildWhitespaceNotFollowedByNonWhitespaceDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "  ".
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("  ", 2), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Finalize.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector));

  // Should match "  " - at EOS, negative lookahead for \S passes.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 2);
}

TEST(Streaming, WhitespaceNegLookahead_ImmediateEval) {
  // Pattern: \s+(?!\S)
  // Input: "  x" (all in one chunk)
  // Expected: match " " at [0,1) - first space passes lookahead (followed by
  // space), second space fails lookahead (followed by x).
  //
  // Tests that lookahead is evaluated at each accepting state, not just the
  // final one. Immediate lookahead when next char is available within the
  // same chunk.
  auto data = BuildWhitespaceNotFollowedByNonWhitespaceDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("  x");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  // First space only - second space's lookahead failed.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

TEST(Streaming, WhitespaceNegLookahead_AllFail) {
  // Pattern: \s+(?!\S)
  // Input: " x" (single space followed by non-whitespace)
  // Expected: no matches - the only space is followed by non-whitespace.
  auto data = BuildWhitespaceNotFollowedByNonWhitespaceDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL(" x");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 0);
}

TEST(Streaming, WhitespaceNegLookahead_MultipleSections) {
  // Pattern: \s+(?!\S)
  // Input: "a  b  " (multiple whitespace sections, different outcomes)
  // Expected:
  //   - "  " at [1,2) - first space of section 1 passes, second fails
  //   - "  " at [4,6) - at EOS, both spaces pass
  auto data = BuildWhitespaceNotFollowedByNonWhitespaceDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("a  b  ");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 2);  // First space only
  EXPECT_EQ(collector.matches[1].first, 4);
  EXPECT_EQ(collector.matches[1].second, 6);  // Both trailing spaces
}

TEST(Streaming, WhitespaceNegLookahead_ManySpacesBeforeNonWS) {
  // Pattern: \s+(?!\S)
  // Input: "     x" (5 spaces then x)
  // Expected: match "    " at [0,4) - first 4 spaces pass, 5th fails.
  auto data = BuildWhitespaceNotFollowedByNonWhitespaceDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("     x");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 4);  // First 4 spaces
}

TEST(Streaming, WhitespaceNegLookahead_ChunkBoundaryVariants) {
  // Pattern: \s+(?!\S)
  // Input: "   x" split as " " + "  x" (1 + 3)
  // Test chunk boundary handling with immediate lookahead evaluation.
  auto data = BuildWhitespaceNotFollowedByNonWhitespaceDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // First chunk: single space at boundary (can't evaluate lookahead).
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(" ", 1), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Second chunk: "  x"
  // When this arrives:
  // - Pending lookahead from chunk 1 is resolved: next is ' ' (passes)
  // - Then we process two more spaces and x.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("  x", 3), 1,
      /*stride=*/NULL, MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector));

  // The first two spaces pass lookahead, third space fails.
  // So we get match at [0, 2).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 2);
}

//===----------------------------------------------------------------------===//
// Overlapping Partial Match Tests
//===----------------------------------------------------------------------===//

// Build a DFA that matches exactly "ab".
std::vector<uint8_t> BuildAbDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(3);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(1, 'b', 2);
  builder.SetAccepting(2);
  return builder.Build();
}

TEST(Matching, OverlappingPartialMatch) {
  // Pattern: ab
  // Input: "aab"
  // Expected: match at [1,3) = "ab"
  // The first 'a' starts a partial match, second 'a' fails it,
  // but the second 'a' starts a new match that succeeds.
  auto data = BuildAbDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("aab");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 3);
}

// Build a DFA that matches exactly "abc".
std::vector<uint8_t> BuildAbcOnlyDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(4);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(1, 'b', 2);
  builder.AddTransition(2, 'c', 3);
  builder.SetAccepting(3);
  return builder.Build();
}

TEST(Matching, RepeatedPrefixFailure) {
  // Pattern: abc
  // Input: "ababc"
  // Expected: match at [2,5) = "abc"
  auto data = BuildAbcOnlyDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("ababc");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 5);
}

// Build a DFA that matches exactly "aab".
std::vector<uint8_t> BuildAabDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(4);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(1, 'a', 2);
  builder.AddTransition(2, 'b', 3);
  builder.SetAccepting(3);
  return builder.Build();
}

TEST(Matching, MultipleFalseStarts) {
  // Pattern: aab
  // Input: "aaaab"
  // Expected: match at [2,5) = "aab"
  auto data = BuildAabDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("aaaab");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 5);
}

//===----------------------------------------------------------------------===//
// Review Round 2 Test Cases (Codex + Gemini 3 Pro)
//===----------------------------------------------------------------------===//

// Test "aaab" overlap - requires proper backtracking to match_start + 1.
TEST(Matching, OverlappingPrefixBacktrack) {
  // Pattern: aab
  // Input: "aaab"
  // Expected: match at [1,4) = "aab"
  // This tests the case where we have "aa" then fail on third 'a'.
  // We must restart from position 1 (not position 2) to find the match.
  auto data = BuildAabDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("aaab");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Test accept-then-fail-later scenario.
TEST(Matching, AcceptThenFailLater) {
  // Build DFA: matches "a" or "ab"
  // State 0 --'a'--> State 1 (accepting)
  // State 1 --'b'--> State 2 (accepting)
  // Input: "ac"
  // Expected: match "a" at [0,1), then nothing (c doesn't match)
  TestDfaBuilder builder;
  builder.SetNumStates(3);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(1, 'b', 2);
  builder.SetAccepting(1);
  builder.SetAccepting(2);
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("ac");
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  // Should match "a" at [0,1). The 'c' causes transition failure from state 1,
  // but we had an accept at state 1, so we emit that match.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

// Test streaming with accept-then-fail across chunk boundary.
TEST(Streaming, AcceptThenFailAcrossChunk) {
  // Same DFA as above: matches "a" or "ab"
  TestDfaBuilder builder;
  builder.SetNumStates(3);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(1, 'b', 2);
  builder.SetAccepting(1);
  builder.SetAccepting(2);
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "a" (in accepting state 1)
  const uint8_t chunk1[] = {'a'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk1, sizeof(chunk1)),
      0, /*stride=*/NULL, MatchCollector::Callback, &collector));

  // Chunk 2: "c" (fails transition from state 1)
  const uint8_t chunk2[] = {'c'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk2, sizeof(chunk2)),
      sizeof(chunk1), /*stride=*/NULL, MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector));

  // Should match "a" at [0,1).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

//===----------------------------------------------------------------------===//
// Additional Edge Case Tests
//===----------------------------------------------------------------------===//

// Test NEG_CHAR lookahead with ASCII (should still work after pseudo-byte fix).
TEST(Lookahead, NegCharAscii) {
  // Pattern: a(?!b) - 'a' not followed by 'b'
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  builder.SetLookaheadNegChar(1, 'b');
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // "ab" - should NOT match (b follows a)
  {
    MatchCollector collector;
    iree_string_view_t text = IREE_SVL("ab");
    IREE_EXPECT_OK(iree_tokenizer_regex_exec(
        &dfa, text, MatchCollector::Callback, &collector));
    EXPECT_EQ(collector.matches.size(), 0);
  }

  // "ac" - should match 'a' at [0,1)
  {
    MatchCollector collector;
    iree_string_view_t text = IREE_SVL("ac");
    IREE_EXPECT_OK(iree_tokenizer_regex_exec(
        &dfa, text, MatchCollector::Callback, &collector));
    ASSERT_EQ(collector.matches.size(), 1);
    EXPECT_EQ(collector.matches[0].first, 0);
    EXPECT_EQ(collector.matches[0].second, 1);
  }
}

// Test Unicode backtracking - does it handle codepoint boundaries correctly?
TEST(Matching, UnicodeBacktrack) {
  // Build DFA: matches two Unicode letters in a row
  TestDfaBuilder builder;
  builder.SetNumStates(3);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_UNICODE);
  builder.AddTransition(0, IREE_TOKENIZER_UTIL_REGEX_PSEUDO_LETTER, 1);
  builder.AddTransition(1, IREE_TOKENIZER_UTIL_REGEX_PSEUDO_LETTER, 2);
  builder.SetAccepting(2);
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // Input: "1é2éé" - should match "éé" at end
  // '1' = 1 byte, 'é' = 2 bytes, '2' = 1 byte, 'é' = 2 bytes, 'é' = 2 bytes
  // Byte positions: 1[0], é[1-2], 2[3], é[4-5], é[6-7]
  // Match should be at [4, 8)
  MatchCollector collector;
  const uint8_t input[] = {'1', 0xC3, 0xA9, '2', 0xC3, 0xA9, 0xC3, 0xA9};
  iree_string_view_t text = {(const char*)input, sizeof(input)};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, text, MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 4);
  EXPECT_EQ(collector.matches[0].second, 8);
}

// Test empty chunk handling.
TEST(Streaming, EmptyChunk) {
  auto data = BuildLowerAlphaDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed: "ab" then empty then "cd"
  const uint8_t chunk1[] = {'a', 'b'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk1, sizeof(chunk1)),
      0, /*stride=*/NULL, MatchCollector::Callback, &collector));

  // Empty chunk (pass nullptr with 0 length)
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk1, 0),
      sizeof(chunk1), /*stride=*/NULL, MatchCollector::Callback, &collector));

  const uint8_t chunk2[] = {'c', 'd'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk2, sizeof(chunk2)),
      sizeof(chunk1), /*stride=*/NULL, MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector));

  // Should match "abcd" at [0,4)
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Test multiple consecutive lookahead deferrals.
TEST(Streaming, MultipleLookaheadDefers) {
  // Pattern: a(?!b) - 'a' not followed by 'b'
  // Input: "a" in tiny 1-byte chunks, followed by "c"
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  builder.SetLookaheadNegChar(1, 'b');
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed 'a' alone - should defer lookahead
  const uint8_t a[] = {'a'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)a, 1), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));
  EXPECT_EQ(collector.matches.size(), 0);  // Not emitted yet

  // Feed 'c' - lookahead passes (c != b), should emit match
  const uint8_t c[] = {'c'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)c, 1), 1,
      /*stride=*/NULL, MatchCollector::Callback, &collector));

  // Finalize
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector));

  // Should have match "a" at [0,1)
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

//===----------------------------------------------------------------------===//
// UTF-8 Validation Tests
//===----------------------------------------------------------------------===//

// Invalid continuation byte: 2-byte sequence with non-continuation second byte.
TEST(Utf8Validation, InvalidContinuationByte) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // 0xC3 expects continuation (0x80-0xBF), but we give 0x40 ('@').
  // Should decode as replacement char (doesn't match), then '@' (doesn't
  // match).
  const uint8_t input[] = {0xC3, 0x40, 'b', 'c'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)input, sizeof(input)), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector));

  // 0xC3 becomes replacement char (no match), 0x40='@' (no match), "bc"
  // matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Overlong encoding: 2-byte encoding of ASCII (should be 1 byte).
TEST(Utf8Validation, OverlongTwoByte) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // 0xC0 0x80 is overlong encoding of U+0000 (NUL).
  // 0xC1 0xBF is overlong encoding of U+007F (DEL).
  // Both should become replacement char.
  const uint8_t input[] = {0xC0, 0x80, 'a', 'b'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)input, sizeof(input)), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector));

  // Overlong sequence becomes replacement char (no match), then "ab" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Overlong encoding: 3-byte encoding of 2-byte range.
TEST(Utf8Validation, OverlongThreeByte) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // 0xE0 0x80 0x80 is overlong encoding of U+0000.
  // Valid 3-byte sequences start at U+0800, so 0xE0 must have 2nd byte >= 0xA0.
  const uint8_t input[] = {0xE0, 0x80, 0x80, 'x', 'y'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)input, sizeof(input)), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector));

  // Overlong becomes replacement char, then "xy" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 3);
  EXPECT_EQ(collector.matches[0].second, 5);
}

// UTF-16 surrogate codepoints (U+D800-U+DFFF) are invalid in UTF-8.
TEST(Utf8Validation, SurrogateCodepoint) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // 0xED 0xA0 0x80 encodes U+D800 (high surrogate start).
  // 0xED 0xBF 0xBF encodes U+DFFF (low surrogate end).
  // Both are invalid and should become replacement char.
  const uint8_t input[] = {0xED, 0xA0, 0x80, 'z'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)input, sizeof(input)), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector));

  // Surrogate becomes replacement char, then "z" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 3);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Invalid leading byte (0xFF is never valid in UTF-8).
TEST(Utf8Validation, InvalidLeadingByte) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // 0xFF and 0xFE are never valid UTF-8 leading bytes.
  const uint8_t input[] = {0xFF, 0xFE, 'a', 'b'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)input, sizeof(input)), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector));

  // 0xFF and 0xFE become replacement chars, then "ab" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Continuation byte without leading byte (0x80-0xBF alone).
TEST(Utf8Validation, StrayContiunationByte) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // 0x80 alone is invalid (continuation byte without leader).
  const uint8_t input[] = {0x80, 0x85, 'c', 'd'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)input, sizeof(input)), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector));

  // Each stray continuation byte becomes replacement char, then "cd" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 4);
}

//===----------------------------------------------------------------------===//
// Cross-Chunk Backtracking Tests
//===----------------------------------------------------------------------===//

// Tests for the rewind buffer that enables proper backtracking when a partial
// match starts in chunk A and fails in chunk B.

TEST(CrossChunkBacktrack, AabPatternCrossChunk) {
  // Pattern: "aab"
  // Input: "aa" + "ab" (split so match_start=0 is in chunk 1)
  // When we process "ab" in chunk 2:
  //   - 'a' transitions from state 2 to... NO_TRANSITION (pattern expects 'b')
  //   - We need to backtrack to match_start + 1 = 1
  //   - Position 1 is in chunk 1, so we use the rewind buffer
  // Expected: match "aab" at [1,4)
  auto data = BuildAabDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "aa"
  const uint8_t chunk1[] = {'a', 'a'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk1, sizeof(chunk1)),
      0, /*stride=*/NULL, MatchCollector::Callback, &collector));

  // Chunk 2: "ab"
  const uint8_t chunk2[] = {'a', 'b'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk2, sizeof(chunk2)),
      sizeof(chunk1), /*stride=*/NULL, MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector));

  // Should match "aab" at [1,4).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 4);
}

TEST(CrossChunkBacktrack, AbaPatternCrossChunk) {
  // Pattern: "aba"
  // Input: "ab" + "aba" (split chunks)
  // Chunk 1 processing:
  //   - 'a' starts match at position 0
  //   - 'b' continues match
  // Chunk 2 processing:
  //   - 'a' completes match, emit [0,3)
  //   - Continue from position 3, see 'b' (can't start "aba")
  //   - Position 4 is 'a' but input ends before pattern completes
  // Expected: one match "aba" at [0,3) from "ab" + "a"

  // Build a DFA that matches "aba".
  TestDfaBuilder builder;
  builder.SetNumStates(4);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(1, 'b', 2);
  builder.AddTransition(2, 'a', 3);
  builder.SetAccepting(3);
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "ab"
  const uint8_t chunk1[] = {'a', 'b'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk1, sizeof(chunk1)),
      0, /*stride=*/NULL, MatchCollector::Callback, &collector));

  // Chunk 2: "aba"
  const uint8_t chunk2[] = {'a', 'b', 'a'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk2, sizeof(chunk2)),
      sizeof(chunk1), /*stride=*/NULL, MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector));

  // Should match "aba" at [0,3) from chunk1 "ab" + chunk2 first 'a'.
  // After emitting this match, we continue from position 3.
  // Position 3 is 'b' which can't start "aba", position 4 is 'a' but there's
  // no more input to complete the pattern. So only one match.
  // (Non-overlapping semantics - tokenization resumes from match end.)
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 3);
}

TEST(CrossChunkBacktrack, SingleByteChunks) {
  // Pattern: "aab"
  // Input: "a" + "a" + "a" + "b" (extreme fragmentation)
  // Tests that the rewind buffer correctly accumulates across many chunks.
  auto data = BuildAabDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "aaab" one byte at a time.
  size_t offset = 0;
  const uint8_t bytes[] = {'a', 'a', 'a', 'b'};
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_string_view((const char*)&bytes[i], 1), offset,
        /*stride=*/NULL, MatchCollector::Callback, &collector));
    offset += 1;
  }

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, offset, MatchCollector::Callback, &collector));

  // Should match "aab" at [1,4).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 4);
}

TEST(CrossChunkBacktrack, BacktrackMultipleTimes) {
  // Pattern: "aab"
  // Input: "aaaab" split as "aaa" + "ab"
  // This requires multiple backtracks to find the match.
  auto data = BuildAabDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "aaa"
  const uint8_t chunk1[] = {'a', 'a', 'a'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk1, sizeof(chunk1)),
      0, /*stride=*/NULL, MatchCollector::Callback, &collector));

  // Chunk 2: "ab"
  const uint8_t chunk2[] = {'a', 'b'};
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view((const char*)chunk2, sizeof(chunk2)),
      sizeof(chunk1), /*stride=*/NULL, MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector));

  // Should match "aab" at [2,5).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 5);
}

//===----------------------------------------------------------------------===//
// Backtrack State Reset Tests
//===----------------------------------------------------------------------===//

// These tests verify that executor state is properly reset on backtrack.
// Both within-chunk and within-buffer backtrack paths must clear state like
// has_accept_fallback and best_branch_idx/best_match_end to prevent stale
// state from leaking into subsequent match attempts.

// Build a DFA that matches a+(?!X)|a+ (one or more 'a' with lookahead and
// fallback).
// States: 0 (start) --'a'--> 1 (accept, self-loop with lookahead (?!X) and
// fallback)
std::vector<uint8_t> BuildOnePlusAWithFallbackDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.AddTransition(1, 'a', 1);  // Self-loop for a+.
  builder.SetAccepting(1);
  builder.SetLookaheadNegCharWithFallback(1, 'X');  // (?!X) with fallback.
  return builder.Build();
}

// Build a DFA that matches \s+(?!\S)|\s+ (whitespace with negative lookahead
// and fallback branch).
//
// This pattern is used in GPT-2/Llama tokenizers. The semantics are:
// - Match one or more whitespace characters
// - If followed by whitespace (or EOS), lookahead passes -> use lookahead match
// - If followed by non-whitespace, lookahead fails -> use fallback match
//
// The difference from BuildWhitespaceNotFollowedByNonWhitespaceDfa is that
// this version ALWAYS matches (via fallback), while that one rejects when
// followed by non-whitespace.
//
// States: 0 (start) --\s--> 1 (accept with neg lookahead for \S + fallback)
//         1 --\s--> 1 (self-loop)
std::vector<uint8_t> BuildWhitespaceWithFallbackDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  // \s matches space, tab, newline, carriage return.
  builder.AddTransition(0, ' ', 1);
  builder.AddTransition(0, '\t', 1);
  builder.AddTransition(0, '\n', 1);
  builder.AddTransition(0, '\r', 1);
  builder.AddTransition(1, ' ', 1);
  builder.AddTransition(1, '\t', 1);
  builder.AddTransition(1, '\n', 1);
  builder.AddTransition(1, '\r', 1);
  builder.SetAccepting(1);
  // (?!\S) with fallback to |\s+.
  builder.SetLookaheadNegShorthandWithFallback(
      1, IREE_TOKENIZER_UTIL_REGEX_SHORTHAND_S);
  return builder.Build();
}

TEST(BacktrackStateReset, WithinChunkBacktrackClearsFallback) {
  // Pattern: a+(?!X)|a+ - matches one or more 'a' not followed by 'X', OR just
  // a+ (fallback).
  //
  // Input: "aXab"
  //
  // This exercises the within-chunk backtrack path. At position 0, we match 'a'
  // and reach the accepting state with fallback. At position 1, we see 'X'
  // which causes lookahead to FAIL ((?!X) rejects 'X'). We then emit using
  // fallback at [0,1) and reset.
  //
  // After reset, we try from position 1 ('X'), which doesn't match. Then
  // position 2 ('a') starts a new match. Position 3 ('b') causes NO_TRANSITION
  // but lookahead PASSES ('b' != 'X'), so we emit [2,3).
  //
  // The bug would cause stale fallback state from the first match to interfere
  // with the second match. This test verifies the fix works correctly.
  auto data = BuildOnePlusAWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view("aXab"),
                                           MatchCollector::Callback,
                                           &collector));

  // Should have two matches: [0,1) and [2,3).
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
}

TEST(BacktrackStateReset, SingleChunkAaXa) {
  // Pattern: a+(?!X)|a+ on input "aaXa" (single chunk).
  //
  // This is the single-chunk reference for StreamingBacktrackWithFallback.
  // Expected matches:
  // - [0,1): Position 0 'a' followed by 'a' (not 'X') → lookahead passes
  // - [1,2): Position 1 'a' followed by 'X' → lookahead fails, use fallback
  // - [3,4): Position 3 'a' at EOS → lookahead passes
  auto data = BuildOnePlusAWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view("aaXa"),
                                           MatchCollector::Callback,
                                           &collector));

  // Should have three matches.
  ASSERT_EQ(collector.matches.size(), 3);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);
  EXPECT_EQ(collector.matches[2].first, 3);
  EXPECT_EQ(collector.matches[2].second, 4);
}

TEST(BacktrackStateReset, BacktrackAfterFailedLookahead) {
  // Pattern: a+(?!X)|a+
  //
  // Input: "aXaXab"
  //
  // Multiple matches where lookahead fails multiple times:
  // - [0,1): 'a' at 0, lookahead fails at 'X', use fallback
  // - [2,3): 'a' at 2, lookahead fails at 'X', use fallback
  // - [4,5): 'a' at 4, lookahead passes at 'b', use lookahead accept
  //
  // This verifies fallback state doesn't leak across multiple backtracks.
  auto data = BuildOnePlusAWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  IREE_EXPECT_OK(
      iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view("aXaXab"),
                                MatchCollector::Callback, &collector));

  // Should have three matches.
  ASSERT_EQ(collector.matches.size(), 3);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
  EXPECT_EQ(collector.matches[2].first, 4);
  EXPECT_EQ(collector.matches[2].second, 5);
}

TEST(BacktrackStateReset, StreamingBacktrackWithFallback) {
  // Pattern: a+(?!X)|a+ on input "aa" + "Xa" (split across chunks).
  //
  // This is the streaming version of SingleChunkAaXa - both must produce
  // the same 3 matches. The split tests cross-chunk resume handling.
  //
  // Processing:
  // - Position 0: 'a' matches, lookahead checks 'a' at 1 (not 'X') → PASS [0,1)
  // - Position 1: 'a' matches, lookahead needs position 2 → DEFER
  // - Chunk 2: Pending lookahead checks 'X' at 2 → FAIL, emit [0,1)
  // - Resume from position 1 via rewind buffer (cross-chunk resume)
  // - Position 1: 'a' matches, lookahead checks 'X' at 2 → FAIL via fallback
  // - Position 2: 'X' → NO_TRANSITION, emit [1,2) via fallback
  // - Position 2: 'X' doesn't match from start
  // - Position 3: 'a' matches, EOS lookahead passes → emit [3,4)
  auto data = BuildOnePlusAWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "aa".
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("aa", 2), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Chunk 2: "Xa".
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("Xa", 2), 2, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector));

  // Should have three matches (same as SingleChunkAaXa):
  // - [0,1): Position 0 'a' where lookahead passed
  // - [1,2): Position 1 'a' where lookahead failed, use fallback
  // - [3,4): Position 3 'a' where EOS lookahead passes
  ASSERT_EQ(collector.matches.size(), 3);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);
  EXPECT_EQ(collector.matches[2].first, 3);
  EXPECT_EQ(collector.matches[2].second, 4);
}

//===----------------------------------------------------------------------===//
// Shorthand Fallback Streaming Tests
//===----------------------------------------------------------------------===//

// These tests verify streaming behavior for patterns with shorthand-class
// negative lookahead AND fallback branches, like \s+(?!\S)|\s+.
//
// This is distinct from the character-based fallback tests above (a+(?!X)|a+)
// because shorthand classes (\S, \s, \d, etc.) have different evaluation
// semantics than single-character lookahead.
//
// The pattern \s+(?!\S)|\s+ is used in GPT-2/Llama tokenizers. The key behavior
// (as verified in regex_test.cc LookaheadFallback_* tests) is:
// - At each accepting position, evaluate lookahead against next char
// - If lookahead PASSES (next is not \S), update last_accept
// - If lookahead FAILS (next is \S), record fallback but don't update
// last_accept
// - When match ends (NO_TRANSITION), emit using last_accept if it exists
// - Then restart from end of emitted match, potentially emitting more via
// fallback
//
// Example: "  x" produces TWO matches [0,1) and [1,2), not one match [0,2).
// This is because the pattern prefers positions where lookahead passes.

TEST(ShorthandFallbackStreaming, ChunkBoundaryMidWhitespaceRun) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "   x" split as "  " + " x" (chunk boundary mid-whitespace)
  //
  // Key insight: When deferred lookahead FAILS, we emit using the previous
  // last_accept, not the deferred position. This matches regex_test.cc
  // behavior.
  //
  // Processing:
  // - Chunk 1 "  ":
  //   - Position 0: ' ' matches, lookahead at 1 (' ' not \S) -> PASS,
  //   last_accept=1
  //   - Position 1: ' ' matches, lookahead needs pos 2 -> DEFER
  // - Chunk 2 " x":
  //   - Resolve deferred: pos 2 is ' ' (not \S) -> PASS, last_accept=2
  //   - Position 2: ' ' matches, lookahead at 3 ('x' is \S) -> FAIL
  //   - Position 3: 'x' NO_TRANSITION -> emit [0,2) using last_accept
  //   - Restart at position 2
  //   - Position 2: ' ' matches, lookahead at 3 ('x') -> FAIL
  //   - Position 3: NO_TRANSITION -> emit [2,3) via fallback
  //
  // Expected: [0,2) and [2,3) - matches regex_test.cc LookaheadFallback
  // behavior
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: two spaces.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("  ", 2), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Chunk 2: space then 'x'.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(" x", 2), 2, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector));

  // Two matches: lookahead-passed match, then fallback for final space.
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 2);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
}

TEST(ShorthandFallbackStreaming, LookaheadDeferredThenFails) {
  // Pattern: \s+(?!\S)|\s+
  // Input: " " + "x" (single space, lookahead deferred, then fails)
  //
  // This tests the critical path where:
  // 1. Lookahead is deferred at chunk boundary (can't see next char)
  // 2. Next chunk reveals non-whitespace, causing lookahead to fail
  // 3. Fallback must be used correctly
  //
  // Processing:
  // - Chunk 1 " ": space matches, lookahead needs pos 1 -> DEFER
  // - Chunk 2 "x": pending lookahead checks 'x' (\S) -> FAIL
  //   Use fallback to emit [0,1)
  //
  // Expected: [0,1)
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: single space at chunk boundary.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(" ", 1), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Chunk 2: 'x' causes deferred lookahead to fail.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("x", 1), 1, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector));

  // Single space matches via fallback.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

TEST(ShorthandFallbackStreaming, LookaheadDeferredThenPasses) {
  // Pattern: \s+(?!\S)|\s+
  // Input: " " + " " (lookahead deferred, then passes)
  //
  // Contrast with above: here deferred lookahead PASSES.
  //
  // Processing:
  // - Chunk 1 " ": space matches, lookahead needs pos 1 -> DEFER
  // - Chunk 2 " ": pending lookahead checks ' ' (\s, not \S) -> PASS
  //   Update last_accept = 1, continue matching
  // - Position 1: ' ' matches, lookahead needs pos 2 -> DEFER to finalize
  // - Finalize: EOS means lookahead passes -> emit [0,2)
  //
  // Expected: [0,2)
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: single space.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(" ", 1), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Chunk 2: another space.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(" ", 1), 1, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector));

  // Both spaces match (lookahead passed).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 2);
}

TEST(ShorthandFallbackStreaming, BacktrackAcrossChunksWithFallback) {
  // Pattern: \s+(?!\S)|\s+ on input "  xy  ".
  //
  // After emitting [0,1), the rewind buffer allows restart at position 1.
  // Streaming must produce same results as single-chunk processing.
  //
  // Processing:
  // - Chunk 1 "  ":
  //   - Pos 0: ' ', lookahead at 1 (' ' not \S) -> PASS, last_accept=1
  //   - Pos 1: ' ', lookahead needs pos 2 -> DEFER
  // - Chunk 2 "xy  ":
  //   - Resolve deferred: 'x' at pos 2 IS \S -> FAIL
  //   - Emit [0,1) using last_accept
  //   - Resume at pos 1 via rewind buffer
  //   - Pos 1: ' ', lookahead at 2 ('x' IS \S) -> FAIL, emit [1,2) via fallback
  //   - Pos 2,3: 'xy' don't match
  //   - Pos 4,5: ' ' matches, last at EOS
  // - Finalize: EOS -> PASS, emit [4,6)
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: two spaces.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("  ", 2), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Chunk 2: "xy  " - non-whitespace then whitespace.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("xy  ", 4), 2,
      /*stride=*/NULL, MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 6, MatchCollector::Callback, &collector));

  // Three matches (same as single-chunk "  xy  "):
  ASSERT_EQ(collector.matches.size(), 3);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);  // Lookahead passed at pos 0.
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);  // Fallback at pos 1.
  EXPECT_EQ(collector.matches[2].first, 4);
  EXPECT_EQ(collector.matches[2].second, 6);  // Trailing spaces at EOS.
}

TEST(ShorthandFallbackStreaming, MultipleSectionsAcrossManyChunks) {
  // Pattern: \s+(?!\S)|\s+ on input "a  b  c".
  //
  // Streaming must produce same results as single-chunk processing.
  // The rewind buffer allows restart at any position in a previous chunk.
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed chunks one at a time.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("a ", 2), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(" b", 2), 2, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("  ", 2), 4, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("c", 1), 6, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 7, MatchCollector::Callback, &collector));

  // Four matches (same as single-chunk "a  b  c"):
  // - [1,2): first space after 'a' (lookahead passed: next is ' ')
  // - [2,3): second space (fallback: next is 'b')
  // - [4,5): first space of "  " section (lookahead passed: next is ' ')
  // - [5,6): second space (fallback: next is 'c')
  ASSERT_EQ(collector.matches.size(), 4);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 2);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
  EXPECT_EQ(collector.matches[2].first, 4);
  EXPECT_EQ(collector.matches[2].second, 5);
  EXPECT_EQ(collector.matches[3].first, 5);
  EXPECT_EQ(collector.matches[3].second, 6);
}

TEST(ShorthandFallbackStreaming, SingleByteChunksStressTest) {
  // Pattern: \s+(?!\S)|\s+ on input "  x  " fed byte-by-byte.
  //
  // Streaming must produce same results as single-chunk processing.
  // The rewind buffer preserves bytes for cross-chunk resume.
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  const char* input = "  x  ";
  for (size_t i = 0; i < 5; ++i) {
    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_string_view(&input[i], 1), i,
        /*stride=*/NULL, MatchCollector::Callback, &collector));
  }

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 5, MatchCollector::Callback, &collector));

  // Three matches (same as single-chunk "  x  "):
  // - [0,1): first space (lookahead passed: next is ' ')
  // - [1,2): second space (fallback: next is 'x')
  // - [3,5): trailing spaces at EOS
  ASSERT_EQ(collector.matches.size(), 3);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);
  EXPECT_EQ(collector.matches[2].first, 3);
  EXPECT_EQ(collector.matches[2].second, 5);
}

TEST(ShorthandFallbackStreaming, FallbackStateNotLeakedOnBacktrack) {
  // Pattern: \s+(?!\S)|\s+
  // Input: " x y"
  //
  // This specifically tests that fallback state from a failed match doesn't
  // leak into subsequent match attempts. The executor must properly reset:
  // - has_accept_fallback
  // - last_accept_fallback
  //
  // Processing:
  // - Pos 0: ' ' matches, lookahead at pos 1 ('x') -> FAIL
  //   Emit [0,1) via fallback, reset state for next match
  // - Pos 1: 'x' doesn't match, skip
  // - Pos 2: ' ' matches, lookahead at pos 3 ('y') -> FAIL
  //   Emit [2,3) via fallback
  //
  // If fallback state leaked, the second match might incorrectly use stale
  // fallback position from the first match.
  //
  // Expected: [0,1), [2,3)
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  IREE_EXPECT_OK(iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view(" x y"),
                                           MatchCollector::Callback,
                                           &collector));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
}

TEST(ShorthandFallbackStreaming, StreamingFallbackStateNotLeaked) {
  // Pattern: \s+(?!\S)|\s+
  // Input: " " + "x " + "y" (streaming version of above)
  //
  // Same scenario as FallbackStateNotLeakedOnBacktrack but with chunk
  // boundaries at interesting positions.
  //
  // Expected: [0,1), [2,3)
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: space with deferred lookahead.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(" ", 1), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Chunk 2: 'x' fails lookahead, then ' ' starts new match.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("x ", 2), 1, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Chunk 3: 'y' fails lookahead for second match.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("y", 1), 3, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
}

TEST(ShorthandFallbackStreaming, TabAndNewlineHandling) {
  // Pattern: \s+(?!\S)|\s+ on input "\t\nx".
  //
  // Verify that shorthand \s correctly matches tab/newline, not just space.
  // Streaming must produce same results as single-chunk processing.
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: tab + newline.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("\t\n", 2), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));

  // Chunk 2: 'x'.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("x", 1), 2, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector));

  // Two matches (same as single-chunk "\t\nx"):
  // - [0,1): tab (lookahead passed: next is '\n', not \S)
  // - [1,2): newline (fallback: next is 'x', which IS \S)
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);
}

//===----------------------------------------------------------------------===//
// Anchor State Tests
//===----------------------------------------------------------------------===//

// Build a DFA that matches "^Hello" - "Hello" only at start of input.
// States: 0 (start at pos 0) --H-e-l-l-o--> 5 (accept)
//         1 (dead state for pos > 0, no transitions)
//
// The key is start_state=0 but unanchored_start_state=1 (dead state).
// This ensures ^ anchor semantics: only matches at position 0.
std::vector<uint8_t> BuildStartAnchoredHelloDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(7);  // 0-5 for path, 6 for dead state.
  builder.SetStartState(0);
  builder.SetUnanchoredStartState(6);  // Dead state for pos > 0.
  // Note: No HAS_ANCHORS flag needed - the different start states alone give
  // us start anchor semantics. HAS_ANCHORS is for per-state anchor bitmaps.

  // Path: 0 --'H'--> 1 --'e'--> 2 --'l'--> 3 --'l'--> 4 --'o'--> 5 (accept)
  builder.AddTransition(0, 'H', 1);
  builder.AddTransition(1, 'e', 2);
  builder.AddTransition(2, 'l', 3);
  builder.AddTransition(3, 'l', 4);
  builder.AddTransition(4, 'o', 5);
  builder.SetAccepting(5);

  // State 6 is the dead state: no transitions, not accepting.
  // Any character from state 6 leads to NO_TRANSITION.

  return builder.Build();
}

TEST(AnchorState, StartAnchorResetsInMatch) {
  // After matching a start-anchored pattern (^), the executor must reset
  // in_match=false so callers don't incorrectly believe they're mid-match
  // when processing subsequent input.
  auto data = BuildStartAnchoredHelloDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "Hello World" - should match "Hello" at position 0.
  const char* text = "Hello World";
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view(text, strlen(text)), 0,
      /*stride=*/NULL, MatchCollector::Callback, &collector));

  // Critical assertion: After processing, in_match must be false.
  // The match at [0,5) was emitted; there's no ongoing partial match.
  EXPECT_FALSE(state.in_match) << "in_match should be false after anchored "
                                  "match completes";

  // Verify the match was found correctly.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 5);  // "Hello"

  // Finalize should not produce additional matches.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, strlen(text), MatchCollector::Callback, &collector));
  EXPECT_EQ(collector.matches.size(), 1);  // Still just one match.
}

TEST(AnchorState, StartAnchorStreamingResetsInMatch) {
  // Streaming variant: Feed input in chunks, verify state is correct.
  // Pattern: ^Hello
  // Input: "He" + "llo World"
  auto data = BuildStartAnchoredHelloDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "He" - partial match, in_match should be true.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("He", 2), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));
  EXPECT_TRUE(state.in_match) << "Should be mid-match after 'He'";
  EXPECT_EQ(collector.matches.size(), 0) << "No match emitted yet";

  // Chunk 2: "llo World" - completes match, in_match should become false.
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("llo World", 9), 2,
      /*stride=*/NULL, MatchCollector::Callback, &collector));

  // Critical: After match completes, in_match must be false.
  EXPECT_FALSE(state.in_match) << "in_match should be false after anchored "
                                  "match completes (streaming)";

  // Verify match.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 5);  // "Hello"
}

//===----------------------------------------------------------------------===//
// Consecutive Newline Tests
//===----------------------------------------------------------------------===//

// These tests verify that consecutive newlines are matched SEPARATELY when
// followed by non-whitespace, as required by patterns like \s+(?!\S)|\s+.
//
// Pattern: \s+(?!\S)|\s+
// Input: "\n\nA" (two newlines followed by 'A')
//
// Expected behavior:
// - First '\n': lookahead checks next char ('\n'), which is NOT \S -> PASS
//   Record accept at position 1, continue matching
// - Second '\n': lookahead checks next char ('A'), which IS \S -> FAIL
//   Don't update last_accept, but record fallback at position 2
// - 'A': NO_TRANSITION from whitespace state
//   Emit match at last_accept (position 1, NOT 2)
// - Resume at position 1, match second '\n'
// - 'A': NO_TRANSITION, emit fallback match at position 2
//
// Result: Two separate matches [0,1) and [1,2), NOT one combined [0,2)

TEST(ConsecutiveNewlines, TwoNewlinesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\n\nA"
  //
  // First '\n' passes lookahead (followed by whitespace).
  // Second '\n' fails lookahead (followed by 'A'), uses fallback.
  // Result: two separate matches.
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  IREE_EXPECT_OK(
      iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view("\n\nA"),
                                MatchCollector::Callback, &collector));

  // Should have TWO matches: [0,1) and [1,2)
  // NOT one combined match [0,2)
  ASSERT_EQ(collector.matches.size(), 2)
      << "Consecutive newlines before text should produce separate matches";
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);  // First '\n'
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);  // Second '\n'
}

TEST(ConsecutiveNewlines, TwoNewlinesAtEndOfInput) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "A\n\n"
  //
  // At EOS, negative lookahead (?!\S) passes (no character = not \S).
  // So both newlines should be combined into one match.
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  IREE_EXPECT_OK(
      iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view("A\n\n"),
                                MatchCollector::Callback, &collector));

  // Should have ONE match for both newlines: [1,3)
  // (At EOS, lookahead passes for all positions)
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 3);  // Both '\n\n'
}

TEST(ConsecutiveNewlines, ThreeNewlinesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\n\n\nA"
  //
  // Expected:
  // - First two '\n': lookahead passes (followed by whitespace)
  // - Third '\n': lookahead fails (followed by 'A')
  // Result: Match [0,2) for first two, then [2,3) for third
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  IREE_EXPECT_OK(
      iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view("\n\n\nA"),
                                MatchCollector::Callback, &collector));

  // Should have TWO matches: [0,2) and [2,3)
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 2);  // First two '\n\n'
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);  // Third '\n'
}

TEST(ConsecutiveNewlines, StreamingTwoNewlinesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\n" + "\nA" (split across chunks)
  //
  // This tests the streaming case where the lookahead decision is deferred.
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: first '\n' - lookahead deferred (at chunk boundary).
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("\n", 1), 0, /*stride=*/NULL,
      MatchCollector::Callback, &collector));

  // Chunk 2: second '\n' followed by 'A'.
  // Deferred lookahead resolves: next char is '\n' (passes).
  // Second '\n' fails lookahead (followed by 'A').
  IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_string_view("\nA", 2), 1,
      /*stride=*/NULL, MatchCollector::Callback, &collector));

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector));

  // Should have TWO matches: [0,1) and [1,2)
  ASSERT_EQ(collector.matches.size(), 2)
      << "Streaming consecutive newlines should produce separate matches";
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);  // First '\n'
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);  // Second '\n'
}

//===----------------------------------------------------------------------===//
// Byte-by-Byte Streaming Tests
//===----------------------------------------------------------------------===//

// These tests verify that the regex executor produces correct results when
// processing input one byte at a time. This is the most challenging case for
// streaming because every character boundary is also a chunk boundary.

TEST(StreamingByteByByte, TwoNewlinesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\n\nA" fed byte-by-byte
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed each byte separately.
  const char* input = "\n\nA";
  for (size_t i = 0; i < 3; ++i) {
    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_string_view(&input[i], 1), i,
        /*stride=*/NULL, MatchCollector::Callback, &collector))
        << "Failed at byte " << i;
  }

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector));

  // Should have TWO matches: [0,1) and [1,2)
  ASSERT_EQ(collector.matches.size(), 2)
      << "Byte-by-byte: consecutive newlines should produce separate matches";
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);
}

TEST(StreamingByteByByte, ThreeNewlinesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\n\n\nA" fed byte-by-byte
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed each byte separately.
  const char* input = "\n\n\nA";
  for (size_t i = 0; i < 4; ++i) {
    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_string_view(&input[i], 1), i,
        /*stride=*/NULL, MatchCollector::Callback, &collector))
        << "Failed at byte " << i;
  }

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector));

  // Should have TWO matches: [0,2) and [2,3)
  // First two newlines pass lookahead, third fails.
  ASSERT_EQ(collector.matches.size(), 2)
      << "Byte-by-byte: three newlines should produce two matches";
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 2);  // First two '\n\n'
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);  // Third '\n'
}

TEST(StreamingByteByByte, TwoSpacesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "  x" fed byte-by-byte
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  const char* input = "  x";
  for (size_t i = 0; i < 3; ++i) {
    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_string_view(&input[i], 1), i,
        /*stride=*/NULL, MatchCollector::Callback, &collector))
        << "Failed at byte " << i;
  }

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector));

  // Should have TWO matches: [0,1) and [1,2)
  ASSERT_EQ(collector.matches.size(), 2)
      << "Byte-by-byte: two spaces before text should produce separate matches";
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 1);
  EXPECT_EQ(collector.matches[1].second, 2);
}

TEST(StreamingByteByByte, TrailingWhitespace) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "A  " fed byte-by-byte
  // At EOS, lookahead passes for all whitespace.
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  const char* input = "A  ";
  for (size_t i = 0; i < 3; ++i) {
    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_string_view(&input[i], 1), i,
        /*stride=*/NULL, MatchCollector::Callback, &collector))
        << "Failed at byte " << i;
  }

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector));

  // Should have ONE match: [1,3) - both trailing spaces combined.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 3);
}

TEST(StreamingByteByByte, MixedWhitespaceAndText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "a  b" fed byte-by-byte
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  const char* input = "a  b";
  for (size_t i = 0; i < 4; ++i) {
    IREE_EXPECT_OK(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_string_view(&input[i], 1), i,
        /*stride=*/NULL, MatchCollector::Callback, &collector))
        << "Failed at byte " << i;
  }

  IREE_EXPECT_OK(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector));

  // Should have TWO matches: [1,2) and [2,3)
  // First space passes lookahead (next is space), second fails (next is 'b').
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 2);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
}

//===----------------------------------------------------------------------===//
// Streaming with Various Chunk Sizes
//===----------------------------------------------------------------------===//

// Helper to test a pattern with various chunk sizes.
void TestWithChunkSizes(
    const iree_tokenizer_regex_dfa_t* dfa, const char* input,
    const std::vector<std::pair<size_t, size_t>>& expected_matches,
    const std::vector<size_t>& chunk_sizes) {
  size_t input_len = strlen(input);

  for (size_t chunk_size : chunk_sizes) {
    MatchCollector collector;
    iree_tokenizer_regex_exec_state_t state;
    iree_tokenizer_regex_exec_initialize(&state, dfa);

    size_t offset = 0;
    while (offset < input_len) {
      size_t remaining = input_len - offset;
      size_t this_chunk = (chunk_size < remaining) ? chunk_size : remaining;
      IREE_ASSERT_OK(iree_tokenizer_regex_exec_feed(
          dfa, &state, iree_make_string_view(input + offset, this_chunk),
          offset, /*stride=*/NULL, MatchCollector::Callback, &collector))
          << "Failed at offset " << offset << " with chunk_size " << chunk_size;
      offset += this_chunk;
    }

    IREE_ASSERT_OK(iree_tokenizer_regex_exec_finalize(
        dfa, &state, input_len, MatchCollector::Callback, &collector))
        << "Finalize failed with chunk_size " << chunk_size;

    ASSERT_EQ(collector.matches.size(), expected_matches.size())
        << "Wrong match count with chunk_size " << chunk_size;
    for (size_t i = 0; i < expected_matches.size(); ++i) {
      EXPECT_EQ(collector.matches[i].first, expected_matches[i].first)
          << "Match " << i << " start mismatch with chunk_size " << chunk_size;
      EXPECT_EQ(collector.matches[i].second, expected_matches[i].second)
          << "Match " << i << " end mismatch with chunk_size " << chunk_size;
    }
  }
}

TEST(StreamingChunkSizes, TwoNewlinesBeforeText) {
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // Expected: [0,1) and [1,2)
  std::vector<std::pair<size_t, size_t>> expected = {{0, 1}, {1, 2}};
  TestWithChunkSizes(&dfa, "\n\nA", expected, {1, 2, 3, 4, 5, 10, 100});
}

TEST(StreamingChunkSizes, ThreeNewlinesBeforeText) {
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // Expected: [0,2) and [2,3)
  std::vector<std::pair<size_t, size_t>> expected = {{0, 2}, {2, 3}};
  TestWithChunkSizes(&dfa, "\n\n\nA", expected, {1, 2, 3, 4, 5, 10, 100});
}

TEST(StreamingChunkSizes, MixedContent) {
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  IREE_ASSERT_OK(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa));

  // "a  b  " -> matches at [1,2), [2,3), [4,6)
  // First space passes (next is space), second fails (next is 'b').
  // Trailing spaces all pass (at EOS).
  std::vector<std::pair<size_t, size_t>> expected = {{1, 2}, {2, 3}, {4, 6}};
  TestWithChunkSizes(&dfa, "a  b  ", expected, {1, 2, 3, 4, 5, 6, 10});
}

}  // namespace
