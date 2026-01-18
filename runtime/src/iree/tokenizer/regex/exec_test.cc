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
    lookahead_[state] = {IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_CHAR, reject_char};
    flags_ |= IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD;
  }

  // Set lookahead for a state (negative shorthand class).
  void SetLookaheadNegShorthand(uint16_t state,
                                iree_tokenizer_regex_shorthand_t shorthand) {
    lookahead_[state] = {IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_SHORTHAND,
                         (uint8_t)shorthand};
    flags_ |= IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD;
  }

  // Set lookahead for a state with fallback (for patterns like a+(?!b)|a+).
  void SetLookaheadNegCharWithFallback(uint16_t state, uint8_t reject_char) {
    lookahead_[state] = {IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_CHAR_WITH_FALLBACK,
                         reject_char};
    flags_ |= IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD;
  }

  // Set lookahead for a state with fallback (shorthand class).
  void SetLookaheadNegShorthandWithFallback(
      uint16_t state, iree_tokenizer_regex_shorthand_t shorthand) {
    lookahead_[state] = {
        IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_SHORTHAND_WITH_FALLBACK,
        (uint8_t)shorthand};
    flags_ |= IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD;
  }

  std::vector<uint8_t> Build() {
    bool has_lookahead =
        (flags_ & IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD) != 0;

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
    header->magic = IREE_TOKENIZER_REGEX_DFA_MAGIC;
    header->version = IREE_TOKENIZER_REGEX_DFA_VERSION;
    header->flags = flags_;
    header->num_states = num_states_;
    header->num_accepting = (uint16_t)accepting_.size();
    header->start_state = start_state_;
    header->unanchored_start_state = start_state_;

    // Fill transition table (default to NO_TRANSITION).
    auto* trans = (uint16_t*)(data.data() + header_size);
    for (size_t i = 0; i < (size_t)num_states_ * 256; ++i) {
      trans[i] = IREE_TOKENIZER_REGEX_NO_TRANSITION;
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
          la[i] = {IREE_TOKENIZER_REGEX_LOOKAHEAD_NONE, 0};
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
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE);
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
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

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
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(DfaLoad, InvalidMagic) {
  auto data = BuildAbcDfa();
  // Corrupt magic.
  data[0] = 'X';
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  EXPECT_TRUE(iree_status_is_failed_precondition(status));
  iree_status_ignore(status);
}

TEST(DfaLoad, UnsupportedVersion) {
  auto data = BuildAbcDfa();
  // Set version to 99.
  auto* header = (iree_tokenizer_regex_dfa_header_t*)data.data();
  header->version = 99;
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  EXPECT_TRUE(iree_status_is_unimplemented(status));
  iree_status_ignore(status);
}

TEST(DfaLoad, TruncatedData) {
  auto data = BuildAbcDfa();
  // Only provide half the data.
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size() / 2), &dfa);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
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
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Validation Tests
//===----------------------------------------------------------------------===//

TEST(DfaValidate, ValidDfa) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));
}

TEST(DfaValidate, AcceptingCountMismatch) {
  auto data = BuildAbcDfa();
  // Corrupt accepting count in header.
  auto* header = (iree_tokenizer_regex_dfa_header_t*)data.data();
  header->num_accepting = 5;  // Wrong - only 1 accepting state.

  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  EXPECT_TRUE(iree_status_is_data_loss(status));
  iree_status_ignore(status);
}

TEST(DfaValidate, LookaheadNegClassRejected) {
  // LOOKAHEAD_NEG_CLASS is defined but not implemented.
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD);
  auto data = builder.Build();

  // Manually patch the lookahead type to NEG_CLASS.
  size_t header_size = sizeof(iree_tokenizer_regex_dfa_header_t);
  size_t trans_size = 2 * 256 * sizeof(uint16_t);
  size_t bitmap_size = sizeof(uint64_t);
  auto* lookahead =
      (iree_tokenizer_regex_lookahead_t*)(data.data() + header_size +
                                          trans_size + bitmap_size);
  lookahead[1].type = IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_CLASS;
  lookahead[1].data = 0;

  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa);
  EXPECT_TRUE(iree_status_is_unimplemented(status));
  iree_status_ignore(status);
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
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
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
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Step Function Tests
//===----------------------------------------------------------------------===//

TEST(DfaStep, BasicTransitions) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // Follow the "abc" path.
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 0, 'a'), 1);
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 1, 'b'), 2);
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 2, 'c'), 3);

  // Non-matching transitions.
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 0, 'x'),
            IREE_TOKENIZER_REGEX_NO_TRANSITION);
  EXPECT_EQ(iree_tokenizer_regex_dfa_step(&dfa, 1, 'a'),
            IREE_TOKENIZER_REGEX_NO_TRANSITION);
}

//===----------------------------------------------------------------------===//
// Execution Tests
//===----------------------------------------------------------------------===//

TEST(DfaExec, ExactMatch) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("abc");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 3);
}

TEST(DfaExec, MultipleMatches) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("abcXabc");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 3);
  EXPECT_EQ(collector.matches[1].first, 4);
  EXPECT_EQ(collector.matches[1].second, 7);
}

TEST(DfaExec, Digits) {
  auto data = BuildDigitsDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("abc123xyz456");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 3);
  EXPECT_EQ(collector.matches[0].second, 6);
  EXPECT_EQ(collector.matches[1].first, 9);
  EXPECT_EQ(collector.matches[1].second, 12);
}

TEST(DfaExec, NoMatch) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("xyz");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

  EXPECT_EQ(collector.matches.size(), 0);
}

TEST(DfaExec, EmptyInput) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

  EXPECT_EQ(collector.matches.size(), 0);
}

TEST(DfaExec, PartialMatch) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  // "ab" is a partial match for "abc" - should not match.
  iree_string_view_t text = IREE_SVL("ab");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

  EXPECT_EQ(collector.matches.size(), 0);
}

TEST(DfaExec, LeftmostLongestMatching) {
  auto data = BuildLowerAlphaDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("hello world");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("aXbYc");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("abd");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // Feed "hello world" in two chunks: "hello " and "world".
  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  const char* chunk1 = "hello ";
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state,
      iree_make_const_byte_span((const uint8_t*)chunk1, strlen(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // "hello" should be emitted after processing the space.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 5);

  const char* chunk2 = "world";
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state,
      iree_make_const_byte_span((const uint8_t*)chunk2, strlen(chunk2)),
      strlen(chunk1), MatchCollector::Callback, &collector)));

  // "world" hasn't been emitted yet (no terminator seen).
  EXPECT_EQ(collector.matches.size(), 1);

  // Finalize to flush pending match.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, strlen(chunk1) + strlen(chunk2), MatchCollector::Callback,
      &collector)));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[1].first, 6);
  EXPECT_EQ(collector.matches[1].second, 11);
}

TEST(DfaStreaming, MatchAcrossChunks) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // Feed "abc" as "a", "b", "c".
  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"a", 1), 0,
      MatchCollector::Callback, &collector)));
  EXPECT_EQ(collector.matches.size(), 0);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"b", 1), 1,
      MatchCollector::Callback, &collector)));
  EXPECT_EQ(collector.matches.size(), 0);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"c", 1), 2,
      MatchCollector::Callback, &collector)));
  // Still no match emitted - we're in accepting state but waiting for more.
  EXPECT_EQ(collector.matches.size(), 0);

  // Finalize emits the match.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  iree_host_size_t count = 0;
  iree_string_view_t text = IREE_SVL("one two three");
  EXPECT_TRUE(iree_status_is_ok(
      iree_tokenizer_regex_count_matches(&dfa, text, &count)));
  EXPECT_EQ(count, 3);
}

TEST(DfaHasMatch, Found) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  bool has_match = false;
  iree_string_view_t text = IREE_SVL("XXXabcYYY");
  EXPECT_TRUE(iree_status_is_ok(
      iree_tokenizer_regex_has_match(&dfa, text, &has_match)));
  EXPECT_TRUE(has_match);
}

TEST(DfaHasMatch, NotFound) {
  auto data = BuildAbcDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  bool has_match = false;
  iree_string_view_t text = IREE_SVL("XXXYYY");
  EXPECT_TRUE(iree_status_is_ok(
      iree_tokenizer_regex_has_match(&dfa, text, &has_match)));
  EXPECT_FALSE(has_match);
}

//===----------------------------------------------------------------------===//
// UTF-8 Streaming Tests
//===----------------------------------------------------------------------===//

// Build a DFA that matches \p{L}+ (one or more letters) with Unicode flag.
// Uses pseudo-byte PSEUDO_LETTER (0x80) for non-ASCII letters.
std::vector<uint8_t> BuildUnicodeLettersDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE);
  // ASCII letters.
  builder.AddTransitionRange(0, 'a', 'z', 1);
  builder.AddTransitionRange(0, 'A', 'Z', 1);
  builder.AddTransitionRange(1, 'a', 'z', 1);
  builder.AddTransitionRange(1, 'A', 'Z', 1);
  // Unicode letters via pseudo-byte.
  builder.AddTransition(0, IREE_TOKENIZER_REGEX_PSEUDO_LETTER, 1);
  builder.AddTransition(1, IREE_TOKENIZER_REGEX_PSEUDO_LETTER, 1);
  builder.SetAccepting(1);
  return builder.Build();
}

TEST(Streaming, SplitUtf8_TwoByte) {
  // Test: Split 2-byte UTF-8 sequence (é = 0xC3 0xA9) across chunks.
  // Input: "café" split as "caf\xC3" + "\xA9"
  // Expected: Match "café" as single word (positions 0-5, since é is 2 bytes).
  auto data = BuildUnicodeLettersDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "caf" + first byte of é.
  const uint8_t chunk1[] = {'c', 'a', 'f', 0xC3};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // Feed second byte of é.
  const uint8_t chunk2[] = {0xA9};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  // Finalize.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector)));

  // Should have matched "café" (5 bytes: c, a, f, 0xC3, 0xA9).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 5);
}

TEST(Streaming, SplitUtf8_ThreeByte) {
  // Test: Split 3-byte UTF-8 sequence (中 = 0xE4 0xB8 0xAD) across chunks.
  // Input: "hi中" split as "hi\xE4\xB8" + "\xAD"
  auto data = BuildUnicodeLettersDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "hi" + first two bytes of 中.
  const uint8_t chunk1[] = {'h', 'i', 0xE4, 0xB8};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // Feed last byte of 中.
  const uint8_t chunk2[] = {0xAD};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector)));

  // Should have matched "hi中" (5 bytes: h, i, 0xE4, 0xB8, 0xAD).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 5);
}

TEST(Streaming, SplitUtf8_FourByte) {
  // Test: Split 4-byte UTF-8 sequence (😀 = 0xF0 0x9F 0x98 0x80) across chunks.
  // Feed each byte in a separate chunk.
  auto data = BuildUnicodeLettersDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "hi" + each byte of 😀 separately.
  const uint8_t chunk1[] = {'h', 'i', 0xF0};
  const uint8_t chunk2[] = {0x9F};
  const uint8_t chunk3[] = {0x98};
  const uint8_t chunk4[] = {0x80};

  size_t offset = 0;
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), offset,
      MatchCollector::Callback, &collector)));
  offset += sizeof(chunk1);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)), offset,
      MatchCollector::Callback, &collector)));
  offset += sizeof(chunk2);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk3, sizeof(chunk3)), offset,
      MatchCollector::Callback, &collector)));
  offset += sizeof(chunk3);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk4, sizeof(chunk4)), offset,
      MatchCollector::Callback, &collector)));
  offset += sizeof(chunk4);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, offset, MatchCollector::Callback, &collector)));

  // 😀 is category Symbol (Emoticons), not Letter, so won't match.
  // "hi" should match (2 bytes).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 2);
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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "a".
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"a", 1), 0,
      MatchCollector::Callback, &collector)));

  // Feed "b".
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"b", 1), 1,
      MatchCollector::Callback, &collector)));

  // Finalize.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector)));

  // Should NOT match - 'b' follows 'a'.
  EXPECT_EQ(collector.matches.size(), 0);
}

TEST(Streaming, LookaheadAcrossChunk_Accept) {
  // Pattern: a(?!b) - match 'a' NOT followed by 'b'
  // Input: "ac" split as "a" + "c"
  // Expected: match "a" at [0,1)
  auto data = BuildANotFollowedByBDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "a".
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"a", 1), 0,
      MatchCollector::Callback, &collector)));

  // Feed "c".
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"c", 1), 1,
      MatchCollector::Callback, &collector)));

  // Finalize.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "a".
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"a", 1), 0,
      MatchCollector::Callback, &collector)));

  // Finalize (no more input).
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 1, MatchCollector::Callback, &collector)));

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
  builder.SetLookaheadNegShorthand(1, IREE_TOKENIZER_REGEX_SHORTHAND_S);
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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "  " (two spaces).
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"  ", 2), 0,
      MatchCollector::Callback, &collector)));

  // Feed "x".
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"x", 1), 2,
      MatchCollector::Callback, &collector)));

  // Finalize.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "  ".
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"  ", 2), 0,
      MatchCollector::Callback, &collector)));

  // Finalize.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector)));

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
  // This tests immediate lookahead evaluation when next char is available
  // within the same chunk. Critical regression test for the fix that evaluates
  // lookahead at EACH accepting state, not just the final one.
  auto data = BuildWhitespaceNotFollowedByNonWhitespaceDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("  x");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL(" x");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("a  b  ");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("     x");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // First chunk: single space at boundary (can't evaluate lookahead).
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)" ", 1), 0,
      MatchCollector::Callback, &collector)));

  // Second chunk: "  x"
  // When this arrives:
  // - Pending lookahead from chunk 1 is resolved: next is ' ' (passes)
  // - Then we process two more spaces and x.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"  x", 3), 1,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("aab");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("ababc");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("aaaab");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 5);
}

//===----------------------------------------------------------------------===//
// Review Round 2 Test Cases (Codex + Gemini 3 Pro)
//===----------------------------------------------------------------------===//

// Build a DFA that matches PSEUDO_LETTER (Unicode letters via \p{L}).
std::vector<uint8_t> BuildUnicodeLetterDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE);
  // Match any Unicode letter (pseudo-byte 0x80).
  builder.AddTransition(0, IREE_TOKENIZER_REGEX_PSEUDO_LETTER, 1);
  builder.AddTransition(1, IREE_TOKENIZER_REGEX_PSEUDO_LETTER, 1);
  builder.SetAccepting(1);
  return builder.Build();
}

// Partial UTF-8 transition failure should retry from start state.
TEST(Streaming, SplitUtf8TransitionFailure_Retry) {
  // Pattern: \p{L}+ (Unicode letters)
  // Input: "1é" where '1' doesn't match \p{L}, but 'é' does.
  // Split: chunk1 = "1\xC3", chunk2 = "\xA9" (é = 0xC3 0xA9)
  // Expected: match "é" at position 1 (bytes 1-3).
  //
  // Bug scenario (before fix):
  // - '1' at pos 0: no transition from start, skip
  // - partial 0xC3 buffered at chunk end
  // - chunk2: complete é decoded, but we're at start state
  // - transition succeeds (letter), but if it had failed from a partial match,
  //   we need to retry from start.
  //
  // More specific test: "x" starts a match that "é" fails, then "é" should
  // start a new match.

  // Build DFA: matches "x" followed by ASCII letter, OR just Unicode letter.
  // State 0 (start) --'x'--> State 1
  // State 1 --[a-z]--> State 2 (accepting)
  // State 0 --PSEUDO_LETTER--> State 3 (accepting)
  TestDfaBuilder builder;
  builder.SetNumStates(4);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE);
  builder.AddTransition(0, 'x', 1);
  builder.AddTransitionRange(1, 'a', 'z', 2);  // x followed by lowercase
  builder.AddTransition(0, IREE_TOKENIZER_REGEX_PSEUDO_LETTER,
                        3);  // Or just letter
  builder.SetAccepting(2);
  builder.SetAccepting(3);
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // Input: "x" + "é" (split). 'x' starts match, 'é' fails (not a-z),
  // but 'é' should start new match as PSEUDO_LETTER.
  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "x\xC3" (x + first byte of é)
  const uint8_t chunk1[] = {'x', 0xC3};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: "\xA9" (second byte of é)
  const uint8_t chunk2[] = {0xA9};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector)));

  // Should match "é" at position 1 (bytes 1-3).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 1);   // Start at byte 1
  EXPECT_EQ(collector.matches[0].second, 3);  // End at byte 3
}

// Lookahead peek on split UTF-8 should defer, not decode as replacement.
TEST(Streaming, LookaheadPeekSplitUtf8_Defer) {
  // Pattern: a(?!\p{L}) - 'a' not followed by Unicode letter
  // Input: "aé" where é = 0xC3 0xA9
  // Split: chunk1 = "a\xC3", chunk2 = "\xA9"
  //
  // At end of chunk1: 'a' is accepting with lookahead (?!\p{L})
  // Next byte is 0xC3 (incomplete UTF-8).
  // Bug: If we decode 0xC3 alone, we get replacement char (not a letter),
  //      so lookahead passes incorrectly.
  // Fix: Detect incomplete UTF-8 and defer lookahead to next chunk.
  // Expected: NO match (é IS a letter, so lookahead rejects).

  // Build DFA: matches 'a' with negative lookahead for PSEUDO_LETTER.
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE |
                   IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  // Negative lookahead: reject if next char is PSEUDO_LETTER (0x80).
  // The letter 'é' maps to PSEUDO_LETTER, so this tests rejection.
  builder.SetLookaheadNegChar(1, IREE_TOKENIZER_REGEX_PSEUDO_LETTER);
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "a\xC3" (a + first byte of é)
  const uint8_t chunk1[] = {'a', 0xC3};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: "\xA9" (second byte of é)
  const uint8_t chunk2[] = {0xA9};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector)));

  // Should have NO matches: 'a' followed by letter 'é', lookahead rejects.
  EXPECT_EQ(collector.matches.size(), 0);
}

// Test "aaab" overlap - requires proper backtracking to match_start + 1.
TEST(Matching, OverlappingPrefixBacktrack) {
  // Pattern: aab
  // Input: "aaab"
  // Expected: match at [1,4) = "aab"
  // This tests the case where we have "aa" then fail on third 'a'.
  // We must restart from position 1 (not position 2) to find the match.
  auto data = BuildAabDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("aaab");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_string_view_t text = IREE_SVL("ac");
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "a" (in accepting state 1)
  const uint8_t chunk1[] = {'a'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: "c" (fails transition from state 1)
  const uint8_t chunk2[] = {'c'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector)));

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
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  builder.SetLookaheadNegChar(1, 'b');
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // "ab" - should NOT match (b follows a)
  {
    MatchCollector collector;
    iree_string_view_t text = IREE_SVL("ab");
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
        &dfa, text, MatchCollector::Callback, &collector)));
    EXPECT_EQ(collector.matches.size(), 0);
  }

  // "ac" - should match 'a' at [0,1)
  {
    MatchCollector collector;
    iree_string_view_t text = IREE_SVL("ac");
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
        &dfa, text, MatchCollector::Callback, &collector)));
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
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE);
  builder.AddTransition(0, IREE_TOKENIZER_REGEX_PSEUDO_LETTER, 1);
  builder.AddTransition(1, IREE_TOKENIZER_REGEX_PSEUDO_LETTER, 2);
  builder.SetAccepting(2);
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // Input: "1é2éé" - should match "éé" at end
  // '1' = 1 byte, 'é' = 2 bytes, '2' = 1 byte, 'é' = 2 bytes, 'é' = 2 bytes
  // Byte positions: 1[0], é[1-2], 2[3], é[4-5], é[6-7]
  // Match should be at [4, 8)
  MatchCollector collector;
  const uint8_t input[] = {'1', 0xC3, 0xA9, '2', 0xC3, 0xA9, 0xC3, 0xA9};
  iree_string_view_t text = {(const char*)input, sizeof(input)};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec(
      &dfa, text, MatchCollector::Callback, &collector)));

  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 4);
  EXPECT_EQ(collector.matches[0].second, 8);
}

// Test empty chunk handling.
TEST(Streaming, EmptyChunk) {
  auto data = BuildLowerAlphaDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed: "ab" then empty then "cd"
  const uint8_t chunk1[] = {'a', 'b'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // Empty chunk (pass nullptr with 0 length)
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, 0), sizeof(chunk1),
      MatchCollector::Callback, &collector)));

  const uint8_t chunk2[] = {'c', 'd'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector)));

  // Should match "abcd" at [0,4)
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Test 4-byte UTF-8 split in various ways.
TEST(Streaming, FourByteUtf8SplitVariants) {
  // Pattern: matches Unicode letters
  auto data = BuildUnicodeLetterDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // 𝄞 (Musical G Clef) = U+1D11E = F0 9D 84 9E (4 bytes)
  // Note: U+1D11E is in the "Symbol" category, not "Letter", so won't match.
  // Use 𐀀 (U+10000) = F0 90 80 80, also Symbol/Other.
  // Let's use a CJK character: 𠀀 (U+20000) = F0 A0 80 80, which is a Letter.
  const uint8_t cjk[] = {0xF0, 0xA0, 0x80, 0x80};

  // Split 1-3
  {
    MatchCollector collector;
    iree_tokenizer_regex_exec_state_t state;
    iree_tokenizer_regex_exec_initialize(&state, &dfa);

    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_const_byte_span(cjk, 1), 0,
        MatchCollector::Callback, &collector)));
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_const_byte_span(cjk + 1, 3), 1,
        MatchCollector::Callback, &collector)));
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
        &dfa, &state, 4, MatchCollector::Callback, &collector)));

    ASSERT_EQ(collector.matches.size(), 1);
    EXPECT_EQ(collector.matches[0].first, 0);
    EXPECT_EQ(collector.matches[0].second, 4);
  }

  // Split 2-2
  {
    MatchCollector collector;
    iree_tokenizer_regex_exec_state_t state;
    iree_tokenizer_regex_exec_initialize(&state, &dfa);

    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_const_byte_span(cjk, 2), 0,
        MatchCollector::Callback, &collector)));
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_const_byte_span(cjk + 2, 2), 2,
        MatchCollector::Callback, &collector)));
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
        &dfa, &state, 4, MatchCollector::Callback, &collector)));

    ASSERT_EQ(collector.matches.size(), 1);
  }

  // Split 3-1
  {
    MatchCollector collector;
    iree_tokenizer_regex_exec_state_t state;
    iree_tokenizer_regex_exec_initialize(&state, &dfa);

    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_const_byte_span(cjk, 3), 0,
        MatchCollector::Callback, &collector)));
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_const_byte_span(cjk + 3, 1), 3,
        MatchCollector::Callback, &collector)));
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
        &dfa, &state, 4, MatchCollector::Callback, &collector)));

    ASSERT_EQ(collector.matches.size(), 1);
  }
}

// Test multiple consecutive lookahead deferrals.
TEST(Streaming, MultipleLookaheadDefers) {
  // Pattern: a(?!b) - 'a' not followed by 'b'
  // Input: "a" in tiny 1-byte chunks, followed by "c"
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  builder.SetLookaheadNegChar(1, 'b');
  auto data = builder.Build();

  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed 'a' alone - should defer lookahead
  const uint8_t a[] = {'a'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(a, 1), 0,
      MatchCollector::Callback, &collector)));
  EXPECT_EQ(collector.matches.size(), 0);  // Not emitted yet

  // Feed 'c' - lookahead passes (c != b), should emit match
  const uint8_t c[] = {'c'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(c, 1), 1,
      MatchCollector::Callback, &collector)));

  // Finalize
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // 0xC3 expects continuation (0x80-0xBF), but we give 0x40 ('@').
  // Should decode as replacement char (doesn't match), then '@' (doesn't
  // match).
  const uint8_t input[] = {0xC3, 0x40, 'b', 'c'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(input, sizeof(input)), 0,
      MatchCollector::Callback, &collector)));
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // 0xC0 0x80 is overlong encoding of U+0000 (NUL).
  // 0xC1 0xBF is overlong encoding of U+007F (DEL).
  // Both should become replacement char.
  const uint8_t input[] = {0xC0, 0x80, 'a', 'b'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(input, sizeof(input)), 0,
      MatchCollector::Callback, &collector)));
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector)));

  // Overlong sequence becomes replacement char (no match), then "ab" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Overlong encoding: 3-byte encoding of 2-byte range.
TEST(Utf8Validation, OverlongThreeByte) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // 0xE0 0x80 0x80 is overlong encoding of U+0000.
  // Valid 3-byte sequences start at U+0800, so 0xE0 must have 2nd byte >= 0xA0.
  const uint8_t input[] = {0xE0, 0x80, 0x80, 'x', 'y'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(input, sizeof(input)), 0,
      MatchCollector::Callback, &collector)));
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector)));

  // Overlong becomes replacement char, then "xy" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 3);
  EXPECT_EQ(collector.matches[0].second, 5);
}

// UTF-16 surrogate codepoints (U+D800-U+DFFF) are invalid in UTF-8.
TEST(Utf8Validation, SurrogateCodepoint) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // 0xED 0xA0 0x80 encodes U+D800 (high surrogate start).
  // 0xED 0xBF 0xBF encodes U+DFFF (low surrogate end).
  // Both are invalid and should become replacement char.
  const uint8_t input[] = {0xED, 0xA0, 0x80, 'z'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(input, sizeof(input)), 0,
      MatchCollector::Callback, &collector)));
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector)));

  // Surrogate becomes replacement char, then "z" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 3);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Partial UTF-8 at end of stream should become replacement char.
TEST(Utf8Validation, PartialAtEOS) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // 'a' 'b' then incomplete 2-byte sequence (0xC3 alone).
  const uint8_t input[] = {'a', 'b', 0xC3};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(input, sizeof(input)), 0,
      MatchCollector::Callback, &collector)));
  // Finalize should handle the partial UTF-8.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector)));

  // "ab" should match at [0,2). The partial 0xC3 becomes replacement char.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 2);
}

// Partial 3-byte UTF-8 at end of stream.
TEST(Utf8Validation, PartialThreeByteAtEOS) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // Incomplete 3-byte sequence (0xE4 0xB8 without third byte).
  const uint8_t input[] = {'x', 0xE4, 0xB8};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(input, sizeof(input)), 0,
      MatchCollector::Callback, &collector)));
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector)));

  // "x" matches at [0,1). Partial sequence becomes replacement char.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

// Invalid leading byte (0xFF is never valid in UTF-8).
TEST(Utf8Validation, InvalidLeadingByte) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // 0xFF and 0xFE are never valid UTF-8 leading bytes.
  const uint8_t input[] = {0xFF, 0xFE, 'a', 'b'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(input, sizeof(input)), 0,
      MatchCollector::Callback, &collector)));
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector)));

  // 0xFF and 0xFE become replacement chars, then "ab" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 4);
}

// Continuation byte without leading byte (0x80-0xBF alone).
TEST(Utf8Validation, StrayContiunationByte) {
  auto data = BuildLowerAlphaUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  // 0x80 alone is invalid (continuation byte without leader).
  const uint8_t input[] = {0x80, 0x85, 'c', 'd'};

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(input, sizeof(input)), 0,
      MatchCollector::Callback, &collector)));
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(input), MatchCollector::Callback, &collector)));

  // Each stray continuation byte becomes replacement char, then "cd" matches.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 4);
}

//===----------------------------------------------------------------------===//
// Multi-Chunk UTF-8 Lookahead Tests
//===----------------------------------------------------------------------===//

// Build a DFA for "a(?!b)" with Unicode mode enabled.
// This allows testing lookahead evaluation when the lookahead character
// is a multi-byte UTF-8 sequence split across chunks.
std::vector<uint8_t> BuildANotFollowedByBUnicodeDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  builder.SetLookaheadNegChar(1, 'b');
  return builder.Build();
}

TEST(Streaming, LookaheadWithUtf8SplitAcrossThreeChunks) {
  // When pending lookahead needs to peek at the next character, and that
  // character is a multi-byte UTF-8 sequence split across more than 2 chunks,
  // we must wait until the sequence is complete before evaluating the
  // lookahead.
  //
  // Pattern: a(?!b) in Unicode mode
  // Input: "a" + 😀 (4-byte emoji: 0xF0 0x9F 0x98 0x80)
  // Split: chunk1 = "a\xF0", chunk2 = "\x9F", chunk3 = "\x98\x80"
  //
  // After chunk1: 'a' accepted with pending lookahead, 0xF0 buffered
  // After chunk2: Can't complete sequence yet (have 2 of 4 bytes)
  // After chunk3: Complete emoji, evaluate lookahead (emoji != 'b'), accept
  //
  // The bug (before fix): chunk2 would use REPLACEMENT_CHAR and clear
  // pending_lookahead prematurely.
  auto data = BuildANotFollowedByBUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "a" + first byte of 4-byte emoji.
  const uint8_t chunk1[] = {'a', 0xF0};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // No match yet - lookahead pending.
  EXPECT_EQ(collector.matches.size(), 0);
  EXPECT_TRUE(state.pending_lookahead);

  // Chunk 2: second byte of emoji - still incomplete (2 of 4 bytes).
  const uint8_t chunk2[] = {0x9F};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  // Still no match - lookahead should still be pending (this is the bug test).
  EXPECT_EQ(collector.matches.size(), 0);
  EXPECT_TRUE(state.pending_lookahead);  // Lookahead still deferred

  // Chunk 3: remaining bytes of emoji (bytes 3-4).
  const uint8_t chunk3[] = {0x98, 0x80};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk3, sizeof(chunk3)),
      sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback, &collector)));

  // Now lookahead should be resolved (emoji != 'b', so passes).
  EXPECT_FALSE(state.pending_lookahead);

  // Finalize.
  size_t total = sizeof(chunk1) + sizeof(chunk2) + sizeof(chunk3);
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, total, MatchCollector::Callback, &collector)));

  // Should match "a" at [0,1).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
}

//===----------------------------------------------------------------------===//
// EOS Lookahead with Invalid UTF-8 Tests
//===----------------------------------------------------------------------===//

// Build a DFA for "a(?!\S)" - 'a' NOT followed by non-whitespace.
// With Unicode mode enabled for UTF-8 handling.
std::vector<uint8_t> BuildANotFollowedByNonWhitespaceUnicodeDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.SetFlags(IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE);
  builder.AddTransition(0, 'a', 1);
  builder.SetAccepting(1);
  builder.SetLookaheadNegShorthand(1, IREE_TOKENIZER_REGEX_SHORTHAND_S);
  return builder.Build();
}

TEST(Streaming, EOSLookaheadWithIncompleteUtf8_Rejects) {
  // When there's incomplete UTF-8 at EOS and a pending lookahead, the
  // lookahead must be evaluated against REPLACEMENT_CHAR, not unconditionally
  // passed.
  //
  // Pattern: a(?!\S) - 'a' NOT followed by non-whitespace
  // Input: "a" + 0xC3 (incomplete 2-byte UTF-8 sequence)
  //
  // At finalize:
  // - Incomplete UTF-8 becomes REPLACEMENT_CHAR
  // - REPLACEMENT_CHAR is non-whitespace (maps to PSEUDO_OTHER)
  // - Lookahead (?!\S) means "not followed by non-whitespace"
  // - REPLACEMENT_CHAR IS non-whitespace, so lookahead REJECTS
  //
  // Before fix: Lookahead unconditionally passed at EOS, giving a match (wrong)
  // After fix: Lookahead evaluated against REPLACEMENT_CHAR, rejects, no match
  auto data = BuildANotFollowedByNonWhitespaceUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "a" + incomplete UTF-8 lead byte.
  const uint8_t chunk[] = {'a', 0xC3};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk, sizeof(chunk)), 0,
      MatchCollector::Callback, &collector)));

  // Verify state: lookahead pending, partial UTF-8 buffered.
  EXPECT_TRUE(state.pending_lookahead);
  EXPECT_EQ(state.utf8_partial_length, 1);

  // Finalize with incomplete UTF-8.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk), MatchCollector::Callback, &collector)));

  // Should NOT match - REPLACEMENT_CHAR (non-whitespace) rejects the lookahead.
  EXPECT_EQ(collector.matches.size(), 0);
}

TEST(Streaming, EOSLookaheadWithIncompleteUtf8_Accepts) {
  // Complementary test: (?!b) should PASS when incomplete UTF-8 at EOS,
  // because REPLACEMENT_CHAR (via codepoint_to_byte -> PSEUDO_OTHER) != 'b'.
  //
  // Pattern: a(?!b) - 'a' NOT followed by 'b'
  // Input: "a" + 0xC3 (incomplete 2-byte UTF-8)
  // Expected: Match "a" - REPLACEMENT_CHAR != 'b'
  auto data = BuildANotFollowedByBUnicodeDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "a" + incomplete UTF-8.
  const uint8_t chunk[] = {'a', 0xC3};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk, sizeof(chunk)), 0,
      MatchCollector::Callback, &collector)));

  // Finalize.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk), MatchCollector::Callback, &collector)));

  // Should match "a" at [0,1) - REPLACEMENT_CHAR != 'b', lookahead passes.
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "aa"
  const uint8_t chunk1[] = {'a', 'a'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: "ab"
  const uint8_t chunk2[] = {'a', 'b'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "ab"
  const uint8_t chunk1[] = {'a', 'b'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: "aba"
  const uint8_t chunk2[] = {'a', 'b', 'a'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed "aaab" one byte at a time.
  size_t offset = 0;
  const uint8_t bytes[] = {'a', 'a', 'a', 'b'};
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_const_byte_span(&bytes[i], 1), offset,
        MatchCollector::Callback, &collector)));
    offset += 1;
  }

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, offset, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "aaa"
  const uint8_t chunk1[] = {'a', 'a', 'a'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk1, sizeof(chunk1)), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: "ab"
  const uint8_t chunk2[] = {'a', 'b'};
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span(chunk2, sizeof(chunk2)),
      sizeof(chunk1), MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, sizeof(chunk1) + sizeof(chunk2), MatchCollector::Callback,
      &collector)));

  // Should match "aab" at [2,5).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 2);
  EXPECT_EQ(collector.matches[0].second, 5);
}

//===----------------------------------------------------------------------===//
// Backtrack State Reset Tests (regression tests for loom-djxu)
//===----------------------------------------------------------------------===//

// These tests verify that executor state is properly reset on backtrack.
// Bug: The within-chunk and within-buffer backtrack paths were not clearing
// has_accept_fallback and best_branch_idx/best_match_end, which could cause
// stale state to leak into subsequent match attempts.

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
      1, IREE_TOKENIZER_REGEX_SHORTHAND_S);
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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  EXPECT_TRUE(iree_status_is_ok(
      iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view("aXab"),
                                MatchCollector::Callback, &collector)));

  // Should have two matches: [0,1) and [2,3).
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  EXPECT_TRUE(iree_status_is_ok(
      iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view("aXaXab"),
                                MatchCollector::Callback, &collector)));

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
  // Pattern: a+(?!X)|a+
  //
  // Input: "aa" + "Xa" (split across chunks)
  //
  // Processing:
  // - Position 0: 'a' matches, lookahead checks position 1 ('a' != 'X') → PASS
  //   Record last_accept = 1
  // - Position 1: 'a' matches, lookahead needs position 2 → DEFER to next chunk
  // - Chunk 2: Pending lookahead checks position 2 ('X' == 'X') → FAIL
  //   Emit using last successful accept [0,1)
  // - Position 2: 'X' doesn't match
  // - Position 3: 'a' matches, EOS lookahead passes → emit [3,4)
  auto data = BuildOnePlusAWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: "aa".
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"aa", 2), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: "Xa".
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"Xa", 2), 2,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector)));

  // Should have two matches:
  // - [0,1): First 'a' where lookahead passed (second 'a' is not 'X')
  // - [3,4): Final 'a' where EOS lookahead passes
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 3);
  EXPECT_EQ(collector.matches[1].second, 4);
}

//===----------------------------------------------------------------------===//
// Shorthand Fallback Streaming Tests (regression tests for loom-mg68)
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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: two spaces.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"  ", 2), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: space then 'x'.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)" x", 2), 2,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: single space at chunk boundary.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)" ", 1), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: 'x' causes deferred lookahead to fail.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"x", 1), 1,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: single space.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)" ", 1), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: another space.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)" ", 1), 1,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 2, MatchCollector::Callback, &collector)));

  // Both spaces match (lookahead passed).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 2);
}

TEST(ShorthandFallbackStreaming, BacktrackAcrossChunksWithFallback) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "  " + "xy  " (backtrack after lookahead fails, then new match)
  //
  // CRITICAL STREAMING BEHAVIOR: Once a chunk is consumed, we cannot restart
  // within it. After emitting [0,1), we'd want to restart at pos 1 but chunk 1
  // is gone. We continue from chunk 2's first available position (pos 2).
  //
  // Processing:
  // - Chunk 1 "  ":
  //   - Pos 0: ' ', lookahead at 1 (' ' not \S) -> PASS, last_accept=1
  //   - Pos 1: ' ', lookahead needs pos 2 -> DEFER
  // - Chunk 2 "xy  ":
  //   - Resolve deferred: 'x' at pos 2 IS \S -> FAIL
  //   - Emit [0,1) using last_accept=1
  //   - Cannot restart at pos 1 (chunk 1 consumed), skip to pos 2
  //   - Pos 2,3: 'xy' don't match
  //   - Pos 4: ' ', lookahead at 5 (' ' not \S) -> PASS, last_accept=5
  //   - Pos 5: ' ', lookahead needs pos 6 -> DEFER
  // - Finalize: EOS not \S -> PASS, last_accept=6, emit [4,6)
  //
  // Expected: [0,1), [4,6) - only two matches (second space in chunk 1 is lost)
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: two spaces.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"  ", 2), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: "xy  " - non-whitespace then whitespace.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"xy  ", 4), 2,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 6, MatchCollector::Callback, &collector)));

  // Two matches: first space (lookahead passed), trailing spaces at EOS.
  // Second space in chunk 1 is lost - cannot restart within consumed chunk.
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);  // Lookahead passed at pos 0.
  EXPECT_EQ(collector.matches[1].first, 4);
  EXPECT_EQ(collector.matches[1].second, 6);  // Trailing spaces at EOS.
}

TEST(ShorthandFallbackStreaming, MultipleSectionsAcrossManyChunks) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "a " + " b" + "  " + "c" = "a  b  c" (multiple whitespace sections)
  //
  // STREAMING BEHAVIOR: Cannot restart within consumed chunks.
  //
  // Processing:
  // - Chunk 1 "a ": 'a' skipped, pos 1 ' ' matches, lookahead needs 2 -> DEFER
  // - Chunk 2 " b":
  //   - Resolve deferred: pos 2 ' ' not \S -> PASS, last_accept=2
  //   - Pos 2: ' ', lookahead at 3 ('b' is \S) -> FAIL
  //   - Pos 3: 'b' NO_TRANSITION -> emit [1,2)
  //   - Restart at pos 2 (in current chunk) -> works!
  //   - Pos 2: ' ', lookahead at 3 ('b') -> FAIL
  //   - Pos 3: NO_TRANSITION -> emit [2,3) via fallback
  //   - 'b' skipped
  // - Chunk 3 "  ":
  //   - Pos 4: ' ', lookahead at 5 (' ') -> PASS, last_accept=5
  //   - Pos 5: ' ', lookahead needs 6 -> DEFER
  // - Chunk 4 "c":
  //   - Resolve deferred: 'c' is \S -> FAIL
  //   - Emit [4,5) using last_accept
  //   - Cannot restart at pos 5 (chunk 3 consumed)
  //
  // Expected: [1,2), [2,3), [4,5) - three matches (pos 5 space lost)
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Feed chunks one at a time.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"a ", 2), 0,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)" b", 2), 2,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"  ", 2), 4,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"c", 1), 6,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 7, MatchCollector::Callback, &collector)));

  // Three matches - last space at pos 5 is lost (chunk 3 consumed).
  ASSERT_EQ(collector.matches.size(), 3);
  EXPECT_EQ(collector.matches[0].first, 1);
  EXPECT_EQ(collector.matches[0].second, 2);  // Lookahead passed at pos 1.
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);  // Fallback for space before 'b'.
  EXPECT_EQ(collector.matches[2].first, 4);
  EXPECT_EQ(collector.matches[2].second, 5);  // Lookahead passed at pos 4.
}

TEST(ShorthandFallbackStreaming, SingleByteChunksStressTest) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "  x  " fed byte-by-byte
  //
  // STREAMING BEHAVIOR: With single-byte chunks, we can NEVER restart within
  // a consumed chunk. After emitting [0,1), we cannot restart at pos 1.
  //
  // Processing:
  // - Chunk 0 ' ': lookahead needs 1 -> DEFER
  // - Chunk 1 ' ': resolve deferred (PASSES), last_accept=1, lookahead needs 2
  //   -> DEFER
  // - Chunk 2 'x': resolve deferred (FAILS), emit [0,1) using last_accept
  //   - Cannot restart at pos 1 (chunk consumed)
  //   - 'x' doesn't match
  // - Chunk 3 ' ': lookahead needs 4 -> DEFER
  // - Chunk 4 ' ': resolve deferred (PASSES), last_accept=4, lookahead needs 5
  //   -> DEFER
  // - Finalize: EOS -> PASS, last_accept=5, emit [3,5)
  //
  // Expected: [0,1), [3,5) - only two matches (second space at pos 1 is lost)
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  const char* input = "  x  ";
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
        &dfa, &state, iree_make_const_byte_span((const uint8_t*)&input[i], 1),
        i, MatchCollector::Callback, &collector)));
  }

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 5, MatchCollector::Callback, &collector)));

  // Two matches - second space at pos 1 is lost (cannot restart in consumed
  // chunk).
  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);  // Lookahead passed at pos 0.
  EXPECT_EQ(collector.matches[1].first, 3);
  EXPECT_EQ(collector.matches[1].second, 5);  // Trailing spaces at EOS.
}

TEST(ShorthandFallbackStreaming, FallbackStateNotLeakedOnBacktrack) {
  // Pattern: \s+(?!\S)|\s+
  // Input: " x y" (regression test for loom-djxu with shorthand)
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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  EXPECT_TRUE(iree_status_is_ok(
      iree_tokenizer_regex_exec(&dfa, iree_make_cstring_view(" x y"),
                                MatchCollector::Callback, &collector)));

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
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: space with deferred lookahead.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)" ", 1), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: 'x' fails lookahead, then ' ' starts new match.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"x ", 2), 1,
      MatchCollector::Callback, &collector)));

  // Chunk 3: 'y' fails lookahead for second match.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"y", 1), 3,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 4, MatchCollector::Callback, &collector)));

  ASSERT_EQ(collector.matches.size(), 2);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);
  EXPECT_EQ(collector.matches[1].first, 2);
  EXPECT_EQ(collector.matches[1].second, 3);
}

TEST(ShorthandFallbackStreaming, TabAndNewlineHandling) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\t\n" + "x" (tab and newline, then non-whitespace)
  //
  // Verify that shorthand \s correctly matches tab/newline, not just space.
  //
  // STREAMING BEHAVIOR: After emitting [0,1), we cannot restart at pos 1
  // because chunk 1 ("\t\n") is consumed.
  //
  // Processing:
  // - Chunk 1 "\t\n":
  //   - pos 0: '\t' matches, lookahead at 1 ('\n' not \S) -> PASS,
  //   last_accept=1
  //   - pos 1: '\n' matches, lookahead needs 2 -> DEFER
  // - Chunk 2 "x":
  //   - Resolve deferred: 'x' is \S -> FAIL
  //   - Emit [0,1) using last_accept
  //   - Cannot restart at pos 1 (chunk 1 consumed)
  //   - 'x' doesn't match
  //
  // Expected: [0,1) - only one match (newline at pos 1 is lost)
  auto data = BuildWhitespaceWithFallbackDfa();
  iree_tokenizer_regex_dfa_t dfa;
  ASSERT_TRUE(iree_status_is_ok(iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(data.data(), data.size()), &dfa)));

  MatchCollector collector;
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, &dfa);

  // Chunk 1: tab + newline.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"\t\n", 2), 0,
      MatchCollector::Callback, &collector)));

  // Chunk 2: 'x'.
  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_feed(
      &dfa, &state, iree_make_const_byte_span((const uint8_t*)"x", 1), 2,
      MatchCollector::Callback, &collector)));

  EXPECT_TRUE(iree_status_is_ok(iree_tokenizer_regex_exec_finalize(
      &dfa, &state, 3, MatchCollector::Callback, &collector)));

  // One match - newline at pos 1 is lost (cannot restart in consumed chunk).
  ASSERT_EQ(collector.matches.size(), 1);
  EXPECT_EQ(collector.matches[0].first, 0);
  EXPECT_EQ(collector.matches[0].second, 1);  // Tab (lookahead passed).
}

}  // namespace
