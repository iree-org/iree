// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_REGEX_INTERNAL_DFA_H_
#define IREE_TOKENIZER_REGEX_INTERNAL_DFA_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/tokenizer/regex/exec.h"
#include "iree/tokenizer/regex/internal/nfa.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// DFA State Set (for subset construction)
//===----------------------------------------------------------------------===//

// A set of NFA states, represented as a bitset.
// Used during subset construction to track which NFA states are active.
typedef struct iree_tokenizer_regex_nfa_state_set_t {
  uint64_t* bits;            // Bitset storage (1 bit per NFA state).
  uint32_t capacity;         // Number of uint64_t elements in bits array.
  uint32_t nfa_state_count;  // Total NFA states for bounds checking.
} iree_tokenizer_regex_nfa_state_set_t;

//===----------------------------------------------------------------------===//
// DFA State
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_regex_dfa_state_t {
  uint32_t id;

  // Transitions for each input byte (0-255).
  // Value is target DFA state ID, or UINT32_MAX for no transition.
  uint32_t transitions[256];

  // Whether this is an accepting state.
  bool is_accepting;

  // Whether this accepting state also has a non-lookahead fallback path.
  // This happens with alternation patterns like \s+(?!\S)|\s+ where the DFA
  // state is reachable via both branches. When true, the executor must track
  // both lookahead-passed and fallback positions, preferring lookahead-passed.
  bool has_fallback;

  // PCRE-compatible match selection flag.
  // True if alternation has a non-lookahead branch BEFORE the first lookahead
  // branch. When true AND has_fallback, prefer longer match (independent branch
  // extends greedily). When false AND has_fallback, prefer lookahead-passed
  // position (fallback branches are companions to the lookahead branch).
  // See nfa.h accept struct comment for full documentation on limitations.
  bool has_early_no_lookahead;

  // Anchor requirements.
  // A state requires start anchor if ALL paths to it went through ^.
  // A state requires end anchor if it's accepting and needs $ to be valid.
  bool requires_start_anchor;
  bool requires_end_anchor;

  // Lookahead info (if accepting).
  bool has_lookahead;
  iree_tokenizer_regex_lookahead_type_t lookahead_type;
  uint8_t lookahead_data;

  // Branch tracking for PCRE-compatible alternation priority.
  // Each bit represents an alternation branch (bit 0 = highest priority).
  // Used to implement left-to-right alternation precedence in DFA execution.
  uint64_t alive_branches;      // Which branches can reach this state.
  uint64_t accepting_branches;  // Which branches accept at this state.

  // Exact codepoint range transitions.
  // These are checked BEFORE pseudo-byte fallback at runtime.
  // Each entry maps a codepoint range to a target DFA state.
  uint8_t range_count;  // Number of range transitions (0-MAX).
  struct {
    uint32_t start;  // Range start codepoint (inclusive).
    uint32_t end;    // Range end codepoint (inclusive).
    uint32_t
        target_id;  // Target DFA state ID (UINT32_MAX if not yet resolved).
  } range_transitions[IREE_TOKENIZER_REGEX_MAX_CHAR_CLASS_RANGES];

  // The NFA state set this DFA state represents (for deduplication).
  iree_tokenizer_regex_nfa_state_set_t nfa_states;
} iree_tokenizer_regex_dfa_state_t;

//===----------------------------------------------------------------------===//
// DFA Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_regex_dfa_build_t {
  // All DFA states.
  iree_tokenizer_regex_dfa_state_t** states;
  uint32_t state_count;
  uint32_t state_capacity;

  // Start state for position 0 (includes anchored paths).
  iree_tokenizer_regex_dfa_state_t* start;
  // Start state for position > 0 (excludes ^ paths).
  // May be same as start if pattern has no start anchor.
  iree_tokenizer_regex_dfa_state_t* unanchored_start;

  // Flags from NFA.
  bool uses_unicode;
  bool has_lookahead;
  bool has_anchors;

  // Arena for allocations.
  iree_arena_allocator_t* arena;
} iree_tokenizer_regex_dfa_build_t;

//===----------------------------------------------------------------------===//
// DFA Construction API
//===----------------------------------------------------------------------===//

// Maximum number of DFA states (limited by 16-bit state IDs in binary format).
#define IREE_TOKENIZER_REGEX_DFA_MAX_STATES 65534

// Maximum iterations for DFA construction loop.
// This is a safety limit to prevent hangs from pathological patterns that
// might slip past parse-time validation. Set conservatively high enough to
// allow complex legitimate patterns while preventing minute-long hangs.
#define IREE_TOKENIZER_REGEX_DFA_MAX_ITERATIONS 100000

// Builds a DFA from an NFA using subset construction.
//
// |nfa| is the source NFA.
// |arena| is used for all DFA allocations.
// |out_dfa| receives the constructed DFA.
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - IREE_STATUS_RESOURCE_EXHAUSTED if state limit exceeded or memory fails.
iree_status_t iree_tokenizer_regex_dfa_build(
    const iree_tokenizer_regex_nfa_t* nfa, iree_arena_allocator_t* arena,
    iree_tokenizer_regex_dfa_build_t* out_dfa);

//===----------------------------------------------------------------------===//
// DFA Serialization API
//===----------------------------------------------------------------------===//

// Serializes a DFA to the binary format defined in exec.h.
//
// |dfa| is the DFA to serialize.
// |allocator| is used to allocate the output buffer.
// |out_data| receives the serialized data.
// |out_size| receives the size of serialized data.
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - IREE_STATUS_RESOURCE_EXHAUSTED if memory allocation fails.
iree_status_t iree_tokenizer_regex_dfa_serialize(
    const iree_tokenizer_regex_dfa_build_t* dfa, iree_allocator_t allocator,
    uint8_t** out_data, iree_host_size_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_REGEX_INTERNAL_DFA_H_
