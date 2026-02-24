// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_UTIL_REGEX_INTERNAL_NFA_H_
#define IREE_TOKENIZER_UTIL_REGEX_INTERNAL_NFA_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/tokenizer/regex/exec.h"
#include "iree/tokenizer/regex/internal/ast.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// NFA State Types
//===----------------------------------------------------------------------===//

typedef enum iree_tokenizer_regex_nfa_state_type_e {
  // Epsilon transition(s) - no input consumed.
  IREE_TOKENIZER_UTIL_REGEX_NFA_EPSILON,
  // Match a single byte.
  IREE_TOKENIZER_UTIL_REGEX_NFA_MATCH_BYTE,
  // Match a character class (bitmap + pseudo-byte mask).
  IREE_TOKENIZER_UTIL_REGEX_NFA_MATCH_CLASS,
  // Accepting state.
  IREE_TOKENIZER_UTIL_REGEX_NFA_ACCEPT,
  // Start anchor (^).
  IREE_TOKENIZER_UTIL_REGEX_NFA_ANCHOR_START,
  // End anchor ($).
  IREE_TOKENIZER_UTIL_REGEX_NFA_ANCHOR_END,
} iree_tokenizer_regex_nfa_state_type_t;

// Maximum number of alternation branches supported.
// Limited to 255 because branch_index is uint8_t for memory efficiency.
// Patterns with >255 branches should use a trie-based approach instead.
#define IREE_TOKENIZER_UTIL_REGEX_MAX_ALTERNATION_BRANCHES 255

//===----------------------------------------------------------------------===//
// NFA State Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_regex_nfa_state_t {
  iree_tokenizer_regex_nfa_state_type_t type;
  uint32_t id;

  union {
    // NFA_EPSILON: up to 2 epsilon transitions.
    struct {
      struct iree_tokenizer_regex_nfa_state_t* out1;
      struct iree_tokenizer_regex_nfa_state_t* out2;
    } epsilon;

    // NFA_MATCH_BYTE: single byte + next state.
    struct {
      uint8_t byte;
      struct iree_tokenizer_regex_nfa_state_t* out;
    } match_byte;

    // NFA_MATCH_CLASS: character class matching data.
    //
    // Cache-optimized layout (fits in 2 cache lines):
    //   - out pointer at offset 0 (always accessed first)
    //   - bitmap at offset 8 (ASCII matching)
    //   - pseudo_mask + range_count at offset 40 (fits in 4 bytes)
    //   - ranges at offset 44 (exact Unicode ranges, accessed only when
    //   present)
    //
    // The common case (0 ranges) accesses bytes 0-43, fitting in one cache
    // line. Full structure with max ranges still fits in 2 cache lines.
    //
    // Ranges are stored sorted by start codepoint to enable early-exit:
    // if codepoint < ranges[i].start, it cannot match any remaining ranges.
    struct {
      struct iree_tokenizer_regex_nfa_state_t* out;  // Next state on match.
      // 256-bit bitmap for bytes 0-255.
      uint8_t bitmap[32];
      uint16_t pseudo_mask;  // Pseudo-bytes 0x80-0x87.
      // Number of exact codepoint ranges.
      uint8_t range_count;
      uint8_t reserved;  // Padding.
      iree_tokenizer_regex_codepoint_range_t
          ranges[IREE_TOKENIZER_UTIL_REGEX_MAX_CHAR_CLASS_RANGES];
    } match_class;

    // NFA_ACCEPT: optional lookahead with PCRE-compatible branch tracking.
    //
    // The branch_index field provides PCRE-compatible match selection by
    // tracking which alternation branch each accept came from. During DFA
    // extraction, we compare branch indices of lookahead vs non-lookahead
    // accepts to determine preference:
    //   - \s+(?!\S)|\s+ : lookahead at branch 0, fallback at branch 1
    //     -> min_no_lookahead (1) > min_lookahead (0) -> prefer
    //     lookahead-passed
    //   - \s*[\r\n]+|\s+(?!\S)|\s+ : no-lookahead at branch 0, lookahead at 1
    //     -> min_no_lookahead (0) < min_lookahead (1) -> prefer longer match
    //
    // LIMITATIONS (not seen in real tokenizers, but could arise):
    //   - Interleaved alternations: a|ab(?!c)|ab where branch priority matters
    //     beyond longest-match semantics
    //   - Nested alternations with lookahead: (a|b(?!c)|d)|(e|f) - branch index
    //     is relative to innermost alternation, not global
    //   - Mixed positive/negative lookahead in same alternation
    struct {
      bool has_lookahead;
      iree_tokenizer_regex_lookahead_type_t lookahead_type;
      uint8_t lookahead_data;
      // Branch index within the containing alternation (0-based).
      // Used to compute has_early_no_lookahead per-DFA-state.
      // Accepts outside alternations have branch_index = 0.
      uint8_t branch_index;
    } accept;

    // NFA_ANCHOR_START, NFA_ANCHOR_END: next state.
    struct iree_tokenizer_regex_nfa_state_t* anchor_out;
  } data;
} iree_tokenizer_regex_nfa_state_t;

//===----------------------------------------------------------------------===//
// NFA Fragment (for composition during Thompson construction)
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_regex_nfa_fragment_t {
  iree_tokenizer_regex_nfa_state_t* start;
  // List of dangling pointers that need to be patched.
  // These are state pointers that point to the "next" state which hasn't
  // been created yet.
  iree_tokenizer_regex_nfa_state_t*** patch_list;
  iree_host_size_t patch_count;
  iree_host_size_t patch_capacity;

  // Per-fragment lookahead tracking.
  // This allows alternation branches to have different lookahead requirements.
  // When patching to accept, fragments with lookahead get a separate accept
  // state from those without, enabling patterns like `\s+(?!\S)|\s+` to work.
  // lookahead_type == NONE means no lookahead required.
  iree_tokenizer_regex_lookahead_type_t lookahead_type;
  uint8_t lookahead_data;
} iree_tokenizer_regex_nfa_fragment_t;

//===----------------------------------------------------------------------===//
// NFA Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_regex_nfa_t {
  // Start and accept states.
  iree_tokenizer_regex_nfa_state_t* start;
  iree_tokenizer_regex_nfa_state_t* accept;

  // All states (for traversal/serialization).
  iree_tokenizer_regex_nfa_state_t** states;
  uint32_t state_count;
  uint32_t state_capacity;

  // Flags derived from AST analysis.
  // Any \p{} or non-ASCII matching.
  bool uses_unicode;
  bool has_lookahead;  // Any (?!...) patterns.
  bool has_anchors;    // Any ^ or $ anchors.

  // Pending lookahead info (extracted from AST, applied to accept state).
  // Only one lookahead per pattern is supported; last one wins.
  iree_tokenizer_regex_lookahead_type_t pending_lookahead_type;
  uint8_t pending_lookahead_data;

  // Allocator for state storage.
  iree_arena_allocator_t* arena;
} iree_tokenizer_regex_nfa_t;

//===----------------------------------------------------------------------===//
// NFA API
//===----------------------------------------------------------------------===//

// Builds an NFA from an AST using Thompson's construction.
//
// |ast| is the root AST node.
// |arena| is used for all NFA state allocations.
// |out_nfa| receives the constructed NFA.
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - IREE_STATUS_RESOURCE_EXHAUSTED if memory allocation fails.
iree_status_t iree_tokenizer_regex_nfa_build(
    const iree_tokenizer_regex_ast_node_t* ast, iree_arena_allocator_t* arena,
    iree_tokenizer_regex_nfa_t* out_nfa);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_UTIL_REGEX_INTERNAL_NFA_H_
