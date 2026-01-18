// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/regex/internal/nfa.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/regex/internal/lexer.h"

//===----------------------------------------------------------------------===//
// State Allocation
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_regex_nfa_state_allocate(
    iree_tokenizer_regex_nfa_t* nfa, iree_tokenizer_regex_nfa_state_type_t type,
    iree_tokenizer_regex_nfa_state_t** out_state) {
  *out_state = NULL;

  // Allocate state.
  iree_tokenizer_regex_nfa_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(nfa->arena, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->type = type;
  state->id = nfa->state_count;

  // Add to state list.
  if (nfa->state_count >= nfa->state_capacity) {
    iree_host_size_t new_capacity =
        nfa->state_capacity == 0 ? 64 : nfa->state_capacity * 2;
    iree_tokenizer_regex_nfa_state_t** new_states = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        nfa->arena, new_capacity * sizeof(iree_tokenizer_regex_nfa_state_t*),
        (void**)&new_states));
    if (nfa->states) {
      memcpy(new_states, nfa->states,
             nfa->state_count * sizeof(iree_tokenizer_regex_nfa_state_t*));
    }
    nfa->states = new_states;
    nfa->state_capacity = (uint32_t)new_capacity;
  }
  nfa->states[nfa->state_count++] = state;

  *out_state = state;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Fragment Helpers
//===----------------------------------------------------------------------===//

// Initializes a fragment with a single state.
static void iree_tokenizer_regex_nfa_fragment_initialize(
    iree_tokenizer_regex_nfa_fragment_t* frag,
    iree_tokenizer_regex_nfa_state_t* start) {
  memset(frag, 0, sizeof(*frag));
  frag->start = start;
  frag->lookahead_type = IREE_TOKENIZER_REGEX_LOOKAHEAD_NONE;
  frag->lookahead_data = 0;
}

// Returns true if fragment has a lookahead requirement.
static inline bool iree_tokenizer_regex_nfa_fragment_has_lookahead(
    const iree_tokenizer_regex_nfa_fragment_t* frag) {
  return frag->lookahead_type != IREE_TOKENIZER_REGEX_LOOKAHEAD_NONE;
}

// Adds a patch pointer to the fragment's patch list.
static iree_status_t iree_tokenizer_regex_nfa_fragment_add_patch(
    iree_tokenizer_regex_nfa_t* nfa, iree_tokenizer_regex_nfa_fragment_t* frag,
    iree_tokenizer_regex_nfa_state_t** ptr) {
  if (frag->patch_count >= frag->patch_capacity) {
    iree_host_size_t new_capacity =
        frag->patch_capacity == 0 ? 8 : frag->patch_capacity * 2;
    iree_tokenizer_regex_nfa_state_t*** new_list = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        nfa->arena, new_capacity * sizeof(iree_tokenizer_regex_nfa_state_t**),
        (void**)&new_list));
    if (frag->patch_list) {
      memcpy(new_list, frag->patch_list,
             frag->patch_count * sizeof(iree_tokenizer_regex_nfa_state_t**));
    }
    frag->patch_list = new_list;
    frag->patch_capacity = new_capacity;
  }
  frag->patch_list[frag->patch_count++] = ptr;
  return iree_ok_status();
}

// Patches all dangling pointers to point to the given state.
static void iree_tokenizer_regex_nfa_fragment_patch(
    iree_tokenizer_regex_nfa_fragment_t* frag,
    iree_tokenizer_regex_nfa_state_t* target) {
  for (iree_host_size_t i = 0; i < frag->patch_count; ++i) {
    *frag->patch_list[i] = target;
  }
}

// Merges patch lists from src fragment into dest.
static iree_status_t iree_tokenizer_regex_nfa_fragment_merge_patches(
    iree_tokenizer_regex_nfa_t* nfa, iree_tokenizer_regex_nfa_fragment_t* dest,
    const iree_tokenizer_regex_nfa_fragment_t* src) {
  for (iree_host_size_t i = 0; i < src->patch_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
        nfa, dest, src->patch_list[i]));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Shorthand to Bitmap Conversion
//===----------------------------------------------------------------------===//

static void iree_tokenizer_regex_nfa_shorthand_to_bitmap(
    iree_tokenizer_regex_shorthand_t shorthand, uint8_t* bitmap,
    uint8_t* pseudo_mask) {
  memset(bitmap, 0, 32);
  *pseudo_mask = 0;

  switch (shorthand) {
    case IREE_TOKENIZER_REGEX_SHORTHAND_d:
      for (int c = '0'; c <= '9'; ++c) bitmap[c >> 3] |= (1u << (c & 7));
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_D:
      memset(bitmap, 0xFF, 32);
      for (int c = '0'; c <= '9'; ++c) bitmap[c >> 3] &= ~(1u << (c & 7));
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_w:
      for (int c = 'a'; c <= 'z'; ++c) bitmap[c >> 3] |= (1u << (c & 7));
      for (int c = 'A'; c <= 'Z'; ++c) bitmap[c >> 3] |= (1u << (c & 7));
      for (int c = '0'; c <= '9'; ++c) bitmap[c >> 3] |= (1u << (c & 7));
      bitmap['_' >> 3] |= (1u << ('_' & 7));
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_W:
      memset(bitmap, 0xFF, 32);
      for (int c = 'a'; c <= 'z'; ++c) bitmap[c >> 3] &= ~(1u << (c & 7));
      for (int c = 'A'; c <= 'Z'; ++c) bitmap[c >> 3] &= ~(1u << (c & 7));
      for (int c = '0'; c <= '9'; ++c) bitmap[c >> 3] &= ~(1u << (c & 7));
      bitmap['_' >> 3] &= ~(1u << ('_' & 7));
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_s:
      bitmap[' ' >> 3] |= (1u << (' ' & 7));
      bitmap['\t' >> 3] |= (1u << ('\t' & 7));
      bitmap['\r' >> 3] |= (1u << ('\r' & 7));
      bitmap['\n' >> 3] |= (1u << ('\n' & 7));
      bitmap['\f' >> 3] |= (1u << ('\f' & 7));
      bitmap['\v' >> 3] |= (1u << ('\v' & 7));
      *pseudo_mask = (1u << (IREE_TOKENIZER_REGEX_PSEUDO_WHITESPACE - 0x80));
      break;
    case IREE_TOKENIZER_REGEX_SHORTHAND_S:
      memset(bitmap, 0xFF, 32);
      bitmap[' ' >> 3] &= ~(1u << (' ' & 7));
      bitmap['\t' >> 3] &= ~(1u << ('\t' & 7));
      bitmap['\r' >> 3] &= ~(1u << ('\r' & 7));
      bitmap['\n' >> 3] &= ~(1u << ('\n' & 7));
      bitmap['\f' >> 3] &= ~(1u << ('\f' & 7));
      bitmap['\v' >> 3] &= ~(1u << ('\v' & 7));
      *pseudo_mask =
          0xFF & ~(1u << (IREE_TOKENIZER_REGEX_PSEUDO_WHITESPACE - 0x80));
      break;
  }
}

//===----------------------------------------------------------------------===//
// Case-Insensitive Expansion
//===----------------------------------------------------------------------===//

// Expands a byte to include both cases if ASCII letter.
static void iree_tokenizer_regex_nfa_expand_case_byte(uint8_t byte,
                                                      uint8_t* bitmap) {
  bitmap[byte >> 3] |= (1u << (byte & 7));
  if (byte >= 'a' && byte <= 'z') {
    uint8_t upper = byte - 32;
    bitmap[upper >> 3] |= (1u << (upper & 7));
  } else if (byte >= 'A' && byte <= 'Z') {
    uint8_t lower = byte + 32;
    bitmap[lower >> 3] |= (1u << (lower & 7));
  }
}

// Expands a bitmap to include both cases for all ASCII letters.
static void iree_tokenizer_regex_nfa_expand_case_bitmap(uint8_t* bitmap) {
  for (int c = 'a'; c <= 'z'; ++c) {
    if (bitmap[c >> 3] & (1u << (c & 7))) {
      uint8_t upper = c - 32;
      bitmap[upper >> 3] |= (1u << (upper & 7));
    }
  }
  for (int c = 'A'; c <= 'Z'; ++c) {
    if (bitmap[c >> 3] & (1u << (c & 7))) {
      uint8_t lower = c + 32;
      bitmap[lower >> 3] |= (1u << (lower & 7));
    }
  }
}

//===----------------------------------------------------------------------===//
// Thompson Construction - Build Fragment from AST Node
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_regex_nfa_build_node(
    iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_ast_node_t* node,
    iree_tokenizer_regex_nfa_fragment_t* out_frag);

// Builds fragment for literal byte.
static iree_status_t iree_tokenizer_regex_nfa_build_literal(
    iree_tokenizer_regex_nfa_t* nfa, uint8_t byte, bool case_insensitive,
    iree_tokenizer_regex_nfa_fragment_t* out_frag) {
  if (case_insensitive &&
      ((byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z'))) {
    // Expand to character class with both cases.
    iree_tokenizer_regex_nfa_state_t* state = NULL;
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
        nfa, IREE_TOKENIZER_REGEX_NFA_MATCH_CLASS, &state));
    memset(state->data.match_class.bitmap, 0, 32);
    iree_tokenizer_regex_nfa_expand_case_byte(byte,
                                              state->data.match_class.bitmap);
    state->data.match_class.pseudo_mask = 0;
    iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
        nfa, out_frag, &state->data.match_class.out));
    return iree_ok_status();
  }

  iree_tokenizer_regex_nfa_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
      nfa, IREE_TOKENIZER_REGEX_NFA_MATCH_BYTE, &state));
  state->data.match_byte.byte = byte;

  iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
      nfa, out_frag, &state->data.match_byte.out));
  return iree_ok_status();
}

// Builds fragment for character class.
static iree_status_t iree_tokenizer_regex_nfa_build_char_class(
    iree_tokenizer_regex_nfa_t* nfa, const uint8_t* bitmap,
    uint16_t pseudo_mask, const iree_tokenizer_regex_codepoint_range_t* ranges,
    uint8_t range_count, bool case_insensitive,
    iree_tokenizer_regex_nfa_fragment_t* out_frag) {
  iree_tokenizer_regex_nfa_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
      nfa, IREE_TOKENIZER_REGEX_NFA_MATCH_CLASS, &state));

  memcpy(state->data.match_class.bitmap, bitmap, 32);
  state->data.match_class.pseudo_mask = pseudo_mask;
  state->data.match_class.range_count = range_count;
  for (uint8_t i = 0; i < range_count; ++i) {
    state->data.match_class.ranges[i] = ranges[i];
  }

  if (case_insensitive) {
    iree_tokenizer_regex_nfa_expand_case_bitmap(state->data.match_class.bitmap);
  }

  if (pseudo_mask != 0 || range_count > 0) {
    nfa->uses_unicode = true;
  }

  iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
      nfa, out_frag, &state->data.match_class.out));
  return iree_ok_status();
}

// Builds fragment for dot (any character except newline).
static iree_status_t iree_tokenizer_regex_nfa_build_dot(
    iree_tokenizer_regex_nfa_t* nfa,
    iree_tokenizer_regex_nfa_fragment_t* out_frag) {
  iree_tokenizer_regex_nfa_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
      nfa, IREE_TOKENIZER_REGEX_NFA_MATCH_CLASS, &state));

  // Match any byte except \n and \r.
  memset(state->data.match_class.bitmap, 0xFF, 32);
  state->data.match_class.bitmap['\n' >> 3] &= ~(1u << ('\n' & 7));
  state->data.match_class.bitmap['\r' >> 3] &= ~(1u << ('\r' & 7));
  // Match all pseudo-bytes (Unicode characters including script blocks).
  state->data.match_class.pseudo_mask =
      (1u << IREE_TOKENIZER_REGEX_PSEUDO_COUNT) - 1;
  nfa->uses_unicode = true;

  iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
      nfa, out_frag, &state->data.match_class.out));
  return iree_ok_status();
}

// Builds fragment for concatenation.
static iree_status_t iree_tokenizer_regex_nfa_build_concat(
    iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_ast_node_t* node,
    iree_tokenizer_regex_nfa_fragment_t* out_frag) {
  if (node->data.compound.child_count == 0) {
    iree_tokenizer_regex_nfa_fragment_initialize(out_frag, NULL);
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_build_node(
      nfa, node->data.compound.children[0], out_frag));

  for (iree_host_size_t i = 1; i < node->data.compound.child_count; ++i) {
    iree_tokenizer_regex_nfa_fragment_t next;
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_build_node(
        nfa, node->data.compound.children[i], &next));
    if (!next.start) {
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, NULL);
      return iree_ok_status();
    }

    // Patch result's outputs to next's start.
    iree_tokenizer_regex_nfa_fragment_patch(out_frag, next.start);

    // Result now has next's patch list and lookahead info.
    out_frag->patch_list = next.patch_list;
    out_frag->patch_count = next.patch_count;
    out_frag->patch_capacity = next.patch_capacity;

    // Propagate lookahead from the next fragment (if it has one).
    if (iree_tokenizer_regex_nfa_fragment_has_lookahead(&next)) {
      out_frag->lookahead_type = next.lookahead_type;
      out_frag->lookahead_data = next.lookahead_data;
    }
  }

  return iree_ok_status();
}

// Helper to allocate an accept state with optional lookahead.
static iree_status_t iree_tokenizer_regex_nfa_accept_state_allocate(
    iree_tokenizer_regex_nfa_t* nfa, bool has_lookahead,
    iree_tokenizer_regex_lookahead_type_t lookahead_type,
    uint8_t lookahead_data, uint8_t branch_index,
    iree_tokenizer_regex_nfa_state_t** out_state) {
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
      nfa, IREE_TOKENIZER_REGEX_NFA_ACCEPT, out_state));

  (*out_state)->data.accept.has_lookahead = has_lookahead;
  (*out_state)->data.accept.lookahead_type = lookahead_type;
  (*out_state)->data.accept.lookahead_data = lookahead_data;
  (*out_state)->data.accept.branch_index = branch_index;
  return iree_ok_status();
}

// Builds fragment for alternation.
// Handles mixed lookahead requirements by creating separate accept states.
static iree_status_t iree_tokenizer_regex_nfa_build_alternation(
    iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_ast_node_t* node,
    iree_tokenizer_regex_nfa_fragment_t* out_frag) {
  if (node->data.compound.child_count == 0) {
    iree_tokenizer_regex_nfa_fragment_initialize(out_frag, NULL);
    return iree_ok_status();
  }

  if (node->data.compound.child_count == 1) {
    return iree_tokenizer_regex_nfa_build_node(
        nfa, node->data.compound.children[0], out_frag);
  }

  // Validate branch count fits in uint8_t (branch_index field).
  if (node->data.compound.child_count >
      IREE_TOKENIZER_REGEX_MAX_ALTERNATION_BRANCHES) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "alternation has %" PRIhsz
        " branches but maximum supported is %d; "
        "consider using a trie-based approach for large keyword lists",
        node->data.compound.child_count,
        IREE_TOKENIZER_REGEX_MAX_ALTERNATION_BRANCHES);
  }

  // First, build all children to check for mixed lookahead requirements.
  iree_host_size_t child_count = node->data.compound.child_count;
  iree_tokenizer_regex_nfa_fragment_t* children = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      nfa->arena, child_count * sizeof(iree_tokenizer_regex_nfa_fragment_t),
      (void**)&children));

  bool has_lookahead_branch = false;
  bool has_non_lookahead_branch = false;
  bool has_conflicting_lookaheads = false;
  iree_tokenizer_regex_lookahead_type_t first_lookahead_type =
      IREE_TOKENIZER_REGEX_LOOKAHEAD_NONE;
  uint8_t first_lookahead_data = 0;

  for (iree_host_size_t i = 0; i < child_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_build_node(
        nfa, node->data.compound.children[i], &children[i]));
    if (!children[i].start) {
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, NULL);
      return iree_ok_status();
    }

    if (iree_tokenizer_regex_nfa_fragment_has_lookahead(&children[i])) {
      if (!has_lookahead_branch) {
        first_lookahead_type = children[i].lookahead_type;
        first_lookahead_data = children[i].lookahead_data;
      } else {
        // Check if this lookahead conflicts with the first one.
        if (children[i].lookahead_type != first_lookahead_type ||
            children[i].lookahead_data != first_lookahead_data) {
          has_conflicting_lookaheads = true;
        }
      }
      has_lookahead_branch = true;
    } else {
      has_non_lookahead_branch = true;
    }
  }

  // Create split state.
  iree_tokenizer_regex_nfa_state_t* split = NULL;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
      nfa, IREE_TOKENIZER_REGEX_NFA_EPSILON, &split));

  iree_tokenizer_regex_nfa_fragment_initialize(out_frag, split);

  // Check if we need separate accept states:
  // - Mixed lookahead/no-lookahead branches (for PCRE-compatible matching)
  // - All-lookahead but with different constraints (conflicting)
  // In either case, we create per-branch accept states so DFA extraction
  // can compute min branch indices for PCRE-compatible match selection.
  bool needs_separate_accepts =
      (has_lookahead_branch && has_non_lookahead_branch) ||
      has_conflicting_lookaheads;

  if (needs_separate_accepts) {
    // Create SEPARATE accept states for each branch to preserve branch index.
    // This is necessary for PCRE-compatible match selection: we need to know
    // which branches are actually represented in each DFA state to compute
    // min_lookahead_branch vs min_no_lookahead_branch at runtime.
    //
    // For conflicting lookaheads (a(?!b)|a(?!c)): DFA extraction detects if
    // paths converge and reports an error.
    //
    // For mixed lookahead/non-lookahead (\s*[\r\n]+|\s+(?!\S)|\s+): DFA
    // extraction computes has_early_no_lookahead per-state based on which
    // branches' accept states are actually present.
    for (iree_host_size_t i = 0; i < child_count; ++i) {
      iree_tokenizer_regex_nfa_state_t* accept = NULL;
      if (iree_tokenizer_regex_nfa_fragment_has_lookahead(&children[i])) {
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_accept_state_allocate(
            nfa, true, children[i].lookahead_type, children[i].lookahead_data,
            (uint8_t)i, &accept));
      } else {
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_accept_state_allocate(
            nfa, false, IREE_TOKENIZER_REGEX_LOOKAHEAD_NONE, 0, (uint8_t)i,
            &accept));
      }
      iree_tokenizer_regex_nfa_fragment_patch(&children[i], accept);
    }

    // Wire up splits.
    if (child_count == 2) {
      split->data.epsilon.out1 = children[0].start;
      split->data.epsilon.out2 = children[1].start;
    } else {
      iree_tokenizer_regex_nfa_state_t* current_split = split;
      for (iree_host_size_t i = 0; i < child_count; ++i) {
        if (i == child_count - 1) {
          current_split->data.epsilon.out2 = children[i].start;
        } else if (i == 0) {
          current_split->data.epsilon.out1 = children[i].start;
        } else {
          iree_tokenizer_regex_nfa_state_t* next_split = NULL;
          IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
              nfa, IREE_TOKENIZER_REGEX_NFA_EPSILON, &next_split));
          current_split->data.epsilon.out2 = next_split;
          current_split = next_split;
          current_split->data.epsilon.out1 = children[i].start;
        }
      }
    }

    // Result has no patches - everything is already connected to accept.
    out_frag->lookahead_type = IREE_TOKENIZER_REGEX_LOOKAHEAD_NONE;
    return iree_ok_status();
  }

  // No mixed lookahead - normal alternation with merged patches.
  if (child_count == 2) {
    split->data.epsilon.out1 = children[0].start;
    split->data.epsilon.out2 = children[1].start;

    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_merge_patches(
        nfa, out_frag, &children[0]));
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_merge_patches(
        nfa, out_frag, &children[1]));

    // Propagate lookahead if all branches have the same one.
    if (has_lookahead_branch && !has_non_lookahead_branch) {
      out_frag->lookahead_type = first_lookahead_type;
      out_frag->lookahead_data = first_lookahead_data;
    }
    return iree_ok_status();
  }

  // N-way alternation.
  iree_tokenizer_regex_nfa_state_t* current_split = split;
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    if (i == child_count - 1) {
      current_split->data.epsilon.out2 = children[i].start;
    } else if (i == 0) {
      current_split->data.epsilon.out1 = children[i].start;
    } else {
      iree_tokenizer_regex_nfa_state_t* next_split = NULL;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
          nfa, IREE_TOKENIZER_REGEX_NFA_EPSILON, &next_split));
      current_split->data.epsilon.out2 = next_split;
      current_split = next_split;
      current_split->data.epsilon.out1 = children[i].start;
    }

    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_merge_patches(
        nfa, out_frag, &children[i]));
  }

  // Propagate lookahead if all branches have the same one.
  if (has_lookahead_branch && !has_non_lookahead_branch) {
    out_frag->lookahead_type = first_lookahead_type;
    out_frag->lookahead_data = first_lookahead_data;
  }

  return iree_ok_status();
}

// Builds fragment for quantifier.
static iree_status_t iree_tokenizer_regex_nfa_build_quantifier(
    iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_ast_node_t* node,
    iree_tokenizer_regex_nfa_fragment_t* out_frag) {
  uint16_t min = node->data.quantifier.min;
  uint16_t max = node->data.quantifier.max;

  // Special case: {0} - matches nothing (empty).
  if (max == 0) {
    iree_tokenizer_regex_nfa_state_t* state = NULL;
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
        nfa, IREE_TOKENIZER_REGEX_NFA_EPSILON, &state));
    iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
        nfa, out_frag, &state->data.epsilon.out1));
    return iree_ok_status();
  }

  // Build required repetitions (min copies).
  memset(out_frag, 0, sizeof(*out_frag));

  for (uint16_t i = 0; i < min; ++i) {
    iree_tokenizer_regex_nfa_fragment_t child;
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_build_node(
        nfa, node->data.quantifier.child, &child));
    if (!child.start) {
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, NULL);
      return iree_ok_status();
    }

    if (i == 0) {
      *out_frag = child;
    } else {
      iree_tokenizer_regex_nfa_fragment_patch(out_frag, child.start);
      out_frag->patch_list = child.patch_list;
      out_frag->patch_count = child.patch_count;
      out_frag->patch_capacity = child.patch_capacity;
    }
  }

  // Handle optional/unbounded part.
  if (max == UINT16_MAX) {
    // * or + semantics: add loop.
    iree_tokenizer_regex_nfa_fragment_t child;
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_build_node(
        nfa, node->data.quantifier.child, &child));
    if (!child.start) {
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, NULL);
      return iree_ok_status();
    }

    iree_tokenizer_regex_nfa_state_t* loop_split = NULL;
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
        nfa, IREE_TOKENIZER_REGEX_NFA_EPSILON, &loop_split));

    loop_split->data.epsilon.out1 = child.start;  // Loop into child.
    iree_tokenizer_regex_nfa_fragment_patch(&child,
                                            loop_split);  // Child loops back.

    if (min == 0) {
      // * - split is the start.
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, loop_split);
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
          nfa, out_frag, &loop_split->data.epsilon.out2));
    } else {
      // + - patch previous to split.
      iree_tokenizer_regex_nfa_fragment_patch(out_frag, loop_split);
      out_frag->patch_list = NULL;
      out_frag->patch_count = 0;
      out_frag->patch_capacity = 0;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
          nfa, out_frag, &loop_split->data.epsilon.out2));
    }
  } else if (max > min) {
    // {n,m} where m > n: add optional copies.
    for (uint16_t i = min; i < max; ++i) {
      iree_tokenizer_regex_nfa_fragment_t child;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_build_node(
          nfa, node->data.quantifier.child, &child));
      if (!child.start) {
        iree_tokenizer_regex_nfa_fragment_initialize(out_frag, NULL);
        return iree_ok_status();
      }

      iree_tokenizer_regex_nfa_state_t* opt_split = NULL;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
          nfa, IREE_TOKENIZER_REGEX_NFA_EPSILON, &opt_split));

      opt_split->data.epsilon.out1 = child.start;  // Take child.

      if (i == min && min == 0) {
        // First optional, no required - split is start.
        iree_tokenizer_regex_nfa_fragment_initialize(out_frag, opt_split);
      } else {
        // Patch previous to split.
        iree_tokenizer_regex_nfa_fragment_patch(out_frag, opt_split);
      }

      // Merge child's patches and split's skip.
      out_frag->patch_list = NULL;
      out_frag->patch_count = 0;
      out_frag->patch_capacity = 0;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_merge_patches(
          nfa, out_frag, &child));
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
          nfa, out_frag, &opt_split->data.epsilon.out2));
    }
  }

  return iree_ok_status();
}

// Main dispatch for building a node.
static iree_status_t iree_tokenizer_regex_nfa_build_node(
    iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_ast_node_t* node,
    iree_tokenizer_regex_nfa_fragment_t* out_frag) {
  if (!node) {
    iree_tokenizer_regex_nfa_fragment_initialize(out_frag, NULL);
    return iree_ok_status();
  }

  switch (node->type) {
    case IREE_TOKENIZER_REGEX_AST_EMPTY: {
      // Empty expression - epsilon transition.
      iree_tokenizer_regex_nfa_state_t* state = NULL;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
          nfa, IREE_TOKENIZER_REGEX_NFA_EPSILON, &state));
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
          nfa, out_frag, &state->data.epsilon.out1));
      return iree_ok_status();
    }

    case IREE_TOKENIZER_REGEX_AST_LITERAL:
      return iree_tokenizer_regex_nfa_build_literal(
          nfa, node->data.literal, node->case_insensitive, out_frag);

    case IREE_TOKENIZER_REGEX_AST_DOT:
      return iree_tokenizer_regex_nfa_build_dot(nfa, out_frag);

    case IREE_TOKENIZER_REGEX_AST_CHAR_CLASS:
      return iree_tokenizer_regex_nfa_build_char_class(
          nfa, node->data.char_class.bitmap, node->data.char_class.pseudo_mask,
          node->data.char_class.ranges, node->data.char_class.range_count,
          node->case_insensitive, out_frag);

    case IREE_TOKENIZER_REGEX_AST_SHORTHAND: {
      uint8_t bitmap[32];
      uint8_t pseudo_mask;
      iree_tokenizer_regex_nfa_shorthand_to_bitmap(
          (iree_tokenizer_regex_shorthand_t)node->data.shorthand, bitmap,
          &pseudo_mask);
      // Shorthands have no ranges - they use bitmap + pseudo_mask.
      return iree_tokenizer_regex_nfa_build_char_class(
          nfa, bitmap, pseudo_mask, NULL, 0, node->case_insensitive, out_frag);
    }

    case IREE_TOKENIZER_REGEX_AST_UNICODE_PROP: {
      // Unicode property - matches both ASCII chars AND pseudo-byte.
      iree_tokenizer_regex_nfa_state_t* state = NULL;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
          nfa, IREE_TOKENIZER_REGEX_NFA_MATCH_CLASS, &state));
      memset(state->data.match_class.bitmap, 0, 32);

      // Use the Unicode library to classify ASCII characters.
      uint8_t pseudo = node->data.unicode_pseudo_byte;
      for (int c = 0; c < 128; ++c) {
        bool matches = false;
        switch (pseudo) {
          case IREE_TOKENIZER_REGEX_PSEUDO_LETTER:
            matches = iree_unicode_is_letter((uint32_t)c);
            break;
          case IREE_TOKENIZER_REGEX_PSEUDO_NUMBER:
            matches = iree_unicode_is_number((uint32_t)c);
            break;
          case IREE_TOKENIZER_REGEX_PSEUDO_PUNCT:
            matches = iree_unicode_is_punctuation((uint32_t)c);
            break;
          case IREE_TOKENIZER_REGEX_PSEUDO_MARK:
            matches = iree_unicode_is_mark((uint32_t)c);
            break;
          case IREE_TOKENIZER_REGEX_PSEUDO_SYMBOL:
            matches = iree_unicode_is_symbol((uint32_t)c);
            break;
          case IREE_TOKENIZER_REGEX_PSEUDO_SEPARATOR:
            matches = iree_unicode_is_separator((uint32_t)c);
            break;
          case IREE_TOKENIZER_REGEX_PSEUDO_OTHER:
            matches = iree_unicode_is_other((uint32_t)c);
            break;
        }
        if (matches) {
          state->data.match_class.bitmap[c >> 3] |= (1u << (c & 7));
        }
      }

      state->data.match_class.pseudo_mask = (1u << (pseudo - 0x80));
      state->data.match_class.range_count = 0;
      nfa->uses_unicode = true;
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
          nfa, out_frag, &state->data.match_class.out));
      return iree_ok_status();
    }

    case IREE_TOKENIZER_REGEX_AST_ANCHOR_START: {
      iree_tokenizer_regex_nfa_state_t* state = NULL;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
          nfa, IREE_TOKENIZER_REGEX_NFA_ANCHOR_START, &state));
      nfa->has_anchors = true;
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
          nfa, out_frag, &state->data.anchor_out));
      return iree_ok_status();
    }

    case IREE_TOKENIZER_REGEX_AST_ANCHOR_END: {
      iree_tokenizer_regex_nfa_state_t* state = NULL;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
          nfa, IREE_TOKENIZER_REGEX_NFA_ANCHOR_END, &state));
      nfa->has_anchors = true;
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
          nfa, out_frag, &state->data.anchor_out));
      return iree_ok_status();
    }

    case IREE_TOKENIZER_REGEX_AST_CONCAT:
      return iree_tokenizer_regex_nfa_build_concat(nfa, node, out_frag);

    case IREE_TOKENIZER_REGEX_AST_ALTERNATION:
      return iree_tokenizer_regex_nfa_build_alternation(nfa, node, out_frag);

    case IREE_TOKENIZER_REGEX_AST_QUANTIFIER:
      return iree_tokenizer_regex_nfa_build_quantifier(nfa, node, out_frag);

    case IREE_TOKENIZER_REGEX_AST_GROUP:
      return iree_tokenizer_regex_nfa_build_node(nfa, node->data.group_child,
                                                 out_frag);

    case IREE_TOKENIZER_REGEX_AST_NEG_LOOKAHEAD: {
      // Negative lookahead - extracts constraint and stores on fragment.
      nfa->has_lookahead = true;

      // Create epsilon transition (lookahead is checked at DFA level).
      iree_tokenizer_regex_nfa_state_t* state = NULL;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
          nfa, IREE_TOKENIZER_REGEX_NFA_EPSILON, &state));
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, state);
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_fragment_add_patch(
          nfa, out_frag, &state->data.epsilon.out1));

      // Extract lookahead info from the child node and store on fragment.
      const iree_tokenizer_regex_ast_node_t* child = node->data.lookahead_child;
      if (child) {
        switch (child->type) {
          case IREE_TOKENIZER_REGEX_AST_LITERAL:
            out_frag->lookahead_type = IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_CHAR;
            out_frag->lookahead_data = child->data.literal;
            break;
          case IREE_TOKENIZER_REGEX_AST_SHORTHAND:
            out_frag->lookahead_type =
                IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_SHORTHAND;
            out_frag->lookahead_data = child->data.shorthand;
            break;
          case IREE_TOKENIZER_REGEX_AST_CHAR_CLASS:
            // Character class lookahead requires bitmap storage which isn't
            // implemented. Fail at compile time rather than load time.
            return iree_make_status(
                IREE_STATUS_UNIMPLEMENTED,
                "negative lookahead with character class not supported; "
                "use shorthand (\\S, \\d, \\w) or single character instead");
          case IREE_TOKENIZER_REGEX_AST_DOT:
            return iree_make_status(
                IREE_STATUS_UNIMPLEMENTED,
                "negative lookahead with dot (.) not supported; "
                "dot lookahead only makes sense at end-of-string");
          case IREE_TOKENIZER_REGEX_AST_UNICODE_PROP:
            return iree_make_status(
                IREE_STATUS_UNIMPLEMENTED,
                "negative lookahead with Unicode property not supported; "
                "use shorthand (\\S, \\d, \\w) or single character instead");
          default:
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "unsupported node type %d in negative lookahead",
                (int)child->type);
        }
      }

      return iree_ok_status();
    }

    default:
      iree_tokenizer_regex_nfa_fragment_initialize(out_frag, NULL);
      return iree_ok_status();
  }
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_regex_nfa_build(
    const iree_tokenizer_regex_ast_node_t* ast, iree_arena_allocator_t* arena,
    iree_tokenizer_regex_nfa_t* out_nfa) {
  if (!ast || !arena || !out_nfa) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  memset(out_nfa, 0, sizeof(*out_nfa));
  out_nfa->arena = arena;
  out_nfa->pending_lookahead_type = IREE_TOKENIZER_REGEX_LOOKAHEAD_NONE;
  out_nfa->pending_lookahead_data = 0;

  // Build NFA fragment from AST.
  iree_tokenizer_regex_nfa_fragment_t frag;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_regex_nfa_build_node(out_nfa, ast, &frag));
  if (!frag.start) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to build NFA from AST");
  }

  out_nfa->start = frag.start;

  // If fragment has no dangling outputs, accept states were already created
  // (e.g., by mixed-lookahead alternation). No need to create another.
  if (frag.patch_count == 0) {
    // Find an accept state for out_nfa->accept (used for traversal).
    for (uint32_t i = 0; i < out_nfa->state_count; ++i) {
      if (out_nfa->states[i]->type == IREE_TOKENIZER_REGEX_NFA_ACCEPT) {
        out_nfa->accept = out_nfa->states[i];
        break;
      }
    }
    return iree_ok_status();
  }

  // Allocate accept state with lookahead from the fragment.
  // For patterns without mixed-lookahead alternation, has_early_no_lookahead
  // is false (no alternation means no "early" non-lookahead branch to prefer).
  iree_tokenizer_regex_nfa_state_t* accept = NULL;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_allocate(
      out_nfa, IREE_TOKENIZER_REGEX_NFA_ACCEPT, &accept));

  // Apply fragment's lookahead info to the accept state.
  if (iree_tokenizer_regex_nfa_fragment_has_lookahead(&frag)) {
    accept->data.accept.has_lookahead = true;
    accept->data.accept.lookahead_type = frag.lookahead_type;
    accept->data.accept.lookahead_data = frag.lookahead_data;
  } else {
    accept->data.accept.has_lookahead = false;
    accept->data.accept.lookahead_type = IREE_TOKENIZER_REGEX_LOOKAHEAD_NONE;
    accept->data.accept.lookahead_data = 0;
  }
  // No alternation means single branch at index 0.
  accept->data.accept.branch_index = 0;

  // Patch all dangling outputs to accept.
  iree_tokenizer_regex_nfa_fragment_patch(&frag, accept);

  out_nfa->accept = accept;

  return iree_ok_status();
}
