// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/regex/internal/dfa.h"

#include <string.h>

#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// NFA State Set Operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_regex_nfa_state_set_init(
    iree_arena_allocator_t* arena, uint32_t nfa_state_count,
    iree_tokenizer_regex_nfa_state_set_t* out_set) {
  out_set->nfa_state_count = nfa_state_count;
  out_set->capacity = (nfa_state_count + 63) / 64;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, out_set->capacity * sizeof(uint64_t), (void**)&out_set->bits));
  memset(out_set->bits, 0, out_set->capacity * sizeof(uint64_t));
  return iree_ok_status();
}

static void iree_tokenizer_regex_nfa_state_set_clear(
    iree_tokenizer_regex_nfa_state_set_t* set) {
  memset(set->bits, 0, set->capacity * sizeof(uint64_t));
}

static void iree_tokenizer_regex_nfa_state_set_add(
    iree_tokenizer_regex_nfa_state_set_t* set, uint32_t state_id) {
  if (state_id < set->nfa_state_count) {
    set->bits[state_id / 64] |= (1ULL << (state_id % 64));
  }
}

static bool iree_tokenizer_regex_nfa_state_set_contains(
    const iree_tokenizer_regex_nfa_state_set_t* set, uint32_t state_id) {
  if (state_id >= set->nfa_state_count) return false;
  return (set->bits[state_id / 64] & (1ULL << (state_id % 64))) != 0;
}

static bool iree_tokenizer_regex_nfa_state_set_is_empty(
    const iree_tokenizer_regex_nfa_state_set_t* set) {
  for (uint32_t i = 0; i < set->capacity; ++i) {
    if (set->bits[i] != 0) return false;
  }
  return true;
}

static bool iree_tokenizer_regex_nfa_state_set_equals(
    const iree_tokenizer_regex_nfa_state_set_t* a,
    const iree_tokenizer_regex_nfa_state_set_t* b) {
  if (a->capacity != b->capacity) return false;
  return memcmp(a->bits, b->bits, a->capacity * sizeof(uint64_t)) == 0;
}

static void iree_tokenizer_regex_nfa_state_set_copy(
    iree_tokenizer_regex_nfa_state_set_t* dst,
    const iree_tokenizer_regex_nfa_state_set_t* src) {
  memcpy(dst->bits, src->bits, src->capacity * sizeof(uint64_t));
}

// FNV-1a hash for state sets.
static uint64_t iree_tokenizer_regex_nfa_state_set_hash(
    const iree_tokenizer_regex_nfa_state_set_t* set) {
  uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
  const uint8_t* bytes = (const uint8_t*)set->bits;
  iree_host_size_t byte_count = set->capacity * sizeof(uint64_t);
  for (iree_host_size_t i = 0; i < byte_count; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;  // FNV prime
  }
  return hash;
}

//===----------------------------------------------------------------------===//
// NFA State Set Utilities
//===----------------------------------------------------------------------===//

// Returns true if the NFA state set contains any consuming states (MATCH_BYTE
// or MATCH_CLASS). If false, the set cannot consume input and represents a
// dead-end in the DFA.
static bool iree_tokenizer_regex_nfa_state_set_has_consuming(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_set_t* set) {
  for (uint32_t i = 0; i < nfa->state_count; ++i) {
    if (!iree_tokenizer_regex_nfa_state_set_contains(set, i)) continue;
    const iree_tokenizer_regex_nfa_state_t* state = nfa->states[i];
    if (state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_MATCH_BYTE ||
        state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_MATCH_CLASS) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Epsilon Closure
//===----------------------------------------------------------------------===//

// Anchor tracking for epsilon closure.
typedef struct iree_tokenizer_regex_anchor_info_t {
  // Set of ACCEPT state IDs that require end anchor.
  // An accept requires end anchor only if ALL paths to it go through
  // ANCHOR_END. Computed via two-pass: (accepts in full closure) - (accepts in
  // unanchored closure).
  iree_tokenizer_regex_nfa_state_set_t* end_anchor_accepts;
  // Set of states reachable WITHOUT going through ANCHOR_END.
  // Used to determine which accepts don't require end anchor.
  iree_tokenizer_regex_nfa_state_set_t* unanchored_reachable;
} iree_tokenizer_regex_anchor_info_t;

// Worklist item for iterative epsilon closure.
// Tracks the state to visit and whether we reached it via ANCHOR_END.
typedef struct iree_tokenizer_regex_epsilon_work_item_t {
  const iree_tokenizer_regex_nfa_state_t* state;
  bool through_end_anchor;
} iree_tokenizer_regex_epsilon_work_item_t;

// Computes the epsilon closure of a single NFA state into |out_set|.
// Uses an iterative worklist to avoid stack overflow on deep patterns.
// The set acts as the visited marker.
//
// |nfa| is the source NFA.
// |initial_state| is the state to start the closure from.
// |at_start| controls whether ANCHOR_START transitions can be followed:
//   - true: we're computing the initial DFA start state (position 0)
//   - false: we're computing a target after consuming input (position > 0)
// |worklist| is pre-allocated workspace of size >= nfa->state_count.
// |out_set| receives the epsilon closure (states reachable via epsilon).
// |anchor_info| optionally tracks which accepts were reached via ANCHOR_END.
static void iree_tokenizer_regex_epsilon_closure_single_with_anchors(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_t* initial_state, bool at_start,
    iree_tokenizer_regex_epsilon_work_item_t* worklist,
    iree_tokenizer_regex_nfa_state_set_t* out_set,
    iree_tokenizer_regex_anchor_info_t* anchor_info) {
  if (!initial_state) return;

  // Initialize worklist with the starting state.
  iree_host_size_t worklist_head = 0;
  iree_host_size_t worklist_tail = 0;
  worklist[worklist_tail++] = (iree_tokenizer_regex_epsilon_work_item_t){
      .state = initial_state,
      .through_end_anchor = false,
  };

  while (worklist_head < worklist_tail) {
    // Pop next item from worklist.
    iree_tokenizer_regex_epsilon_work_item_t item = worklist[worklist_head++];
    const iree_tokenizer_regex_nfa_state_t* state = item.state;
    bool through_end_anchor = item.through_end_anchor;

    if (!state) continue;
    if (iree_tokenizer_regex_nfa_state_set_contains(out_set, state->id)) {
      continue;
    }

    iree_tokenizer_regex_nfa_state_set_add(out_set, state->id);

    // Track accept states reached via end anchor.
    if (state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_ACCEPT &&
        through_end_anchor && anchor_info && anchor_info->end_anchor_accepts) {
      iree_tokenizer_regex_nfa_state_set_add(anchor_info->end_anchor_accepts,
                                             state->id);
    }

    // Follow epsilon-like transitions by pushing to worklist.
    // We check if successors are already visited before pushing to avoid
    // redundant worklist entries. This prevents overflow with dense epsilon
    // graphs (e.g., from `(a?){100}` where each state has 2 epsilon
    // successors).
    if (state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_EPSILON) {
      if (state->data.epsilon.out1 &&
          !iree_tokenizer_regex_nfa_state_set_contains(
              out_set, state->data.epsilon.out1->id)) {
        worklist[worklist_tail++] = (iree_tokenizer_regex_epsilon_work_item_t){
            .state = state->data.epsilon.out1,
            .through_end_anchor = through_end_anchor,
        };
      }
      if (state->data.epsilon.out2 &&
          !iree_tokenizer_regex_nfa_state_set_contains(
              out_set, state->data.epsilon.out2->id)) {
        worklist[worklist_tail++] = (iree_tokenizer_regex_epsilon_work_item_t){
            .state = state->data.epsilon.out2,
            .through_end_anchor = through_end_anchor,
        };
      }
    } else if (state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_ANCHOR_START) {
      // Start anchor can only be traversed at position 0 (beginning of input).
      // After consuming any character, we're at position > 0, so start anchor
      // transitions become dead ends. This makes patterns like `a^b` correctly
      // fail to match and `^a|b` only match 'a' at the actual start.
      if (at_start && state->data.anchor_out &&
          !iree_tokenizer_regex_nfa_state_set_contains(
              out_set, state->data.anchor_out->id)) {
        worklist[worklist_tail++] = (iree_tokenizer_regex_epsilon_work_item_t){
            .state = state->data.anchor_out,
            .through_end_anchor = through_end_anchor,
        };
      }
    } else if (state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_ANCHOR_END) {
      // End anchor - mark that subsequent accept states require end anchor.
      if (state->data.anchor_out &&
          !iree_tokenizer_regex_nfa_state_set_contains(
              out_set, state->data.anchor_out->id)) {
        worklist[worklist_tail++] = (iree_tokenizer_regex_epsilon_work_item_t){
            .state = state->data.anchor_out,
            .through_end_anchor = true,  // Now through end anchor.
        };
      }
    }
    // MATCH_BYTE, MATCH_CLASS, ACCEPT: no epsilon successors to follow.
  }
}

// Computes the epsilon closure of a state set with anchor tracking.
// |at_start| controls whether ANCHOR_START transitions can be followed.
// |worklist| is pre-allocated workspace of size >= nfa->state_count.
static void iree_tokenizer_regex_epsilon_closure_set_with_anchors(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_set_t* input_set, bool at_start,
    iree_tokenizer_regex_epsilon_work_item_t* worklist,
    iree_tokenizer_regex_nfa_state_set_t* out_set,
    iree_tokenizer_regex_anchor_info_t* anchor_info) {
  iree_tokenizer_regex_nfa_state_set_clear(out_set);
  if (anchor_info && anchor_info->end_anchor_accepts) {
    iree_tokenizer_regex_nfa_state_set_clear(anchor_info->end_anchor_accepts);
  }
  for (uint32_t i = 0; i < nfa->state_count; ++i) {
    if (iree_tokenizer_regex_nfa_state_set_contains(input_set, i)) {
      iree_tokenizer_regex_epsilon_closure_single_with_anchors(
          nfa, nfa->states[i], at_start, worklist, out_set, anchor_info);
    }
  }
}

//===----------------------------------------------------------------------===//
// Unanchored Epsilon Closure (for two-pass anchor tracking)
//===----------------------------------------------------------------------===//

// Computes epsilon closure while treating ANCHOR_END as a dead-end.
// Used to find states reachable WITHOUT going through end anchor.
// This is the second pass of the two-pass anchor tracking algorithm:
//   Pass 1: Full closure (all reachable states)
//   Pass 2: Unanchored closure (states reachable without ANCHOR_END)
//   Result: end_anchor_accepts = (accepts in pass1) - (accepts in pass2)
//
// |worklist| is just state IDs (simpler than the anchor-tracking version).
static void iree_tokenizer_regex_epsilon_closure_single_ignoring_end_anchor(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_t* initial_state, bool at_start,
    uint32_t* worklist, iree_tokenizer_regex_nfa_state_set_t* out_set) {
  if (!initial_state) return;

  // Initialize worklist with the starting state.
  iree_host_size_t worklist_head = 0;
  iree_host_size_t worklist_tail = 0;
  worklist[worklist_tail++] = initial_state->id;

  while (worklist_head < worklist_tail) {
    // Pop next state ID from worklist.
    uint32_t state_id = worklist[worklist_head++];
    if (state_id >= nfa->state_count) continue;

    const iree_tokenizer_regex_nfa_state_t* state = nfa->states[state_id];
    if (!state) continue;
    if (iree_tokenizer_regex_nfa_state_set_contains(out_set, state->id)) {
      continue;
    }

    iree_tokenizer_regex_nfa_state_set_add(out_set, state->id);

    // Follow epsilon-like transitions EXCEPT ANCHOR_END.
    if (state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_EPSILON) {
      if (state->data.epsilon.out1 &&
          !iree_tokenizer_regex_nfa_state_set_contains(
              out_set, state->data.epsilon.out1->id)) {
        worklist[worklist_tail++] = state->data.epsilon.out1->id;
      }
      if (state->data.epsilon.out2 &&
          !iree_tokenizer_regex_nfa_state_set_contains(
              out_set, state->data.epsilon.out2->id)) {
        worklist[worklist_tail++] = state->data.epsilon.out2->id;
      }
    } else if (state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_ANCHOR_START) {
      // Start anchor: same rules as full closure.
      if (at_start && state->data.anchor_out &&
          !iree_tokenizer_regex_nfa_state_set_contains(
              out_set, state->data.anchor_out->id)) {
        worklist[worklist_tail++] = state->data.anchor_out->id;
      }
    }
    // ANCHOR_END: treat as dead-end (don't follow its successor).
    // MATCH_BYTE, MATCH_CLASS, ACCEPT: no epsilon successors.
  }
}

// Computes the unanchored closure of a state set (ignoring ANCHOR_END).
static void iree_tokenizer_regex_epsilon_closure_set_ignoring_end_anchor(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_set_t* input_set, bool at_start,
    uint32_t* worklist, iree_tokenizer_regex_nfa_state_set_t* out_set) {
  iree_tokenizer_regex_nfa_state_set_clear(out_set);
  for (uint32_t i = 0; i < nfa->state_count; ++i) {
    if (iree_tokenizer_regex_nfa_state_set_contains(input_set, i)) {
      iree_tokenizer_regex_epsilon_closure_single_ignoring_end_anchor(
          nfa, nfa->states[i], at_start, worklist, out_set);
    }
  }
}

// Computes end_anchor_accepts using two-pass algorithm.
// An accept state requires end anchor only if it's in the full closure
// but NOT in the unanchored closure (i.e., only reachable via ANCHOR_END).
static void iree_tokenizer_regex_compute_end_anchor_accepts(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_set_t* full_closure,
    const iree_tokenizer_regex_nfa_state_set_t* unanchored_closure,
    iree_tokenizer_regex_nfa_state_set_t* end_anchor_accepts) {
  iree_tokenizer_regex_nfa_state_set_clear(end_anchor_accepts);
  for (uint32_t i = 0; i < nfa->state_count; ++i) {
    if (!iree_tokenizer_regex_nfa_state_set_contains(full_closure, i)) continue;
    const iree_tokenizer_regex_nfa_state_t* state = nfa->states[i];
    if (state->type != IREE_TOKENIZER_UTIL_REGEX_NFA_ACCEPT) continue;
    // Accept is in full closure. Is it also reachable without ANCHOR_END?
    if (!iree_tokenizer_regex_nfa_state_set_contains(unanchored_closure, i)) {
      // Only reachable via anchored path - requires end anchor.
      iree_tokenizer_regex_nfa_state_set_add(end_anchor_accepts, i);
    }
  }
}

//===----------------------------------------------------------------------===//
// Move Function
//===----------------------------------------------------------------------===//

// Computes move(state_set, byte): the set of NFA states reachable from
// |state_set| by consuming |byte|.
static void iree_tokenizer_regex_nfa_move(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_set_t* state_set, uint8_t byte,
    iree_tokenizer_regex_nfa_state_set_t* out_set) {
  iree_tokenizer_regex_nfa_state_set_clear(out_set);

  for (uint32_t i = 0; i < nfa->state_count; ++i) {
    if (!iree_tokenizer_regex_nfa_state_set_contains(state_set, i)) continue;

    const iree_tokenizer_regex_nfa_state_t* state = nfa->states[i];
    switch (state->type) {
      case IREE_TOKENIZER_UTIL_REGEX_NFA_MATCH_BYTE:
        if (state->data.match_byte.byte == byte) {
          iree_tokenizer_regex_nfa_state_set_add(
              out_set, state->data.match_byte.out->id);
        }
        break;

      case IREE_TOKENIZER_UTIL_REGEX_NFA_MATCH_CLASS: {
        bool matches = false;
        if (byte < 0x80) {
          // Regular byte: check bitmap.
          matches = (state->data.match_class.bitmap[byte / 8] &
                     (1 << (byte % 8))) != 0;
        } else if (byte >= 0x80 &&
                   byte < 0x80 + IREE_TOKENIZER_UTIL_REGEX_PSEUDO_COUNT) {
          // Pseudo-byte: check pseudo_mask.
          matches =
              (state->data.match_class.pseudo_mask & (1 << (byte - 0x80))) != 0;
        }
        if (matches) {
          iree_tokenizer_regex_nfa_state_set_add(
              out_set, state->data.match_class.out->id);
        }
        break;
      }

      default:
        // Other state types don't consume input.
        break;
    }
  }
}

//===----------------------------------------------------------------------===//
// DFA State Management
//===----------------------------------------------------------------------===//

// Hash table bucket for state set deduplication.
typedef struct iree_tokenizer_regex_dfa_state_bucket_t {
  iree_tokenizer_regex_dfa_state_t* state;
  struct iree_tokenizer_regex_dfa_state_bucket_t* next;
} iree_tokenizer_regex_dfa_state_bucket_t;

// Hash table for finding existing DFA states by NFA state set.
typedef struct iree_tokenizer_regex_dfa_state_map_t {
  iree_tokenizer_regex_dfa_state_bucket_t** buckets;
  uint32_t bucket_count;
  uint32_t state_count;
  iree_arena_allocator_t* arena;
} iree_tokenizer_regex_dfa_state_map_t;

static iree_status_t iree_tokenizer_regex_dfa_state_map_init(
    iree_arena_allocator_t* arena, uint32_t bucket_count,
    iree_tokenizer_regex_dfa_state_map_t* out_map) {
  out_map->bucket_count = bucket_count;
  out_map->state_count = 0;
  out_map->arena = arena;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, bucket_count * sizeof(iree_tokenizer_regex_dfa_state_bucket_t*),
      (void**)&out_map->buckets));
  memset(out_map->buckets, 0,
         bucket_count * sizeof(iree_tokenizer_regex_dfa_state_bucket_t*));
  return iree_ok_status();
}

// Resizes the hash table to the given bucket count.
// Old bucket array is abandoned (arena handles deallocation).
static iree_status_t iree_tokenizer_regex_dfa_state_map_resize(
    iree_tokenizer_regex_dfa_state_map_t* map, uint32_t new_bucket_count) {
  iree_tokenizer_regex_dfa_state_bucket_t** new_buckets;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      map->arena,
      new_bucket_count * sizeof(iree_tokenizer_regex_dfa_state_bucket_t*),
      (void**)&new_buckets));
  memset(new_buckets, 0,
         new_bucket_count * sizeof(iree_tokenizer_regex_dfa_state_bucket_t*));

  // Rehash all entries from old buckets.
  for (uint32_t i = 0; i < map->bucket_count; ++i) {
    iree_tokenizer_regex_dfa_state_bucket_t* bucket = map->buckets[i];
    while (bucket) {
      iree_tokenizer_regex_dfa_state_bucket_t* next = bucket->next;
      uint64_t hash =
          iree_tokenizer_regex_nfa_state_set_hash(&bucket->state->nfa_states);
      uint32_t new_index = hash % new_bucket_count;
      bucket->next = new_buckets[new_index];
      new_buckets[new_index] = bucket;
      bucket = next;
    }
  }

  map->buckets = new_buckets;
  map->bucket_count = new_bucket_count;
  return iree_ok_status();
}

// Looks up or creates a DFA state for the given NFA state set.
// Returns the existing state if found, or NULL if not found (and out_new=true).
static iree_tokenizer_regex_dfa_state_t*
iree_tokenizer_regex_dfa_state_map_find(
    iree_tokenizer_regex_dfa_state_map_t* map,
    const iree_tokenizer_regex_nfa_state_set_t* nfa_states) {
  uint64_t hash = iree_tokenizer_regex_nfa_state_set_hash(nfa_states);
  uint32_t bucket_idx = hash % map->bucket_count;

  iree_tokenizer_regex_dfa_state_bucket_t* bucket = map->buckets[bucket_idx];
  while (bucket) {
    if (iree_tokenizer_regex_nfa_state_set_equals(&bucket->state->nfa_states,
                                                  nfa_states)) {
      return bucket->state;
    }
    bucket = bucket->next;
  }
  return NULL;
}

static iree_status_t iree_tokenizer_regex_dfa_state_map_insert(
    iree_tokenizer_regex_dfa_state_map_t* map,
    iree_tokenizer_regex_dfa_state_t* state) {
  // Resize if load factor exceeds 0.75 (state_count * 4 > bucket_count * 3).
  if (map->state_count * 4 > map->bucket_count * 3) {
    IREE_RETURN_IF_ERROR(
        iree_tokenizer_regex_dfa_state_map_resize(map, map->bucket_count * 2));
  }

  uint64_t hash = iree_tokenizer_regex_nfa_state_set_hash(&state->nfa_states);
  uint32_t bucket_idx = hash % map->bucket_count;

  iree_tokenizer_regex_dfa_state_bucket_t* new_bucket;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      map->arena, sizeof(iree_tokenizer_regex_dfa_state_bucket_t),
      (void**)&new_bucket));
  new_bucket->state = state;
  new_bucket->next = map->buckets[bucket_idx];
  map->buckets[bucket_idx] = new_bucket;
  ++map->state_count;
  return iree_ok_status();
}

// Creates a new DFA state with the given NFA state set.
static iree_status_t iree_tokenizer_regex_dfa_state_create(
    iree_arena_allocator_t* arena,
    const iree_tokenizer_regex_nfa_state_set_t* nfa_states, uint32_t id,
    iree_tokenizer_regex_dfa_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_regex_dfa_state_t* state;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, sizeof(iree_tokenizer_regex_dfa_state_t),
                              (void**)&state));

  state->id = id;
  state->is_accepting = false;
  state->has_lookahead = false;
  state->lookahead_type = IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NONE;
  state->lookahead_data = 0;
  state->has_fallback = false;
  state->has_early_no_lookahead = false;
  state->requires_start_anchor = false;
  state->requires_end_anchor = false;
  state->alive_branches = 0;
  state->accepting_branches = 0;
  state->range_count = 0;

  // Initialize transitions to no-transition.
  for (int i = 0; i < 256; ++i) {
    state->transitions[i] = UINT32_MAX;
  }

  // Copy NFA state set.
  state->nfa_states.nfa_state_count = nfa_states->nfa_state_count;
  state->nfa_states.capacity = nfa_states->capacity;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, nfa_states->capacity * sizeof(uint64_t),
                              (void**)&state->nfa_states.bits));
  iree_tokenizer_regex_nfa_state_set_copy(&state->nfa_states, nfa_states);

  *out_state = state;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Branch Reachability
//===----------------------------------------------------------------------===//

// Computes which alternation branches are reachable from each NFA state.
// Uses fixed-point iteration: a state can reach any branch reachable from its
// successors. Accept states initialize with their own branch_index.
//
// This enables computing alive_branches for DFA states: a branch is alive in a
// DFA state if any NFA state in the set can reach an accept for that branch.
static iree_status_t iree_tokenizer_regex_compute_branch_reachability(
    const iree_tokenizer_regex_nfa_t* nfa, iree_arena_allocator_t* arena,
    uint64_t** out_reachable) {
  uint64_t* reachable;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, nfa->state_count * sizeof(uint64_t), (void**)&reachable));

  // Initialize: accept states can reach their own branch.
  for (uint32_t i = 0; i < nfa->state_count; ++i) {
    reachable[i] = 0;
    if (nfa->states[i]->type == IREE_TOKENIZER_UTIL_REGEX_NFA_ACCEPT) {
      reachable[i] = (1ULL << nfa->states[i]->data.accept.branch_index);
    }
  }

  // Fixed-point: propagate reachability forward through transitions.
  // A state can reach any branch reachable from its successors.
  // Mathematically bounded to O(N) iterations where N is state count, but
  // we add an explicit limit as defense against pathological patterns.
  bool changed = true;
  uint32_t iterations = 0;
  while (changed) {
    if (++iterations > IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_ITERATIONS) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "branch reachability exceeded iteration limit (%d); "
          "pattern may be pathological",
          IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_ITERATIONS);
    }
    changed = false;
    for (uint32_t i = 0; i < nfa->state_count; ++i) {
      uint64_t old = reachable[i];
      const iree_tokenizer_regex_nfa_state_t* state = nfa->states[i];

      switch (state->type) {
        case IREE_TOKENIZER_UTIL_REGEX_NFA_EPSILON:
          if (state->data.epsilon.out1) {
            reachable[i] |= reachable[state->data.epsilon.out1->id];
          }
          if (state->data.epsilon.out2) {
            reachable[i] |= reachable[state->data.epsilon.out2->id];
          }
          break;
        case IREE_TOKENIZER_UTIL_REGEX_NFA_MATCH_BYTE:
          if (state->data.match_byte.out) {
            reachable[i] |= reachable[state->data.match_byte.out->id];
          }
          break;
        case IREE_TOKENIZER_UTIL_REGEX_NFA_MATCH_CLASS:
          if (state->data.match_class.out) {
            reachable[i] |= reachable[state->data.match_class.out->id];
          }
          break;
        case IREE_TOKENIZER_UTIL_REGEX_NFA_ANCHOR_START:
        case IREE_TOKENIZER_UTIL_REGEX_NFA_ANCHOR_END:
          if (state->data.anchor_out) {
            reachable[i] |= reachable[state->data.anchor_out->id];
          }
          break;
        case IREE_TOKENIZER_UTIL_REGEX_NFA_ACCEPT:
          // Accept states don't have outgoing transitions.
          break;
      }

      if (reachable[i] != old) changed = true;
    }
  }

  *out_reachable = reachable;
  return iree_ok_status();
}

// Computes alive_branches and accepting_branches for a DFA state from its NFA
// state set.
static void iree_tokenizer_regex_compute_branch_masks(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_set_t* nfa_states,
    const uint64_t* nfa_reachable_branches,
    iree_tokenizer_regex_dfa_state_t* dfa_state) {
  dfa_state->alive_branches = 0;
  dfa_state->accepting_branches = 0;

  for (uint32_t i = 0; i < nfa->state_count; ++i) {
    if (!iree_tokenizer_regex_nfa_state_set_contains(nfa_states, i)) continue;

    // This NFA state contributes its reachable branches to alive_branches.
    dfa_state->alive_branches |= nfa_reachable_branches[i];

    // If it's an accept state, add its branch to accepting_branches.
    if (nfa->states[i]->type == IREE_TOKENIZER_UTIL_REGEX_NFA_ACCEPT) {
      dfa_state->accepting_branches |=
          (1ULL << nfa->states[i]->data.accept.branch_index);
    }
  }
}

//===----------------------------------------------------------------------===//
// Lookahead Extraction
//===----------------------------------------------------------------------===//

// Extracts lookahead info from accepting NFA states in the set.
//
// For alternation patterns like `\s+(?!\S)|\s+`, the DFA state may contain
// multiple NFA accept states - some with lookahead, some without. The correct
// semantic is:
//   - If ANY accepting NFA state has NO lookahead → accept unconditionally
//   - Only if ALL accepting states have lookahead → lookahead is required
//
// This is because a path through a non-lookahead branch succeeds regardless
// of what follows.
//
// Returns an error if multiple accepting states have DIFFERENT lookahead
// constraints. This detects patterns like `a(?!b)|a(?!c)` where the same
// input reaches accept states with conflicting constraints.
static iree_status_t iree_tokenizer_regex_extract_lookahead(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_set_t* nfa_states,
    iree_tokenizer_regex_dfa_state_t* dfa_state) {
  bool found_accept_without_lookahead = false;
  bool found_accept_with_lookahead = false;
  // Track minimum branch index for lookahead and non-lookahead accepts.
  // Used to compute has_early_no_lookahead per-DFA-state for PCRE semantics.
  uint8_t min_lookahead_branch = UINT8_MAX;
  uint8_t min_no_lookahead_branch = UINT8_MAX;
  iree_tokenizer_regex_lookahead_type_t first_lookahead_type =
      IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NONE;
  uint8_t first_lookahead_data = 0;

  for (uint32_t i = 0; i < nfa->state_count; ++i) {
    if (!iree_tokenizer_regex_nfa_state_set_contains(nfa_states, i)) continue;

    const iree_tokenizer_regex_nfa_state_t* state = nfa->states[i];
    if (state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_ACCEPT) {
      dfa_state->is_accepting = true;

      if (state->data.accept.has_lookahead) {
        // Track minimum branch index for lookahead accepts.
        if (state->data.accept.branch_index < min_lookahead_branch) {
          min_lookahead_branch = state->data.accept.branch_index;
        }
        if (!found_accept_with_lookahead) {
          // First lookahead found - record it.
          first_lookahead_type = state->data.accept.lookahead_type;
          first_lookahead_data = state->data.accept.lookahead_data;
        } else {
          // Subsequent lookahead - check for conflict.
          if (state->data.accept.lookahead_type != first_lookahead_type ||
              state->data.accept.lookahead_data != first_lookahead_data) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "alternation branches have conflicting lookahead constraints "
                "(e.g., a(?!b)|a(?!c) is not supported)");
          }
        }
        found_accept_with_lookahead = true;
      } else {
        // Track minimum branch index for non-lookahead accepts.
        if (state->data.accept.branch_index < min_no_lookahead_branch) {
          min_no_lookahead_branch = state->data.accept.branch_index;
        }
        found_accept_without_lookahead = true;
      }
    }
  }

  // Handle lookahead based on which NFA accept paths exist.
  if (found_accept_with_lookahead) {
    dfa_state->has_lookahead = true;
    dfa_state->lookahead_data = first_lookahead_data;

    if (found_accept_without_lookahead) {
      // Mixed alternation: both lookahead and non-lookahead branches merge.
      // Example: \s+(?!\S)|\s+ - first branch has lookahead, second doesn't.
      // The executor must track BOTH positions and prefer lookahead-passed.
      // Use the WITH_FALLBACK lookahead type to signal this.
      dfa_state->has_fallback = true;
      // Compute has_early_no_lookahead based on branch indices of accepts
      // actually present in this DFA state. This correctly handles cases like
      // \s*[\r\n]+|\s+(?!\S)|\s+ where branch 0 may or may not contribute.
      dfa_state->has_early_no_lookahead =
          min_no_lookahead_branch < min_lookahead_branch;
      switch (first_lookahead_type) {
        case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CHAR:
          dfa_state->lookahead_type =
              IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CHAR_WITH_FALLBACK;
          break;
        case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_SHORTHAND:
          dfa_state->lookahead_type =
              IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_SHORTHAND_WITH_FALLBACK;
          break;
        case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CLASS:
          dfa_state->lookahead_type =
              IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CLASS_WITH_FALLBACK;
          break;
        default:
          dfa_state->lookahead_type = first_lookahead_type;
          break;
      }
    } else {
      // Only lookahead branches - no fallback needed.
      dfa_state->lookahead_type = first_lookahead_type;
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Anchor Extraction
//===----------------------------------------------------------------------===//

// Extracts end anchor info from accepting NFA states.
//
// Similar logic to lookahead: if ALL accepting NFA states in the DFA state
// went through ANCHOR_END, then the DFA state requires end anchor.
// If ANY accepting state didn't go through end anchor, accept without check.
static void iree_tokenizer_regex_extract_end_anchor(
    const iree_tokenizer_regex_nfa_t* nfa,
    const iree_tokenizer_regex_nfa_state_set_t* nfa_states,
    const iree_tokenizer_regex_nfa_state_set_t* end_anchor_accepts,
    iree_tokenizer_regex_dfa_state_t* dfa_state) {
  if (!dfa_state->is_accepting) return;

  bool found_accept_without_end_anchor = false;
  bool found_accept_with_end_anchor = false;

  for (uint32_t i = 0; i < nfa->state_count; ++i) {
    if (!iree_tokenizer_regex_nfa_state_set_contains(nfa_states, i)) continue;

    const iree_tokenizer_regex_nfa_state_t* state = nfa->states[i];
    if (state->type == IREE_TOKENIZER_UTIL_REGEX_NFA_ACCEPT) {
      if (iree_tokenizer_regex_nfa_state_set_contains(end_anchor_accepts, i)) {
        found_accept_with_end_anchor = true;
      } else {
        found_accept_without_end_anchor = true;
      }
    }
  }

  // Require end anchor only if ALL accepting states went through ANCHOR_END.
  dfa_state->requires_end_anchor =
      found_accept_with_end_anchor && !found_accept_without_end_anchor;
}

//===----------------------------------------------------------------------===//
// Subset Construction
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_regex_dfa_build(
    const iree_tokenizer_regex_nfa_t* nfa, iree_arena_allocator_t* arena,
    iree_tokenizer_regex_dfa_build_t* out_dfa) {
  memset(out_dfa, 0, sizeof(*out_dfa));
  out_dfa->arena = arena;
  out_dfa->uses_unicode = nfa->uses_unicode;
  out_dfa->has_lookahead = nfa->has_lookahead;
  out_dfa->has_anchors = nfa->has_anchors;

  // Compute branch reachability for all NFA states.
  // This enables tracking which alternation branches are alive in each DFA
  // state, which is needed for PCRE-compatible priority-based matching.
  uint64_t* nfa_reachable_branches;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_compute_branch_reachability(
      nfa, arena, &nfa_reachable_branches));

  // Allocate initial state storage.
  out_dfa->state_capacity = 256;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena,
      out_dfa->state_capacity * sizeof(iree_tokenizer_regex_dfa_state_t*),
      (void**)&out_dfa->states));

  // Create hash table for state deduplication.
  iree_tokenizer_regex_dfa_state_map_t state_map;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_regex_dfa_state_map_init(arena, 1024, &state_map));

  // Working sets for subset construction.
  iree_tokenizer_regex_nfa_state_set_t current_set;
  iree_tokenizer_regex_nfa_state_set_t move_set;
  iree_tokenizer_regex_nfa_state_set_t closure_set;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_set_init(
      arena, nfa->state_count, &current_set));
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_set_init(
      arena, nfa->state_count, &move_set));
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_set_init(
      arena, nfa->state_count, &closure_set));

  // Anchor tracking using two-pass algorithm.
  // Pass 1: Full closure (all reachable states).
  // Pass 2: Unanchored closure (states reachable without ANCHOR_END).
  // Result: end_anchor_accepts = (accepts in full) - (accepts in unanchored).
  iree_tokenizer_regex_nfa_state_set_t end_anchor_accepts;
  iree_tokenizer_regex_nfa_state_set_t unanchored_reachable;
  iree_tokenizer_regex_anchor_info_t anchor_info = {0};
  if (nfa->has_anchors) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_set_init(
        arena, nfa->state_count, &end_anchor_accepts));
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_set_init(
        arena, nfa->state_count, &unanchored_reachable));
    anchor_info.end_anchor_accepts = &end_anchor_accepts;
    anchor_info.unanchored_reachable = &unanchored_reachable;
  }

  // Allocate worklist for iterative epsilon closure.
  // Size is 2 * state_count to handle worst-case fan-in (each state pushed
  // twice before duplicate check).
  iree_tokenizer_regex_epsilon_work_item_t* epsilon_worklist;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena,
      2 * nfa->state_count * sizeof(iree_tokenizer_regex_epsilon_work_item_t),
      (void**)&epsilon_worklist));

  // Simpler worklist for unanchored closure (just state IDs).
  uint32_t* unanchored_worklist = NULL;
  if (nfa->has_anchors) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, 2 * nfa->state_count * sizeof(uint32_t),
                            (void**)&unanchored_worklist));
  }

  // Compute epsilon closure of NFA start state.
  // at_start=true because we're at position 0, so ANCHOR_START is valid.
  iree_tokenizer_regex_nfa_state_set_clear(&current_set);
  iree_tokenizer_regex_epsilon_closure_single_with_anchors(
      nfa, nfa->start, /*at_start=*/true, epsilon_worklist, &current_set, NULL);

  // For anchors, use two-pass algorithm to compute end_anchor_accepts.
  // Pass 1: Full closure (done above).
  // Pass 2: Unanchored closure (ignoring ANCHOR_END transitions).
  // Result: end_anchor_accepts = (accepts in full) - (accepts in unanchored).
  if (nfa->has_anchors) {
    iree_tokenizer_regex_nfa_state_set_clear(&unanchored_reachable);
    iree_tokenizer_regex_epsilon_closure_single_ignoring_end_anchor(
        nfa, nfa->start, /*at_start=*/true, unanchored_worklist,
        &unanchored_reachable);
    iree_tokenizer_regex_compute_end_anchor_accepts(
        nfa, &current_set, &unanchored_reachable, &end_anchor_accepts);
  }

  // Create start DFA state.
  iree_tokenizer_regex_dfa_state_t* start_state;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_dfa_state_create(
      arena, &current_set, 0, &start_state));
  iree_tokenizer_regex_compute_branch_masks(
      nfa, &current_set, nfa_reachable_branches, start_state);
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_regex_extract_lookahead(nfa, &current_set, start_state));

  // Extract end anchor requirement for start state (if it's accepting).
  if (nfa->has_anchors) {
    iree_tokenizer_regex_extract_end_anchor(nfa, &current_set,
                                            &end_anchor_accepts, start_state);
  }

  out_dfa->states[0] = start_state;
  out_dfa->start = start_state;
  out_dfa->state_count = 1;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_regex_dfa_state_map_insert(&state_map, start_state));

  // Compute unanchored start state (for matching at position > 0).
  // This excludes paths through ANCHOR_START, so patterns like ^a|b
  // will only allow 'b' matches at non-zero positions.
  iree_tokenizer_regex_nfa_state_set_t unanchored_set;
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_nfa_state_set_init(
      arena, nfa->state_count, &unanchored_set));
  iree_tokenizer_regex_epsilon_closure_single_with_anchors(
      nfa, nfa->start, /*at_start=*/false, epsilon_worklist, &unanchored_set,
      NULL);

  // Determine if pattern requires start anchor. This is true when:
  // 1. NFA starts directly with ANCHOR_START, OR
  // 2. Unanchored closure has no consuming states (all paths go through ^)
  // For patterns like ^a|^b, the unanchored set contains only epsilon/anchor
  // states with no way to consume input, so it's effectively a dead-end.
  if (nfa->has_anchors) {
    if (nfa->start &&
        nfa->start->type == IREE_TOKENIZER_UTIL_REGEX_NFA_ANCHOR_START) {
      start_state->requires_start_anchor = true;
    } else if (!iree_tokenizer_regex_nfa_state_set_has_consuming(
                   nfa, &unanchored_set)) {
      // Unanchored start is a dead-end - pattern can only match at position 0.
      start_state->requires_start_anchor = true;
    }
  }

  // Check if unanchored closure is same as anchored closure.
  if (iree_tokenizer_regex_nfa_state_set_equals(&current_set,
                                                &unanchored_set)) {
    // Same set - use the same start state.
    out_dfa->unanchored_start = start_state;
  } else {
    // Different set - look up or create a separate state.
    iree_tokenizer_regex_dfa_state_t* unanchored_start =
        iree_tokenizer_regex_dfa_state_map_find(&state_map, &unanchored_set);
    if (!unanchored_start) {
      // Create new state for unanchored start.
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_dfa_state_create(
          arena, &unanchored_set, out_dfa->state_count, &unanchored_start));
      iree_tokenizer_regex_compute_branch_masks(
          nfa, &unanchored_set, nfa_reachable_branches, unanchored_start);
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_extract_lookahead(
          nfa, &unanchored_set, unanchored_start));
      if (nfa->has_anchors) {
        // Two-pass anchor tracking for unanchored start state.
        iree_tokenizer_regex_nfa_state_set_clear(&unanchored_reachable);
        iree_tokenizer_regex_epsilon_closure_set_ignoring_end_anchor(
            nfa, &unanchored_set, /*at_start=*/false, unanchored_worklist,
            &unanchored_reachable);
        iree_tokenizer_regex_compute_end_anchor_accepts(
            nfa, &unanchored_set, &unanchored_reachable, &end_anchor_accepts);
        iree_tokenizer_regex_extract_end_anchor(
            nfa, &unanchored_set, &end_anchor_accepts, unanchored_start);
      }
      out_dfa->states[out_dfa->state_count] = unanchored_start;
      out_dfa->state_count++;
      IREE_RETURN_IF_ERROR(iree_tokenizer_regex_dfa_state_map_insert(
          &state_map, unanchored_start));
    }
    out_dfa->unanchored_start = unanchored_start;
  }

  // Worklist: DFA states that need their transitions computed.
  // We use a simple array and process states in order.
  uint32_t worklist_head = 0;
  uint32_t iterations = 0;

  while (worklist_head < out_dfa->state_count) {
    // Safety limit to prevent hangs from pathological patterns.
    if (++iterations > IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_ITERATIONS) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "DFA construction exceeded iteration limit (%d); "
                              "pattern may cause exponential state growth",
                              IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_ITERATIONS);
    }

    iree_tokenizer_regex_dfa_state_t* current_dfa_state =
        out_dfa->states[worklist_head++];

    // Compute transitions for each input byte.
    // We iterate 0-255 (regular bytes) plus pseudo-bytes 0x80-0x8B.
    for (int byte = 0; byte < 256; ++byte) {
      // Skip continuation bytes (0x8C-0xBF) that aren't pseudo-bytes.
      // These can't appear as standalone input.
      if (byte >= 0x80 + IREE_TOKENIZER_UTIL_REGEX_PSEUDO_COUNT && byte <= 0xBF)
        continue;
      // Also skip invalid lead bytes (0xC0, 0xC1, 0xF5-0xFF).
      if (byte == 0xC0 || byte == 0xC1) continue;
      if (byte >= 0xF5) continue;

      // Compute move(current_nfa_states, byte).
      iree_tokenizer_regex_nfa_move(nfa, &current_dfa_state->nfa_states,
                                    (uint8_t)byte, &move_set);

      if (iree_tokenizer_regex_nfa_state_set_is_empty(&move_set)) {
        // No transition for this byte.
        continue;
      }

      // Compute epsilon closure of move set.
      // at_start=false because we've consumed input, so ANCHOR_START is
      // invalid. This makes patterns like `a^b` correctly produce dead-end
      // transitions.
      iree_tokenizer_regex_epsilon_closure_set_with_anchors(
          nfa, &move_set, /*at_start=*/false, epsilon_worklist, &closure_set,
          NULL);

      // Look up or create DFA state for this closure.
      iree_tokenizer_regex_dfa_state_t* target_state =
          iree_tokenizer_regex_dfa_state_map_find(&state_map, &closure_set);

      if (!target_state) {
        // Check state limit.
        if (out_dfa->state_count >= IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_STATES) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "DFA state limit exceeded (%d max)",
                                  IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_STATES);
        }

        // Expand state array if needed.
        if (out_dfa->state_count >= out_dfa->state_capacity) {
          // Overflow is impossible: MAX_STATES (65534) bounds state_count,
          // so capacity never exceeds 131072 (next power of 2), far below
          // UINT32_MAX/2. The initial capacity of 256 ensures non-zero.
          IREE_ASSERT(out_dfa->state_capacity > 0 &&
                      out_dfa->state_capacity <= UINT32_MAX / 2);
          uint32_t new_capacity = out_dfa->state_capacity * 2;
          iree_tokenizer_regex_dfa_state_t** new_states;
          IREE_RETURN_IF_ERROR(iree_arena_allocate(
              arena, new_capacity * sizeof(iree_tokenizer_regex_dfa_state_t*),
              (void**)&new_states));
          memcpy(
              new_states, out_dfa->states,
              out_dfa->state_count * sizeof(iree_tokenizer_regex_dfa_state_t*));
          out_dfa->states = new_states;
          out_dfa->state_capacity = new_capacity;
        }

        // Create new DFA state.
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_dfa_state_create(
            arena, &closure_set, out_dfa->state_count, &target_state));
        iree_tokenizer_regex_compute_branch_masks(
            nfa, &closure_set, nfa_reachable_branches, target_state);
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_extract_lookahead(
            nfa, &closure_set, target_state));

        // Extract end anchor requirement using two-pass algorithm.
        if (nfa->has_anchors) {
          iree_tokenizer_regex_nfa_state_set_clear(&unanchored_reachable);
          iree_tokenizer_regex_epsilon_closure_set_ignoring_end_anchor(
              nfa, &move_set, /*at_start=*/false, unanchored_worklist,
              &unanchored_reachable);
          iree_tokenizer_regex_compute_end_anchor_accepts(
              nfa, &closure_set, &unanchored_reachable, &end_anchor_accepts);
          iree_tokenizer_regex_extract_end_anchor(
              nfa, &closure_set, &end_anchor_accepts, target_state);
        }

        out_dfa->states[out_dfa->state_count] = target_state;
        out_dfa->state_count++;
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_dfa_state_map_insert(
            &state_map, target_state));
      }

      // Record transition.
      current_dfa_state->transitions[byte] = target_state->id;
    }

    // Collect exact codepoint range transitions from NFA states.
    // For each NFA match_class state that has ranges, we compute the target
    // DFA state and record the range transition.
    for (uint32_t i = 0; i < nfa->state_count; ++i) {
      if (!iree_tokenizer_regex_nfa_state_set_contains(
              &current_dfa_state->nfa_states, i))
        continue;

      const iree_tokenizer_regex_nfa_state_t* nfa_state = nfa->states[i];
      if (nfa_state->type != IREE_TOKENIZER_UTIL_REGEX_NFA_MATCH_CLASS)
        continue;
      if (nfa_state->data.match_class.range_count == 0) continue;

      // This NFA state has exact ranges. Compute the target DFA state.
      // The target is the epsilon closure of the match_class.out state.
      iree_tokenizer_regex_nfa_state_set_clear(&move_set);
      iree_tokenizer_regex_nfa_state_set_add(
          &move_set, nfa_state->data.match_class.out->id);

      iree_tokenizer_regex_epsilon_closure_set_with_anchors(
          nfa, &move_set, /*at_start=*/false, epsilon_worklist, &closure_set,
          NULL);

      // Look up or create target DFA state.
      iree_tokenizer_regex_dfa_state_t* range_target =
          iree_tokenizer_regex_dfa_state_map_find(&state_map, &closure_set);

      if (!range_target) {
        // Check state limit.
        if (out_dfa->state_count >= IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_STATES) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "DFA state limit exceeded (%d max)",
                                  IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_STATES);
        }

        // Expand state array if needed.
        if (out_dfa->state_count >= out_dfa->state_capacity) {
          // Overflow is impossible: MAX_STATES (65534) bounds state_count,
          // so capacity never exceeds 131072 (next power of 2), far below
          // UINT32_MAX/2. The initial capacity of 256 ensures non-zero.
          IREE_ASSERT(out_dfa->state_capacity > 0 &&
                      out_dfa->state_capacity <= UINT32_MAX / 2);
          uint32_t new_capacity = out_dfa->state_capacity * 2;
          iree_tokenizer_regex_dfa_state_t** new_states;
          IREE_RETURN_IF_ERROR(iree_arena_allocate(
              arena, new_capacity * sizeof(iree_tokenizer_regex_dfa_state_t*),
              (void**)&new_states));
          memcpy(
              new_states, out_dfa->states,
              out_dfa->state_count * sizeof(iree_tokenizer_regex_dfa_state_t*));
          out_dfa->states = new_states;
          out_dfa->state_capacity = new_capacity;
        }

        // Create new DFA state.
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_dfa_state_create(
            arena, &closure_set, out_dfa->state_count, &range_target));
        iree_tokenizer_regex_compute_branch_masks(
            nfa, &closure_set, nfa_reachable_branches, range_target);
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_extract_lookahead(
            nfa, &closure_set, range_target));

        // Extract end anchor requirement using two-pass algorithm.
        if (nfa->has_anchors) {
          iree_tokenizer_regex_nfa_state_set_clear(&unanchored_reachable);
          iree_tokenizer_regex_epsilon_closure_set_ignoring_end_anchor(
              nfa, &move_set, /*at_start=*/false, unanchored_worklist,
              &unanchored_reachable);
          iree_tokenizer_regex_compute_end_anchor_accepts(
              nfa, &closure_set, &unanchored_reachable, &end_anchor_accepts);
          iree_tokenizer_regex_extract_end_anchor(
              nfa, &closure_set, &end_anchor_accepts, range_target);
        }

        out_dfa->states[out_dfa->state_count] = range_target;
        out_dfa->state_count++;
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_dfa_state_map_insert(
            &state_map, range_target));
      }

      // Add all ranges from this NFA state to the DFA state.
      for (uint8_t r = 0; r < nfa_state->data.match_class.range_count; ++r) {
        const iree_tokenizer_regex_codepoint_range_t* nfa_range =
            &nfa_state->data.match_class.ranges[r];

        // Check if we already have this range (deduplication).
        bool found = false;
        for (uint8_t existing = 0; existing < current_dfa_state->range_count;
             ++existing) {
          if (current_dfa_state->range_transitions[existing].start ==
                  nfa_range->start &&
              current_dfa_state->range_transitions[existing].end ==
                  nfa_range->end &&
              current_dfa_state->range_transitions[existing].target_id ==
                  range_target->id) {
            found = true;
            break;
          }
        }

        if (!found) {
          if (current_dfa_state->range_count >=
              IREE_TOKENIZER_UTIL_REGEX_MAX_CHAR_CLASS_RANGES) {
            return iree_make_status(
                IREE_STATUS_RESOURCE_EXHAUSTED,
                "DFA state exceeds %d range transitions",
                IREE_TOKENIZER_UTIL_REGEX_MAX_CHAR_CLASS_RANGES);
          }
          uint8_t idx = current_dfa_state->range_count++;
          current_dfa_state->range_transitions[idx].start = nfa_range->start;
          current_dfa_state->range_transitions[idx].end = nfa_range->end;
          current_dfa_state->range_transitions[idx].target_id =
              range_target->id;
        }
      }
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Serialization
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_regex_dfa_serialize(
    const iree_tokenizer_regex_dfa_build_t* dfa, iree_allocator_t allocator,
    uint8_t** out_data, iree_host_size_t* out_size) {
  // Validate state count fits in uint16_t.
  if (dfa->state_count > IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_STATES) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED, "DFA has too many states: %u > %u",
        dfa->state_count, IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_STATES);
  }

  // Count accepting states, ranges, and build header flags.
  uint16_t num_states = (uint16_t)dfa->state_count;
  uint16_t num_accepting = 0;
  uint16_t num_ranges = 0;
  bool has_branch_tracking = false;
  iree_tokenizer_regex_dfa_flags_t flags =
      IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_NONE;
  if (dfa->uses_unicode) {
    flags |= IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_UNICODE;
  }
  for (uint32_t i = 0; i < dfa->state_count; ++i) {
    if (dfa->states[i]->is_accepting) {
      num_accepting++;
      if (dfa->states[i]->has_lookahead) {
        flags |= IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD;
      }
    }
    if (dfa->states[i]->requires_start_anchor ||
        dfa->states[i]->requires_end_anchor) {
      flags |= IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS;
    }
    // Check if branch tracking is needed (more than one branch alive).
    if (iree_math_count_ones_u64(dfa->states[i]->alive_branches) > 1) {
      has_branch_tracking = true;
    }
    // Count range transitions.
    num_ranges += dfa->states[i]->range_count;
  }
  if (has_branch_tracking) {
    flags |= IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_BRANCHES;
  }
  if (num_ranges > 0) {
    flags |= IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_RANGES;
  }

  // Calculate layout with overflow checking.
  // Conditional counts: 0 disables the section.
  iree_host_size_t bitmap_qwords = iree_host_align(num_states, 64) / 64;
  iree_host_size_t lookahead_count =
      (flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD) ? num_states
                                                                 : 0;
  iree_host_size_t anchor_bitmap_count =
      (flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS) ? 2 : 0;
  iree_host_size_t branch_array_count =
      (flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_BRANCHES) ? 2 : 0;
  iree_host_size_t range_header_count =
      (flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_RANGES) ? 1 : 0;
  iree_host_size_t range_array_count =
      (flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_RANGES) ? num_ranges : 0;

  iree_host_size_t total_size = 0;
  iree_host_size_t transitions_offset = 0;
  iree_host_size_t accepting_bitmap_offset = 0;
  iree_host_size_t lookahead_offset = 0;
  iree_host_size_t anchor_bitmaps_offset = 0;
  iree_host_size_t branches_offset = 0;
  iree_host_size_t range_header_offset = 0;
  iree_host_size_t range_transitions_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_tokenizer_regex_dfa_header_t), &total_size,
      // Transitions: num_states rows of 256 uint16_t values.
      IREE_STRUCT_ARRAY_FIELD(num_states, 256, uint16_t, &transitions_offset),
      // Accepting bitmap: ceil(num_states/64) qwords.
      IREE_STRUCT_FIELD(bitmap_qwords, uint64_t, &accepting_bitmap_offset),
      // Lookahead (optional): one entry per state.
      IREE_STRUCT_FIELD(lookahead_count, iree_tokenizer_regex_lookahead_t,
                        &lookahead_offset),
      // Anchor bitmaps (optional): 2 bitmaps of bitmap_qwords each.
      IREE_STRUCT_ARRAY_FIELD(anchor_bitmap_count, bitmap_qwords, uint64_t,
                              &anchor_bitmaps_offset),
      // Branches (optional): 2 arrays of num_states each.
      IREE_STRUCT_ARRAY_FIELD(branch_array_count, num_states, uint64_t,
                              &branches_offset),
      // Range header (optional): num_ranges + reserved.
      IREE_STRUCT_FIELD(range_header_count, uint32_t, &range_header_offset),
      // Range transitions (optional).
      IREE_STRUCT_FIELD(range_array_count,
                        iree_tokenizer_regex_dfa_range_transition_t,
                        &range_transitions_offset)));

  // Allocate output buffer.
  uint8_t* data;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&data));
  memset(data, 0, total_size);

  // Write header.
  iree_tokenizer_regex_dfa_header_t* header =
      (iree_tokenizer_regex_dfa_header_t*)data;
  header->magic = IREE_TOKENIZER_UTIL_REGEX_DFA_MAGIC;
  header->version = IREE_TOKENIZER_UTIL_REGEX_DFA_VERSION;
  header->flags = flags;
  header->num_states = num_states;
  header->num_accepting = num_accepting;
  header->start_state = 0;  // Start is always state 0.
  header->unanchored_start_state = (uint16_t)dfa->unanchored_start->id;

  // Write transition table.
  uint16_t* transitions = (uint16_t*)(data + transitions_offset);
  for (uint32_t state_idx = 0; state_idx < dfa->state_count; ++state_idx) {
    const iree_tokenizer_regex_dfa_state_t* state = dfa->states[state_idx];
    uint16_t* state_trans = &transitions[state_idx * 256];
    for (int byte = 0; byte < 256; ++byte) {
      if (state->transitions[byte] == UINT32_MAX) {
        state_trans[byte] = IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION;
      } else {
        state_trans[byte] = (uint16_t)state->transitions[byte];
      }
    }
  }

  // Write accepting bitmap.
  uint64_t* accepting_bitmap = (uint64_t*)(data + accepting_bitmap_offset);
  for (uint32_t state_idx = 0; state_idx < dfa->state_count; ++state_idx) {
    if (dfa->states[state_idx]->is_accepting) {
      accepting_bitmap[state_idx / 64] |= (1ULL << (state_idx % 64));
    }
  }

  // Write lookahead table (if any).
  if (flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD) {
    iree_tokenizer_regex_lookahead_t* lookahead =
        (iree_tokenizer_regex_lookahead_t*)(data + lookahead_offset);
    for (uint32_t state_idx = 0; state_idx < dfa->state_count; ++state_idx) {
      const iree_tokenizer_regex_dfa_state_t* state = dfa->states[state_idx];
      if (state->has_lookahead) {
        lookahead[state_idx].type = (uint8_t)state->lookahead_type;
        lookahead[state_idx].data = state->lookahead_data;
        // Serialize PCRE-compatibility flag for match selection.
        lookahead[state_idx].flags =
            state->has_early_no_lookahead
                ? IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_FLAG_HAS_EARLY_NO_LOOKAHEAD
                : IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_FLAG_NONE;
        lookahead[state_idx].reserved = 0;
      } else {
        lookahead[state_idx].type = IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NONE;
        lookahead[state_idx].data = 0;
        lookahead[state_idx].flags =
            IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_FLAG_NONE;
        lookahead[state_idx].reserved = 0;
      }
    }
  }

  // Write anchor bitmaps (if any).
  if (flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS) {
    // Start anchor bitmap: states that require match to start at position 0.
    uint64_t* start_anchor_bitmap = (uint64_t*)(data + anchor_bitmaps_offset);
    for (uint32_t state_idx = 0; state_idx < dfa->state_count; ++state_idx) {
      if (dfa->states[state_idx]->requires_start_anchor) {
        start_anchor_bitmap[state_idx / 64] |= (1ULL << (state_idx % 64));
      }
    }

    // End anchor bitmap: accepting states that require match at end of input.
    // Follows immediately after start anchor bitmap.
    uint64_t* end_anchor_bitmap = (uint64_t*)(data + anchor_bitmaps_offset +
                                              bitmap_qwords * sizeof(uint64_t));
    for (uint32_t state_idx = 0; state_idx < dfa->state_count; ++state_idx) {
      if (dfa->states[state_idx]->requires_end_anchor) {
        end_anchor_bitmap[state_idx / 64] |= (1ULL << (state_idx % 64));
      }
    }
  }

  // Write branch tracking arrays (if any).
  if (flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_BRANCHES) {
    // Write alive_branches array (one uint64_t per state).
    uint64_t* alive_branches = (uint64_t*)(data + branches_offset);
    for (uint32_t state_idx = 0; state_idx < dfa->state_count; ++state_idx) {
      alive_branches[state_idx] = dfa->states[state_idx]->alive_branches;
    }

    // Write accepting_branches array (one uint64_t per state).
    // Follows immediately after alive_branches.
    uint64_t* accepting_branches =
        (uint64_t*)(data + branches_offset +
                    (iree_host_size_t)num_states * sizeof(uint64_t));
    for (uint32_t state_idx = 0; state_idx < dfa->state_count; ++state_idx) {
      accepting_branches[state_idx] =
          dfa->states[state_idx]->accepting_branches;
    }
  }

  // Write range transitions (if any).
  if (flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_RANGES) {
    // Write range header: num_ranges (u16) + reserved (u16).
    uint16_t* range_header = (uint16_t*)(data + range_header_offset);
    range_header[0] = num_ranges;
    range_header[1] = 0;  // Reserved.

    // Write range transitions as flat array.
    iree_tokenizer_regex_dfa_range_transition_t* range_transitions =
        (iree_tokenizer_regex_dfa_range_transition_t*)(data +
                                                       range_transitions_offset);
    uint16_t range_idx = 0;
    for (uint32_t state_idx = 0; state_idx < dfa->state_count; ++state_idx) {
      const iree_tokenizer_regex_dfa_state_t* state = dfa->states[state_idx];
      for (uint8_t r = 0; r < state->range_count; ++r) {
        range_transitions[range_idx].from_state = (uint16_t)state_idx;
        range_transitions[range_idx].target_state =
            (uint16_t)state->range_transitions[r].target_id;
        range_transitions[range_idx].start = state->range_transitions[r].start;
        range_transitions[range_idx].end = state->range_transitions[r].end;
        ++range_idx;
      }
    }
  }

  *out_data = data;
  *out_size = total_size;
  return iree_ok_status();
}
