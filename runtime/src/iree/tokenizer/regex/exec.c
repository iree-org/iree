// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/regex/exec.h"

#include "iree/base/internal/math.h"
#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Unicode Category to Pseudo-Byte Mapping
//===----------------------------------------------------------------------===//

// Maps a Unicode codepoint to either its ASCII byte value or a pseudo-byte
// representing its Unicode category. This allows the DFA to match Unicode
// category patterns (\p{L}, \p{N}, etc.) while operating on byte-level
// transitions.
//
// For ASCII (0x00-0x7F): returns the byte value directly.
// For non-ASCII: returns the pseudo-byte for the codepoint's category.
//
// Note on \s vs \p{Z}:
//   - PSEUDO_WHITESPACE (0x87) is for \s matching (Unicode White_Space
//   property)
//   - PSEUDO_SEPARATOR (0x85) is for \p{Z} matching (Separator category)
//   - White_Space includes some control characters not in Z category
//   - We check whitespace first to ensure \s matches non-ASCII whitespace
//
// Note on exact codepoint ranges:
//   This function maps codepoints to CATEGORY pseudo-bytes only.
//   Exact codepoint range matching (e.g., [一-龥]) is handled separately
//   through the DFA's range checking mechanism, not through pseudo-bytes.
static IREE_ATTRIBUTE_ALWAYS_INLINE inline uint8_t
iree_tokenizer_regex_codepoint_to_byte(uint32_t codepoint) {
  // ASCII bytes pass through directly.
  if (codepoint < 0x80) return (uint8_t)codepoint;

  // Handle Unicode replacement character (invalid UTF-8) consistently.
  // Map to OTHER to avoid false matches on category patterns.
  if (codepoint == IREE_UNICODE_REPLACEMENT_CHAR) {
    return IREE_TOKENIZER_UTIL_REGEX_PSEUDO_OTHER;
  }

  // Check whitespace BEFORE category - White_Space property is separate from
  // the Z (Separator) category. This ensures \s matches non-ASCII whitespace
  // like NBSP (U+00A0), ideographic space (U+3000), etc.
  if (iree_unicode_is_whitespace(codepoint)) {
    return IREE_TOKENIZER_UTIL_REGEX_PSEUDO_WHITESPACE;
  }

  // Map Unicode category to pseudo-byte.
  iree_unicode_category_t category = iree_unicode_category(codepoint);
  if (category & IREE_UNICODE_CATEGORY_LETTER) {
    return IREE_TOKENIZER_UTIL_REGEX_PSEUDO_LETTER;
  }
  if (category & IREE_UNICODE_CATEGORY_NUMBER) {
    return IREE_TOKENIZER_UTIL_REGEX_PSEUDO_NUMBER;
  }
  if (category & IREE_UNICODE_CATEGORY_PUNCTUATION) {
    return IREE_TOKENIZER_UTIL_REGEX_PSEUDO_PUNCT;
  }
  if (category & IREE_UNICODE_CATEGORY_MARK) {
    return IREE_TOKENIZER_UTIL_REGEX_PSEUDO_MARK;
  }
  if (category & IREE_UNICODE_CATEGORY_SYMBOL) {
    return IREE_TOKENIZER_UTIL_REGEX_PSEUDO_SYMBOL;
  }
  if (category & IREE_UNICODE_CATEGORY_SEPARATOR) {
    return IREE_TOKENIZER_UTIL_REGEX_PSEUDO_SEPARATOR;
  }
  // Default to OTHER for unclassified codepoints.
  return IREE_TOKENIZER_UTIL_REGEX_PSEUDO_OTHER;
}

//===----------------------------------------------------------------------===//
// Lookahead Checking
//===----------------------------------------------------------------------===//

// Checks if a codepoint matches the specified shorthand class.
static inline bool iree_tokenizer_regex_shorthand_matches(
    iree_tokenizer_regex_shorthand_t shorthand, uint32_t codepoint) {
  switch (shorthand) {
    case IREE_TOKENIZER_UTIL_REGEX_SHORTHAND_s:
      return iree_unicode_is_whitespace(codepoint);
    case IREE_TOKENIZER_UTIL_REGEX_SHORTHAND_S:
      return !iree_unicode_is_whitespace(codepoint);
    case IREE_TOKENIZER_UTIL_REGEX_SHORTHAND_d:
      // \d matches ASCII digits only (common regex behavior).
      return codepoint >= '0' && codepoint <= '9';
    case IREE_TOKENIZER_UTIL_REGEX_SHORTHAND_D:
      return !(codepoint >= '0' && codepoint <= '9');
    case IREE_TOKENIZER_UTIL_REGEX_SHORTHAND_w:
      // \w matches [a-zA-Z0-9_].
      return (codepoint >= 'a' && codepoint <= 'z') ||
             (codepoint >= 'A' && codepoint <= 'Z') ||
             (codepoint >= '0' && codepoint <= '9') || codepoint == '_';
    case IREE_TOKENIZER_UTIL_REGEX_SHORTHAND_W:
      return !((codepoint >= 'a' && codepoint <= 'z') ||
               (codepoint >= 'A' && codepoint <= 'Z') ||
               (codepoint >= '0' && codepoint <= '9') || codepoint == '_');
    default:
      return false;
  }
}

// Checks if the lookahead condition at an accepting state should reject the
// match. Returns true if the match should be rejected (negative lookahead
// matched), false if the match should be accepted.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline bool
iree_tokenizer_regex_lookahead_rejects(const iree_tokenizer_regex_dfa_t* dfa,
                                       uint16_t state,
                                       uint32_t next_codepoint) {
  if (!dfa->lookahead) return false;

  const iree_tokenizer_regex_lookahead_t* la = &dfa->lookahead[state];
  switch (la->type) {
    case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NONE:
      return false;

    case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CHAR:
    case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CHAR_WITH_FALLBACK:
      // Negative char lookahead: reject if next char equals specified char.
      // Convert codepoint to DFA byte for comparison (handles pseudo-bytes).
      return iree_tokenizer_regex_codepoint_to_byte(next_codepoint) == la->data;

    case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_SHORTHAND:
    case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_SHORTHAND_WITH_FALLBACK:
      // Negative shorthand lookahead: reject if next char matches shorthand.
      return iree_tokenizer_regex_shorthand_matches(
          (iree_tokenizer_regex_shorthand_t)la->data, next_codepoint);

    case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CLASS:
    case IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CLASS_WITH_FALLBACK:
      // Class lookahead requires bitmap lookup - not implemented.
      // Validation should reject DFAs with this lookahead type.
      // If we get here, validation was bypassed - fail safe by not rejecting.
      return false;

    default:
      return false;
  }
}

// Returns true if the lookahead type has a fallback path (for mixed
// lookahead/non-lookahead alternation like \s+(?!\S)|\s+).
static inline bool iree_tokenizer_regex_lookahead_has_fallback(
    const iree_tokenizer_regex_dfa_t* dfa, uint16_t state) {
  if (!dfa->lookahead) return false;
  uint8_t type = dfa->lookahead[state].type;
  return type == IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CHAR_WITH_FALLBACK ||
         type ==
             IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_SHORTHAND_WITH_FALLBACK ||
         type == IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CLASS_WITH_FALLBACK;
}

//===----------------------------------------------------------------------===//
// DFA Loading
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_regex_dfa_load(
    iree_const_byte_span_t data, iree_tokenizer_regex_dfa_t* out_dfa) {
  if (!out_dfa) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_dfa is NULL");
  }

  memset(out_dfa, 0, sizeof(*out_dfa));

  // Validate minimum size for header.
  if (data.data_length < sizeof(iree_tokenizer_regex_dfa_header_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "DFA data too small: %" PRIhsz " bytes < header %zu", data.data_length,
        sizeof(iree_tokenizer_regex_dfa_header_t));
  }

  // Validate alignment for zero-copy access.
  // The accepting_bitmap uses uint64_t requiring 8-byte alignment.
  // Transition table uses uint16_t requiring 2-byte alignment.
  // The stricter requirement (8 bytes) applies to the entire buffer.
  if (((uintptr_t)data.data % 8) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "DFA data must be 8-byte aligned (got alignment %zu)",
        (size_t)((uintptr_t)data.data % 8));
  }

  const iree_tokenizer_regex_dfa_header_t* header =
      (const iree_tokenizer_regex_dfa_header_t*)data.data;

  // Validate magic number.
  if (header->magic != IREE_TOKENIZER_UTIL_REGEX_DFA_MAGIC) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "invalid DFA magic: 0x%08X (expected 0x%08X)",
                            header->magic, IREE_TOKENIZER_UTIL_REGEX_DFA_MAGIC);
  }

  // Validate version.
  if (header->version < 1 ||
      header->version > IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_VERSION) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported DFA version: %u (supported: 1-%u)",
                            header->version,
                            IREE_TOKENIZER_UTIL_REGEX_DFA_MAX_VERSION);
  }

  // Validate state count.
  if (header->num_states == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DFA has zero states");
  }

  // Validate state count doesn't collide with NO_TRANSITION sentinel.
  // max valid state is num_states-1, which must be < 0xFFFF.
  if (header->num_states >= IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "num_states %u collides with NO_TRANSITION sentinel "
        "(max supported: %u)",
        header->num_states, IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION - 1);
  }

  // Validate start state.
  if (header->start_state >= header->num_states) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "start state %u out of range [0, %u)",
                            header->start_state, header->num_states);
  }

  // Validate unanchored start state.
  if (header->unanchored_start_state >= header->num_states) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unanchored start state %u out of range [0, %u)",
                            header->unanchored_start_state, header->num_states);
  }

  // Calculate expected data size (base size, ranges checked separately).
  iree_host_size_t expected_size = iree_tokenizer_regex_dfa_binary_size(
      header->num_states, header->flags, 0);

  if (data.data_length < expected_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DFA data truncated: %" PRIhsz
                            " bytes < expected %" PRIhsz,
                            data.data_length, expected_size);
  }

  // Set up pointers into the binary data (zero-copy).
  // Defense-in-depth: validate bounds at each pointer advance in case the
  // expected_size calculation has a bug or header values are corrupt.
  const uint8_t* ptr = data.data + sizeof(iree_tokenizer_regex_dfa_header_t);
  const uint8_t* data_end = data.data + data.data_length;

// Macro for bounds checking at each pointer advance.
#define CHECK_DFA_BOUNDS(size, section_name)                            \
  do {                                                                  \
    if ((iree_host_size_t)(data_end - ptr) < (size)) {                  \
      return iree_make_status(                                          \
          IREE_STATUS_INVALID_ARGUMENT,                                 \
          "DFA truncated at %s: need %" PRIhsz " bytes, have %" PRIhsz, \
          (section_name), (size), (iree_host_size_t)(data_end - ptr));  \
    }                                                                   \
  } while (0)

  out_dfa->header = header;

  // Transitions: num_states * 256 * sizeof(uint16_t)
  iree_host_size_t transitions_size =
      (iree_host_size_t)header->num_states * 256 * sizeof(uint16_t);
  CHECK_DFA_BOUNDS(transitions_size, "transitions");
  out_dfa->transitions = (const uint16_t*)ptr;
  ptr += transitions_size;

  iree_host_size_t bitmap_qwords = iree_host_align(header->num_states, 64) / 64;
  iree_host_size_t bitmap_size = bitmap_qwords * sizeof(uint64_t);

  // Accepting bitmap.
  CHECK_DFA_BOUNDS(bitmap_size, "accepting_bitmap");
  out_dfa->accepting_bitmap = (const uint64_t*)ptr;
  ptr += bitmap_size;

  if (header->flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_LOOKAHEAD) {
    iree_host_size_t lookahead_size = (iree_host_size_t)header->num_states *
                                      sizeof(iree_tokenizer_regex_lookahead_t);
    CHECK_DFA_BOUNDS(lookahead_size, "lookahead");
    out_dfa->lookahead = (const iree_tokenizer_regex_lookahead_t*)ptr;
    ptr += lookahead_size;
  } else {
    out_dfa->lookahead = NULL;
  }

  if (header->flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_ANCHORS) {
    CHECK_DFA_BOUNDS(bitmap_size, "start_anchor_bitmap");
    out_dfa->start_anchor_bitmap = (const uint64_t*)ptr;
    ptr += bitmap_size;
    CHECK_DFA_BOUNDS(bitmap_size, "end_anchor_bitmap");
    out_dfa->end_anchor_bitmap = (const uint64_t*)ptr;
    ptr += bitmap_size;
  } else {
    out_dfa->start_anchor_bitmap = NULL;
    out_dfa->end_anchor_bitmap = NULL;
  }

  if (header->flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_BRANCHES) {
    iree_host_size_t branches_size =
        (iree_host_size_t)header->num_states * sizeof(uint64_t);
    CHECK_DFA_BOUNDS(branches_size, "alive_branches");
    out_dfa->alive_branches = (const uint64_t*)ptr;
    ptr += branches_size;
    CHECK_DFA_BOUNDS(branches_size, "accepting_branches");
    out_dfa->accepting_branches = (const uint64_t*)ptr;
    ptr += branches_size;
  } else {
    out_dfa->alive_branches = NULL;
    out_dfa->accepting_branches = NULL;
  }

#undef CHECK_DFA_BOUNDS

  if (header->flags & IREE_TOKENIZER_UTIL_REGEX_DFA_FLAG_HAS_RANGES) {
    // Read range count and validate extended size.
    if (data.data_length <
        (iree_host_size_t)(ptr - data.data) + sizeof(uint32_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "DFA data truncated: missing range header");
    }
    out_dfa->num_ranges = *(const uint16_t*)ptr;
    ptr += sizeof(uint32_t);  // num_ranges (u16) + reserved (u16)

    // Verify we have enough space for all ranges.
    iree_host_size_t ranges_size =
        out_dfa->num_ranges *
        sizeof(iree_tokenizer_regex_dfa_range_transition_t);
    if (data.data_length < (iree_host_size_t)(ptr - data.data) + ranges_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "DFA data truncated: expected %u ranges",
                              out_dfa->num_ranges);
    }
    out_dfa->ranges = (const iree_tokenizer_regex_dfa_range_transition_t*)ptr;
  } else {
    out_dfa->ranges = NULL;
    out_dfa->num_ranges = 0;
  }

  // Validate the loaded DFA to catch structural issues early.
  return iree_tokenizer_regex_dfa_validate(out_dfa);
}

// Forward declaration for use in validation.
static inline bool iree_tokenizer_regex_requires_end_anchor(
    const iree_tokenizer_regex_dfa_t* dfa, uint16_t state);

iree_status_t iree_tokenizer_regex_dfa_validate(
    const iree_tokenizer_regex_dfa_t* dfa) {
  if (!dfa || !dfa->header) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "DFA is NULL");
  }

  const uint16_t num_states = dfa->header->num_states;

  // Count accepting states and verify against header.
  uint16_t accepting_count = 0;
  for (uint16_t state = 0; state < num_states; ++state) {
    if (iree_tokenizer_regex_dfa_is_accepting(dfa, state)) {
      ++accepting_count;
    }
  }

  if (accepting_count != dfa->header->num_accepting) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "accepting state count mismatch: bitmap has %u, header says %u",
        accepting_count, dfa->header->num_accepting);
  }

  // Forbid immediate empty-string matches.
  // If the start state is accepting without an end anchor requirement,
  // the pattern could match empty strings at every position, causing
  // infinite loops (e.g., a*, (a|)).
  //
  // Patterns like ^.*$ are safe because even though the start state is
  // accepting (via .* matching empty), the end anchor defers acceptance
  // until end of input - no intermediate empty matches occur. The pattern
  // must also have transitions from start, otherwise it can only match empty
  // (e.g., ^$ has accepting start with end anchor but no transitions).
  if (iree_tokenizer_regex_dfa_is_accepting(dfa, dfa->header->start_state)) {
    bool has_end_anchor =
        iree_tokenizer_regex_requires_end_anchor(dfa, dfa->header->start_state);
    if (!has_end_anchor) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "start state %u is accepting without end anchor - "
          "empty-string matches are not supported",
          dfa->header->start_state);
    }
    // Even with end anchor, if start has no transitions, pattern only matches
    // empty strings (e.g., ^$).
    bool has_transition = false;
    for (int byte = 0; byte < 256 && !has_transition; ++byte) {
      if (iree_tokenizer_regex_dfa_step(dfa, dfa->header->start_state,
                                        (uint8_t)byte, (uint8_t)byte) !=
          IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION) {
        has_transition = true;
      }
    }
    if (!has_transition) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "start state %u is accepting with no transitions - "
          "pattern can only match empty strings",
          dfa->header->start_state);
    }
  }

  // Validate all transitions reference valid states.
  for (uint16_t state = 0; state < num_states; ++state) {
    for (int byte = 0; byte < 256; ++byte) {
      uint16_t next = iree_tokenizer_regex_dfa_step(dfa, state, (uint8_t)byte,
                                                    (uint8_t)byte);
      if (next != IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION &&
          next >= num_states) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "transition from state %u on byte 0x%02X "
                                "references invalid state %u",
                                state, byte, next);
      }
    }
  }

  // Validate lookahead data if present.
  if (dfa->lookahead) {
    for (uint16_t state = 0; state < num_states; ++state) {
      uint8_t type = dfa->lookahead[state].type;
      if (type > IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CLASS_WITH_FALLBACK) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid lookahead type %u at state %u", type,
                                state);
      }
      // LOOKAHEAD_NEG_CLASS requires bitmap support not yet implemented.
      if (type == IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CLASS ||
          type == IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NEG_CLASS_WITH_FALLBACK) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "LOOKAHEAD_NEG_CLASS at state %u requires "
                                "bitmap support (not yet implemented)",
                                state);
      }
    }
  }

  // Validate range transitions if present.
  if (dfa->ranges && dfa->num_ranges > 0) {
    for (uint16_t i = 0; i < dfa->num_ranges; ++i) {
      const iree_tokenizer_regex_dfa_range_transition_t* range =
          &dfa->ranges[i];
      if (range->from_state >= num_states) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "range %u: from_state %u >= num_states %u", i,
                                range->from_state, num_states);
      }
      if (range->target_state >= num_states) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "range %u: target_state %u >= num_states %u", i,
                                range->target_state, num_states);
      }
      if (range->start > range->end) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "range %u: start U+%04X > end U+%04X", i,
                                range->start, range->end);
      }
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Stride (DFA execution acceleration)
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_regex_stride_allocate(
    const iree_tokenizer_regex_dfa_t* dfa, iree_allocator_t allocator,
    iree_tokenizer_regex_stride_t** out_stride) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_stride = NULL;

  uint16_t num_states = dfa->header->num_states;
  iree_host_size_t boring_bitmap_size = (num_states + 7) / 8;

  // Compute slab layout.
  iree_host_size_t total_size = 0;
  iree_host_size_t boring_bitmap_offset = 0;
  iree_host_size_t self_loop_category_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          sizeof(iree_tokenizer_regex_stride_t), &total_size,
          IREE_STRUCT_FIELD(boring_bitmap_size, uint8_t, &boring_bitmap_offset),
          IREE_STRUCT_FIELD(num_states, uint8_t, &self_loop_category_offset)));

  iree_tokenizer_regex_stride_t* stride = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&stride));
  memset(stride, 0, total_size);

  stride->num_states = num_states;
  stride->boring_bitmap_offset = boring_bitmap_offset;
  stride->self_loop_category_offset = self_loop_category_offset;

  // Step 1: Compute byte equivalence classes for ASCII bytes.
  // Two bytes are equivalent if they produce the same transition in EVERY
  // state. We use fingerprinting: compute a hash of each byte's transition
  // column, then assign the same category to bytes with identical columns.
  //
  // First pass: for each byte, compute its transition column fingerprint.
  // We store columns temporarily for comparison (128 bytes * num_states * 2 =
  // manageable since num_states is typically 20-50).
  uint8_t category_assignment[128];
  memset(category_assignment, 0xFF, sizeof(category_assignment));
  uint8_t next_category = 0;

  for (uint8_t byte_a = 0; byte_a < 128; ++byte_a) {
    if (category_assignment[byte_a] != 0xFF) continue;  // Already assigned.
    category_assignment[byte_a] = next_category;

    // Compare byte_a's column against all unassigned bytes.
    for (uint8_t byte_b = byte_a + 1; byte_b < 128; ++byte_b) {
      if (category_assignment[byte_b] != 0xFF) continue;

      // Check if byte_a and byte_b have identical transitions in all states.
      bool equivalent = true;
      for (uint16_t s = 0; s < num_states; ++s) {
        if (dfa->transitions[(iree_host_size_t)s * 256 + byte_a] !=
            dfa->transitions[(iree_host_size_t)s * 256 + byte_b]) {
          equivalent = false;
          break;
        }
      }
      if (equivalent) {
        category_assignment[byte_b] = next_category;
      }
    }
    ++next_category;
  }

  memcpy(stride->category_table, category_assignment,
         sizeof(stride->category_table));
  stride->category_count = next_category;

  // Step 2: Compute boring bitmap.
  // A state is boring when it needs no bookkeeping: non-accepting, no
  // lookahead, no anchors, and all branches alive.
  uint8_t* boring_bitmap =
      (uint8_t*)((uintptr_t)stride + stride->boring_bitmap_offset);
  for (uint16_t s = 0; s < num_states; ++s) {
    bool boring = true;
    boring &= !iree_tokenizer_regex_dfa_is_accepting(dfa, s);
    if (dfa->lookahead) {
      boring &=
          (dfa->lookahead[s].type == IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NONE);
    }
    if (dfa->start_anchor_bitmap) {
      boring &= (dfa->start_anchor_bitmap[s / 64] & (1ULL << (s % 64))) == 0;
    }
    if (dfa->end_anchor_bitmap) {
      boring &= (dfa->end_anchor_bitmap[s / 64] & (1ULL << (s % 64))) == 0;
    }
    if (dfa->alive_branches) {
      // A state is boring for branch purposes only if all branches are still
      // alive (no branch has died, so no commit check can trigger).
      boring &= (dfa->alive_branches[s / 64] == ~0ULL);
    }
    if (boring) {
      boring_bitmap[s / 8] |= (uint8_t)(1 << (s % 8));
    }
  }

  // Step 3: Compute self-loop categories.
  // For each boring state, check if it has a self-loop on exactly one category.
  // self_loop_category[S] = C means ALL bytes in category C transition S→S.
  // 0xFF means no usable self-loop.
  uint8_t* self_loop_category =
      (uint8_t*)((uintptr_t)stride + stride->self_loop_category_offset);
  for (uint16_t s = 0; s < num_states; ++s) {
    if (!iree_tokenizer_regex_stride_is_boring(stride, s)) {
      self_loop_category[s] = 0xFF;
      continue;
    }

    // Find which category (if any) this state self-loops on.
    int16_t loop_category = -1;
    bool valid = true;
    for (uint8_t byte_idx = 0; byte_idx < 128; ++byte_idx) {
      if (dfa->transitions[(iree_host_size_t)s * 256 + byte_idx] == s) {
        uint8_t category = stride->category_table[byte_idx];
        if (loop_category == -1) {
          loop_category = category;
        } else if (category != (uint8_t)loop_category) {
          // Self-loops on multiple categories — can't use Level 2.
          valid = false;
          break;
        }
      }
    }

    if (!valid || loop_category < 0) {
      self_loop_category[s] = 0xFF;
      continue;
    }

    // Verify ALL bytes in this category self-loop (not just some).
    bool all_loop = true;
    for (uint8_t byte_idx = 0; byte_idx < 128; ++byte_idx) {
      if (stride->category_table[byte_idx] == (uint8_t)loop_category) {
        if (dfa->transitions[(iree_host_size_t)s * 256 + byte_idx] != s) {
          all_loop = false;
          break;
        }
      }
    }

    self_loop_category[s] = all_loop ? (uint8_t)loop_category : 0xFF;
  }

  *out_stride = stride;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_tokenizer_regex_stride_free(iree_tokenizer_regex_stride_t* stride,
                                      iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(allocator, stride);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Streaming Executor
//===----------------------------------------------------------------------===//

void iree_tokenizer_regex_exec_initialize(
    iree_tokenizer_regex_exec_state_t* state,
    const iree_tokenizer_regex_dfa_t* dfa) {
  IREE_TRACE_ZONE_BEGIN(z0);
  state->dfa_state = dfa->header->start_state;
  state->match_start = 0;
  state->last_accept = 0;
  state->in_match = false;
  state->has_accept = false;
  // No pending lookahead at start.
  state->pending_lookahead = false;
  state->pending_lookahead_state = 0;
  state->pending_match_end = 0;
  // No fallback accept at start.
  state->has_accept_fallback = false;
  state->last_accept_fallback = 0;
  // No best candidate at start (64 = no candidate).
  state->best_branch_idx = 64;
  state->best_match_end = 0;
  // Rewind buffer starts empty.
  state->rewind_buffer_length = 0;
  state->rewind_buffer_start = 0;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Lookahead Helpers (for emit_match)
//===----------------------------------------------------------------------===//

// Helper to check if state has the "early no-lookahead" PCRE-compatibility
// flag. When true and both lookahead-passed and fallback positions exist,
// prefer longer. When false, prefer lookahead-passed position.
static inline bool iree_tokenizer_regex_has_early_no_lookahead(
    const iree_tokenizer_regex_dfa_t* dfa, uint16_t state) {
  if (!dfa->lookahead) return false;
  return iree_any_bit_set(
      dfa->lookahead[state].flags,
      IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_FLAG_HAS_EARLY_NO_LOOKAHEAD);
}

// Internal helper to emit a match if we have a valid accepting position.
//
// When deciding between lookahead-passed and fallback positions:
// - If has_early_no_lookahead is true, prefer LONGER match (independent branch
//   extends greedily). Example: \s*[\r\n]+|\s+(?!\S)|\s+ with input "\n\n"
// - If has_early_no_lookahead is false, prefer LOOKAHEAD-PASSED position
//   (fallback branch is companion to lookahead). Example: \s+(?!\S)|\s+ with "
//   x"
//
// See nfa.h accept struct comment for full documentation on limitations.
// Emits a match if we have a valid accepting position, returning the match end
// position via out_match_end for callers that need to resume from that point.
//
// The out_match_end parameter is set to the emitted match's end position when
// a match is emitted, or left unchanged if no match is emitted. This explicit
// output replaces the previous pattern of mutating last_accept as a side
// effect, making the data flow clearer for resume logic.
static inline iree_status_t iree_tokenizer_regex_emit_match(
    iree_tokenizer_regex_exec_state_t* state,
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_match_callback_fn_t callback, void* user_data,
    iree_host_size_t* out_match_end) {
  if (!callback) return iree_ok_status();

  iree_host_size_t match_end = 0;
  bool has_match = false;

  if (state->has_accept && state->has_accept_fallback) {
    // Both lookahead-passed and fallback positions exist.
    // Decision depends on whether an early non-lookahead branch exists.
    bool prefer_longer =
        iree_tokenizer_regex_has_early_no_lookahead(dfa, state->dfa_state) &&
        state->last_accept_fallback > state->last_accept;

    if (prefer_longer) {
      // Early non-lookahead branch extends further - use it.
      // Example: \s*[\r\n]+ in "\s*[\r\n]+|\s+(?!\S)|\s+" with input "\n\n".
      match_end = state->last_accept_fallback;
    } else {
      // All non-lookahead branches are companions (come after lookahead).
      // Prefer lookahead-passed position (PCRE backtracking semantics).
      // Example: \s+ in "\s+(?!\S)|\s+" with input "   x" prefers "  " not " ".
      match_end = state->last_accept;
    }
    has_match = true;
  } else if (state->has_accept) {
    // Only lookahead-passed position.
    match_end = state->last_accept;
    has_match = true;
  } else if (state->has_accept_fallback) {
    // Only fallback position.
    match_end = state->last_accept_fallback;
    has_match = true;
  }

  if (has_match) {
    iree_tokenizer_regex_match_t match = {
        .start = state->match_start,
        .end = match_end,
    };
    if (out_match_end) *out_match_end = match_end;
    IREE_RETURN_IF_ERROR(callback(user_data, match));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lookahead Helpers
//===----------------------------------------------------------------------===//

// Helper to check if state has lookahead that needs next character.
static inline bool iree_tokenizer_regex_has_lookahead(
    const iree_tokenizer_regex_dfa_t* dfa, uint16_t state) {
  if (!dfa->lookahead) return false;
  return dfa->lookahead[state].type != IREE_TOKENIZER_UTIL_REGEX_LOOKAHEAD_NONE;
}

//===----------------------------------------------------------------------===//
// Anchor Checking
//===----------------------------------------------------------------------===//

// Returns true if the given state requires start anchor (^ pattern).
// A state with start anchor can only be reached if matching starts at position
// 0.
static inline bool iree_tokenizer_regex_requires_start_anchor(
    const iree_tokenizer_regex_dfa_t* dfa, uint16_t state) {
  if (!dfa->start_anchor_bitmap) return false;
  return (dfa->start_anchor_bitmap[state / 64] & (1ULL << (state % 64))) != 0;
}

// Returns true if the given accepting state requires end anchor ($ pattern).
// A state with end anchor only produces a valid match at end of input.
static inline bool iree_tokenizer_regex_requires_end_anchor(
    const iree_tokenizer_regex_dfa_t* dfa, uint16_t state) {
  if (!dfa->end_anchor_bitmap) return false;
  return (dfa->end_anchor_bitmap[state / 64] & (1ULL << (state % 64))) != 0;
}

//===----------------------------------------------------------------------===//
// Branch-Aware Execution Helpers (PCRE-compatible alternation)
//===----------------------------------------------------------------------===//

// Check if DFA has branch tracking data for PCRE-compatible alternation.
static inline bool iree_tokenizer_regex_has_branches(
    const iree_tokenizer_regex_dfa_t* dfa) {
  return dfa->alive_branches != NULL && dfa->accepting_branches != NULL;
}

// Update best candidate if a higher-priority branch accepts at this state.
// Returns true if we should try to commit (all higher-priority branches dead).
//
// |lookahead_confirmed| indicates whether any lookahead at this state has been
// evaluated and passed. When false (lookahead pending or failed), we skip
// recording best_match_end but still check if we can commit a previously
// recorded candidate.
static inline bool iree_tokenizer_regex_update_best_candidate(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, uint16_t dfa_state,
    iree_host_size_t match_end, bool lookahead_confirmed) {
  if (!iree_tokenizer_regex_has_branches(dfa)) return false;

  uint64_t accepting = dfa->accepting_branches[dfa_state];
  if (!accepting) return false;

  // Find highest-priority accepting branch (lowest bit set).
  int highest_accepting = iree_math_count_trailing_zeros_u64(accepting);

  // Only record the match position if lookahead has been confirmed.
  // This prevents committing matches where lookahead was deferred (pending)
  // or failed. Without this check, patterns like `a(?!b)|ab` could incorrectly
  // match "a" when lookahead is deferred at chunk boundary and later fails.
  if (lookahead_confirmed) {
    if (highest_accepting < state->best_branch_idx) {
      // Higher priority branch accepts - update best candidate.
      state->best_branch_idx = (uint8_t)highest_accepting;
      state->best_match_end = match_end;
    } else if (highest_accepting == state->best_branch_idx) {
      // Same branch accepting again - extend greedily.
      state->best_match_end = match_end;
    }
  }

  // Check if we can commit: the best branch is no longer accepting AND
  // no higher-priority branches can take over.
  // Higher priority = lower branch index.
  if (state->best_branch_idx < 64) {
    uint64_t alive = dfa->alive_branches[dfa_state];
    uint64_t higher_priority_mask = (1ULL << state->best_branch_idx) - 1;

    // Commit when:
    // 1. No higher-priority branches are alive (they can't win anymore)
    // 2. The best branch is no longer ACCEPTING at this state (it accepted
    //    earlier and the accepting path has died, even if other paths remain)
    uint64_t best_branch_mask = 1ULL << state->best_branch_idx;
    bool best_branch_accepting = (accepting & best_branch_mask) != 0;

    if ((alive & higher_priority_mask) == 0 && !best_branch_accepting) {
      return true;
    }
  }
  return false;
}

// Reset best candidate tracking after emitting a match.
static inline void iree_tokenizer_regex_reset_best_candidate(
    iree_tokenizer_regex_exec_state_t* state) {
  state->best_branch_idx = 64;  // No candidate.
  state->best_match_end = 0;
}

//===----------------------------------------------------------------------===//
// State Management Helpers
//===----------------------------------------------------------------------===//

// Defers lookahead evaluation to the next chunk or finalize.
//
// This is used when we reach an accepting state at the end of a chunk.
// The next codepoint needed for lookahead evaluation is in the next chunk.
//
// The pending state is resolved in one of two ways:
//   - Next exec_feed() call: Uses the first codepoint of the new chunk.
//   - exec_finalize() call: Negative lookahead passes at EOS (nothing follows).
//
// Parameters:
//   state: Executor state to update.
//   accepting_state: The DFA state that has the lookahead condition.
//   match_end: The byte position where the match would end if accepted.
static inline void iree_tokenizer_regex_defer_lookahead(
    iree_tokenizer_regex_exec_state_t* state, uint16_t accepting_state,
    iree_host_size_t match_end) {
  state->pending_lookahead = true;
  state->pending_lookahead_state = accepting_state;
  state->pending_match_end = match_end;
}

// Resets executor state to begin searching for a new match.
//
// This is called after:
//   1. Successfully emitting a match (resume_pos = end of emitted match), or
//   2. Failed partial match with backtracking (resume_pos = match_start + 1).
//
// The executor returns to the DFA start state and clears match tracking.
// The next character processed will potentially begin a new match.
//
// Parameters:
//   state: Executor state to reset.
//   dfa: DFA for accessing start state.
//   resume_position: Absolute byte position to resume matching from.
//                    This becomes match_start for loop control logic.
static inline void iree_tokenizer_regex_reset_to_start(
    iree_tokenizer_regex_exec_state_t* state,
    const iree_tokenizer_regex_dfa_t* dfa, iree_host_size_t resume_position) {
  // Use unanchored start state if resuming at position > 0.
  // This prevents anchored patterns like ^a from matching mid-string.
  state->dfa_state = (resume_position == 0)
                         ? dfa->header->start_state
                         : dfa->header->unanchored_start_state;
  state->match_start = resume_position;
  state->has_accept = false;
  state->has_accept_fallback = false;
  state->in_match = false;
  // Reset branch tracking for new match attempt.
  iree_tokenizer_regex_reset_best_candidate(state);
}

// Evaluates an accepting state, handling lookahead, end anchor, and fallback.
// Called when the DFA enters an accepting state. Updates state->has_accept,
// state->last_accept, state->has_accept_fallback, state->last_accept_fallback,
// and potentially defers lookahead evaluation.
static inline void iree_tokenizer_regex_evaluate_accepting_state(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, uint16_t accepting_state,
    iree_host_size_t accept_pos, iree_string_view_t peek_view,
    iree_host_size_t peek_offset) {
  if (iree_tokenizer_regex_has_lookahead(dfa, accepting_state)) {
    // Record fallback for patterns like \s+(?!\S)|\s+.
    bool has_fallback =
        iree_tokenizer_regex_lookahead_has_fallback(dfa, accepting_state);
    if (has_fallback) {
      state->last_accept_fallback = accept_pos;
      state->has_accept_fallback = true;
    }

    if (peek_offset < peek_view.size) {
      // Next character is available - evaluate lookahead now.
      iree_host_size_t peek_position = peek_offset;
      uint32_t next_codepoint =
          iree_unicode_utf8_decode(peek_view, &peek_position);
      bool rejected = iree_tokenizer_regex_lookahead_rejects(
          dfa, accepting_state, next_codepoint);
      if (!rejected) {
        state->last_accept = accept_pos;
        state->has_accept = true;
        state->pending_lookahead = false;
      }
    } else {
      // At boundary - defer lookahead to next chunk or finalize.
      iree_tokenizer_regex_defer_lookahead(state, accepting_state, accept_pos);
    }
  } else if (iree_tokenizer_regex_requires_end_anchor(dfa, accepting_state)) {
    // End anchor state: defer acceptance until finalize confirms end of input.
  } else {
    // No lookahead or end anchor - accept unconditionally.
    state->has_accept = true;
    state->last_accept = accept_pos;
  }
}

// Forward declaration for process_rewind_buffer (called by
// backtrack_via_buffer and resume_via_buffer).
static iree_status_t iree_tokenizer_regex_process_rewind_buffer(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, iree_host_size_t next_start_abs);

// Resumes matching from a position in a previous chunk after emitting a match.
// After a successful match, we restart matching from the match end position.
// If that position is in a previous chunk (before base_offset), we use the
// rewind buffer to replay those bytes. Returns IREE_STATUS_OUT_OF_RANGE if
// |resume_position| is not in the rewind buffer.
static iree_status_t iree_tokenizer_regex_resume_via_buffer(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state,
    iree_host_size_t resume_position) {
  iree_host_size_t buffer_end_abs =
      state->rewind_buffer_start + state->rewind_buffer_length;

  if (resume_position < state->rewind_buffer_start ||
      resume_position >= buffer_end_abs) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "streaming regex resume failed: "
                            "resume_position=%" PRIhsz
                            " not in rewind buffer "
                            "[%" PRIhsz ", %" PRIhsz ")",
                            resume_position, state->rewind_buffer_start,
                            buffer_end_abs);
  }

  return iree_tokenizer_regex_process_rewind_buffer(dfa, state,
                                                    resume_position);
}

// Attempts cross-chunk backtracking using the rewind buffer.
//
// When a partial match fails and match_start is in a previous chunk, we need
// to restart matching from match_start + 1. This function checks if the rewind
// buffer contains the necessary bytes and processes them if so.
//
// Parameters:
//   dfa: The DFA being executed.
//   state: Executor state (match_start must be set before calling).
//   base_offset: Absolute byte offset of the current chunk's first byte.
//   context: Human-readable context for error message (e.g., "main loop").
//
// Returns:
//   - IREE_STATUS_OK if backtracking succeeded (state is updated for
//     continuing with current chunk from position 0).
//   - IREE_STATUS_OUT_OF_RANGE if match_start is not in the rewind buffer.
static iree_status_t iree_tokenizer_regex_backtrack_via_buffer(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, iree_host_size_t base_offset,
    const char* context) {
  iree_host_size_t buffer_end_abs =
      state->rewind_buffer_start + state->rewind_buffer_length;

  // Check if match_start is within the rewind buffer.
  if (state->match_start < state->rewind_buffer_start ||
      state->match_start >= buffer_end_abs) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "streaming regex backtrack failed (%s): "
                            "match_start=%" PRIhsz
                            " not in rewind buffer "
                            "[%" PRIhsz ", %" PRIhsz ")",
                            context, state->match_start,
                            state->rewind_buffer_start, buffer_end_abs);
  }

  // Calculate next start position (match_start + 1 codepoint).
  iree_host_size_t local_match_start =
      state->match_start - state->rewind_buffer_start;
  iree_host_size_t next_start =
      local_match_start + iree_unicode_utf8_sequence_length(
                              state->rewind_buffer[local_match_start]);
  iree_host_size_t next_start_abs = state->rewind_buffer_start + next_start;

  // Process buffer bytes to set up state for continuing with current chunk.
  return iree_tokenizer_regex_process_rewind_buffer(dfa, state, next_start_abs);
}

// Processes bytes from the rewind buffer for cross-chunk backtracking.
//
// When a partial match fails and match_start is in a previous chunk, we need
// to restart matching from match_start + 1. If those bytes are in the rewind
// buffer, this function processes them through the DFA to set up the correct
// state for continuing with the current chunk.
//
// This is called when:
//   1. A partial match failed (NO_TRANSITION encountered)
//   2. match_start is in a previous chunk (match_start < base_offset)
//   3. match_start is within the rewind buffer range
//
// The function steps through the buffered bytes, updating the DFA state and
// match tracking. After returning, the main loop continues processing the
// current chunk from position 0 with the updated state.
//
// Parameters:
//   dfa: The DFA being executed.
//   state: Executor state to update.
//   next_start_abs: Absolute position to start from (match_start + 1).
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - Error status if a callback fails.
static iree_status_t iree_tokenizer_regex_process_rewind_buffer(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, iree_host_size_t next_start_abs) {
  // Find the starting position within the buffer.
  iree_host_size_t buffer_offset = next_start_abs - state->rewind_buffer_start;
  iree_host_size_t buffer_end = state->rewind_buffer_length;

  // Reset state for new match attempt.
  state->dfa_state = (next_start_abs == 0)
                         ? dfa->header->start_state
                         : dfa->header->unanchored_start_state;
  state->match_start = next_start_abs;
  state->in_match = false;
  state->has_accept = false;
  state->has_accept_fallback = false;
  iree_tokenizer_regex_reset_best_candidate(state);

  // Process buffer bytes through the DFA.
  // This is a simplified version of the main loop - we don't emit matches here
  // because matches that complete within the buffer would have been found in
  // a previous chunk. We're just setting up state to continue into the current
  // chunk.

  // Create string view for entire rewind buffer - reused across iterations.
  const iree_string_view_t buffer_view = iree_make_string_view(
      (const char*)state->rewind_buffer, state->rewind_buffer_length);

  while (buffer_offset < buffer_end) {
    iree_host_size_t abs_pos = state->rewind_buffer_start + buffer_offset;

    // Decode UTF-8 codepoint and map to pseudo-byte for DFA transition.
    // Complete UTF-8 is guaranteed by the caller, so rewind buffer only
    // contains complete codepoints.
    uint32_t codepoint = iree_unicode_utf8_decode(buffer_view, &buffer_offset);
    uint8_t byte = iree_tokenizer_regex_codepoint_to_byte(codepoint);

    // Step the DFA (check exact codepoint ranges first, then byte lookup).
    uint16_t next_state =
        iree_tokenizer_regex_dfa_step(dfa, state->dfa_state, codepoint, byte);

    if (next_state == IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION) {
      // Failed transition - if we were in a match, we need to backtrack.
      if (state->in_match) {
        // Restart from match_start + 1 codepoint within buffer.
        iree_host_size_t local_match_start =
            state->match_start - state->rewind_buffer_start;
        iree_host_size_t next_start =
            local_match_start + iree_unicode_utf8_sequence_length(
                                    state->rewind_buffer[local_match_start]);

        if (next_start < buffer_end) {
          // Continue backtracking within buffer.
          // Clear all match state including fallback and branch tracking.
          buffer_offset = next_start;
          iree_host_size_t restart_abs =
              state->rewind_buffer_start + next_start;
          state->dfa_state = (restart_abs == 0)
                                 ? dfa->header->start_state
                                 : dfa->header->unanchored_start_state;
          state->match_start = restart_abs;
          state->in_match = false;
          state->has_accept = false;
          state->has_accept_fallback = false;
          iree_tokenizer_regex_reset_best_candidate(state);
          continue;
        }
        // Past end of buffer - main loop will handle the rest.
        // Clear all match state before falling through to retry.
        state->in_match = false;
        state->has_accept = false;
        state->has_accept_fallback = false;
        iree_tokenizer_regex_reset_best_candidate(state);
      }
      // Try current byte from start state.
      // Note: buffer_offset is already advanced past this character by decode.
      state->dfa_state = (abs_pos == 0) ? dfa->header->start_state
                                        : dfa->header->unanchored_start_state;
      state->match_start = abs_pos;
      continue;
    }

    // Valid transition.
    if (!state->in_match) {
      state->match_start = abs_pos;
      state->in_match = true;
      state->has_accept = false;
    }
    state->dfa_state = next_state;

    // Track accepting states (but don't emit - that happens in main loop).
    // buffer_offset is already advanced past this character by decode.
    if (iree_tokenizer_regex_dfa_is_accepting(dfa, next_state)) {
      iree_host_size_t accept_pos = state->rewind_buffer_start + buffer_offset;
      iree_tokenizer_regex_evaluate_accepting_state(
          dfa, state, next_state, accept_pos, buffer_view, buffer_offset);
    }
  }

  return iree_ok_status();
}

// Updates the rewind buffer at the end of chunk processing.
//
// When streaming across chunk boundaries, proper leftmost-longest matching
// requires the ability to backtrack to match_start + 1 if a partial match
// fails. If match_start is in a previous chunk, those bytes are normally lost.
// The rewind buffer preserves bytes from match_start onward so backtracking
// can succeed in the next chunk.
//
// This function is called at the end of exec_feed() and handles three cases:
//
// 1. No active match (in_match = false):
//    Clear the buffer. No backtracking will be needed.
//
// 2. Match started in current chunk (match_start >= base_offset):
//    Copy bytes from match_start to end of chunk into the buffer.
//    These bytes will be available for backtracking in the next chunk.
//
// 3. Match started in previous chunk (match_start < base_offset):
//    Append current chunk bytes to the existing buffer contents.
//    The buffer should already contain bytes from match_start onward
//    (copied by the previous chunk's call to this function).
//
// If the buffer is too small to hold all needed bytes, we keep the most recent
// IREE_TOKENIZER_UTIL_REGEX_REWIND_BUFFER_SIZE bytes. Backtracking will fail
// with OUT_OF_RANGE if match_start falls outside the buffer range.
//
// Parameters:
//   state: Executor state with rewind buffer to update.
//   chunk: The chunk that was just processed.
//   base_offset: Absolute byte offset of the chunk's first byte.
static void iree_tokenizer_regex_update_rewind_buffer(
    iree_tokenizer_regex_exec_state_t* state, iree_string_view_t chunk,
    iree_host_size_t base_offset) {
  iree_host_size_t chunk_end = base_offset + chunk.size;

  if (!state->in_match) {
    // No active match - clear the buffer.
    state->rewind_buffer_length = 0;
    state->rewind_buffer_start = chunk_end;
    return;
  }

  // We're in a partial match - preserve bytes from match_start onward.
  if (state->match_start >= base_offset) {
    // Match started in this chunk - copy from match_start to end.
    iree_host_size_t local_start = state->match_start - base_offset;
    iree_host_size_t bytes_to_copy = chunk.size - local_start;
    if (bytes_to_copy <= sizeof(state->rewind_buffer)) {
      memcpy(state->rewind_buffer, chunk.data + local_start, bytes_to_copy);
      state->rewind_buffer_length = bytes_to_copy;
      state->rewind_buffer_start = state->match_start;
    } else {
      // Buffer too small - keep the most recent bytes.
      iree_host_size_t skip = bytes_to_copy - sizeof(state->rewind_buffer);
      memcpy(state->rewind_buffer, chunk.data + local_start + skip,
             sizeof(state->rewind_buffer));
      state->rewind_buffer_length = sizeof(state->rewind_buffer);
      state->rewind_buffer_start = state->match_start + skip;
    }
    return;
  }

  // Match started in a previous chunk - append this chunk's bytes.
  iree_host_size_t buffer_end =
      state->rewind_buffer_start + state->rewind_buffer_length;
  if (buffer_end != base_offset) {
    // Gap between buffer and chunk - shouldn't happen in normal streaming.
    // Leave buffer as-is; backtracking will fail with an appropriate error.
    return;
  }

  // Append current chunk to buffer.
  // Guard against state corruption: buffer_length must not exceed buffer size.
  IREE_ASSERT(state->rewind_buffer_length <= sizeof(state->rewind_buffer));
  iree_host_size_t space_left =
      sizeof(state->rewind_buffer) - state->rewind_buffer_length;
  if (chunk.size <= space_left) {
    // Entire chunk fits - simple append.
    memcpy(state->rewind_buffer + state->rewind_buffer_length, chunk.data,
           chunk.size);
    state->rewind_buffer_length += chunk.size;
    IREE_ASSERT(state->rewind_buffer_length <= sizeof(state->rewind_buffer));
    return;
  }

  // Chunk doesn't fit - shift buffer to make room.
  // Invariant: chunk.size > space_left (otherwise we'd have returned).
  IREE_ASSERT(chunk.size > space_left);
  iree_host_size_t shift_needed = chunk.size - space_left;
  if (shift_needed < state->rewind_buffer_length) {
    // Shift out oldest bytes to make room for new chunk.
    memmove(state->rewind_buffer, state->rewind_buffer + shift_needed,
            state->rewind_buffer_length - shift_needed);
    state->rewind_buffer_length -= shift_needed;
    state->rewind_buffer_start += shift_needed;
    memcpy(state->rewind_buffer + state->rewind_buffer_length, chunk.data,
           chunk.size);
    state->rewind_buffer_length += chunk.size;
    IREE_ASSERT(state->rewind_buffer_length <= sizeof(state->rewind_buffer));
  } else {
    // Entire buffer replaced by this chunk's most recent bytes.
    iree_host_size_t copy_start = chunk.size - sizeof(state->rewind_buffer);
    memcpy(state->rewind_buffer, chunk.data + copy_start,
           sizeof(state->rewind_buffer));
    state->rewind_buffer_length = sizeof(state->rewind_buffer);
    state->rewind_buffer_start = base_offset + copy_start;
    IREE_ASSERT(state->rewind_buffer_length <= sizeof(state->rewind_buffer));
  }
}

// Updates pending lookahead for an accepting state.
//
// With quantifiers like \s+(?!\S), we may visit an accepting state multiple
// times. The lookahead should only be evaluated at the FINAL position where
// the match cannot extend further. This function just records that a lookahead
// needs to be checked; actual evaluation happens at NO_TRANSITION boundary.
//
// Algorithm:
//   Always defer lookahead by recording the pending state. The lookahead will
//   be evaluated when we hit NO_TRANSITION or end of input.
//
// Evaluates pending lookahead at a boundary.
//
// Called when we hit NO_TRANSITION and need to resolve any pending lookahead.
// At this point, we know the match cannot extend further, so we can safely
// evaluate the lookahead condition.
//
// Returns:
//   true if lookahead passed (or no lookahead pending), false if it failed.
static inline bool iree_tokenizer_regex_resolve_pending_lookahead(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, iree_string_view_t chunk_view,
    iree_host_size_t peek_offset) {
  if (!state->pending_lookahead) {
    return true;  // No pending lookahead - always passes.
  }

  // Check if we're at end of chunk.
  if (peek_offset >= chunk_view.size) {
    // At end of chunk with pending lookahead - keep pending for next chunk.
    // The lookahead will be resolved when more data arrives or at finalize.
    return true;  // Don't resolve yet.
  }

  // Decode the next character for lookahead evaluation.
  // Complete UTF-8 is guaranteed by the caller, so we can decode directly.
  iree_host_size_t decode_position = peek_offset;
  uint32_t next_codepoint =
      iree_unicode_utf8_decode(chunk_view, &decode_position);

  // Evaluate the lookahead condition.
  bool rejected = iree_tokenizer_regex_lookahead_rejects(
      dfa, state->pending_lookahead_state, next_codepoint);

  if (!rejected) {
    // Lookahead passed - record acceptance.
    state->last_accept = state->pending_match_end;
    state->has_accept = true;
  }

  state->pending_lookahead = false;
  return !rejected;
}

iree_status_t iree_tokenizer_regex_exec_feed(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, iree_string_view_t chunk,
    iree_host_size_t base_offset, const iree_tokenizer_regex_stride_t* stride,
    iree_tokenizer_regex_match_callback_fn_t callback, void* user_data) {
  // Check for start anchor constraint: if the pattern has a start anchor (^),
  // we can only start matching at position 0. If we're in a later chunk
  // (base_offset > 0) and haven't started a match yet, no match is possible.
  const bool requires_start =
      iree_tokenizer_regex_requires_start_anchor(dfa, dfa->header->start_state);
  if (requires_start && base_offset > 0 && !state->in_match) {
    // Start anchor required but we're past position 0 with no match - skip.
    return iree_ok_status();
  }

  iree_host_size_t position = 0;

  // Resolve pending lookahead from previous chunk if we have new data.
  if (state->pending_lookahead && chunk.size > 0) {
    iree_tokenizer_regex_resolve_pending_lookahead(dfa, state, chunk, 0);
  }

  // Main processing loop.
  while (position < chunk.size) {
    iree_host_size_t char_start = position;
    iree_host_size_t abs_pos = base_offset + char_start;

    // Decode UTF-8 codepoint and map to pseudo-byte for DFA transition.
    // ASCII fast path: For bytes 0x00-0x7F, codepoint == byte == pseudo-byte.
    // This avoids function call overhead and the 8-branch category classifier.
    uint32_t codepoint = 0;
    uint8_t byte = 0;
    uint8_t first_byte = (uint8_t)chunk.data[position];
    if (IREE_LIKELY(first_byte < 0x80)) {
      // ASCII: trivial decode, no category classification needed.
      codepoint = first_byte;
      byte = first_byte;
      ++position;

      // Stride acceleration: when in a match and the transition lands in a
      // boring state (non-accepting, no lookahead, no anchors, all branches
      // alive), skip all per-byte bookkeeping and run a tight inner loop.
      if (stride && state->in_match) {
        // Direct byte-table lookup (ranges never match ASCII codepoints).
        uint16_t stride_next =
            dfa->transitions[(iree_host_size_t)state->dfa_state * 256 + byte];
        if (stride_next != IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION &&
            iree_tokenizer_regex_stride_is_boring(stride, stride_next)) {
          state->dfa_state = stride_next;

          // Level 2: self-loop run scanning. If this state loops on itself for
          // all bytes in the same category, advance until the category changes.
          uint8_t category = stride->category_table[byte];
          if (iree_tokenizer_regex_stride_self_loop_category(
                  stride, stride_next) == category) {
            while (position < chunk.size) {
              uint8_t b = (uint8_t)chunk.data[position];
              if (b >= 0x80 || stride->category_table[b] != category) break;
              ++position;
            }
          }

          // Level 1: tight loop for subsequent ASCII boring transitions.
          while (position < chunk.size) {
            uint8_t b = (uint8_t)chunk.data[position];
            if (IREE_UNLIKELY(b >= 0x80)) break;
            uint16_t next =
                dfa->transitions[(iree_host_size_t)state->dfa_state * 256 + b];
            if (IREE_UNLIKELY(next == IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION))
              break;
            if (IREE_UNLIKELY(
                    !iree_tokenizer_regex_stride_is_boring(stride, next)))
              break;
            state->dfa_state = next;
            ++position;

            // Level 2: self-loop for new state.
            uint8_t category_inner = stride->category_table[b];
            if (iree_tokenizer_regex_stride_self_loop_category(stride, next) ==
                category_inner) {
              while (position < chunk.size) {
                uint8_t b2 = (uint8_t)chunk.data[position];
                if (b2 >= 0x80 || stride->category_table[b2] != category_inner)
                  break;
                ++position;
              }
            }
          }
          continue;
        }
      }
    } else {
      // Non-ASCII: full UTF-8 decode and Unicode category classification.
      codepoint = iree_unicode_utf8_decode(chunk, &position);
      byte = iree_tokenizer_regex_codepoint_to_byte(codepoint);
    }

    // Perform DFA transition (check exact ranges first, then byte/pseudo-byte).
    uint16_t next_state =
        iree_tokenizer_regex_dfa_step(dfa, state->dfa_state, codepoint, byte);

    if (next_state == IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION) {
      // No transition - end of current match attempt.
      // First, resolve any pending lookahead at the current boundary.
      iree_tokenizer_regex_resolve_pending_lookahead(dfa, state, chunk,
                                                     char_start);

      if (state->has_accept || state->has_accept_fallback) {
        // Emit the match we found. emit_match prefers lookahead-passed
        // position over fallback for patterns like \s+(?!\S)|\s+.
        iree_host_size_t match_end = 0;
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_emit_match(
            state, dfa, callback, user_data, &match_end));

        // With start anchor (^), only one match at position 0 is possible.
        // Once we've emitted that match, no more matches can occur.
        // Reset state before returning so callers see in_match=false.
        if (requires_start) {
          state->in_match = false;
          state->has_accept = false;
          state->has_accept_fallback = false;
          iree_tokenizer_regex_reset_best_candidate(state);
          return iree_ok_status();
        }

        // Resume scanning from the end of the emitted match, not from the
        // failure position. Characters between match_end and char_start may
        // start new matches.
        iree_host_size_t resume_position = match_end;
        iree_tokenizer_regex_reset_to_start(state, dfa, resume_position);
        if (resume_position >= base_offset) {
          position = resume_position - base_offset;
        } else {
          // Match ended in previous chunk - replay from rewind buffer.
          IREE_RETURN_IF_ERROR(iree_tokenizer_regex_resume_via_buffer(
              dfa, state, resume_position));
          position = 0;
        }
        continue;
      } else if (state->in_match) {
        // Failed partial match - need to restart from match_start + 1.
        state->in_match = false;

        // With start anchor (^), matching can only start at position 0.
        // If we're already past position 0 and the partial match failed,
        // there's no point in trying further positions.
        if (requires_start && state->match_start > 0) {
          return iree_ok_status();
        }

        // Can we backtrack to match_start + 1?
        // Only possible if match_start is in the current chunk.
        if (state->match_start >= base_offset) {
          // match_start is in this chunk - we can backtrack within it.
          iree_host_size_t match_start_local = state->match_start - base_offset;
          // Advance by one codepoint from match_start.
          iree_host_size_t next_start =
              match_start_local + iree_unicode_utf8_sequence_length(
                                      (uint8_t)chunk.data[match_start_local]);

          // With start anchor, don't restart if next_start > 0.
          if (requires_start && next_start > 0) {
            return iree_ok_status();
          }

          if (next_start < chunk.size) {
            // Restart from next_start (match_start + 1 codepoint).
            // Use reset_to_start to properly clear all match state including
            // fallback acceptance and branch tracking.
            iree_host_size_t restart_abs = base_offset + next_start;
            iree_tokenizer_regex_reset_to_start(state, dfa, restart_abs);
            position = next_start;
            continue;
          }
          // Exhausted all positions in this chunk. In streaming mode, the next
          // chunk will continue. In one-shot mode, finalize handles cleanup.
          break;
        }

        // match_start is in a previous chunk - use rewind buffer.
        IREE_RETURN_IF_ERROR(iree_tokenizer_regex_backtrack_via_buffer(
            dfa, state, base_offset, "main loop"));

        // With start anchor, don't restart if we've moved past position 0.
        if (requires_start && state->match_start > 0) {
          return iree_ok_status();
        }

        // Continue processing current chunk from the beginning.
        position = 0;
        continue;
      } else {
        // Not in a match - just try current char from start state.
        // With start anchor, give up if not at position 0.
        if (requires_start && abs_pos > 0) {
          return iree_ok_status();
        }
        // If we already tried, advance.
        if (state->match_start == abs_pos) {
          continue;
        }
        state->match_start = abs_pos;
        position = char_start;
        continue;
      }
    }

    // Valid transition.
    if (!state->in_match) {
      // With start anchor (^), only start a match at position 0.
      // Skip this character if we're past the start.
      if (requires_start && abs_pos > 0) {
        continue;
      }
      // Starting a new potential match.
      state->match_start = abs_pos;
      state->in_match = true;
      state->has_accept = false;
    }

    state->dfa_state = next_state;

    // Check if this is an accepting state.
    // If the accepting state requires start anchor, only accept if match
    // started at position 0 (otherwise ^a in ^a|b would match 'a' anywhere).
    if (iree_tokenizer_regex_dfa_is_accepting(dfa, next_state) &&
        (!iree_tokenizer_regex_requires_start_anchor(dfa, next_state) ||
         state->match_start == 0)) {
      iree_host_size_t abs_end = base_offset + position;
      iree_tokenizer_regex_evaluate_accepting_state(dfa, state, next_state,
                                                    abs_end, chunk, position);
    }

    // Branch-aware commit check: If we have branch tracking data and
    // all higher-priority branches are dead, emit the best candidate now
    // instead of waiting for NO_TRANSITION.
    //
    // We only record match positions when lookahead has been confirmed at this
    // exact position. This prevents the race condition where lookahead is
    // deferred at chunk boundary and the branch fast-path commits a match
    // before lookahead can be evaluated.
    iree_host_size_t branch_check_end = base_offset + position;
    bool lookahead_confirmed =
        !iree_tokenizer_regex_has_lookahead(dfa, next_state) ||
        (state->has_accept && state->last_accept == branch_check_end);
    if (iree_tokenizer_regex_update_best_candidate(
            dfa, state, next_state, branch_check_end, lookahead_confirmed)) {
      // Higher-priority branches are dead - emit best candidate.
      // Use best_match_end as the match end position.
      if (state->best_branch_idx < 64) {
        iree_tokenizer_regex_match_t match = {
            .start = state->match_start,
            .end = state->best_match_end,
        };
        if (callback) {
          IREE_RETURN_IF_ERROR(callback(user_data, match));
        }

        // With start anchor (^), only one match at position 0 is possible.
        // Reset state before returning so callers see in_match=false.
        if (requires_start) {
          state->in_match = false;
          state->has_accept = false;
          state->has_accept_fallback = false;
          iree_tokenizer_regex_reset_best_candidate(state);
          return iree_ok_status();
        }

        // Resume scanning from the end of the emitted match.
        iree_host_size_t resume_position = state->best_match_end;
        iree_tokenizer_regex_reset_to_start(state, dfa, resume_position);
        state->last_accept = resume_position;
        if (resume_position >= base_offset) {
          position = resume_position - base_offset;
        } else {
          // Match ended in previous chunk - replay from rewind buffer.
          IREE_RETURN_IF_ERROR(iree_tokenizer_regex_resume_via_buffer(
              dfa, state, resume_position));
          position = 0;
        }
        continue;
      }
    }
  }

  // Preserve bytes for potential cross-chunk backtracking in next chunk.
  iree_tokenizer_regex_update_rewind_buffer(state, chunk, base_offset);

  return iree_ok_status();
}

iree_status_t iree_tokenizer_regex_exec_finalize(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, iree_host_size_t total_length,
    iree_tokenizer_regex_match_callback_fn_t callback, void* user_data) {
  // Resolve pending lookahead at end of stream.
  // With no following character, negative lookahead unconditionally passes:
  //   - a(?!b) at EOS: no 'b' follows, so passes
  //   - \s+(?!\S) at EOS: no non-whitespace follows, so passes
  //
  // Only use pending_match_end if we don't already have a better
  // accept position from a different pattern branch. This can happen when:
  //   1. Whitespace pattern defers lookahead at chunk boundary
  //   2. A different branch (e.g., letter pattern) subsequently matches
  //   3. The letter match sets last_accept to a longer position
  //   4. We must not overwrite that better position with pending_match_end
  if (state->pending_lookahead) {
    // Only use pending lookahead position if no better accept exists,
    // or if pending extends beyond current accept (same match, extended).
    if (!state->has_accept || state->pending_match_end > state->last_accept) {
      state->last_accept = state->pending_match_end;
      state->has_accept = true;
    }
    state->pending_lookahead = false;
  }

  // Handle end anchor ($) at end of stream.
  // If we're in an accepting state with end anchor requirement and at end of
  // input, the match is now valid.
  if (state->in_match && !state->has_accept && !state->has_accept_fallback &&
      iree_tokenizer_regex_dfa_is_accepting(dfa, state->dfa_state) &&
      iree_tokenizer_regex_requires_end_anchor(dfa, state->dfa_state)) {
    // End anchor satisfied - we're at end of input.
    state->last_accept = total_length;
    state->has_accept = true;
  }

  // Emit any pending match.
  if (state->has_accept || state->has_accept_fallback) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_regex_emit_match(
        state, dfa, callback, user_data, /*out_match_end=*/NULL));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Convenience Wrappers
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_regex_exec(
    const iree_tokenizer_regex_dfa_t* dfa, iree_string_view_t text,
    iree_tokenizer_regex_match_callback_fn_t callback, void* user_data) {
  iree_tokenizer_regex_exec_state_t state;
  iree_tokenizer_regex_exec_initialize(&state, dfa);

  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_exec_feed(
      dfa, &state, text, 0, /*stride=*/NULL, callback, user_data));

  return iree_tokenizer_regex_exec_finalize(dfa, &state, text.size, callback,
                                            user_data);
}

// Callback for counting matches.
typedef struct iree_tokenizer_regex_count_state_t {
  iree_host_size_t count;
} iree_tokenizer_regex_count_state_t;

static iree_status_t iree_tokenizer_regex_count_callback(
    void* user_data, iree_tokenizer_regex_match_t match) {
  (void)match;
  iree_tokenizer_regex_count_state_t* count_state =
      (iree_tokenizer_regex_count_state_t*)user_data;
  count_state->count++;
  return iree_ok_status();
}

iree_status_t iree_tokenizer_regex_count_matches(
    const iree_tokenizer_regex_dfa_t* dfa, iree_string_view_t text,
    iree_host_size_t* out_count) {
  if (!out_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_count is NULL");
  }

  iree_tokenizer_regex_count_state_t count_state = {.count = 0};
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_exec(
      dfa, text, iree_tokenizer_regex_count_callback, &count_state));
  *out_count = count_state.count;
  return iree_ok_status();
}

// Callback for has_match (aborts after first match).
static iree_status_t iree_tokenizer_regex_has_match_callback(
    void* user_data, iree_tokenizer_regex_match_t match) {
  (void)match;
  bool* found = (bool*)user_data;
  *found = true;
  // Return a "cancelled" status to abort execution early.
  return iree_status_from_code(IREE_STATUS_CANCELLED);
}

iree_status_t iree_tokenizer_regex_has_match(
    const iree_tokenizer_regex_dfa_t* dfa, iree_string_view_t text,
    bool* out_has_match) {
  if (!out_has_match) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_has_match is NULL");
  }

  *out_has_match = false;
  iree_status_t status = iree_tokenizer_regex_exec(
      dfa, text, iree_tokenizer_regex_has_match_callback, out_has_match);

  // CANCELLED means we found a match and aborted early - that's success.
  if (iree_status_is_cancelled(status)) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  return status;
}
