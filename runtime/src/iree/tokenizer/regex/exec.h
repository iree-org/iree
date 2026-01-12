// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// DFA regex executor for tokenizer pre-tokenization.
//
// This is the minimal runtime component of the regex engine. It executes
// pre-compiled DFAs to find pattern matches in UTF-8 text. The executor is
// designed for streaming operation with O(1) state and zero heap allocation.
//
// Architecture:
//   The regex system is split into two libraries:
//   1. Executor (regex/exec.h) - Minimal runtime for executing pre-compiled DFA
//   2. Compiler (regex/compile.h) - Pattern string to DFA compilation
//
//   This separation allows embedding minimal runtime code when DFAs are
//   pre-compiled at build time, while still supporting on-demand compilation
//   from JSON tokenizer configs at runtime when needed.
//
// Supported Features (covers 95%+ of HuggingFace tokenizers):
//   - Literals: a, ab, abc
//   - Any character: .
//   - Character classes: [abc], [a-z], [^abc]
//   - Escapes: \r, \n, \t, \f, \v, \a, \e, \0, \\, \[, etc.
//   - Shorthand classes: \d, \D, \w, \W, \s, \S
//   - Unicode categories: \p{L}, \p{N}, \p{P}, \p{M}, \p{S}, \p{Z}, \p{C}
//   - Quantifiers: *, +, ?, {n}, {n,}, {n,m}
//   - Alternation: a|b|c
//   - Grouping: (...), (?:...)
//   - Negative lookahead: (?!x), (?!\S) (end of branch, single char/shorthand)
//   - Anchors: ^, $
//
// NOT Supported (rare, complex, breaks DFA properties):
//   - Backreferences: \1, \2 (requires backtracking)
//   - Positive lookahead: (?=...) (complex state machine)
//   - Lookbehind: (?<=...), (?<!...) (requires backward scan)
//   - Named groups: (?P<name>...) (capture not needed)
//   - Atomic groups: (?>...) (backtracking construct)
//   - Recursion: (?R) (requires stack)
//   - Negated Unicode: \P{...} (use [^\p{...}] instead)
//
// Shorthand Class Semantics:
//   - \d = [0-9] (ASCII digits only, NOT Unicode Nd category)
//   - \D = [^0-9] (anything except ASCII digits)
//   - \w = [a-zA-Z0-9_] (ASCII word chars only, NOT Unicode letters/digits)
//   - \W = [^a-zA-Z0-9_] (anything except ASCII word chars)
//   - \s = ASCII whitespace + Unicode White_Space property (via
//   PSEUDO_WHITESPACE)
//   - \S = non-whitespace (complement of \s)
//
//   Note: \d and \w are ASCII-only for compatibility with Python's re module
//   default behavior and most tokenizer patterns. Use \p{N} and [\p{L}\p{N}_]
//   for Unicode-aware digit/word matching.
//
// Matching Semantics:
//   - Leftmost-longest: Matches start as early as possible, longest wins
//   - Search mode: The executor finds matches anywhere in input (not anchored)
//   - After failed partial match, executor rescans from match_start + 1
//   - Example: Pattern "ab" on "aab" finds match at position 1
//
// Streaming:
//   Streaming mode uses an internal 256-byte rewind buffer to handle
//   cross-chunk backtracking. When a partial match starts in chunk A and fails
//   in chunk B, the executor can backtrack properly using buffered bytes from
//   chunk A.
//
//   This covers all practical tokenizer patterns (whitespace runs, word
//   boundaries, punctuation). If a pattern requires backtracking more than 256
//   bytes, exec_feed() returns IREE_STATUS_OUT_OF_RANGE.
//
// Why DFA:
//   - O(n) guaranteed time complexity (patterns run millions of times)
//   - Streamable execution with checkpoint/resume support
//   - Pre-compilable to compact binary format
//   - No backtracking means predictable, bounded memory usage
//
// API Organization:
//
//   Step function (inline, lowest level):
//     iree_tokenizer_regex_dfa_step() - single transition lookup
//
//   Streaming executor (core API):
//     iree_tokenizer_regex_exec_initialize() - initialize state
//     iree_tokenizer_regex_exec_feed() - process chunk, emit matches
//     iree_tokenizer_regex_exec_finalize() - flush pending match
//
//   Convenience wrappers:
//     iree_tokenizer_regex_exec() - full-text execution
//
// Thread Safety:
//   - DFA loading: Thread-safe (read-only on input).
//   - DFA execution: Thread-safe for same DFA with different state/text.
//     Multiple threads can execute the same DFA concurrently as long as each
//     has its own exec_state_t.

#ifndef IREE_TOKENIZER_REGEX_EXEC_H_
#define IREE_TOKENIZER_REGEX_EXEC_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Binary Format Constants
//===----------------------------------------------------------------------===//

// Magic number for DFA binary format: "ITDF" (IREE Tokenizer DFA).
#define IREE_TOKENIZER_REGEX_DFA_MAGIC                              \
  (((uint32_t)'I') | ((uint32_t)'T' << 8) | ((uint32_t)'D' << 16) | \
   ((uint32_t)'F' << 24))

// Current DFA binary format version.
// Increment when making breaking changes to the format.
//
// Layout: header + transitions + bitmap + lookahead + anchors
//   - Dense 256-entry transition table per state (512 bytes/state)
//   - Maximum 65534 states (65535 reserved for NO_TRANSITION sentinel)
//
// ALIGNMENT: DFA data must be 8-byte aligned for zero-copy loading.
// The accepting_bitmap uses uint64_t which requires 8-byte alignment on
// strict architectures (ARM, SPARC). Unaligned data will be rejected by
// iree_tokenizer_regex_dfa_load() with INVALID_ARGUMENT status.
#define IREE_TOKENIZER_REGEX_DFA_VERSION 1

// Maximum supported DFA version (for forward compatibility).
#define IREE_TOKENIZER_REGEX_DFA_MAX_VERSION 1

// Sentinel value for "no transition" in the transition table.
#define IREE_TOKENIZER_REGEX_NO_TRANSITION 0xFFFFu

//===----------------------------------------------------------------------===//
// Unicode Pseudo-Bytes
//===----------------------------------------------------------------------===//

// The DFA operates on bytes (0-255), but we need to match Unicode categories
// (\p{L}, \p{N}, etc.) which span multiple codepoints. We use "pseudo-bytes"
// in the range 0x80-0x8F (invalid as UTF-8 lead bytes) to represent categories.
//
// At runtime, the executor decodes UTF-8 codepoints, classifies them via
// iree_unicode_category(), and looks up transitions using the pseudo-byte.
// ASCII bytes (0x00-0x7F) are used directly.
//
// Example: For pattern \p{L}+, the DFA has transitions on PSEUDO_LETTER.
// When executing on "Hello", 'H' (0x48) uses direct lookup, but on "Héllo",
// the 'é' (U+00E9) is classified as Letter and uses PSEUDO_LETTER transition.

typedef enum iree_tokenizer_regex_pseudo_byte_e {
  // Unicode categories (0x80-0x87).
  IREE_TOKENIZER_REGEX_PSEUDO_LETTER = 0x80,      // \p{L} - Letters
  IREE_TOKENIZER_REGEX_PSEUDO_NUMBER = 0x81,      // \p{N} - Numbers
  IREE_TOKENIZER_REGEX_PSEUDO_PUNCT = 0x82,       // \p{P} - Punctuation
  IREE_TOKENIZER_REGEX_PSEUDO_MARK = 0x83,        // \p{M} - Marks
  IREE_TOKENIZER_REGEX_PSEUDO_SYMBOL = 0x84,      // \p{S} - Symbols
  IREE_TOKENIZER_REGEX_PSEUDO_SEPARATOR = 0x85,   // \p{Z} - Separators
  IREE_TOKENIZER_REGEX_PSEUDO_OTHER = 0x86,       // \p{C} - Other
  IREE_TOKENIZER_REGEX_PSEUDO_WHITESPACE = 0x87,  // \s (non-ASCII whitespace)
  IREE_TOKENIZER_REGEX_PSEUDO_COUNT = 8,
} iree_tokenizer_regex_pseudo_byte_t;

//===----------------------------------------------------------------------===//
// Exact Codepoint Ranges
//===----------------------------------------------------------------------===//

// Maximum number of exact codepoint ranges in a character class.
// This limit keeps the match_class structure cache-friendly while supporting
// the Unicode ranges found in HuggingFace tokenizers (e.g., DeepSeek's
// [一-龥]). Exceeding this limit produces a compile-time error with guidance to
// use \p{L} for broad matching instead of many literal ranges.
#define IREE_TOKENIZER_REGEX_MAX_CHAR_CLASS_RANGES 4

// An exact Unicode codepoint range [start, end] (inclusive).
// Used for precise matching of literal Unicode ranges in character classes
// like [一-龥] (U+4E00-U+9FA5). Ranges are stored sorted by start codepoint
// to enable early-exit optimization during matching.
typedef struct iree_tokenizer_regex_codepoint_range_t {
  uint32_t start;  // First codepoint in range (inclusive).
  uint32_t end;    // Last codepoint in range (inclusive).
} iree_tokenizer_regex_codepoint_range_t;

// A DFA range transition maps a codepoint range to a target state.
// When the executor sees a codepoint in [start, end], it transitions to
// target_state. Range transitions are checked BEFORE pseudo-byte fallback,
// enabling exact matching that would otherwise over-approximate via categories.
//
// Binary format for range transitions section (when HAS_RANGES flag is set):
//   uint16_t num_ranges     // Total number of range transitions
//   uint16_t reserved       // Padding for 4-byte alignment
//   iree_tokenizer_regex_dfa_range_transition_t[num_ranges]  // Flat array
//
// The flat array design enables zero-copy loading without per-DFA allocation.
// At runtime, we linear-search the array for ranges matching the current state.
// This is O(n) in the number of ranges, but tokenizers typically have 0-4
// ranges total, so this is negligible compared to UTF-8 decoding overhead.
typedef struct iree_tokenizer_regex_dfa_range_transition_t {
  uint16_t from_state;    // DFA state this range applies to.
  uint16_t target_state;  // DFA state to transition to on match.
  uint32_t start;         // Range start codepoint (inclusive).
  uint32_t end;           // Range end codepoint (inclusive).
} iree_tokenizer_regex_dfa_range_transition_t;
static_assert(sizeof(iree_tokenizer_regex_dfa_range_transition_t) == 12,
              "range transition must be 12 bytes");

//===----------------------------------------------------------------------===//
// DFA Header
//===----------------------------------------------------------------------===//

// DFA header flags.
typedef enum iree_tokenizer_regex_dfa_flag_bits_e {
  IREE_TOKENIZER_REGEX_DFA_FLAG_NONE = 0,
  // DFA has lookahead data (lookahead table follows accepting bitmap).
  IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD = 1 << 0,
  // DFA uses Unicode pseudo-bytes (requires UTF-8 decoding at runtime).
  IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE = 1 << 1,
  // DFA is case-insensitive (ASCII letters only).
  // NOTE: This flag is INFORMATIONAL ONLY for tooling and debugging.
  // The compiler is responsible for emitting transitions for both cases
  // (e.g., 'a' and 'A' transitions for case-insensitive [a-z]).
  // The executor does NOT perform runtime case folding.
  IREE_TOKENIZER_REGEX_DFA_FLAG_CASE_INSENSITIVE = 1 << 2,
  // DFA has anchor bitmaps (start_anchor and end_anchor bitmaps follow).
  // Start anchor: states that require match to start at position 0.
  // End anchor: accepting states that require match to end at end of input.
  IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_ANCHORS = 1 << 3,
  // DFA has branch tracking bitmasks for PCRE-compatible alternation.
  // alive_branches: which alternation branches can reach each state.
  // accepting_branches: which branches accept at each state.
  IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_BRANCHES = 1 << 4,
  // DFA has exact codepoint range transitions.
  // Ranges are stored per-state and checked before pseudo-byte fallback.
  // This enables precise matching of Unicode ranges like [一-龥].
  IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_RANGES = 1 << 5,
} iree_tokenizer_regex_dfa_flag_bits_t;
typedef uint16_t iree_tokenizer_regex_dfa_flags_t;

// Binary DFA header. All fields are little-endian.
// The header is followed by:
//   1. Transition table: uint16_t[num_states][256]
//   2. Accepting bitmap: uint64_t[ceil(num_states/64)]
//   3. Lookahead table (if HAS_LOOKAHEAD):
//        iree_tokenizer_regex_lookahead_t[num_states]
//   4. Start anchor bitmap (if HAS_ANCHORS): uint64_t[ceil(num_states/64)]
//        States that require matching to start at position 0.
//   5. End anchor bitmap (if HAS_ANCHORS): uint64_t[ceil(num_states/64)]
//        Accepting states that require matching to end at end of input.
//   6. Alive branches (if HAS_BRANCHES): uint64_t[num_states]
//        For each state, which alternation branches can reach it.
//   7. Accepting branches (if HAS_BRANCHES): uint64_t[num_states]
//        For each state, which branches accept at this state.
//   8. Range transitions (if HAS_RANGES):
//        uint16_t num_ranges   // Total range count
//        uint16_t reserved     // Padding for alignment
//        iree_tokenizer_regex_dfa_range_transition_t[num_ranges]  // Flat array
typedef struct iree_tokenizer_regex_dfa_header_t {
  uint32_t magic;          // Must be IREE_TOKENIZER_REGEX_DFA_MAGIC.
  uint16_t version;        // Format version.
  uint16_t flags;          // iree_tokenizer_regex_dfa_flags_t.
  uint16_t num_states;     // Number of states.
  uint16_t num_accepting;  // Number of accepting states (for validation).
  uint16_t start_state;    // Start state for position 0 (includes ^ paths).
  uint16_t unanchored_start_state;  // Start state for position > 0 (no ^).
} iree_tokenizer_regex_dfa_header_t;
static_assert(sizeof(iree_tokenizer_regex_dfa_header_t) == 16,
              "DFA header must be 16 bytes");

//===----------------------------------------------------------------------===//
// Lookahead Support
//===----------------------------------------------------------------------===//

// Lookahead is used to implement negative lookahead assertions like (?!\S).
// Instead of compiling the lookahead pattern to a separate DFA (complex), we
// support only single-character or character-class lookaheads, which covers
// all known HuggingFace tokenizer patterns.
//
// Research finding (via analysis of tiktoken, GPT-2, SentencePiece, custom):
//   - tiktoken/GPT-4, GPT-2, StarCoder: Only use \s+(?!\S)
//   - SentencePiece models: Don't use regex lookahead at all
//   - Custom tokenizers: May use (?!\]), (?![a-zA-Z]) - single char/class
//   - No evidence of complex lookaheads like (?!foo|bar+) in production

typedef enum iree_tokenizer_regex_lookahead_type_e {
  IREE_TOKENIZER_REGEX_LOOKAHEAD_NONE = 0,
  // Negative lookahead for single character: (?!x)
  IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_CHAR = 1,
  // Negative lookahead for shorthand class: (?!\S), (?!\s), (?!\d), etc.
  IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_SHORTHAND = 2,
  // Negative lookahead for character class: (?![abc]), (?![a-z])
  // The class is stored as a 256-bit bitmap in lookahead_data.
  IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_CLASS = 3,

  // Lookahead types with fallback (for alternation patterns like
  // \s+(?!\S)|\s+). When alternation has both lookahead and non-lookahead
  // branches, the DFA state is accepting for BOTH. The executor must track two
  // positions:
  //   1. Last position where lookahead passed (preferred)
  //   2. Last position where fallback would accept (used if lookahead fails)
  // This gives PCRE-like semantics where the lookahead branch has priority.
  IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_CHAR_WITH_FALLBACK = 4,
  IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_SHORTHAND_WITH_FALLBACK = 5,
  IREE_TOKENIZER_REGEX_LOOKAHEAD_NEG_CLASS_WITH_FALLBACK = 6,
} iree_tokenizer_regex_lookahead_type_t;

// Flags for PCRE-compatible match selection in lookahead table.
// See nfa.h accept struct comment for full documentation on limitations.
typedef enum iree_tokenizer_regex_lookahead_flag_bits_e {
  IREE_TOKENIZER_REGEX_LOOKAHEAD_FLAG_NONE = 0,
  // True if alternation has a non-lookahead branch BEFORE the first lookahead.
  // When set, prefer longer match (independent branch extends greedily).
  // When not set, prefer lookahead-passed position (fallback branches).
  IREE_TOKENIZER_REGEX_LOOKAHEAD_FLAG_HAS_EARLY_NO_LOOKAHEAD = 1u << 0,
} iree_tokenizer_regex_lookahead_flag_bits_t;
typedef uint8_t iree_tokenizer_regex_lookahead_flags_t;

// Lookahead data for a single state.
// For LOOKAHEAD_NEG_CHAR: data is the character to NOT match.
// For LOOKAHEAD_NEG_SHORTHAND: data encodes the shorthand type.
// For LOOKAHEAD_NEG_CLASS: data is offset into extended lookahead bitmap table.
typedef struct iree_tokenizer_regex_lookahead_t {
  uint8_t type;  // iree_tokenizer_regex_lookahead_type_t
  uint8_t data;  // Type-specific data
  iree_tokenizer_regex_lookahead_flags_t flags;  // PCRE-compatibility flags
  uint8_t reserved;                              // Padding for alignment
} iree_tokenizer_regex_lookahead_t;
static_assert(sizeof(iree_tokenizer_regex_lookahead_t) == 4,
              "lookahead entry must be 4 bytes");

// Shorthand class identifiers for LOOKAHEAD_NEG_SHORTHAND.
typedef enum iree_tokenizer_regex_shorthand_e {
  IREE_TOKENIZER_REGEX_SHORTHAND_s = 0,  // \s - whitespace
  IREE_TOKENIZER_REGEX_SHORTHAND_S = 1,  // \S - non-whitespace
  IREE_TOKENIZER_REGEX_SHORTHAND_d = 2,  // \d - digit
  IREE_TOKENIZER_REGEX_SHORTHAND_D = 3,  // \D - non-digit
  IREE_TOKENIZER_REGEX_SHORTHAND_w = 4,  // \w - word character
  IREE_TOKENIZER_REGEX_SHORTHAND_W = 5,  // \W - non-word character
} iree_tokenizer_regex_shorthand_t;

//===----------------------------------------------------------------------===//
// Match Result
//===----------------------------------------------------------------------===//

// A single match result from DFA execution.
typedef struct iree_tokenizer_regex_match_t {
  iree_host_size_t start;  // Byte offset of match start (inclusive).
  iree_host_size_t end;    // Byte offset of match end (exclusive).
} iree_tokenizer_regex_match_t;

// Callback invoked with match results during DFA execution.
// |match| contains the byte offsets of the match within the input text.
// Return non-OK status to abort execution.
typedef iree_status_t (*iree_tokenizer_regex_match_callback_fn_t)(
    void* user_data, iree_tokenizer_regex_match_t match);

//===----------------------------------------------------------------------===//
// DFA Handle
//===----------------------------------------------------------------------===//

// Opaque handle to a loaded DFA.
// This is a lightweight view into the binary DFA data (zero-copy when mmap'd).
typedef struct iree_tokenizer_regex_dfa_t {
  const iree_tokenizer_regex_dfa_header_t* header;
  const uint16_t* transitions;       // [num_states][256] transition table.
  const uint64_t* accepting_bitmap;  // Bit i = state i is accepting.
  const iree_tokenizer_regex_lookahead_t* lookahead;  // NULL if no lookahead.
  const uint64_t* start_anchor_bitmap;                // NULL if no anchors.
  const uint64_t* end_anchor_bitmap;                  // NULL if no anchors.
  const uint64_t* alive_branches;                     // NULL if no branches.
  const uint64_t* accepting_branches;                 // NULL if no branches.
  // Range transitions: flat array for zero-copy loading. NULL if no ranges.
  // At runtime, linear-search for ranges matching current state.
  const iree_tokenizer_regex_dfa_range_transition_t* ranges;
  uint16_t num_ranges;  // Number of range transitions (0 if no ranges).
} iree_tokenizer_regex_dfa_t;

//===----------------------------------------------------------------------===//
// Streaming Execution State
//===----------------------------------------------------------------------===//

// State for streaming DFA execution.
//
// This allows feeding text in chunks while maintaining match state across
// calls. The state is O(1) size (~320 bytes) and can be stack-allocated.
//
// Usage:
//   iree_tokenizer_regex_exec_state_t state;
//   iree_tokenizer_regex_exec_initialize(&state, &dfa);
//   while (more_chunks) {
//     iree_tokenizer_regex_exec_feed(&dfa, &state, chunk, offset, cb, ud);
//     offset += chunk.data_length;
//   }
//   iree_tokenizer_regex_exec_finalize(&dfa, &state, total_length, cb, ud);
//
// Cross-Chunk Backtracking:
//   The executor implements leftmost-longest matching, which requires trying
//   alternative match starts when the current path fails. When a partial match
//   begins in chunk A and fails in chunk B, the executor uses an internal
//   256-byte rewind buffer to backtrack properly.
//
//   This handles all practical tokenizer patterns (whitespace, word boundaries,
//   punctuation). If a pattern requires backtracking more than 256 bytes,
//   exec_feed() returns IREE_STATUS_OUT_OF_RANGE.
typedef struct iree_tokenizer_regex_exec_state_t {
  uint16_t dfa_state;            // Current DFA state.
  iree_host_size_t match_start;  // Start position of current potential match.
  iree_host_size_t last_accept;  // End position where lookahead passed (or
                                 // unconditional accept if no lookahead).
  bool in_match;                 // Whether we're tracking a potential match.
  bool has_accept;               // Whether we've seen an accepting state.

  // UTF-8 partial sequence buffer for streaming across chunk boundaries.
  // When a multi-byte UTF-8 sequence is split across chunks, we buffer the
  // incomplete bytes here and complete them on the next chunk.
  uint8_t utf8_partial_buffer[4];  // Incomplete UTF-8 sequence (max 4 bytes).
  uint8_t utf8_partial_length;     // Number of bytes in partial buffer (0-3).

  // Pending lookahead state for chunk boundary handling.
  // When we reach an accepting state with lookahead at chunk end, we can't
  // evaluate the lookahead until we see the next character. We store the
  // pending state here and resolve it on the next chunk or at finalize.
  bool pending_lookahead;              // Waiting for next char for lookahead.
  uint16_t pending_lookahead_state;    // DFA state awaiting lookahead check.
  iree_host_size_t pending_match_end;  // Match end position if accepted.

  // Fallback acceptance for mixed lookahead/non-lookahead alternation.
  // For patterns like \s+(?!\S)|\s+, the DFA state accepts for BOTH branches.
  // We track positions separately:
  //   - last_accept/has_accept: Position where lookahead PASSED
  //   - last_accept_fallback/has_accept_fallback: Position that would accept
  //     without lookahead (fallback)
  // At match boundary, we prefer lookahead-passed position. Only if no
  // lookahead-passed position exists do we use the fallback.
  bool has_accept_fallback;               // Have a fallback accept position.
  iree_host_size_t last_accept_fallback;  // Fallback accept position.

  // Branch-aware best candidate tracking (for PCRE-compatible alternation).
  // Only used when DFA has HAS_BRANCHES flag set.
  // Tracks the highest-priority branch that has accepted so far.
  uint8_t best_branch_idx;  // Lowest accepting branch (0=highest priority,
                            // 64=none).
  iree_host_size_t best_match_end;  // End position of best candidate.

  // Rewind buffer for cross-chunk backtracking.
  //
  // When a partial match starts in chunk A and fails in chunk B, proper
  // leftmost-longest semantics require trying match_start + 1. If match_start
  // is in a previous chunk, we need those bytes to backtrack correctly.
  //
  // At the end of each chunk, if we're in a partial match, we copy bytes from
  // match_start to the end of the chunk into this buffer. On backtrack failure,
  // we can then re-process from the buffer instead of silently skipping
  // matches.
  //
  // The 256-byte size covers all practical tokenizer patterns (whitespace runs,
  // word boundaries, punctuation). If a pattern requires more, exec_feed()
  // returns IREE_STATUS_OUT_OF_RANGE.
  uint8_t rewind_buffer[256];
  iree_host_size_t rewind_buffer_length;  // Valid bytes in buffer.
  iree_host_size_t rewind_buffer_start;   // Absolute offset of buffer[0].
} iree_tokenizer_regex_exec_state_t;

//===----------------------------------------------------------------------===//
// Split Behavior (for pre-tokenization)
//===----------------------------------------------------------------------===//

// How to handle matched delimiters when splitting text.
// These match HuggingFace tokenizers' SplitDelimiterBehavior.
typedef enum iree_tokenizer_regex_split_behavior_e {
  // Discard the matched delimiter entirely.
  // "a]b" with pattern "]" -> ["a", "b"]
  IREE_TOKENIZER_REGEX_SPLIT_REMOVED = 0,
  // Emit the delimiter as its own separate segment.
  // "a]b" with pattern "]" -> ["a", "]", "b"]
  IREE_TOKENIZER_REGEX_SPLIT_ISOLATED = 1,
  // Merge the delimiter with the previous segment.
  // "a]b" with pattern "]" -> ["a]", "b"]
  IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_PREVIOUS = 2,
  // Merge the delimiter with the next segment.
  // "a]b" with pattern "]" -> ["a", "]b"]
  IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT = 3,
  // Merge consecutive delimiters into a single segment.
  // "a]]b" with pattern "]" -> ["a", "]]", "b"]
  IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS = 4,
} iree_tokenizer_regex_split_behavior_t;

// Returns true if the split behavior creates BPE word boundaries.
//
// Word-splitting behaviors (REMOVED, ISOLATED, CONTIGUOUS) create segments that
// should be tokenized independently by the BPE model.
//
// Offset-tracking behaviors (MERGED_WITH_PREVIOUS, MERGED_WITH_NEXT) produce
// segments that are metadata for character-to-token mapping, NOT BPE
// boundaries. For these, the BPE model should see the full text as one unit.
//
// Note: When a ByteLevel transform follows a non-boundary Split in a Sequence,
// the ByteLevel conversion (space → Ġ) makes segments safe for independent
// tokenization because BPE was trained with Ġ at word starts.
static inline bool iree_tokenizer_regex_split_creates_bpe_boundaries(
    iree_tokenizer_regex_split_behavior_t behavior) {
  return behavior != IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_PREVIOUS &&
         behavior != IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT;
}

//===----------------------------------------------------------------------===//
// DFA Loading
//===----------------------------------------------------------------------===//

// Loads a DFA from binary data.
//
// This is a zero-copy operation - the DFA structure points directly into the
// binary data. The caller must ensure |data| remains valid for the lifetime
// of |out_dfa|.
//
// |data| is the binary DFA data (must remain valid).
// |out_dfa| receives the loaded DFA handle.
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - IREE_STATUS_INVALID_ARGUMENT if data is too small or NULL.
//   - IREE_STATUS_FAILED_PRECONDITION if magic number is invalid.
//   - IREE_STATUS_UNIMPLEMENTED if version is unsupported.
iree_status_t iree_tokenizer_regex_dfa_load(
    iree_const_byte_span_t data, iree_tokenizer_regex_dfa_t* out_dfa);

// Validates a loaded DFA for internal consistency.
//
// This performs additional validation beyond load-time checks:
//   - Verifies accepting state count matches bitmap population count
//   - Checks all transitions reference valid states or NO_TRANSITION
//   - Validates lookahead data if present
//
// Returns:
//   - IREE_STATUS_OK if valid.
//   - IREE_STATUS_DATA_LOSS if internal inconsistency detected.
iree_status_t iree_tokenizer_regex_dfa_validate(
    const iree_tokenizer_regex_dfa_t* dfa);

//===----------------------------------------------------------------------===//
// DFA Inspection
//===----------------------------------------------------------------------===//

// Returns the number of states in the DFA.
static inline uint16_t iree_tokenizer_regex_dfa_state_count(
    const iree_tokenizer_regex_dfa_t* dfa) {
  return dfa->header->num_states;
}

// Returns the DFA flags.
static inline iree_tokenizer_regex_dfa_flags_t iree_tokenizer_regex_dfa_flags(
    const iree_tokenizer_regex_dfa_t* dfa) {
  return dfa->header->flags;
}

// Returns the start state index.
static inline uint16_t iree_tokenizer_regex_dfa_start_state(
    const iree_tokenizer_regex_dfa_t* dfa) {
  return dfa->header->start_state;
}

// Returns true if the DFA uses Unicode pseudo-bytes.
static inline bool iree_tokenizer_regex_dfa_uses_unicode(
    const iree_tokenizer_regex_dfa_t* dfa) {
  return (dfa->header->flags & IREE_TOKENIZER_REGEX_DFA_FLAG_UNICODE) != 0;
}

// Returns true if state |index| is an accepting state.
static inline bool iree_tokenizer_regex_dfa_is_accepting(
    const iree_tokenizer_regex_dfa_t* dfa, uint16_t state) {
  if (state >= dfa->header->num_states) return false;
  return (dfa->accepting_bitmap[state / 64] & (1ULL << (state % 64))) != 0;
}

//===----------------------------------------------------------------------===//
// Step Function (lowest level)
//===----------------------------------------------------------------------===//

// Performs a single DFA transition using a byte.
//
// Returns the next state for |current_state| on input |byte|.
// Returns IREE_TOKENIZER_REGEX_NO_TRANSITION if no transition exists.
//
// This is the core primitive - no state mutation, no callbacks, just a lookup.
// For codepoints that may match exact Unicode ranges, use dfa_step_codepoint.
static inline uint16_t iree_tokenizer_regex_dfa_step(
    const iree_tokenizer_regex_dfa_t* dfa, uint16_t current_state,
    uint8_t byte) {
  return dfa->transitions[(iree_host_size_t)current_state * 256 + byte];
}

// Performs a single DFA transition using a codepoint.
//
// Checks exact codepoint range transitions first, then falls back to
// byte/pseudo-byte transitions. Use this when matching patterns that
// contain exact Unicode ranges like [一-龥].
//
// |codepoint| is the Unicode codepoint.
// |byte| is the category pseudo-byte for |codepoint| (from codepoint_to_byte).
//
// Returns the next state or IREE_TOKENIZER_REGEX_NO_TRANSITION.
static inline uint16_t iree_tokenizer_regex_dfa_step_codepoint(
    const iree_tokenizer_regex_dfa_t* dfa, uint16_t current_state,
    uint32_t codepoint, uint8_t byte) {
  // Check exact codepoint range transitions first.
  // Ranges are sorted by start codepoint for early exit.
  if (dfa->num_ranges > 0) {
    for (uint16_t i = 0; i < dfa->num_ranges; ++i) {
      const iree_tokenizer_regex_dfa_range_transition_t* range =
          &dfa->ranges[i];
      if (range->from_state != current_state) continue;
      if (codepoint < range->start) continue;  // Before this range.
      if (codepoint <= range->end) {
        return range->target_state;  // Exact match!
      }
    }
  }

  // Fall back to byte/pseudo-byte transition.
  return dfa->transitions[(iree_host_size_t)current_state * 256 + byte];
}

//===----------------------------------------------------------------------===//
// Streaming Executor
//===----------------------------------------------------------------------===//

// Initializes execution state for a new matching session.
//
// |state| is caller-owned storage for execution state (typically stack).
// |dfa| is the DFA to execute (must remain valid during execution).
void iree_tokenizer_regex_exec_initialize(
    iree_tokenizer_regex_exec_state_t* state,
    const iree_tokenizer_regex_dfa_t* dfa);

// Feeds a chunk of text to the executor, emitting matches as found.
//
// This implements leftmost-longest matching semantics:
//   - Matches start as early as possible in the text
//   - Among matches starting at the same position, the longest is chosen
//   - After a match, scanning resumes from the end of the match
//
// Matches are emitted via |callback| with positions relative to the overall
// stream (using |base_offset| + local position).
//
// |dfa| is the DFA to execute.
// |state| is the execution state (from exec_init, persists across calls).
// |chunk| is the text chunk to process.
// |base_offset| is the byte offset of this chunk in the overall stream.
// |callback| is invoked for each complete match found (may be NULL).
// |user_data| is passed to the callback.
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - Status from callback if it returns non-OK.
iree_status_t iree_tokenizer_regex_exec_feed(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, iree_const_byte_span_t chunk,
    iree_host_size_t base_offset,
    iree_tokenizer_regex_match_callback_fn_t callback, void* user_data);

// Finalizes execution, flushing any pending match at end of input.
//
// Must be called after all chunks have been fed to ensure the final match
// (if any) is emitted. After this call, the state should not be reused
// without calling exec_init again.
//
// |dfa| is the DFA that was executed.
// |state| is the execution state.
// |total_length| is the total length of all input fed (for match end position).
// |callback| is invoked if there's a pending match (may be NULL).
// |user_data| is passed to the callback.
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - Status from callback if it returns non-OK.
iree_status_t iree_tokenizer_regex_exec_finalize(
    const iree_tokenizer_regex_dfa_t* dfa,
    iree_tokenizer_regex_exec_state_t* state, iree_host_size_t total_length,
    iree_tokenizer_regex_match_callback_fn_t callback, void* user_data);

//===----------------------------------------------------------------------===//
// Convenience Wrappers
//===----------------------------------------------------------------------===//

// Executes the DFA on complete text, emitting matches via callback.
//
// This is a thin wrapper around init/feed/finalize for the common case
// where complete text is available.
//
// |dfa| is the DFA to execute.
// |text| is the complete UTF-8 input text.
// |callback| is invoked for each match (may be NULL to just validate).
// |user_data| is passed to the callback.
//
// Returns:
//   - IREE_STATUS_OK on success.
//   - Status from callback if it returns non-OK.
iree_status_t iree_tokenizer_regex_exec(
    const iree_tokenizer_regex_dfa_t* dfa, iree_string_view_t text,
    iree_tokenizer_regex_match_callback_fn_t callback, void* user_data);

// Counts matches without collecting them.
//
// More efficient than exec() with a counting callback when only the count
// is needed.
//
// |dfa| is the DFA to execute.
// |text| is the UTF-8 input text.
// |out_count| receives the number of matches found.
iree_status_t iree_tokenizer_regex_count_matches(
    const iree_tokenizer_regex_dfa_t* dfa, iree_string_view_t text,
    iree_host_size_t* out_count);

// Checks if the pattern matches anywhere in the text.
//
// Short-circuits on first match - more efficient than counting when you
// only need to know if any match exists.
//
// |dfa| is the DFA to execute.
// |text| is the UTF-8 input text.
// |out_has_match| receives true if at least one match was found.
iree_status_t iree_tokenizer_regex_has_match(
    const iree_tokenizer_regex_dfa_t* dfa, iree_string_view_t text,
    bool* out_has_match);

//===----------------------------------------------------------------------===//
// Binary Size Calculation
//===----------------------------------------------------------------------===//

// Calculates the minimum binary size for a DFA with the given parameters.
// Useful for pre-allocating buffers when building DFAs.
//
// |num_states| is the number of states.
// |flags| specifies which optional sections are included.
// |num_ranges| is the number of range transitions (0 if HAS_RANGES not set).
static inline iree_host_size_t iree_tokenizer_regex_dfa_binary_size(
    uint16_t num_states, iree_tokenizer_regex_dfa_flags_t flags,
    uint16_t num_ranges) {
  iree_host_size_t bitmap_qwords =
      iree_host_align(num_states, 64) / 64;  // ceil(num_states / 64)
  iree_host_size_t size = sizeof(iree_tokenizer_regex_dfa_header_t);
  size +=
      (iree_host_size_t)num_states * 256 * sizeof(uint16_t);  // Transitions.
  size += bitmap_qwords * sizeof(uint64_t);                   // Accept bitmap.
  if (flags & IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_LOOKAHEAD) {
    size += num_states * sizeof(iree_tokenizer_regex_lookahead_t);
  }
  if (flags & IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_ANCHORS) {
    size += 2 * bitmap_qwords * sizeof(uint64_t);  // Anchor bitmaps.
  }
  if (flags & IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_BRANCHES) {
    // alive_branches and accepting_branches arrays.
    size += 2 * (iree_host_size_t)num_states * sizeof(uint64_t);
  }
  if (flags & IREE_TOKENIZER_REGEX_DFA_FLAG_HAS_RANGES) {
    // Range header (num_ranges + reserved) + range array.
    size += sizeof(uint32_t);  // Header: uint16_t num_ranges + uint16_t pad.
    size += num_ranges * sizeof(iree_tokenizer_regex_dfa_range_transition_t);
  }
  return size;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_REGEX_EXEC_H_
