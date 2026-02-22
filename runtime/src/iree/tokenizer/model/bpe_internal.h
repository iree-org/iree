// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for BPE model types and inline functions.
// Used by bpe.c and its split-out implementation files.

#ifndef IREE_TOKENIZER_MODEL_BPE_INTERNAL_H_
#define IREE_TOKENIZER_MODEL_BPE_INTERNAL_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/byte_level_tables.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/model/bpe_heap.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Sliding Window Token Entry
//===----------------------------------------------------------------------===//

// Entry in the sliding window representing a token being built.
// Stores the token ID and the byte range it covers in the input segment.
typedef struct iree_tokenizer_bpe_window_token_t {
  // The token ID (-1 for invalid/removed slots).
  int32_t token_id;
  // Starting byte position in segment (inclusive).
  uint32_t start_byte;
  // Ending byte position (exclusive).
  uint32_t end_byte;
} iree_tokenizer_bpe_window_token_t;

// Inverse merge entry: for a token formed by a merge, stores its constituents.
// Base tokens (single-byte, added, not from any merge) store (self, self).
typedef struct iree_tokenizer_bpe_split_entry_t {
  uint32_t left_id;
  uint32_t right_id;
} iree_tokenizer_bpe_split_entry_t;

// Backtracking stack entry: token ID and the byte position where it starts.
// Bit-packed backtrack entry to maintain 8-byte size while supporting large
// vocabs. start_byte is bounded by max_backtrack_segment_bytes (<=4095), so
// we use 12 bits for it and 20 bits for deferred_merge_rank (supports 1M
// tokens, covering all realistic vocabularies).
#define IREE_TOKENIZER_BPE_BACKTRACK_START_BYTE_BITS 12
#define IREE_TOKENIZER_BPE_BACKTRACK_START_BYTE_MASK \
  ((1u << IREE_TOKENIZER_BPE_BACKTRACK_START_BYTE_BITS) - 1)
#define IREE_TOKENIZER_BPE_BACKTRACK_DEFERRED_RANK_BITS 20
#define IREE_TOKENIZER_BPE_BACKTRACK_DEFERRED_RANK_MAX \
  ((1u << IREE_TOKENIZER_BPE_BACKTRACK_DEFERRED_RANK_BITS) - 1)

typedef struct iree_tokenizer_bpe_backtrack_entry_t {
  int32_t token_id;
  // Packed field: lower 12 bits = start_byte, upper 20 bits = deferred_rank.
  // The deferred_merge_rank is the effective_rank that was active when this
  // token was pushed. It tells pair validation at the next position which
  // merges were intentionally deferred due to suffix blocking.
  uint32_t start_byte_and_deferred_rank;
} iree_tokenizer_bpe_backtrack_entry_t;

static inline uint32_t iree_tokenizer_bpe_backtrack_entry_start_byte(
    const iree_tokenizer_bpe_backtrack_entry_t* entry) {
  return entry->start_byte_and_deferred_rank &
         IREE_TOKENIZER_BPE_BACKTRACK_START_BYTE_MASK;
}

static inline uint32_t iree_tokenizer_bpe_backtrack_entry_deferred_rank(
    const iree_tokenizer_bpe_backtrack_entry_t* entry) {
  return entry->start_byte_and_deferred_rank >>
         IREE_TOKENIZER_BPE_BACKTRACK_START_BYTE_BITS;
}

static inline void iree_tokenizer_bpe_backtrack_entry_set(
    iree_tokenizer_bpe_backtrack_entry_t* entry, int32_t token_id,
    uint32_t start_byte, uint32_t deferred_rank) {
  entry->token_id = token_id;
  // Cap deferred_rank at 20-bit max (supports vocabs up to 1M tokens).
  if (deferred_rank > IREE_TOKENIZER_BPE_BACKTRACK_DEFERRED_RANK_MAX) {
    deferred_rank = IREE_TOKENIZER_BPE_BACKTRACK_DEFERRED_RANK_MAX;
  }
  entry->start_byte_and_deferred_rank =
      (start_byte & IREE_TOKENIZER_BPE_BACKTRACK_START_BYTE_MASK) |
      (deferred_rank << IREE_TOKENIZER_BPE_BACKTRACK_START_BYTE_BITS);
}

//===----------------------------------------------------------------------===//
// Backtrack Tables (Model-Side)
//===----------------------------------------------------------------------===//

// Precomputed tables for the O(n) backtracking algorithm.
// Built once during model allocation, read-only after construction.
// All tables are indexed by token ID and sized to vocab_capacity.
typedef struct iree_tokenizer_bpe_backtrack_tables_t {
  // Single allocation for all tables below.
  void* slab;
  iree_tokenizer_bpe_split_entry_t* split_table;  // Inverse merge table.
  // Longest proper prefix token, or UINT32_MAX.
  uint32_t* next_prefix_match;
  uint32_t* effective_rank;   // 0 = non-participant, >0 = BPE-reachable.
  uint64_t* token_reachable;  // Precomputed reachability bits.
} iree_tokenizer_bpe_backtrack_tables_t;

//===----------------------------------------------------------------------===//
// Backtrack State (State-Side)
//===----------------------------------------------------------------------===//

// Runtime state for the O(n) backtracking algorithm.
// Modified during BACKTRACK/BACKTRACK_EMIT phases.
typedef struct iree_tokenizer_bpe_backtrack_state_t {
  iree_host_size_t stack_count;  // Tokens in the stack.
  // Next token to emit from stack.
  iree_host_size_t emit_index;
  // Dirty mask for lazy bitfield reset. Each bit corresponds to a uint64_t
  // word in the backtrack bitfield that was modified (cleared) during the
  // previous segment's backtracking. On segment start, only these words are
  // reset to UINT64_MAX, giving O(backtracks) init instead of O(capacity).
  uint64_t dirty_mask;
} iree_tokenizer_bpe_backtrack_state_t;

//===----------------------------------------------------------------------===//
// Window State (State-Side)
//===----------------------------------------------------------------------===//

// Runtime state for the sliding window used in the O(n log L) algorithm.
// Modified during BYTE_LOOP/FLUSH phases.
typedef struct iree_tokenizer_bpe_window_state_t {
  iree_host_size_t count;  // Tokens currently in window.
  iree_host_size_t start;  // Circular buffer head index.
} iree_tokenizer_bpe_window_state_t;

//===----------------------------------------------------------------------===//
// BPE Model Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_bpe_model_t {
  iree_tokenizer_model_t base;
  iree_allocator_t allocator;
  const iree_tokenizer_vocab_t* vocab;  // NOT owned.

  // Prefix trie for token lookup (owned).
  iree_tokenizer_vocab_trie_t* trie;

  // Merge hash table for O(1) merge lookup (owned).
  iree_tokenizer_vocab_merge_hash_t* merge_hash;

  // Configuration flags (see iree_tokenizer_bpe_flag_bits_t).
  iree_tokenizer_bpe_flags_t flags;

  // End-of-word suffix (e.g., "</w>" for CLIP).
  // Empty if not used. Appended to each segment before tokenization.
  char end_of_word_suffix[16];
  iree_host_size_t end_of_word_suffix_length;

  // Byte-to-token lookup table.
  // Maps each byte (0-255) to its single-byte token ID, or -1 if none.
  // For ByteLevel mode, this maps the UTF-8 encoding of the ByteLevel
  // codepoint.
  int32_t byte_to_token[256];

  // Precomputed byte fallback token IDs for each byte value.
  // Maps byte N to the token ID for "<0xNN>" (the byte-level fallback token).
  // -1 if no such token exists. Built during model construction to avoid
  // snprintf + vocab hash lookup per unknown byte during encoding.
  int32_t byte_fallback_token[256];

  // Precomputed buffer capacities (based on max_token_length).
  // window_capacity is rounded up to power of 2 for fast modulo via bitmask.
  iree_host_size_t window_capacity;
  iree_host_size_t window_capacity_mask;  // = window_capacity - 1
  iree_host_size_t heap_capacity;
  iree_host_size_t max_token_length;

  // Offsets into state slab for buffer access (window+heap path).
  iree_host_size_t window_offset;
  iree_host_size_t heap_offset;

  // Backtracking algorithm tables (owned).
  // Each table is indexed by token ID and sized to vocab_capacity.
  iree_tokenizer_bpe_backtrack_tables_t backtrack_tables;
  iree_host_size_t vocab_capacity;

  // Backtracking encode configuration and state buffer offsets.
  // Segments <= max_backtrack_segment_bytes use the backtracking path;
  // longer segments fall back to the window+heap path.
  iree_host_size_t max_backtrack_segment_bytes;
  iree_host_size_t backtrack_stack_capacity;
  iree_host_size_t backtrack_bitfield_capacity;  // In uint64_t units.
  iree_host_size_t backtrack_stack_offset;
  iree_host_size_t backtrack_bitfield_offset;

  // Word cache capacity (power of 2, or 0 if disabled).
  // Caching exploits Zipf's law: frequent words dominate token positions.
  // Only useful with sufficient vocabulary diversity (>= 256 tokens).
  iree_host_size_t cache_capacity;
  iree_host_size_t cache_capacity_mask;  // = cache_capacity - 1, or 0.
  iree_host_size_t cache_offset;
} iree_tokenizer_bpe_model_t;

//===----------------------------------------------------------------------===//
// Segment Context (State-Side)
//===----------------------------------------------------------------------===//

// Context for the segment currently being encoded.
// All three fields are set at segment start and used together for offset
// calculations. Grouping them makes the relationship explicit and reduces
// the chance of updating one without the others.
typedef struct iree_tokenizer_bpe_segment_context_t {
  // Offset of segment in transform buffer.
  iree_host_size_t base_offset;
  // Size before suffix appending (for clamping).
  iree_host_size_t original_size;
  // Current byte within segment (resumption).
  iree_host_size_t byte_position;
} iree_tokenizer_bpe_segment_context_t;

//===----------------------------------------------------------------------===//
// BPE State Structure
//===----------------------------------------------------------------------===//

// Encoding phase state machine.
// Transitions: SEGMENT_START -> BYTE_LOOP -> FLUSH -> SEGMENT_START (next
// segment)
//              SEGMENT_START -> FAST_PATH_PENDING (if output full on fast path)
//              FAST_PATH_PENDING -> SEGMENT_START (after emitting)
typedef enum iree_tokenizer_bpe_phase_e {
  // Ready to start a new segment. Check for fast-path match first.
  IREE_TOKENIZER_BPE_PHASE_SEGMENT_START = 0,
  // Fast-path token matched but couldn't emit (output was full).
  // Next call should emit fast_path_pending_token_id and complete segment.
  IREE_TOKENIZER_BPE_PHASE_FAST_PATH_PENDING,
  // Processing bytes: adding tokens to window, applying merges, emitting
  // frozen. segment.byte_position tracks progress.
  IREE_TOKENIZER_BPE_PHASE_BYTE_LOOP,
  // All bytes processed. Applying final merges and emitting remaining tokens.
  IREE_TOKENIZER_BPE_PHASE_FLUSH,
  // Backtracking path: running the O(n) algorithm on a segment.
  IREE_TOKENIZER_BPE_PHASE_BACKTRACK,
  // Backtracking path: emitting tokens from the completed stack.
  IREE_TOKENIZER_BPE_PHASE_BACKTRACK_EMIT,
} iree_tokenizer_bpe_phase_t;

// BPE encoding state with trailing window and heap buffers.
// The total allocation size is model->base.state_size bytes.
// Memory layout:
//   [iree_tokenizer_bpe_state_t struct]
//   [window_tokens array]
//   [heap_entries array]
typedef struct iree_tokenizer_bpe_state_t {
  iree_tokenizer_model_state_t base;

  // Current encoding phase (see iree_tokenizer_bpe_phase_t).
  iree_tokenizer_bpe_phase_t phase;

  // Sliding window: circular buffer of tokens being built.
  iree_tokenizer_bpe_window_state_t window;

  // Min-heap of merge candidates ordered by rank.
  iree_tokenizer_bpe_heap_t heap;

  // Current segment context (offset, size, position).
  iree_tokenizer_bpe_segment_context_t segment;

  // Token ID waiting to emit (used in FAST_PATH_PENDING phase).
  int32_t fast_path_pending_token_id;

  // Last emitted token ID, used for UNK fusing across calls.
  int32_t last_emitted_token_id;

  // Backtracking state (used in BACKTRACK/BACKTRACK_EMIT phases).
  iree_tokenizer_bpe_backtrack_state_t backtrack;

  // Trailing buffers (accessed via model offsets):
  // - iree_tokenizer_bpe_window_token_t window_tokens[window_capacity]
  // - iree_tokenizer_bpe_heap_entry_t heap_entries[heap_capacity]
  // - iree_tokenizer_bpe_backtrack_entry_t backtrack_stack[stack_capacity]
  // - uint64_t backtrack_bitfield[bitfield_capacity]
  // - iree_tokenizer_bpe_cache_entry_t cache_entries[cache_capacity]
  // - char suffixed_buffer[suffixed_buffer_capacity]
} iree_tokenizer_bpe_state_t;

//===----------------------------------------------------------------------===//
// Whole-Word Cache
//===----------------------------------------------------------------------===//

// Direct-mapped cache for repeated segment tokenization results.
// Segments that produce the same byte sequence will hash to the same slot
// and return cached token IDs, skipping all trie/backtracking work.
//
// Capacity is a power of 2 for fast modulo via bitmask. 512 entries provides
// 60-68% hit rate on English prose and code (whitespace-segmented words follow
// Zipf's law — a small set of frequent words dominates token positions).

#define IREE_TOKENIZER_BPE_CACHE_CAPACITY 512

// Minimum vocabulary size for caching to be worthwhile. Below this threshold,
// there aren't enough distinct token sequences for the direct-mapped cache to
// achieve meaningful hit rates — the vocabulary simply can't produce the
// variety of multi-token word tokenizations that caching accelerates.
#define IREE_TOKENIZER_BPE_CACHE_MIN_VOCAB_SIZE 256

// Maximum segment byte length that can be cached. Segments longer than this
// bypass the cache (they're rare enough not to affect overall hit rate).
// 32 bytes covers identifiers and words up to the 99.9th percentile of
// English prose and code (mean ~5 bytes, long identifiers ~20-30 bytes).
#define IREE_TOKENIZER_BPE_CACHE_MAX_KEY_BYTES 32

// Maximum token count per cached segment. Segments producing more tokens than
// this are not cached (rare for natural language — most words produce 1-5
// tokens, long/rare words up to ~10).
#define IREE_TOKENIZER_BPE_CACHE_MAX_TOKENS 10

typedef struct iree_tokenizer_bpe_cache_entry_t {
  // FNV-1a hash of segment bytes (0 = empty slot).
  uint32_t key_hash;
  // Byte length of the cached segment.
  uint16_t key_length;
  // Number of tokens produced for this segment.
  uint16_t token_count;
  uint8_t key[IREE_TOKENIZER_BPE_CACHE_MAX_KEY_BYTES];
  iree_tokenizer_token_id_t tokens[IREE_TOKENIZER_BPE_CACHE_MAX_TOKENS];
} iree_tokenizer_bpe_cache_entry_t;
// Size: 4 + 2 + 2 + 32 + 40 = 80 bytes per entry (1.25 cache lines).
// Total: 512 * 80 = 40KB.

// Word-at-a-time hash with Murmur3 finalizer for cache keys.
// Processes 8 bytes per iteration (1 load + 1 XOR + 1 multiply), finishing
// with the Murmur3 64-bit finalizer for thorough bit mixing. For a typical
// 8-byte word this is ~7 ALU ops total vs FNV-1a's 16 per-byte ops.
// Returns non-zero (0 is the empty-slot sentinel).
static inline uint32_t iree_tokenizer_bpe_cache_hash(const uint8_t* data,
                                                     iree_host_size_t length) {
  uint64_t accumulator = length;
  iree_host_size_t offset = 0;
  // Process 8-byte chunks.
  while (offset + 8 <= length) {
    uint64_t word;
    memcpy(&word, data + offset, 8);
    accumulator ^= word;
    accumulator *= 0x9E3779B97F4A7C15ULL;
    offset += 8;
  }
  // Handle remaining 1-7 bytes.
  if (offset < length) {
    uint64_t tail = 0;
    memcpy(&tail, data + offset, length - offset);
    accumulator ^= tail;
  }
  // Murmur3 64-bit finalizer.
  accumulator ^= accumulator >> 33;
  accumulator *= 0xff51afd7ed558ccdULL;
  accumulator ^= accumulator >> 33;
  accumulator *= 0xc4ceb9fe1a85ec53ULL;
  accumulator ^= accumulator >> 33;
  uint32_t result = (uint32_t)accumulator;
  return result ? result : 1;
}

//===----------------------------------------------------------------------===//
// State Buffer Layout
//===----------------------------------------------------------------------===//
//
// The state struct has a trailing allocation with this layout:
//
// ┌───────────────────────────────────────┐
// │ iree_tokenizer_bpe_state_t struct    │  (sizeof(bpe_state_t))
// ├───────────────────────────────────────┤
// │ window_tokens[window_capacity]        │  (model->window_offset)
// ├───────────────────────────────────────┤
// │ heap_entries[heap_capacity]           │  (model->heap_offset)
// ├───────────────────────────────────────┤
// │ backtrack_stack[stack_capacity]       │  (model->backtrack_stack_offset)
// ├───────────────────────────────────────┤
// │ backtrack_bitfield[bitfield_capacity] │ (model->backtrack_bitfield_offset)
// ├───────────────────────────────────────┤
// │ cache_entries[cache_capacity]         │  (model->cache_offset, if enabled)
// └───────────────────────────────────────┘
//
// Total size = model->base.state_size
//
// INVARIANT: All offsets must satisfy:
//   offset + element_count * element_size <= model->base.state_size
//
// Layout is computed in bpe.c:iree_tokenizer_bpe_model_allocate().
//
//===----------------------------------------------------------------------===//
// State Buffer Accessors
//===----------------------------------------------------------------------===//

static inline iree_tokenizer_bpe_window_token_t*
iree_tokenizer_bpe_state_window(iree_tokenizer_bpe_state_t* state,
                                const iree_tokenizer_bpe_model_t* model) {
  return (iree_tokenizer_bpe_window_token_t*)((uint8_t*)state +
                                              model->window_offset);
}

static inline iree_tokenizer_bpe_heap_entry_t* iree_tokenizer_bpe_state_heap(
    iree_tokenizer_bpe_state_t* state,
    const iree_tokenizer_bpe_model_t* model) {
  return (iree_tokenizer_bpe_heap_entry_t*)((uint8_t*)state +
                                            model->heap_offset);
}

static inline iree_tokenizer_bpe_backtrack_entry_t*
iree_tokenizer_bpe_state_backtrack_stack(
    iree_tokenizer_bpe_state_t* state,
    const iree_tokenizer_bpe_model_t* model) {
  return (iree_tokenizer_bpe_backtrack_entry_t*)((uint8_t*)state +
                                                 model->backtrack_stack_offset);
}

static inline uint64_t* iree_tokenizer_bpe_state_backtrack_bitfield(
    iree_tokenizer_bpe_state_t* state,
    const iree_tokenizer_bpe_model_t* model) {
  return (uint64_t*)((uint8_t*)state + model->backtrack_bitfield_offset);
}

static inline iree_tokenizer_bpe_cache_entry_t* iree_tokenizer_bpe_state_cache(
    iree_tokenizer_bpe_state_t* state,
    const iree_tokenizer_bpe_model_t* model) {
  return (iree_tokenizer_bpe_cache_entry_t*)((uint8_t*)state +
                                             model->cache_offset);
}

//===----------------------------------------------------------------------===//
// Window Helpers
//===----------------------------------------------------------------------===//

static inline iree_host_size_t iree_tokenizer_bpe_window_index(
    iree_tokenizer_bpe_state_t* state, const iree_tokenizer_bpe_model_t* model,
    iree_host_size_t logical_index) {
  return (state->window.start + logical_index) & model->window_capacity_mask;
}

static inline iree_tokenizer_bpe_window_token_t* iree_tokenizer_bpe_window_at(
    iree_tokenizer_bpe_state_t* state, const iree_tokenizer_bpe_model_t* model,
    iree_host_size_t logical_index) {
  iree_tokenizer_bpe_window_token_t* window =
      iree_tokenizer_bpe_state_window(state, model);
  return &window[iree_tokenizer_bpe_window_index(state, model, logical_index)];
}

//===----------------------------------------------------------------------===//
// Output Cursor
//===----------------------------------------------------------------------===//

// Tracks position and remaining capacity in the output buffer.
// When offset_ptr is non-NULL, byte ranges are written for each emitted token.
typedef struct iree_tokenizer_bpe_output_cursor_t {
  iree_tokenizer_token_id_t* ptr;  // Next write position.
  // Next offset write position (or NULL).
  iree_tokenizer_offset_t* offset_ptr;
  // Added to per-token byte positions.
  iree_host_size_t segment_base_offset;
  iree_host_size_t remaining;  // Slots available.
} iree_tokenizer_bpe_output_cursor_t;

// Creates an output cursor for a segment at |segment_base_offset| within the
// transform buffer. Pass NULL for |offsets| to skip offset tracking.
static inline iree_tokenizer_bpe_output_cursor_t
iree_tokenizer_bpe_output_cursor_make(iree_tokenizer_token_id_t* base,
                                      iree_tokenizer_offset_t* offsets,
                                      iree_host_size_t segment_base_offset,
                                      iree_host_size_t capacity) {
  iree_tokenizer_bpe_output_cursor_t cursor = {base, offsets,
                                               segment_base_offset, capacity};
  return cursor;
}

// Returns the number of tokens written so far.
static inline iree_host_size_t iree_tokenizer_bpe_output_cursor_count(
    const iree_tokenizer_bpe_output_cursor_t* cursor,
    const iree_tokenizer_token_id_t* base) {
  return (iree_host_size_t)(cursor->ptr - base);
}

// Emits a token to the output buffer with its byte range within the segment.
// The offset written is {segment_base_offset + start_byte,
//                        segment_base_offset + end_byte}.
// Returns false if buffer is full.
static inline bool iree_tokenizer_bpe_emit(
    iree_tokenizer_bpe_output_cursor_t* cursor,
    iree_tokenizer_token_id_t token_id, uint32_t start_byte,
    uint32_t end_byte) {
  if (cursor->remaining == 0) return false;
  *cursor->ptr++ = token_id;
  if (cursor->offset_ptr) {
    cursor->offset_ptr->start = cursor->segment_base_offset + start_byte;
    cursor->offset_ptr->end = cursor->segment_base_offset + end_byte;
    cursor->offset_ptr++;
  }
  --cursor->remaining;
  return true;
}

// Emits a token with byte range and updates last_emitted_token_id.
// Returns false if full.
static inline bool iree_tokenizer_bpe_emit_and_track(
    iree_tokenizer_bpe_state_t* state,
    iree_tokenizer_bpe_output_cursor_t* cursor,
    iree_tokenizer_token_id_t token_id, uint32_t start_byte,
    uint32_t end_byte) {
  if (!iree_tokenizer_bpe_emit(cursor, token_id, start_byte, end_byte))
    return false;
  state->last_emitted_token_id = token_id;
  return true;
}

//===----------------------------------------------------------------------===//
// Bitfield Helpers
//===----------------------------------------------------------------------===//

// Bitfield helpers: the bitfield uses one bit per byte position.
// Initialized to all-ones (optimistic: all positions assumed reachable).
// Bits are cleared when backtracking proves a position is unreachable.

static inline bool iree_tokenizer_bpe_bitfield_is_set(
    const uint64_t* bitfield, iree_host_size_t position) {
  return (bitfield[position / 64] >> (position % 64)) & 1;
}

static inline void iree_tokenizer_bpe_bitfield_clear(
    uint64_t* bitfield, uint64_t* dirty_mask, iree_host_size_t position) {
  iree_host_size_t word_index = position / 64;
  bitfield[word_index] &= ~((uint64_t)1 << (position % 64));
  *dirty_mask |= (uint64_t)1 << word_index;
}

//===----------------------------------------------------------------------===//
// Shared Trie Helpers
//===----------------------------------------------------------------------===//

// Advances a trie cursor by a single raw byte, handling ByteLevel conversion.
// Returns true if the advance succeeded (cursor is at a valid trie node).
static inline bool iree_tokenizer_bpe_trie_advance_byte(
    iree_tokenizer_trie_cursor_t* cursor, uint8_t raw_byte, bool byte_level) {
  if (!byte_level || (raw_byte >= 0x21 && raw_byte <= 0x7E)) {
    return iree_tokenizer_trie_cursor_advance(cursor, raw_byte);
  }
  // ByteLevel non-ASCII: convert to 2-byte UTF-8.
  const iree_tokenizer_byte_level_utf8_t* utf8 =
      &iree_tokenizer_byte_level_utf8[raw_byte];
  if (!iree_tokenizer_trie_cursor_advance(cursor, utf8->bytes[0])) {
    return false;
  }
  if (utf8->length > 1 &&
      !iree_tokenizer_trie_cursor_advance(cursor, utf8->bytes[1])) {
    return false;
  }
  return true;
}

// Finds the longest token matching raw input bytes starting at data[0..size].
// For ByteLevel mode, each raw byte is converted to its ByteLevel UTF-8 form
// (1-2 bytes) before feeding to the trie cursor.
// Returns the matched token_id and the count of raw bytes consumed.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_tokenizer_bpe_backtrack_longest_match(
    const iree_tokenizer_bpe_model_t* model, const uint8_t* data,
    iree_host_size_t size, int32_t* out_token_id,
    iree_host_size_t* out_raw_length) {
  *out_token_id = -1;
  *out_raw_length = 0;

  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  const bool byte_level =
      iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT);

  for (iree_host_size_t raw_consumed = 0; raw_consumed < size; ++raw_consumed) {
    uint8_t byte = data[raw_consumed];
    if (!byte_level || (byte >= 0x21 && byte <= 0x7E)) {
      // Non-ByteLevel or printable ASCII (0x21-0x7E): the ByteLevel mapping is
      // identity for this range (codepoint == byte, single-byte UTF-8), so we
      // advance the trie cursor directly with the raw byte.
      if (!iree_tokenizer_trie_cursor_advance(&cursor, byte)) {
        break;
      }
    } else {
      // Non-printable or high byte: use pre-computed UTF-8 from lookup table.
      // All non-identity ByteLevel mappings produce 2-byte UTF-8.
      const iree_tokenizer_byte_level_utf8_t* utf8 =
          &iree_tokenizer_byte_level_utf8[byte];
      if (!iree_tokenizer_trie_cursor_advance(&cursor, utf8->bytes[0])) {
        break;
      }
      if (utf8->length == 2 &&
          !iree_tokenizer_trie_cursor_advance(&cursor, utf8->bytes[1])) {
        break;
      }
    }

    // Check if current position is a token boundary (greedy longest match).
    // Only accept tokens reachable via BPE merge rules:
    //   - Single raw byte tokens are base vocabulary (always valid).
    //   - Multi-byte tokens must participate in the BPE merge system
    //     (effective_rank > 0): either produced by a merge, or a base token
    //     that serves as a left/right component in at least one merge.
    // This prevents added tokens (e.g., "<|endoftext|>") that exist in the
    // vocabulary but don't participate in BPE from being incorrectly matched.
    int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
    if (token_id >= 0) {
      iree_host_size_t candidate_raw_length = raw_consumed + 1;
      if (candidate_raw_length == 1 ||
          model->backtrack_tables.effective_rank[(uint32_t)token_id] > 0) {
        *out_token_id = token_id;
        *out_raw_length = candidate_raw_length;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Table Construction (bpe_tables.c)
//===----------------------------------------------------------------------===//

// Builds the backtracking tables (split_table, effective_rank,
// token_reachable). Called during model allocation.
iree_status_t iree_tokenizer_bpe_build_backtrack_tables(
    iree_tokenizer_bpe_model_t* model);

// Checks if a token is reachable from its input bytes via BPE merge ordering.
// Uses the precomputed token_reachable bitmap from model load.
bool iree_tokenizer_bpe_is_first_token_reachable(
    const iree_tokenizer_bpe_model_t* model, uint32_t token);

// Checks if a specific decomposition (left + right) of a token is reachable.
// Used by pair validation during backtracking.
bool iree_tokenizer_bpe_is_decomposition_reachable(
    const iree_tokenizer_bpe_model_t* model, uint32_t left, uint32_t right,
    uint32_t token);

// Returns the rightmost leaf (base) token of the split tree rooted at token.
// For a base token (split_table[token] = {token, token}), returns itself.
uint32_t iree_tokenizer_bpe_rightmost_base_token(
    const iree_tokenizer_bpe_model_t* model, uint32_t token);

// Returns the effective_rank when the rightmost boundary base token is first
// consumed by a merge within the subtree rooted at |token|.
// For base tokens, returns UINT32_MAX (never consumed internally).
uint32_t iree_tokenizer_bpe_right_boundary_consumed_rank(
    const iree_tokenizer_bpe_model_t* model, uint32_t token);

//===----------------------------------------------------------------------===//
// Backtracking Algorithm (bpe_backtrack.c)
//===----------------------------------------------------------------------===//

// Runs the O(n) backtracking BPE encode algorithm on a segment.
// Populates the backtrack_stack with the resulting token sequence.
void iree_tokenizer_bpe_backtrack_encode(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    const uint8_t* segment_data, iree_host_size_t segment_size,
    const char* suffix_data, iree_host_size_t suffix_length);

//===----------------------------------------------------------------------===//
// Window/Heap Operations (bpe_window.c)
//===----------------------------------------------------------------------===//

// Pushes a token onto the window.
void iree_tokenizer_bpe_window_push(iree_tokenizer_bpe_state_t* state,
                                    const iree_tokenizer_bpe_model_t* model,
                                    iree_tokenizer_bpe_window_token_t token);

// Pops and returns the front token from the window.
iree_tokenizer_bpe_window_token_t iree_tokenizer_bpe_window_pop_front(
    iree_tokenizer_bpe_state_t* state, const iree_tokenizer_bpe_model_t* model);

// Adds a merge candidate to the heap if a merge exists for the pair.
void iree_tokenizer_bpe_maybe_add_merge(iree_tokenizer_bpe_state_t* state,
                                        const iree_tokenizer_bpe_model_t* model,
                                        iree_host_size_t position);

// Applies all valid merges from the heap until no valid merges remain.
void iree_tokenizer_bpe_apply_pending_merges(
    iree_tokenizer_bpe_state_t* state, const iree_tokenizer_bpe_model_t* model);

// Emits all frozen tokens from the window front.
// Returns false if output fills before all frozen tokens are emitted.
bool iree_tokenizer_bpe_emit_frozen_tokens(
    iree_tokenizer_bpe_state_t* state, const iree_tokenizer_bpe_model_t* model,
    iree_host_size_t current_byte_position,
    iree_tokenizer_bpe_output_cursor_t* cursor);

//===----------------------------------------------------------------------===//
// Cache and Suffix Operations (bpe_cache.c)
//===----------------------------------------------------------------------===//

// Attempts to serve a segment from the word cache.
// Returns true if cache hit and all tokens emitted.
bool iree_tokenizer_bpe_cache_lookup(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment, iree_tokenizer_bpe_output_cursor_t* cursor);

// Populates a word cache entry from the backtrack stack after encoding.
void iree_tokenizer_bpe_cache_populate(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment,
    const iree_tokenizer_bpe_backtrack_entry_t* stack,
    iree_host_size_t token_count);

// Finds the longest token matching segment + suffix bytes via trie lookup.
// Used by both backtracking and encode state machine for end-of-word handling.
void iree_tokenizer_bpe_trie_longest_match_with_suffix(
    const iree_tokenizer_bpe_model_t* model, iree_string_view_t segment,
    iree_string_view_t suffix, int32_t* out_token_id,
    iree_host_size_t* out_length);

// Finds the longest token with ByteLevel transformation applied.
void iree_tokenizer_bpe_trie_longest_match_byte_level(
    const iree_tokenizer_bpe_model_t* model, iree_string_view_t segment,
    int32_t* out_token_id, iree_host_size_t* out_length);

// Applies end-of-word suffix to the last token in the backtrack stack.
void iree_tokenizer_bpe_apply_suffix_to_backtrack(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment);

// Applies end-of-word suffix to the last token in the sliding window.
void iree_tokenizer_bpe_apply_suffix_to_last_window_token(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment);

//===----------------------------------------------------------------------===//
// Encoding State Machine (bpe_encode.c)
//===----------------------------------------------------------------------===//

// Encodes segments to tokens using the resumable state machine.
// These functions implement the iree_tokenizer_model_vtable_t interface.
iree_status_t iree_tokenizer_bpe_state_encode(
    iree_tokenizer_model_state_t* state,
    iree_const_byte_span_t transform_buffer,
    iree_tokenizer_segment_list_t segments,
    iree_tokenizer_token_output_t output,
    iree_host_size_t* out_segments_consumed, iree_host_size_t* out_token_count);

iree_status_t iree_tokenizer_bpe_state_finalize(
    iree_tokenizer_model_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* out_token_count);

bool iree_tokenizer_bpe_state_has_pending(
    const iree_tokenizer_model_state_t* state);

iree_host_size_t iree_tokenizer_bpe_state_reclaim(
    iree_tokenizer_model_state_t* state);

//===----------------------------------------------------------------------===//
// Shared Utilities (bpe.c)
//===----------------------------------------------------------------------===//

// Handles a byte that could not be encoded via the trie.
// Returns the token ID for the byte (fallback, UNK, or -1 for FUSE_UNK).
int32_t iree_tokenizer_bpe_handle_unknown_byte(
    const iree_tokenizer_bpe_model_t* model, uint8_t byte,
    int32_t last_emitted_token_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_MODEL_BPE_INTERNAL_H_
