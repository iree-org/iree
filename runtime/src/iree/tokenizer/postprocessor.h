// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Post-processor for inserting special tokens around encoded sequences.
//
// The post-processor operates on token IDs produced by the model, inserting
// special tokens (BOS, EOS, CLS, SEP, etc.) at sequence boundaries. All
// HuggingFace post-processor types (TemplateProcessing, RobertaProcessing,
// ByteLevel, Sequence) compile down to a single flat precomputed
// representation at parse time.
//
// Design principles:
// - Precomputed: all template types compile to flat prefix/infix/suffix arrays
// - Cache-optimal: entire active template fits in one cache line (44 bytes)
// - Zero heap for data: template token IDs stored inline in the struct
// - Single codepath: no vtable dispatch, just array indexing

#ifndef IREE_TOKENIZER_POSTPROCESSOR_H_
#define IREE_TOKENIZER_POSTPROCESSOR_H_

#include "iree/base/api.h"
#include "iree/tokenizer/types.h"

typedef struct iree_tokenizer_vocab_t iree_tokenizer_vocab_t;

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Post-Processor Template
//===----------------------------------------------------------------------===//

// Maximum number of special tokens across prefix + infix + suffix.
// Real-world models use 1-4 total; 7 is generous for future models.
// Validated at construction time.
#define IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES 7

// Precomputed template for a single-sequence or pair-sequence encoding.
//
// Stores token IDs and type IDs inline in contiguous arrays, indexed by phase:
//   PREFIX: token_ids[0 .. prefix_count-1]
//   INFIX:  token_ids[prefix_count .. prefix_count+infix_count-1]
//   SUFFIX: token_ids[prefix_count+infix_count .. total-1]
//
// sizeof = 44 bytes. Fits in one cache line (64 bytes).
typedef struct iree_tokenizer_postprocessor_template_t {
  // Counts (accessed first during phase transitions).
  uint8_t prefix_count;
  uint8_t infix_count;
  uint8_t suffix_count;

  // Type IDs assigned to sequence tokens (not special tokens).
  // sequence_a_type_id applies to all model-produced tokens in sequence A.
  // sequence_b_type_id applies to all model-produced tokens in sequence B.
  uint8_t sequence_a_type_id;
  uint8_t sequence_b_type_id;

  // Padding to align token_ids to 4 bytes.
  uint8_t _pad[3];

  // Contiguous token IDs: [prefix...][infix...][suffix...].
  // Total entries = prefix_count + infix_count + suffix_count.
  iree_tokenizer_token_id_t token_ids[IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES];

  // Contiguous type IDs for special tokens (same layout as token_ids).
  // Each entry is the type_id/segment_id assigned to the corresponding
  // special token when written to output.
  uint8_t type_ids[IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES];
} iree_tokenizer_postprocessor_template_t;
static_assert(sizeof(iree_tokenizer_postprocessor_template_t) <= 64,
              "template must fit in one cache line");

// Returns the total number of special tokens in the template.
static inline uint8_t iree_tokenizer_postprocessor_template_total_count(
    const iree_tokenizer_postprocessor_template_t* t) {
  return t->prefix_count + t->infix_count + t->suffix_count;
}

//===----------------------------------------------------------------------===//
// Post-Processor
//===----------------------------------------------------------------------===//

// Flags controlling postprocessor behavior.
typedef uint32_t iree_tokenizer_postprocessor_flags_t;
enum iree_tokenizer_postprocessor_flags_e {
  IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE = 0,
  // Trim leading/trailing whitespace from token offsets (ByteLevel and RoBERTa
  // post-processor behavior).
  IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS = 1u << 0,
  // A prefix space was added during pre-tokenization. Affects offset trimming:
  // the first leading space of the first token is preserved rather than trimmed
  // (since it was artificially added and is part of the token's content).
  IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE = 1u << 1,
};

// Post-processor holding precomputed single and pair templates.
// The tokenizer selects one template at encode init time based on flags.
// This is a value type with no heap allocations; it can be embedded directly
// in larger structs (e.g., iree_tokenizer_t).
typedef struct iree_tokenizer_postprocessor_t {
  // Template for single-sequence encoding (always populated).
  iree_tokenizer_postprocessor_template_t single;

  // Template for pair-sequence encoding (zeroed if unsupported).
  // When pair.prefix_count + pair.infix_count + pair.suffix_count == 0 and
  // pair.sequence_a_type_id == 0 and pair.sequence_b_type_id == 0, pair
  // encoding is not supported.
  iree_tokenizer_postprocessor_template_t pair;

  // Behavioral flags (IREE_TOKENIZER_POSTPROCESSOR_FLAG_*).
  iree_tokenizer_postprocessor_flags_t flags;
} iree_tokenizer_postprocessor_t;

// Returns true if the postprocessor supports pair encoding.
static inline bool iree_tokenizer_postprocessor_supports_pair(
    const iree_tokenizer_postprocessor_t* postprocessor) {
  return iree_tokenizer_postprocessor_template_total_count(
             &postprocessor->pair) > 0 ||
         postprocessor->pair.sequence_a_type_id != 0 ||
         postprocessor->pair.sequence_b_type_id != 0;
}

// Initializes a postprocessor from precomputed template data.
// |single_template| is required and specifies the single-sequence template.
// |pair_template| is optional (NULL if pair encoding is not supported).
// |flags| controls offset trimming and prefix space behavior.
// Validates that total piece counts do not exceed MAX_PIECES.
// The postprocessor is caller-owned storage (stack or embedded in struct).
iree_status_t iree_tokenizer_postprocessor_initialize(
    const iree_tokenizer_postprocessor_template_t* single_template,
    const iree_tokenizer_postprocessor_template_t* pair_template,
    iree_tokenizer_postprocessor_flags_t flags,
    iree_tokenizer_postprocessor_t* out_postprocessor);

// Deinitializes a postprocessor. Currently a no-op (no heap resources), but
// provided for API symmetry and future-proofing.
void iree_tokenizer_postprocessor_deinitialize(
    iree_tokenizer_postprocessor_t* postprocessor);

//===----------------------------------------------------------------------===//
// Encode State
//===----------------------------------------------------------------------===//

// Phase of the postprocessor state machine during encoding.
// Tracks which portion of the template is being emitted relative to model
// tokens:
//   PREFIX → SEQUENCE_A → SUFFIX → DONE
// For pair encoding (future), the full sequence is:
//   PREFIX → SEQUENCE_A → INFIX → SEQUENCE_B → SUFFIX → DONE
typedef enum {
  IREE_TOKENIZER_POSTPROCESSOR_PHASE_IDLE = 0,
  IREE_TOKENIZER_POSTPROCESSOR_PHASE_PREFIX,
  IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A,
  IREE_TOKENIZER_POSTPROCESSOR_PHASE_SUFFIX,
  IREE_TOKENIZER_POSTPROCESSOR_PHASE_DONE,
} iree_tokenizer_postprocessor_phase_t;

// Per-encode postprocessor state. Embedded in the tokenizer's encode state.
// When phase is IDLE, all postprocessor operations are no-ops.
typedef struct iree_tokenizer_postprocessor_encode_state_t {
  // Active template for this encode (points into the postprocessor struct).
  const iree_tokenizer_postprocessor_template_t* active_template;
  // Current emission phase.
  iree_tokenizer_postprocessor_phase_t phase;
  // Index within current phase's token array (0..count-1).
  uint8_t position;
  // Cached flags from postprocessor (avoids pointer chasing on hot path).
  iree_tokenizer_postprocessor_flags_t flags;
  // True after the first model token has been processed for offset trimming.
  // Used to handle ADD_PREFIX_SPACE: only the very first model token across
  // the entire encode preserves its leading space.
  bool first_model_token_trimmed;
} iree_tokenizer_postprocessor_encode_state_t;

// Initializes encode state for a given postprocessor and template. If the
// template has no special tokens (total count == 0), the state remains IDLE
// (all operations are no-ops). The template pointer must remain valid for the
// lifetime of the encode state.
static inline void iree_tokenizer_postprocessor_encode_state_initialize(
    const iree_tokenizer_postprocessor_t* postprocessor,
    const iree_tokenizer_postprocessor_template_t* active_template,
    iree_tokenizer_postprocessor_encode_state_t* out_state) {
  memset(out_state, 0, sizeof(*out_state));
  // Cache flags for hot path access.
  out_state->flags = postprocessor->flags;
  uint8_t total =
      iree_tokenizer_postprocessor_template_total_count(active_template);
  if (total > 0) {
    out_state->active_template = active_template;
    out_state->phase = active_template->prefix_count > 0
                           ? IREE_TOKENIZER_POSTPROCESSOR_PHASE_PREFIX
                           : IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A;
  }
}

// Returns true if the postprocessor has pending tokens to emit (prefix or
// suffix phase not yet complete). Used by the tokenizer to report pending work
// in the streaming API.
static inline bool iree_tokenizer_postprocessor_encode_state_has_pending(
    const iree_tokenizer_postprocessor_encode_state_t* state) {
  return state->phase == IREE_TOKENIZER_POSTPROCESSOR_PHASE_PREFIX ||
         state->phase == IREE_TOKENIZER_POSTPROCESSOR_PHASE_SUFFIX;
}

//===----------------------------------------------------------------------===//
// Encode Operations
//===----------------------------------------------------------------------===//

// Emits prefix special tokens into the output if in PREFIX phase.
// Transitions to SEQUENCE_A when all prefix tokens are emitted.
// No-op if the phase is not PREFIX.
// Returns the number of tokens written to output.
iree_host_size_t iree_tokenizer_postprocessor_emit_prefix(
    iree_tokenizer_postprocessor_encode_state_t* state,
    iree_tokenizer_token_output_t output, iree_host_size_t output_offset);

// Assigns type_ids to model-produced tokens based on current sequence phase.
// Uses sequence_a_type_id during SEQUENCE_A (sequence_b_type_id for future pair
// encoding). No-op if type_ids output is NULL or phase is not a sequence phase.
void iree_tokenizer_postprocessor_assign_type_ids(
    const iree_tokenizer_postprocessor_encode_state_t* state,
    iree_tokenizer_token_output_t output, iree_host_size_t offset,
    iree_host_size_t count);

// Transitions from sequence phase to SUFFIX after model finalize.
// If suffix_count is 0, transitions directly to DONE.
// No-op if phase is not SEQUENCE_A.
static inline void iree_tokenizer_postprocessor_begin_suffix(
    iree_tokenizer_postprocessor_encode_state_t* state) {
  if (state->phase != IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A) return;
  state->position = 0;
  state->phase = state->active_template->suffix_count > 0
                     ? IREE_TOKENIZER_POSTPROCESSOR_PHASE_SUFFIX
                     : IREE_TOKENIZER_POSTPROCESSOR_PHASE_DONE;
}

// Emits suffix special tokens into the output if in SUFFIX phase.
// Transitions to DONE when all suffix tokens are emitted.
// No-op if the phase is not SUFFIX.
// Returns the number of tokens written to output.
iree_host_size_t iree_tokenizer_postprocessor_emit_suffix(
    iree_tokenizer_postprocessor_encode_state_t* state,
    iree_tokenizer_token_output_t output, iree_host_size_t output_offset);

//===----------------------------------------------------------------------===//
// Offset Trimming
//===----------------------------------------------------------------------===//

// Trims leading and trailing whitespace from token offsets when
// TRIM_OFFSETS is set in state->flags (ByteLevel and RoBERTa behavior).
//
// Token offsets point into the original input text. When trim_offsets is
// enabled, offsets are adjusted to exclude whitespace that was part of the
// token's encoding but not its semantic content. For ByteLevel tokenizers,
// the Ġ character (U+0120) represents an original space and counts as
// leading whitespace for trimming purposes.
//
// |state| provides cached TRIM_OFFSETS/ADD_PREFIX_SPACE flags and tracks
// whether the first model token has been processed (for streaming).
// |vocab| is required to look up token text for whitespace detection.
// |output| contains the token IDs and offsets to process.
// |model_token_start| is the index of the first model token (after any prefix
// special tokens which have zero offsets and should not be trimmed).
// |model_token_count| is the number of model-produced tokens to process.
//
// No-op if TRIM_OFFSETS is not set in state->flags or output.token_offsets is
// NULL.
void iree_tokenizer_postprocessor_trim_token_offsets(
    iree_tokenizer_postprocessor_encode_state_t* state,
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_token_output_t output,
    iree_host_size_t model_token_start, iree_host_size_t model_token_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_POSTPROCESSOR_H_
