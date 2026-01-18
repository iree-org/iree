// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Token post-processors for special token insertion.
//
// Post-processors control how special tokens (BOS, EOS, CLS, SEP) are inserted
// during encoding. This implements the HuggingFace tokenizers post_processor
// field behavior:
//
// - null/missing: No special tokens added (GPT-2 style)
// - TemplateProcessing: Configurable template-based insertion (BERT, LLaMA)
// - RobertaProcessing: RoBERTa-specific handling
// - Sequence: Chain multiple processors (Llama 3 style)
//
// Usage:
//   iree_tokenizer_postprocessor_t pp;
//   iree_tokenizer_postprocessor_initialize_none(&pp);
//   // ... or parse from JSON ...
//   iree_tokenizer_postprocessor_apply_single(&pp, text, encode_fn, emit_fn,
//   ctx); iree_tokenizer_postprocessor_deinitialize(&pp);

#ifndef IREE_TOKENIZER_POSTPROCESSOR_H_
#define IREE_TOKENIZER_POSTPROCESSOR_H_

#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Postprocessor Types
//===----------------------------------------------------------------------===//

// Post-processor algorithm type.
typedef enum iree_tokenizer_postprocessor_type_e {
  // No post-processing (passthrough - no special tokens added).
  IREE_TOKENIZER_POSTPROCESSOR_NONE = 0,
  // TemplateProcessing: configurable template-based insertion.
  IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE,
  // RobertaProcessing: specialized for RoBERTa models.
  IREE_TOKENIZER_POSTPROCESSOR_ROBERTA,
  // ByteLevel: trim offsets, reverse byte mapping (for Sequence chains).
  IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL,
  // Sequence of post-processors applied in order.
  IREE_TOKENIZER_POSTPROCESSOR_SEQUENCE,
} iree_tokenizer_postprocessor_type_t;

//===----------------------------------------------------------------------===//
// Template Piece (for TemplateProcessing)
//===----------------------------------------------------------------------===//

// Template piece types (matches HuggingFace TemplateProcessing).
typedef enum iree_tokenizer_template_piece_type_e {
  // $A placeholder - encode first input sequence here.
  IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A = 0,
  // $B placeholder - encode second input sequence here (pairs only).
  IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_B = 1,
  // Special token - emit a specific token ID.
  IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL = 2,
} iree_tokenizer_template_piece_type_t;

// Single piece in a template (e.g., one [CLS] or $A).
// Compact 8-byte representation for cache efficiency.
typedef struct iree_tokenizer_template_piece_t {
  // Token ID for SPECIAL type, -1 for SEQUENCE_A/B.
  int32_t token_id;
  // Piece type (iree_tokenizer_template_piece_type_t).
  uint8_t type;
  // Segment/type ID (0 or 1) for attention mask differentiation.
  uint8_t type_id;
  // Reserved for future use.
  uint16_t reserved;
} iree_tokenizer_template_piece_t;
static_assert(sizeof(iree_tokenizer_template_piece_t) == 8,
              "template piece must be 8 bytes");

//===----------------------------------------------------------------------===//
// Type-Specific Configurations
//===----------------------------------------------------------------------===//

// Configuration for TemplateProcessing.
// Templates are stored contiguously: [single_pieces...][pair_pieces...]
typedef struct iree_tokenizer_template_config_t {
  // Number of pieces in single sequence template.
  iree_host_size_t single_count;
  // Number of pieces in pair sequence template (0 if pairs not supported).
  iree_host_size_t pair_count;
  // Heap-allocated template pieces (owned). Layout: [single...|pair...]
  iree_tokenizer_template_piece_t* templates;
  // Allocator used to free templates.
  iree_allocator_t allocator;
} iree_tokenizer_template_config_t;

// Configuration flags for RobertaProcessing.
typedef enum iree_tokenizer_roberta_flag_bits_e {
  IREE_TOKENIZER_ROBERTA_FLAG_DEFAULT = 0,
  // Add prefix space before encoding.
  IREE_TOKENIZER_ROBERTA_FLAG_ADD_PREFIX_SPACE = 1 << 0,
  // Trim whitespace offsets.
  IREE_TOKENIZER_ROBERTA_FLAG_TRIM_OFFSETS = 1 << 1,
} iree_tokenizer_roberta_flag_bits_t;
typedef uint32_t iree_tokenizer_roberta_flags_t;

// Configuration for RobertaProcessing.
// RoBERTa uses format: <s> $A </s> for single, <s> $A </s></s> $B </s> for
// pair.
typedef struct iree_tokenizer_roberta_config_t {
  // CLS token ID (<s>).
  int32_t cls_id;
  // SEP token ID (</s>).
  int32_t sep_id;
  // Behavior flags.
  iree_tokenizer_roberta_flags_t flags;
} iree_tokenizer_roberta_config_t;

// Configuration for ByteLevel post-processor.
// Used in Sequence chains (e.g., Llama 3).
typedef struct iree_tokenizer_byte_level_pp_config_t {
  // Reserved for future use.
  uint32_t flags;
} iree_tokenizer_byte_level_pp_config_t;

// Forward declaration for sequence config.
typedef struct iree_tokenizer_postprocessor_t iree_tokenizer_postprocessor_t;

// Configuration for Sequence post-processor (chain of processors).
typedef struct iree_tokenizer_postprocessor_sequence_config_t {
  // Number of child processors.
  iree_host_size_t count;
  // Heap-allocated array of child processors (owned).
  iree_tokenizer_postprocessor_t* children;
  // Allocator used to free children array.
  iree_allocator_t allocator;
} iree_tokenizer_postprocessor_sequence_config_t;

//===----------------------------------------------------------------------===//
// Postprocessor Value Type
//===----------------------------------------------------------------------===//

// Post-processor instance with inline configuration.
// Most post-processors are value types with optional heap allocations.
// Sequence and Template types own heap-allocated data and must be
// deinitialized.
typedef struct iree_tokenizer_postprocessor_t {
  iree_tokenizer_postprocessor_type_t type;
  union {
    iree_tokenizer_template_config_t template_;
    iree_tokenizer_roberta_config_t roberta;
    iree_tokenizer_byte_level_pp_config_t byte_level;
    iree_tokenizer_postprocessor_sequence_config_t sequence;
  } config;
} iree_tokenizer_postprocessor_t;

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

// Initializes a passthrough post-processor (no special tokens added).
// This is equivalent to HuggingFace's post_processor: null.
static inline void iree_tokenizer_postprocessor_initialize_none(
    iree_tokenizer_postprocessor_t* out_pp) {
  memset(out_pp, 0, sizeof(*out_pp));
  out_pp->type = IREE_TOKENIZER_POSTPROCESSOR_NONE;
}

// Initializes a RoBERTa post-processor.
// Format: <s> $A </s> for single, <s> $A </s></s> $B </s> for pair.
static inline void iree_tokenizer_postprocessor_initialize_roberta(
    int32_t cls_id, int32_t sep_id, iree_tokenizer_roberta_flags_t flags,
    iree_tokenizer_postprocessor_t* out_pp) {
  memset(out_pp, 0, sizeof(*out_pp));
  out_pp->type = IREE_TOKENIZER_POSTPROCESSOR_ROBERTA;
  out_pp->config.roberta.cls_id = cls_id;
  out_pp->config.roberta.sep_id = sep_id;
  out_pp->config.roberta.flags = flags;
}

// Initializes a ByteLevel post-processor (for Sequence chains).
static inline void iree_tokenizer_postprocessor_initialize_byte_level(
    uint32_t flags, iree_tokenizer_postprocessor_t* out_pp) {
  memset(out_pp, 0, sizeof(*out_pp));
  out_pp->type = IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL;
  out_pp->config.byte_level.flags = flags;
}

// Initializes a TemplateProcessing post-processor.
// TAKES OWNERSHIP of |templates| array (move semantics).
// |templates| layout: [single_pieces...][pair_pieces...]
// |single_count| is number of pieces in single template.
// |pair_count| is number of pieces in pair template (0 if not supported).
iree_status_t iree_tokenizer_postprocessor_initialize_template(
    iree_tokenizer_template_piece_t* templates, iree_host_size_t single_count,
    iree_host_size_t pair_count, iree_allocator_t allocator,
    iree_tokenizer_postprocessor_t* out_pp);

// Initializes a Sequence post-processor that chains multiple processors.
// TAKES OWNERSHIP of |children| array (move semantics). The children array is
// moved and the originals are zeroed. Caller must not deinitialize children
// after this call.
iree_status_t iree_tokenizer_postprocessor_initialize_sequence(
    iree_tokenizer_postprocessor_t* children, iree_host_size_t count,
    iree_allocator_t allocator, iree_tokenizer_postprocessor_t* out_pp);

// Deinitializes a post-processor, freeing any owned resources.
// MUST be called for all post-processors regardless of type.
// Recursively frees Sequence children.
void iree_tokenizer_postprocessor_deinitialize(
    iree_tokenizer_postprocessor_t* pp);

//===----------------------------------------------------------------------===//
// Apply API
//===----------------------------------------------------------------------===//

// Callback invoked when a text sequence placeholder ($A or $B) is encountered.
// The callback should encode the text and emit tokens.
typedef iree_status_t (*iree_tokenizer_encode_text_fn_t)(
    void* user_data, iree_string_view_t text);

// Callback invoked to emit a single special token.
typedef iree_status_t (*iree_tokenizer_emit_token_fn_t)(void* user_data,
                                                        int32_t token_id);

// Applies post-processor for single sequence encoding.
//
// Walks the single template (if any) and:
// - Calls emit_token_fn for each special token piece
// - Calls encode_text_fn when $A placeholder is encountered
//
// For NONE type, simply calls encode_text_fn with the full text.
// For Sequence type, applies all child processors in order.
//
// |pp| is the post-processor configuration.
// |text| is the input text to encode.
// |encode_text_fn| is called to encode text sequences.
// |emit_token_fn| is called to emit special tokens.
// |user_data| is passed to callbacks.
//
// Returns IREE_STATUS_OK on success, or error from callbacks.
iree_status_t iree_tokenizer_postprocessor_apply_single(
    const iree_tokenizer_postprocessor_t* pp, iree_string_view_t text,
    iree_tokenizer_encode_text_fn_t encode_text_fn,
    iree_tokenizer_emit_token_fn_t emit_token_fn, void* user_data);

// Applies post-processor for sequence pair encoding.
//
// Walks the pair template (if any) and:
// - Calls emit_token_fn for each special token piece
// - Calls encode_text_fn when $A or $B placeholders are encountered
//
// For NONE type, encodes text_a then text_b with no special tokens.
// For Sequence type, applies all child processors in order.
//
// |pp| is the post-processor configuration.
// |text_a| is the first input sequence.
// |text_b| is the second input sequence.
// |encode_text_fn| is called to encode text sequences.
// |emit_token_fn| is called to emit special tokens.
// |user_data| is passed to callbacks.
//
// Returns IREE_STATUS_OK on success, or error from callbacks.
iree_status_t iree_tokenizer_postprocessor_apply_pair(
    const iree_tokenizer_postprocessor_t* pp, iree_string_view_t text_a,
    iree_string_view_t text_b, iree_tokenizer_encode_text_fn_t encode_text_fn,
    iree_tokenizer_emit_token_fn_t emit_token_fn, void* user_data);

//===----------------------------------------------------------------------===//
// Special Token Emission
//===----------------------------------------------------------------------===//

// Emits all prefix tokens (special tokens before $A) for single encoding.
// Calls emit_token_fn for each prefix token in order.
// Returns immediately if no prefix tokens are configured.
//
// For TemplateProcessing: emits all SPECIAL tokens before SEQUENCE_A.
// For RobertaProcessing: emits cls_id (<s>).
// For Sequence: emits prefix from first child with one.
// For NONE/ByteLevel: emits nothing.
iree_status_t iree_tokenizer_postprocessor_emit_prefix(
    const iree_tokenizer_postprocessor_t* pp,
    iree_tokenizer_emit_token_fn_t emit_token_fn, void* user_data);

// Emits all suffix tokens (special tokens after $A) for single encoding.
// Calls emit_token_fn for each suffix token in order.
// Returns immediately if no suffix tokens are configured.
//
// For TemplateProcessing: emits all SPECIAL tokens after SEQUENCE_A.
// For RobertaProcessing: emits sep_id (</s>).
// For Sequence: emits suffix from last child with one.
// For NONE/ByteLevel: emits nothing.
iree_status_t iree_tokenizer_postprocessor_emit_suffix(
    const iree_tokenizer_postprocessor_t* pp,
    iree_tokenizer_emit_token_fn_t emit_token_fn, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_POSTPROCESSOR_H_
