// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core type definitions for the IREE tokenizer library.
//
// This header defines the fundamental data structures used throughout the
// tokenizer: token entries, merge rules, special token IDs, encoding options,
// and result structures. All types are designed for cache efficiency and
// minimal memory footprint.

#ifndef IREE_TOKENIZER_TYPES_H_
#define IREE_TOKENIZER_TYPES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Token Attributes
//===----------------------------------------------------------------------===//

// Token attribute flags (may be combined).
enum iree_tokenizer_token_attr_bits_e {
  IREE_TOKENIZER_TOKEN_ATTR_NONE = 0u,
  IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN = 1u << 0,  // [UNK] token
  IREE_TOKENIZER_TOKEN_ATTR_CONTROL = 1u << 1,  // Control token (not in text)
  IREE_TOKENIZER_TOKEN_ATTR_BYTE = 1u << 2,     // Single byte fallback <0xNN>
  IREE_TOKENIZER_TOKEN_ATTR_SPECIAL = 1u << 3,  // Special token (BOS/EOS/etc)
  IREE_TOKENIZER_TOKEN_ATTR_UNUSED = 1u << 4,   // Placeholder/unused slot
};
typedef uint16_t iree_tokenizer_token_attr_t;

//===----------------------------------------------------------------------===//
// Token Entry (8 bytes, cache-friendly)
//===----------------------------------------------------------------------===//

// Single token entry in the vocabulary table.
// Designed for cache-efficient sequential access during tokenization.
typedef struct iree_tokenizer_token_t {
  uint32_t string_offset;                  // Offset into string table.
  uint16_t string_length;                  // UTF-8 byte length.
  iree_tokenizer_token_attr_t attributes;  // Attribute flags.
} iree_tokenizer_token_t;
static_assert(sizeof(iree_tokenizer_token_t) == 8,
              "token entry must be 8 bytes");

//===----------------------------------------------------------------------===//
// BPE Merge Entry (8 bytes)
//===----------------------------------------------------------------------===//

// BPE merge rule: left_id + right_id merged to form a new token.
// Stored in rank order (index 0 = highest priority merge).
typedef struct iree_tokenizer_merge_t {
  uint32_t left_id;   // Left token ID.
  uint32_t right_id;  // Right token ID.
} iree_tokenizer_merge_t;
static_assert(sizeof(iree_tokenizer_merge_t) == 8,
              "merge entry must be 8 bytes");

//===----------------------------------------------------------------------===//
// Special Token IDs
//===----------------------------------------------------------------------===//

// Special token type enum (for builder API).
typedef enum iree_tokenizer_special_token_e {
  IREE_TOKENIZER_SPECIAL_TOKEN_UNK = 0,   // Unknown token.
  IREE_TOKENIZER_SPECIAL_TOKEN_BOS = 1,   // Beginning of sequence.
  IREE_TOKENIZER_SPECIAL_TOKEN_EOS = 2,   // End of sequence.
  IREE_TOKENIZER_SPECIAL_TOKEN_PAD = 3,   // Padding.
  IREE_TOKENIZER_SPECIAL_TOKEN_SEP = 4,   // Separator (BERT [SEP]).
  IREE_TOKENIZER_SPECIAL_TOKEN_CLS = 5,   // Classification (BERT [CLS]).
  IREE_TOKENIZER_SPECIAL_TOKEN_MASK = 6,  // Mask token (BERT [MASK]).
  IREE_TOKENIZER_SPECIAL_TOKEN_COUNT = 7,
} iree_tokenizer_special_token_t;

// Well-known special token IDs. Value of -1 indicates not present.
typedef struct iree_tokenizer_special_ids_t {
  int32_t bos;   // Beginning of sequence.
  int32_t eos;   // End of sequence.
  int32_t unk;   // Unknown token.
  int32_t pad;   // Padding.
  int32_t sep;   // Separator (BERT [SEP]).
  int32_t cls;   // Classification (BERT [CLS]).
  int32_t mask;  // Mask token (BERT [MASK]).
} iree_tokenizer_special_ids_t;

// Returns special IDs initialized to -1 (not present).
static inline iree_tokenizer_special_ids_t iree_tokenizer_special_ids_none(
    void) {
  iree_tokenizer_special_ids_t ids = {-1, -1, -1, -1, -1, -1, -1};
  return ids;
}

//===----------------------------------------------------------------------===//
// Unicode Constants
//===----------------------------------------------------------------------===//

// Default Metaspace replacement character: ▁ (Lower One Eighth Block, U+2581).
// Used by SentencePiece and BPE tokenizers to mark word boundaries.
// Spaces are replaced with this character during encoding, and restored during
// decoding. This allows the model to learn word boundaries explicitly.
#define IREE_TOKENIZER_METASPACE_REPLACEMENT 0x2581u

//===----------------------------------------------------------------------===//
// Streaming Callback API
//===----------------------------------------------------------------------===//

// Stack buffer sizes for batched string processing. Tune based on stack budget.
// 16 strings × 16 bytes = 256 bytes for string views.
// With 4KB data buffer, ~4.25KB per processing level.
#ifndef IREE_TOKENIZER_STRING_BATCH_CAPACITY
#define IREE_TOKENIZER_STRING_BATCH_CAPACITY 16
#endif
#ifndef IREE_TOKENIZER_DATA_BATCH_CAPACITY
#define IREE_TOKENIZER_DATA_BATCH_CAPACITY (4 * 1024)
#endif

// Callback invoked with batches of strings during encode/decode.
// |strings| contains views that are valid only for the duration of the call.
// Return non-OK status to abort processing.
typedef iree_status_t (*iree_tokenizer_string_callback_fn_t)(
    void* user_data, iree_string_view_list_t strings);

// Batched token ID list (matches iree_string_view_list_t pattern).
// 64 tokens × 4 bytes = 256 bytes, matching string batch stack budget.
typedef struct iree_tokenizer_id_list_t {
  iree_host_size_t count;
  const int32_t* values;
} iree_tokenizer_id_list_t;

#ifndef IREE_TOKENIZER_TOKEN_BATCH_CAPACITY
#define IREE_TOKENIZER_TOKEN_BATCH_CAPACITY 64
#endif

// Callback invoked with batches of token IDs during encoding.
// |ids| contains values valid only for the duration of the call.
// Return non-OK status to abort processing.
typedef iree_status_t (*iree_tokenizer_token_callback_fn_t)(
    void* user_data, iree_tokenizer_id_list_t ids);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TYPES_H_
