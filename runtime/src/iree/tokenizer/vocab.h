// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable vocabulary for tokenization.
//
// A vocab contains a token array (ID → text mapping), a string table, and a
// hash table for fast lookup (text → ID). Vocabs are constructed via
// iree_tokenizer_vocab_builder and are immutable after construction.
//
// Ownership Model:
// Vocabs use single-owner semantics with ownership transfer (not reference
// counting). A vocab is typically built once and then transferred to a
// tokenizer via iree_tokenizer_*_allocate(), which takes ownership. After
// transfer, the caller must not use the vocab pointer - the tokenizer owns it
// and will free it when the tokenizer is freed. This design avoids the
// overhead of atomic reference counting for objects that are not shared.

#ifndef IREE_TOKENIZER_VOCAB_H_
#define IREE_TOKENIZER_VOCAB_H_

#include "iree/base/api.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Vocab
//===----------------------------------------------------------------------===//

// Opaque immutable vocabulary type.
// Constructed via iree_tokenizer_vocab_builder_build().
typedef struct iree_tokenizer_vocab_t iree_tokenizer_vocab_t;

// Looks up a string in the vocabulary.
// Returns the token ID if found, or -1 if not found.
IREE_API_EXPORT int32_t iree_tokenizer_vocab_lookup(
    const iree_tokenizer_vocab_t* vocab, iree_string_view_t text);

// Returns the text for a token ID.
// Returns an empty string view if token_id is out of range.
IREE_API_EXPORT iree_string_view_t iree_tokenizer_vocab_token_text(
    const iree_tokenizer_vocab_t* vocab, int32_t token_id);

// Returns the score for a token ID (BPE priority, 0.0 for unscored).
// Returns 0.0 if token_id is out of range.
IREE_API_EXPORT float iree_tokenizer_vocab_token_score(
    const iree_tokenizer_vocab_t* vocab, int32_t token_id);

// Returns the attributes for a token ID.
// Returns ATTR_NONE if token_id is out of range.
IREE_API_EXPORT iree_tokenizer_token_attr_t iree_tokenizer_vocab_token_attrs(
    const iree_tokenizer_vocab_t* vocab, int32_t token_id);

// Returns the capacity of the vocabulary (max_token_id + 1).
// This is the size of the token array, suitable for bounds checking and
// iteration. For sparse vocabularies, this may be larger than the actual
// token count due to gap slots marked ATTR_UNUSED.
IREE_API_EXPORT iree_host_size_t
iree_tokenizer_vocab_capacity(const iree_tokenizer_vocab_t* vocab);

// Returns the number of active tokens in the vocabulary.
// For sparse vocabularies, this excludes gap slots (ATTR_UNUSED entries).
IREE_API_EXPORT iree_host_size_t
iree_tokenizer_vocab_token_count(const iree_tokenizer_vocab_t* vocab);

// Returns the special token IDs for the vocabulary.
IREE_API_EXPORT iree_tokenizer_special_ids_t
iree_tokenizer_vocab_special_ids(const iree_tokenizer_vocab_t* vocab);

// Returns the number of BPE merge rules (0 for non-BPE vocabs).
IREE_API_EXPORT iree_host_size_t
iree_tokenizer_vocab_merge_count(const iree_tokenizer_vocab_t* vocab);

// Returns a merge rule by rank (0 = highest priority).
// Returns {0, 0} if rank is out of range or vocab has no merges.
IREE_API_EXPORT iree_tokenizer_merge_t iree_tokenizer_vocab_merge(
    const iree_tokenizer_vocab_t* vocab, iree_host_size_t rank);

// Frees a vocabulary.
//
// Note: This is typically called by iree_tokenizer_free() when the tokenizer
// owns the vocab. Direct calls are only needed when a vocab is created but
// not transferred to a tokenizer (e.g., on error paths before allocation).
IREE_API_EXPORT void iree_tokenizer_vocab_free(iree_tokenizer_vocab_t* vocab);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_VOCAB_H_
