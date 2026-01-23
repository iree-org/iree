// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab_builder.h"

#include "iree/base/api.h"
#include "iree/tokenizer/vocab_hash.h"
#include "iree/tokenizer/vocab_internal.h"

//===----------------------------------------------------------------------===//
// Limits
//===----------------------------------------------------------------------===//

// Maximum number of unused ID slots allowed in a sparse vocabulary.
// This prevents memory exhaustion attacks where a malicious input specifies
// a small number of tokens with extremely large IDs (e.g., ID 0 and ID 2B
// would attempt to allocate ~40GB for just 2 tokens).
// A gap of 1M allows ~12MB of wasted storage, which is acceptable for
// legitimate sparse vocabs like ConvBERT.
#define IREE_TOKENIZER_MAX_VOCAB_SPARSITY 1000000

//===----------------------------------------------------------------------===//
// Internal Structure
//===----------------------------------------------------------------------===//

struct iree_tokenizer_vocab_builder_t {
  iree_allocator_t allocator;

  // Growable token array.
  iree_tokenizer_token_t* tokens;
  float* scores;
  iree_host_size_t token_count;
  iree_host_size_t token_capacity;

  // Target IDs for out-of-order insertion (NULL if sequential).
  int32_t* target_ids;
  bool needs_sort;

  // Growable string table.
  uint8_t* string_data;
  iree_host_size_t string_size;
  iree_host_size_t string_capacity;

  // Special token IDs.
  iree_tokenizer_special_ids_t special_ids;

  // Growable BPE merge array (NULL if no merges added).
  iree_tokenizer_merge_t* merges;
  iree_host_size_t merge_count;
  iree_host_size_t merge_capacity;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Ensures token arrays have capacity for at least |min_capacity| tokens.
static iree_status_t iree_tokenizer_vocab_builder_reserve_tokens(
    iree_tokenizer_vocab_builder_t* builder, iree_host_size_t min_capacity) {
  if (builder->token_capacity >= min_capacity) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Grow by 2x or to min_capacity, whichever is larger.
  // Guard against overflow when doubling.
  iree_host_size_t new_capacity = builder->token_capacity;
  if (new_capacity <= IREE_HOST_SIZE_MAX / 2) {
    new_capacity *= 2;
  }
  if (new_capacity < min_capacity) new_capacity = min_capacity;
  if (new_capacity < 16) new_capacity = 16;

  // Check for overflow in size calculations.
  if (new_capacity > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_token_t)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "token capacity overflow");
  }

  // Reallocate arrays. Note: if a realloc fails after earlier ones succeed,
  // the builder is left in an inconsistent state. This is acceptable since
  // allocation failure is terminal - callers must free the builder on any
  // error return.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_realloc(builder->allocator,
                                 new_capacity * sizeof(iree_tokenizer_token_t),
                                 (void**)&builder->tokens));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_realloc(builder->allocator, new_capacity * sizeof(float),
                             (void**)&builder->scores));
  if (builder->target_ids) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_realloc(builder->allocator,
                                   new_capacity * sizeof(int32_t),
                                   (void**)&builder->target_ids));
  }

  builder->token_capacity = new_capacity;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Ensures string table has capacity for at least |min_capacity| bytes.
static iree_status_t iree_tokenizer_vocab_builder_reserve_strings(
    iree_tokenizer_vocab_builder_t* builder, iree_host_size_t min_capacity) {
  if (builder->string_capacity >= min_capacity) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Grow by 2x or to min_capacity, whichever is larger.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_grow_array(builder->allocator, iree_max(256, min_capacity),
                                /*element_size=*/1, &builder->string_capacity,
                                (void**)&builder->string_data));
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Ensures merge array has capacity for at least |min_capacity| merges.
static iree_status_t iree_tokenizer_vocab_builder_reserve_merges(
    iree_tokenizer_vocab_builder_t* builder, iree_host_size_t min_capacity) {
  if (builder->merge_capacity >= min_capacity) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Grow by 2x or to min_capacity, whichever is larger.
  // Guard against overflow when doubling.
  iree_host_size_t new_capacity = builder->merge_capacity;
  if (new_capacity <= IREE_HOST_SIZE_MAX / 2) {
    new_capacity *= 2;
  }
  if (new_capacity < min_capacity) new_capacity = min_capacity;
  if (new_capacity < 64) new_capacity = 64;

  // Check for overflow in size calculations.
  if (new_capacity > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_merge_t)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "merge capacity overflow");
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_realloc(builder->allocator,
                                 new_capacity * sizeof(iree_tokenizer_merge_t),
                                 (void**)&builder->merges));

  builder->merge_capacity = new_capacity;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Sets a special token ID by type enum.
static void iree_tokenizer_vocab_builder_set_special_id(
    iree_tokenizer_special_ids_t* ids, iree_tokenizer_special_token_t type,
    int32_t token_id) {
  switch (type) {
    case IREE_TOKENIZER_SPECIAL_TOKEN_UNK:
      ids->unk = token_id;
      break;
    case IREE_TOKENIZER_SPECIAL_TOKEN_BOS:
      ids->bos = token_id;
      break;
    case IREE_TOKENIZER_SPECIAL_TOKEN_EOS:
      ids->eos = token_id;
      break;
    case IREE_TOKENIZER_SPECIAL_TOKEN_PAD:
      ids->pad = token_id;
      break;
    case IREE_TOKENIZER_SPECIAL_TOKEN_SEP:
      ids->sep = token_id;
      break;
    case IREE_TOKENIZER_SPECIAL_TOKEN_CLS:
      ids->cls = token_id;
      break;
    case IREE_TOKENIZER_SPECIAL_TOKEN_MASK:
      ids->mask = token_id;
      break;
    default:
      break;
  }
}

// Validates that all special token IDs are either -1 (unset) or in range.
static iree_status_t iree_tokenizer_vocab_builder_validate_special_ids(
    const iree_tokenizer_special_ids_t* ids, int32_t max_token_id) {
  const int32_t* id_array = &ids->bos;
  static const char* names[] = {"bos", "eos", "unk", "pad",
                                "sep", "cls", "mask"};
  (void)names;
  for (int i = 0; i < 7; ++i) {
    int32_t id = id_array[i];
    if (id != -1 && (id < 0 || id > max_token_id)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "special token %s ID %d out of range [0, %d]",
                              names[i], id, max_token_id);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_vocab_builder_allocate(
    iree_host_size_t capacity_hint, iree_allocator_t allocator,
    iree_tokenizer_vocab_builder_t** out_builder) {
  IREE_ASSERT_ARGUMENT(out_builder);
  *out_builder = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate builder struct.
  iree_tokenizer_vocab_builder_t* builder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(iree_tokenizer_vocab_builder_t),
                            (void**)&builder));

  // Initialize.
  memset(builder, 0, sizeof(*builder));
  builder->allocator = allocator;
  builder->special_ids = iree_tokenizer_special_ids_none();

  // Pre-allocate if hint provided.
  if (capacity_hint > 0) {
    iree_status_t status =
        iree_tokenizer_vocab_builder_reserve_tokens(builder, capacity_hint);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_vocab_builder_free(builder);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    // Estimate ~8 bytes per token for strings, with overflow protection.
    iree_host_size_t string_hint = capacity_hint;
    if (string_hint <= IREE_HOST_SIZE_MAX / 8) {
      string_hint *= 8;
    } else {
      string_hint = IREE_HOST_SIZE_MAX;  // Clamp on overflow.
    }
    status = iree_tokenizer_vocab_builder_reserve_strings(builder, string_hint);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_vocab_builder_free(builder);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  *out_builder = builder;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_tokenizer_vocab_builder_add_token(
    iree_tokenizer_vocab_builder_t* builder, iree_string_view_t text,
    float score, iree_tokenizer_token_attr_t attrs) {
  IREE_ASSERT_ARGUMENT(builder);

  // Validate text.size fits in uint16_t (max token length).
  if (text.size > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "token text too long: %" PRIhsz " > %u", text.size,
                            (unsigned)UINT16_MAX);
  }

  // Check for string_size overflow before addition.
  if (builder->string_size > UINT32_MAX - text.size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "string table overflow");
  }

  // Ensure token count fits in int32_t (used for target_ids and token IDs).
  if (builder->token_count >= INT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "vocabulary size limit exceeded");
  }

  // Ensure capacity for one more token.
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_builder_reserve_tokens(
      builder, builder->token_count + 1));

  // Ensure capacity for the string (always reserve at least 1 byte to avoid
  // NULL string_data when all tokens are empty strings).
  iree_host_size_t min_string_capacity = builder->string_size + text.size;
  if (min_string_capacity == 0) min_string_capacity = 1;
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_builder_reserve_strings(
      builder, min_string_capacity));

  // Append string to string table.
  if (text.size > 0) {
    memcpy(builder->string_data + builder->string_size, text.data, text.size);
  }

  // Create token entry.
  iree_tokenizer_token_t token = {
      .string_offset = (uint32_t)builder->string_size,
      .string_length = (uint16_t)text.size,
      .attributes = attrs,
  };

  builder->tokens[builder->token_count] = token;
  builder->scores[builder->token_count] = score;

  // If explicit IDs are being used (target_ids exists), record implicit ID.
  // This allows mixing add_token() and add_token_with_id() freely.
  if (builder->target_ids) {
    builder->target_ids[builder->token_count] = (int32_t)builder->token_count;
  }

  builder->string_size += text.size;
  builder->token_count++;

  return iree_ok_status();
}

iree_status_t iree_tokenizer_vocab_builder_add_token_with_id(
    iree_tokenizer_vocab_builder_t* builder, int32_t token_id,
    iree_string_view_t text, float score, iree_tokenizer_token_attr_t attrs) {
  IREE_ASSERT_ARGUMENT(builder);

  // Lazily allocate target_ids array on first use.
  if (!builder->target_ids) {
    iree_host_size_t capacity =
        builder->token_capacity > 0 ? builder->token_capacity : 16;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(builder->allocator,
                                               capacity * sizeof(int32_t),
                                               (void**)&builder->target_ids));
    // Backfill implicit IDs for any tokens already added via add_token().
    for (iree_host_size_t i = 0; i < builder->token_count; ++i) {
      builder->target_ids[i] = (int32_t)i;
    }
    builder->needs_sort = true;
  }

  // Add the token using the base function.
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_vocab_builder_add_token(builder, text, score, attrs));

  // Record the target ID for the just-added token.
  builder->target_ids[builder->token_count - 1] = token_id;

  return iree_ok_status();
}

// Sorts tokens by their target ID (internal helper).
// Called automatically by build() when explicit IDs were used.
// Supports sparse/non-contiguous IDs - gaps are filled with ATTR_UNUSED.
// Returns max_token_id via out parameter.
static iree_status_t iree_tokenizer_vocab_builder_sort(
    iree_tokenizer_vocab_builder_t* builder, int32_t* out_max_token_id) {
  *out_max_token_id = (int32_t)builder->token_count - 1;
  if (!builder->needs_sort || builder->token_count == 0) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Find max ID and validate no negative IDs.
  int32_t max_id = -1;
  for (iree_host_size_t i = 0; i < builder->token_count; ++i) {
    int32_t id = builder->target_ids[i];
    if (id < 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "negative token ID %d", id);
    }
    if (id > max_id) max_id = id;
  }

  // Guard against max_id + 1 overflowing in int32_t domain before casting.
  if (max_id == INT32_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "max token ID at INT32_MAX limit");
  }
  iree_host_size_t array_size = (iree_host_size_t)(max_id + 1);

  // Validate array_size won't overflow when multiplied by element sizes.
  // Check against iree_tokenizer_token_t (8 bytes) as the largest element type.
  if (array_size > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_token_t)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sparse ID range too large: max_id=%" PRId32,
                            max_id);
  }

  // Check for excessive sparsity to prevent memory exhaustion DoS.
  // The gap between max_id and token_count represents unused ID slots.
  // Only check when max_id + 1 > token_count (no duplicates); duplicate IDs
  // are caught by the permutation check below.
  if (array_size > builder->token_count) {
    iree_host_size_t gap = array_size - builder->token_count;
    if (gap > IREE_TOKENIZER_MAX_VOCAB_SPARSITY) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "vocabulary ID gap too large: %" PRIhsz
          " unused IDs "
          "(max_id=%d, token_count=%" PRIhsz "); limit is %" PRIhsz,
          gap, max_id, builder->token_count,
          (iree_host_size_t)IREE_TOKENIZER_MAX_VOCAB_SPARSITY);
    }
  }

  // Allocate permutation array sized for max_id + 1 (not token_count).
  // This allows sparse IDs - gaps will have IREE_HOST_SIZE_MAX sentinel.
  iree_host_size_t* perm = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(builder->allocator,
                                array_size * sizeof(iree_host_size_t),
                                (void**)&perm));

  // Initialize permutation to invalid sentinel (marks gaps).
  for (iree_host_size_t i = 0; i < array_size; ++i) {
    perm[i] = IREE_HOST_SIZE_MAX;
  }

  // Build permutation: perm[target_id] = source_index.
  for (iree_host_size_t i = 0; i < builder->token_count; ++i) {
    int32_t target = builder->target_ids[i];
    if (perm[target] != IREE_HOST_SIZE_MAX) {
      iree_allocator_free(builder->allocator, perm);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate token ID %d", target);
    }
    perm[target] = i;
  }

  // Allocate new arrays sized for max_id + 1 (including gaps).
  iree_tokenizer_token_t* new_tokens = NULL;
  float* new_scores = NULL;
  iree_status_t status = iree_allocator_malloc(
      builder->allocator, array_size * sizeof(iree_tokenizer_token_t),
      (void**)&new_tokens);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        builder->allocator, array_size * sizeof(float), (void**)&new_scores);
  }
  if (iree_status_is_ok(status)) {
    // Populate arrays - real tokens from perm, gaps marked UNUSED.
    for (iree_host_size_t i = 0; i < array_size; ++i) {
      iree_host_size_t src = perm[i];
      if (src != IREE_HOST_SIZE_MAX) {
        // Real token at this ID.
        new_tokens[i] = builder->tokens[src];
        new_scores[i] = builder->scores[src];
      } else {
        // Gap in ID space - mark as unused.
        new_tokens[i].string_offset = 0;
        new_tokens[i].string_length = 0;
        new_tokens[i].attributes = IREE_TOKENIZER_TOKEN_ATTR_UNUSED;
        new_scores[i] = 0.0f;
      }
    }
    iree_allocator_free(builder->allocator, builder->tokens);
    iree_allocator_free(builder->allocator, builder->scores);
    builder->tokens = new_tokens;
    builder->scores = new_scores;
    builder->token_capacity = array_size;
  } else {
    if (new_tokens) iree_allocator_free(builder->allocator, new_tokens);
  }
  iree_allocator_free(builder->allocator, perm);

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Clear sort state.
  iree_allocator_free(builder->allocator, builder->target_ids);
  builder->target_ids = NULL;
  builder->needs_sort = false;

  *out_max_token_id = max_id;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_tokenizer_vocab_builder_set_special_token(
    iree_tokenizer_vocab_builder_t* builder,
    iree_tokenizer_special_token_t type, int32_t token_id) {
  IREE_ASSERT_ARGUMENT(builder);
  iree_tokenizer_vocab_builder_set_special_id(&builder->special_ids, type,
                                              token_id);
  return iree_ok_status();
}

iree_status_t iree_tokenizer_vocab_builder_add_token_attrs(
    iree_tokenizer_vocab_builder_t* builder, int32_t token_id,
    iree_tokenizer_token_attr_t attrs) {
  IREE_ASSERT_ARGUMENT(builder);

  // Find the token by its target ID.
  // When target_ids is NULL, tokens are in sequential order (index == ID).
  // When target_ids is set, we need to search for the matching ID.
  if (builder->target_ids) {
    for (iree_host_size_t i = 0; i < builder->token_count; ++i) {
      if (builder->target_ids[i] == token_id) {
        builder->tokens[i].attributes |= attrs;
        return iree_ok_status();
      }
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "token ID %d not found in builder", token_id);
  }

  // Sequential mode: ID == index.
  if (token_id < 0 || (iree_host_size_t)token_id >= builder->token_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "token ID %d not found in builder", token_id);
  }
  builder->tokens[token_id].attributes |= attrs;
  return iree_ok_status();
}

iree_status_t iree_tokenizer_vocab_builder_add_merge(
    iree_tokenizer_vocab_builder_t* builder, uint32_t left_id,
    uint32_t right_id) {
  IREE_ASSERT_ARGUMENT(builder);

  // Ensure capacity for one more merge.
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_builder_reserve_merges(
      builder, builder->merge_count + 1));

  // Add the merge.
  builder->merges[builder->merge_count].left_id = left_id;
  builder->merges[builder->merge_count].right_id = right_id;
  builder->merge_count++;

  return iree_ok_status();
}

iree_status_t iree_tokenizer_vocab_builder_build(
    iree_tokenizer_vocab_builder_t* builder,
    iree_tokenizer_vocab_t** out_vocab) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_vocab);
  *out_vocab = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Sort automatically if explicit IDs were used.
  // This handles sparse/non-contiguous IDs by allocating for max_id + 1.
  int32_t max_token_id = (int32_t)builder->token_count - 1;
  if (builder->needs_sort) {
    iree_status_t sort_status =
        iree_tokenizer_vocab_builder_sort(builder, &max_token_id);
    if (!iree_status_is_ok(sort_status)) {
      iree_tokenizer_vocab_builder_free(builder);
      IREE_TRACE_ZONE_END(z0);
      return sort_status;
    }
  }

  // Array size for tokens/scores (may be > token_count for sparse vocabs).
  iree_host_size_t array_size = (iree_host_size_t)max_token_id + 1;

  iree_allocator_t allocator = builder->allocator;
  iree_status_t status = iree_ok_status();

  // Validate special token IDs are in range.
  status = iree_tokenizer_vocab_builder_validate_special_ids(
      &builder->special_ids, max_token_id);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_builder_free(builder);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Build hash table.
  // Pass array_size (not token_count) to iterate over all slots including gaps.
  // Gaps are marked ATTR_UNUSED and skipped during hash insertion.
  iree_tokenizer_vocab_hash_t* hash = NULL;
  iree_const_byte_span_t string_span = {builder->string_data,
                                        builder->string_size};
  status = iree_tokenizer_vocab_hash_build(
      builder->tokens, array_size, string_span,
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, allocator, &hash);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_builder_free(builder);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Allocate vocab struct.
  iree_tokenizer_vocab_t* vocab = NULL;
  status = iree_allocator_malloc(allocator, sizeof(iree_tokenizer_vocab_t),
                                 (void**)&vocab);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_hash_free(hash);
    iree_tokenizer_vocab_builder_free(builder);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Transfer ownership from builder to vocab.
  vocab->allocator = allocator;
  vocab->tokens = builder->tokens;
  vocab->scores = builder->scores;
  vocab->token_count = builder->token_count;
  vocab->max_token_id = max_token_id;
  vocab->string_data = builder->string_data;
  vocab->string_size = builder->string_size;
  vocab->hash = hash;
  vocab->special_ids = builder->special_ids;
  vocab->merges = builder->merges;
  vocab->merge_count = builder->merge_count;

  // Clear builder's pointers so free() doesn't double-free.
  builder->tokens = NULL;
  builder->scores = NULL;
  builder->string_data = NULL;
  builder->merges = NULL;

  // Free builder struct (but not the data we transferred).
  iree_allocator_free(allocator, builder);

  *out_vocab = vocab;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_tokenizer_vocab_builder_free(
    iree_tokenizer_vocab_builder_t* builder) {
  if (!builder) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = builder->allocator;

  if (builder->tokens) {
    iree_allocator_free(allocator, builder->tokens);
  }
  if (builder->scores) {
    iree_allocator_free(allocator, builder->scores);
  }
  if (builder->target_ids) {
    iree_allocator_free(allocator, builder->target_ids);
  }
  if (builder->string_data) {
    iree_allocator_free(allocator, builder->string_data);
  }
  if (builder->merges) {
    iree_allocator_free(allocator, builder->merges);
  }
  iree_allocator_free(allocator, builder);
  IREE_TRACE_ZONE_END(z0);
}
