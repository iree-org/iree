// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab/vocab_builder.h"

#include "iree/base/api.h"
#include "iree/tokenizer/vocab/vocab_hash.h"
#include "iree/tokenizer/vocab/vocab_internal.h"

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

  // Token arrays stored in a single slab allocation via IREE_STRUCT_LAYOUT.
  void* token_slab;                // Combined [tokens][scores] allocation.
  iree_tokenizer_token_t* tokens;  // = token_slab + 0
  float* scores;                   // = token_slab + tokens_size
  iree_host_size_t token_count;
  iree_host_size_t token_capacity;

  // Target IDs for out-of-order insertion (lazily allocated, separate).
  // NULL unless add_token_with_id() used.
  int32_t* target_ids;
  // True if target_ids are in use.
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

  // Maximum token text length seen so far.
  iree_host_size_t max_token_length;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Ensures token arrays have capacity for at least |min_capacity| tokens.
// tokens and scores are stored in a single slab for cache efficiency.
// target_ids is managed separately (lazily allocated for explicit ID mode).
static iree_status_t iree_tokenizer_vocab_builder_reserve_tokens(
    iree_tokenizer_vocab_builder_t* builder, iree_host_size_t min_capacity) {
  if (builder->token_capacity >= min_capacity) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Grow by 2x or to min_capacity, whichever is larger.
  iree_host_size_t new_capacity = 0;
  if (!iree_host_size_checked_mul(builder->token_capacity, 2, &new_capacity)) {
    new_capacity = min_capacity;  // Overflow, just use min_capacity.
  }
  if (new_capacity < min_capacity) new_capacity = min_capacity;
  if (new_capacity < 16) new_capacity = 16;

  // Calculate slab layout: [tokens][scores].
  iree_host_size_t total_size = 0;
  iree_host_size_t tokens_offset = 0;
  iree_host_size_t scores_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      /*base_size=*/0, &total_size,
      IREE_STRUCT_FIELD(new_capacity, iree_tokenizer_token_t, &tokens_offset),
      IREE_STRUCT_FIELD(new_capacity, float, &scores_offset));
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Allocate new slab.
  void* new_slab = NULL;
  status = iree_allocator_malloc(builder->allocator, total_size, &new_slab);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Compute new array pointers.
  uint8_t* base = (uint8_t*)new_slab;
  iree_tokenizer_token_t* new_tokens =
      (iree_tokenizer_token_t*)(base + tokens_offset);
  float* new_scores = (float*)(base + scores_offset);

  // Copy existing data from old slab.
  if (builder->token_count > 0) {
    memcpy(new_tokens, builder->tokens,
           builder->token_count * sizeof(iree_tokenizer_token_t));
    memcpy(new_scores, builder->scores, builder->token_count * sizeof(float));
  }

  // Free old slab and commit new pointers.
  if (builder->token_slab) {
    iree_allocator_free(builder->allocator, builder->token_slab);
  }
  builder->token_slab = new_slab;
  builder->tokens = new_tokens;
  builder->scores = new_scores;
  builder->token_capacity = new_capacity;

  // Grow target_ids if it exists (explicit ID mode).
  if (builder->target_ids) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_realloc_array(builder->allocator, new_capacity,
                                         sizeof(int32_t),
                                         (void**)&builder->target_ids));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Ensures string table has capacity for at least |min_capacity| bytes.
static iree_status_t iree_tokenizer_vocab_builder_reserve_strings(
    iree_tokenizer_vocab_builder_t* builder, iree_host_size_t min_capacity) {
  if (builder->string_capacity >= min_capacity) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
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
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_grow_array(
              builder->allocator, iree_max(64, min_capacity),
              sizeof(iree_tokenizer_merge_t), &builder->merge_capacity,
              (void**)&builder->merges));
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

  // Track maximum token length for buffer sizing.
  if (text.size > builder->max_token_length) {
    builder->max_token_length = text.size;
  }

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
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        builder->allocator, capacity, sizeof(int32_t),
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
      IREE_RETURN_AND_END_ZONE(z0,
                               iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                                "negative token ID %d", id));
    }
    if (id > max_id) max_id = id;
  }

  // Guard against max_id + 1 overflowing in int32_t domain before casting.
  if (max_id == INT32_MAX) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "max token ID at INT32_MAX limit"));
  }
  iree_host_size_t array_size = (iree_host_size_t)(max_id + 1);

  // Validate array_size won't overflow when multiplied by element sizes.
  // Check against iree_tokenizer_token_t (8 bytes) as the largest element
  // type.
  if (array_size > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_token_t)) {
    IREE_RETURN_AND_END_ZONE(
        z0,
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "sparse ID range too large: max_id=%" PRId32, max_id));
  }

  // Check for excessive sparsity to prevent memory exhaustion DoS.
  // The gap between max_id and token_count represents unused ID slots.
  // Only check when max_id + 1 > token_count (no duplicates); duplicate IDs
  // are caught by the permutation check below.
  if (array_size > builder->token_count) {
    iree_host_size_t gap = array_size - builder->token_count;
    if (gap > IREE_TOKENIZER_MAX_VOCAB_SPARSITY) {
      IREE_RETURN_AND_END_ZONE(
          z0, iree_make_status(
                  IREE_STATUS_RESOURCE_EXHAUSTED,
                  "vocabulary ID gap too large: %" PRIhsz " unused IDs "
                  "(max_id=%d, token_count=%" PRIhsz "); limit is %" PRIhsz,
                  gap, max_id, builder->token_count,
                  (iree_host_size_t)IREE_TOKENIZER_MAX_VOCAB_SPARSITY));
    }
  }

  // Allocate permutation array sized for max_id + 1 (not token_count).
  // This allows sparse IDs - gaps will have IREE_HOST_SIZE_MAX sentinel.
  iree_host_size_t* perm = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_array(builder->allocator, array_size,
                                      sizeof(iree_host_size_t), (void**)&perm));

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

  // Allocate new slab sized for max_id + 1 (including gaps).
  iree_host_size_t total_size = 0;
  iree_host_size_t tokens_offset = 0;
  iree_host_size_t scores_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      /*base_size=*/0, &total_size,
      IREE_STRUCT_FIELD(array_size, iree_tokenizer_token_t, &tokens_offset),
      IREE_STRUCT_FIELD(array_size, float, &scores_offset));
  void* new_slab = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(builder->allocator, total_size, &new_slab);
  }
  if (iree_status_is_ok(status)) {
    uint8_t* base = (uint8_t*)new_slab;
    iree_tokenizer_token_t* new_tokens =
        (iree_tokenizer_token_t*)(base + tokens_offset);
    float* new_scores = (float*)(base + scores_offset);

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

    // Free old slab and commit new one.
    iree_allocator_free(builder->allocator, builder->token_slab);
    builder->token_slab = new_slab;
    builder->tokens = new_tokens;
    builder->scores = new_scores;
    builder->token_capacity = array_size;
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

// Finds a token by its target ID and ORs in |attrs|. Returns true if found.
static bool iree_tokenizer_vocab_builder_try_add_token_attrs(
    iree_tokenizer_vocab_builder_t* builder, int32_t token_id,
    iree_tokenizer_token_attr_t attrs) {
  // When target_ids is NULL, tokens are in sequential order (index == ID).
  // When target_ids is set, we need to search for the matching ID.
  if (builder->target_ids) {
    for (iree_host_size_t i = 0; i < builder->token_count; ++i) {
      if (builder->target_ids[i] == token_id) {
        builder->tokens[i].attributes |= attrs;
        return true;
      }
    }
    return false;
  }
  // Sequential mode: ID == index.
  if (token_id < 0 || (iree_host_size_t)token_id >= builder->token_count) {
    return false;
  }
  builder->tokens[token_id].attributes |= attrs;
  return true;
}

iree_status_t iree_tokenizer_vocab_builder_add_token_attrs(
    iree_tokenizer_vocab_builder_t* builder, int32_t token_id,
    iree_tokenizer_token_attr_t attrs) {
  IREE_ASSERT_ARGUMENT(builder);
  if (!iree_tokenizer_vocab_builder_try_add_token_attrs(builder, token_id,
                                                        attrs)) {
    return iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }
  return iree_ok_status();
}

iree_status_t iree_tokenizer_vocab_builder_ensure_token(
    iree_tokenizer_vocab_builder_t* builder, int32_t token_id,
    iree_string_view_t text, float score, iree_tokenizer_token_attr_t attrs) {
  IREE_ASSERT_ARGUMENT(builder);

  // Try to update attributes on existing token first.
  if (iree_tokenizer_vocab_builder_try_add_token_attrs(builder, token_id,
                                                       attrs)) {
    return iree_ok_status();
  }
  // Token doesn't exist - insert it.
  return iree_tokenizer_vocab_builder_add_token_with_id(builder, token_id, text,
                                                        score, attrs);
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

  iree_allocator_t allocator = builder->allocator;

  // Sort automatically if explicit IDs were used.
  // This handles sparse/non-contiguous IDs by allocating for max_id + 1.
  int32_t max_token_id = (int32_t)builder->token_count - 1;
  if (builder->needs_sort) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_tokenizer_vocab_builder_sort(builder, &max_token_id));
  }

  // Array size for tokens/scores (may be > token_count for sparse vocabs).
  iree_host_size_t array_size = (iree_host_size_t)max_token_id + 1;

  // Validate special token IDs are in range.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_vocab_builder_validate_special_ids(
              &builder->special_ids, max_token_id));

  // Calculate hash storage size.
  iree_host_size_t hash_storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_vocab_hash_storage_size(
              array_size, IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT,
              &hash_storage_size));

  // Calculate combined vocab+hash allocation layout.
  // Hash table alignment: the struct contains pointers, so sizeof(void*)
  // is the minimum required alignment.
  iree_host_size_t total_size = 0;
  iree_host_size_t hash_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_tokenizer_vocab_t), &total_size,
              IREE_STRUCT_FIELD_ALIGNED(hash_storage_size, uint8_t,
                                        sizeof(void*), &hash_offset)));

  // Allocate combined vocab struct + hash table.
  void* slab = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, &slab));

  iree_tokenizer_vocab_t* vocab = (iree_tokenizer_vocab_t*)slab;
  iree_tokenizer_vocab_hash_t* hash =
      (iree_tokenizer_vocab_hash_t*)((uint8_t*)slab + hash_offset);

  // Initialize hash table in-place.
  iree_const_byte_span_t string_span = {builder->string_data,
                                        builder->string_size};
  iree_tokenizer_vocab_hash_initialize(
      builder->tokens, array_size, string_span,
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, hash);

  // Transfer ownership from builder to vocab (zero-copy).
  vocab->allocator = allocator;
  vocab->token_slab = builder->token_slab;
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
  vocab->max_token_length = builder->max_token_length;

  // Clear builder's pointers so free() doesn't double-free.
  builder->token_slab = NULL;
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

  iree_allocator_free(allocator, builder->token_slab);
  iree_allocator_free(allocator, builder->target_ids);
  iree_allocator_free(allocator, builder->string_data);
  iree_allocator_free(allocator, builder->merges);
  iree_allocator_free(allocator, builder);
  IREE_TRACE_ZONE_END(z0);
}
