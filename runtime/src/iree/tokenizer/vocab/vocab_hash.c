// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab/vocab_hash.h"

#include "iree/base/api.h"
#include "iree/base/internal/math.h"

// Include intrinsics headers for hardware-accelerated CRC32C.
#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
#if defined(IREE_COMPILER_MSVC)
#include <intrin.h>
#else
#include <nmmintrin.h>  // SSE4.2 CRC32 intrinsics.
#endif                  // IREE_COMPILER_MSVC
#define IREE_TOKENIZER_HAS_CRC32C 1
#endif  // IREE_ARCH_X86_*

//===----------------------------------------------------------------------===//
// Hash Function
//===----------------------------------------------------------------------===//

#if defined(IREE_TOKENIZER_HAS_CRC32C)

// CRC32C hash using SSE4.2 hardware instructions.
// Processes 8 bytes at a time for maximum throughput.
// Returns non-zero hash (0 is reserved for empty slots).
#if defined(IREE_COMPILER_MSVC)
static inline uint32_t iree_tokenizer_hash(iree_string_view_t text) {
#else
__attribute__((target("sse4.2"))) static inline uint32_t iree_tokenizer_hash(
    iree_string_view_t text) {
#endif
  uint32_t crc = 0xFFFFFFFFu;
  const uint8_t* p = (const uint8_t*)text.data;
  iree_host_size_t length = text.size;

#if defined(IREE_ARCH_X86_64)
  // Process 8 bytes at a time on 64-bit.
  // Use memcpy for potentially unaligned access (compiler optimizes to single
  // load on x86 where unaligned access is supported).
  while (length >= 8) {
    uint64_t value = 0;
    memcpy(&value, p, sizeof(value));
    crc = (uint32_t)_mm_crc32_u64(crc, value);
    p += 8;
    length -= 8;
  }
#endif  // IREE_ARCH_X86_64

  // Process 4 bytes at a time.
  while (length >= 4) {
    uint32_t value = 0;
    memcpy(&value, p, sizeof(value));
    crc = _mm_crc32_u32(crc, value);
    p += 4;
    length -= 4;
  }

  // Process remaining bytes.
  while (length > 0) {
    crc = _mm_crc32_u8(crc, *p++);
    length--;
  }

  return crc ? crc : 1;  // Ensure non-zero.
}

#else  // !IREE_TOKENIZER_HAS_CRC32C

// FNV-1a fallback hash function for platforms without CRC32C.
// Returns non-zero hash (0 is reserved for empty slots).
static inline uint32_t iree_tokenizer_hash(iree_string_view_t text) {
  uint32_t hash = 2166136261u;  // FNV offset basis.
  for (iree_host_size_t i = 0; i < text.size; ++i) {
    hash ^= (uint8_t)text.data[i];
    hash *= 16777619u;  // FNV prime.
  }
  return hash ? hash : 1;  // Ensure non-zero.
}

#endif  // IREE_TOKENIZER_HAS_CRC32C

//===----------------------------------------------------------------------===//
// Internal Types
//===----------------------------------------------------------------------===//

// Hash slot: stores hash and token ID for linear probing.
typedef struct iree_tokenizer_vocab_hash_slot_t {
  uint32_t hash;  // 0 = empty slot.
  // Token ID (index into token array).
  int32_t token_id;
} iree_tokenizer_vocab_hash_slot_t;

struct iree_tokenizer_vocab_hash_t {
  iree_allocator_t allocator;
  const iree_tokenizer_token_t* tokens;      // Reference to token array.
  iree_host_size_t token_count;              // Number of tokens in array.
  iree_const_byte_span_t string_table;       // Reference to string data.
  iree_host_size_t slot_count;               // Power of 2.
  iree_host_size_t slot_mask;                // slot_count - 1.
  iree_tokenizer_vocab_hash_slot_t slots[];  // Flexible array member.
};

//===----------------------------------------------------------------------===//
// Internal Helpers
//===----------------------------------------------------------------------===//

// Calculates the slot count for a hash table with the given parameters.
// Returns 0 if the calculation would overflow.
static iree_host_size_t iree_tokenizer_vocab_hash_slot_count(
    iree_host_size_t token_count, uint8_t load_percent) {
  // Clamp load_percent to valid range.
  // Maximum 90% to guarantee at least 10% empty slots for linear probing.
  if (load_percent == 0) load_percent = 1;
  if (load_percent > 90) load_percent = 90;

  // slot_count = ceil(token_count * 100 / load_percent), rounded to power of 2.
  // Use 64-bit arithmetic to prevent overflow for large vocabs.
  uint64_t min_slots_64 =
      ((uint64_t)token_count * 100 + load_percent - 1) / load_percent;
  if (min_slots_64 > (uint64_t)IREE_HOST_SIZE_MAX / 2) {
    return 0;  // Overflow.
  }
  iree_host_size_t slot_count =
      (iree_host_size_t)iree_math_round_up_to_pow2_u64(min_slots_64);
  return slot_count < 4 ? 4 : slot_count;
}

// Gets the string view for a token from the string table.
// Returns empty string view if token's offset/length exceeds string table
// bounds.
static inline iree_string_view_t iree_tokenizer_token_string(
    const iree_tokenizer_token_t* token, iree_const_byte_span_t string_table) {
  // Use subtraction-based bounds check to prevent overflow bypass.
  if (token->string_offset > string_table.data_length ||
      token->string_length > string_table.data_length - token->string_offset) {
    return iree_string_view_empty();
  }
  return iree_make_string_view(
      (const char*)string_table.data + token->string_offset,
      token->string_length);
}

//===----------------------------------------------------------------------===//
// Storage Size / Initialize / Deinitialize
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_vocab_hash_storage_size(
    iree_host_size_t token_count, uint8_t load_percent,
    iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;
  iree_host_size_t slot_count =
      iree_tokenizer_vocab_hash_slot_count(token_count, load_percent);
  if (slot_count == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hash table slot count overflow");
  }
  if (!iree_host_size_checked_mul_add(
          sizeof(iree_tokenizer_vocab_hash_t), slot_count,
          sizeof(iree_tokenizer_vocab_hash_slot_t), out_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hash table storage size overflow");
  }
  return iree_ok_status();
}

void iree_tokenizer_vocab_hash_initialize(
    const iree_tokenizer_token_t* tokens, iree_host_size_t token_count,
    iree_const_byte_span_t string_table, uint8_t load_percent,
    iree_tokenizer_vocab_hash_t* out_hash) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_hash);

  iree_host_size_t slot_count =
      iree_tokenizer_vocab_hash_slot_count(token_count, load_percent);
  iree_host_size_t slot_mask = slot_count - 1;

  // Initialize header. Null allocator indicates embedded storage.
  out_hash->allocator = iree_allocator_null();
  out_hash->tokens = tokens;
  out_hash->token_count = token_count;
  out_hash->string_table = string_table;
  out_hash->slot_count = slot_count;
  out_hash->slot_mask = slot_mask;

  // Zero all slots (hash=0 means empty).
  memset(out_hash->slots, 0,
         slot_count * sizeof(iree_tokenizer_vocab_hash_slot_t));

  // Insert all tokens (skip UNUSED gap slots in sparse vocabs).
  for (iree_host_size_t i = 0; i < token_count; ++i) {
    if (iree_any_bit_set(tokens[i].attributes,
                         IREE_TOKENIZER_TOKEN_ATTR_UNUSED)) {
      continue;
    }
    iree_string_view_t token_string =
        iree_tokenizer_token_string(&tokens[i], string_table);
    uint32_t h = iree_tokenizer_hash(token_string);
    iree_host_size_t slot_index = h & slot_mask;

    // Linear probe to find empty slot.
    while (out_hash->slots[slot_index].hash != 0) {
      slot_index = (slot_index + 1) & slot_mask;
    }

    out_hash->slots[slot_index].hash = h;
    out_hash->slots[slot_index].token_id = (int32_t)i;
  }
  IREE_TRACE_ZONE_END(z0);
}

void iree_tokenizer_vocab_hash_deinitialize(iree_tokenizer_vocab_hash_t* hash) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(hash);
  (void)hash;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Build / Free
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_vocab_hash_build(
    const iree_tokenizer_token_t* tokens, iree_host_size_t token_count,
    iree_const_byte_span_t string_table, uint8_t load_percent,
    iree_allocator_t allocator, iree_tokenizer_vocab_hash_t** out_hash) {
  IREE_ASSERT_ARGUMENT(out_hash);
  *out_hash = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Calculate storage size.
  iree_host_size_t storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_vocab_hash_storage_size(token_count, load_percent,
                                                 &storage_size));

  // Allocate hash table.
  iree_tokenizer_vocab_hash_t* hash = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, storage_size, (void**)&hash));

  // Initialize hash table.
  iree_tokenizer_vocab_hash_initialize(tokens, token_count, string_table,
                                       load_percent, hash);

  // Set allocator for standalone allocation (initialize sets null).
  hash->allocator = allocator;

  *out_hash = hash;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_tokenizer_vocab_hash_free(iree_tokenizer_vocab_hash_t* hash) {
  if (!hash) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = hash->allocator;
  iree_tokenizer_vocab_hash_deinitialize(hash);
  iree_allocator_free(allocator, hash);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Lookup
//===----------------------------------------------------------------------===//

int32_t iree_tokenizer_vocab_hash_lookup(
    const iree_tokenizer_vocab_hash_t* hash, iree_string_view_t text) {
  if (!hash || !hash->tokens) return -1;

  uint32_t h = iree_tokenizer_hash(text);
  iree_host_size_t slot_index = h & hash->slot_mask;

  // Linear probe until we find matching hash or empty slot.
  // Probe limit guards against infinite loops on corrupted hash tables.
  iree_host_size_t probes = 0;
  while (hash->slots[slot_index].hash != 0 && probes < hash->slot_count) {
    if (hash->slots[slot_index].hash == h) {
      // Hash matches, verify string.
      int32_t token_id = hash->slots[slot_index].token_id;
      // Bounds check: token_id must be valid index into token array.
      if (token_id < 0 || (iree_host_size_t)token_id >= hash->token_count) {
        return -1;
      }
      iree_string_view_t token_string = iree_tokenizer_token_string(
          &hash->tokens[token_id], hash->string_table);
      if (token_string.size == text.size &&
          iree_string_view_equal(token_string, text)) {
        return token_id;
      }
    }
    slot_index = (slot_index + 1) & hash->slot_mask;
    ++probes;
  }

  return -1;  // Not found.
}
