// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab_hash.h"

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
  uint32_t hash;     // 0 = empty slot.
  int32_t token_id;  // Token ID (index into token array).
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
// Implementation
//===----------------------------------------------------------------------===//

// Gets the string view for a token from the string table.
static inline iree_string_view_t iree_tokenizer_token_string(
    const iree_tokenizer_token_t* token, iree_const_byte_span_t string_table) {
  return iree_make_string_view(
      (const char*)string_table.data + token->string_offset,
      token->string_length);
}

iree_status_t iree_tokenizer_vocab_hash_build(
    const iree_tokenizer_token_t* tokens, iree_host_size_t token_count,
    iree_const_byte_span_t string_table, uint8_t load_percent,
    iree_allocator_t allocator, iree_tokenizer_vocab_hash_t** out_hash) {
  IREE_ASSERT_ARGUMENT(out_hash);
  *out_hash = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Clamp load_percent to valid range.
  // IMPORTANT: Maximum 90% to guarantee at least 10% empty slots for
  // linear probing termination. 100% load would cause infinite loops
  // when searching for keys that don't exist.
  if (load_percent == 0) load_percent = 1;
  if (load_percent > 90) load_percent = 90;

  // Calculate slot count based on load factor.
  // slot_count = token_count * 100 / load_percent, rounded up to power of 2.
  // Minimum 4 slots to avoid degenerate cases.
  // Use 64-bit arithmetic to prevent overflow for large vocabs.
  uint64_t min_slots_64 =
      ((uint64_t)token_count * 100 + load_percent - 1) / load_percent;
  // Clamp to maximum representable size (prevents overflow in allocation).
  if (min_slots_64 > (uint64_t)IREE_HOST_SIZE_MAX / 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "vocabulary too large for hash table");
  }
  iree_host_size_t min_slots = (iree_host_size_t)min_slots_64;
  iree_host_size_t slot_count =
      (iree_host_size_t)iree_math_round_up_to_pow2_u64(min_slots);
  if (slot_count < 4) slot_count = 4;
  iree_host_size_t slot_mask = slot_count - 1;

  // Allocate hash table with flexible array for slots.
  iree_host_size_t total_size =
      sizeof(iree_tokenizer_vocab_hash_t) +
      slot_count * sizeof(iree_tokenizer_vocab_hash_slot_t);
  iree_tokenizer_vocab_hash_t* hash = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&hash));

  // Initialize.
  hash->allocator = allocator;
  hash->tokens = tokens;
  hash->token_count = token_count;
  hash->string_table = string_table;
  hash->slot_count = slot_count;
  hash->slot_mask = slot_mask;

  // Zero all slots (hash=0 means empty).
  memset(hash->slots, 0, slot_count * sizeof(iree_tokenizer_vocab_hash_slot_t));

  // Insert all tokens (skip UNUSED gap slots in sparse vocabs).
  for (iree_host_size_t i = 0; i < token_count; ++i) {
    // Skip gap slots marked as unused.
    if (iree_any_bit_set(tokens[i].attributes,
                         IREE_TOKENIZER_TOKEN_ATTR_UNUSED)) {
      continue;
    }
    iree_string_view_t token_string =
        iree_tokenizer_token_string(&tokens[i], string_table);
    uint32_t h = iree_tokenizer_hash(token_string);
    iree_host_size_t slot_index = h & slot_mask;

    // Linear probe to find empty slot.
    while (hash->slots[slot_index].hash != 0) {
      slot_index = (slot_index + 1) & slot_mask;
    }

    hash->slots[slot_index].hash = h;
    hash->slots[slot_index].token_id = (int32_t)i;
  }

  *out_hash = hash;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

int32_t iree_tokenizer_vocab_hash_lookup(
    const iree_tokenizer_vocab_hash_t* hash, iree_string_view_t text) {
  if (!hash) return -1;

  uint32_t h = iree_tokenizer_hash(text);
  iree_host_size_t slot_index = h & hash->slot_mask;

  // Linear probe until we find matching hash or empty slot.
  while (hash->slots[slot_index].hash != 0) {
    if (hash->slots[slot_index].hash == h) {
      // Hash matches, verify string.
      int32_t token_id = hash->slots[slot_index].token_id;
      // Bounds check: token_id must be valid index into token array.
      if (token_id < 0 || (iree_host_size_t)token_id >= hash->token_count) {
        return -1;
      }
      iree_string_view_t token_string = iree_tokenizer_token_string(
          &hash->tokens[token_id], hash->string_table);
      if (iree_string_view_equal(token_string, text)) {
        return token_id;
      }
    }
    slot_index = (slot_index + 1) & hash->slot_mask;
  }

  return -1;  // Not found.
}

void iree_tokenizer_vocab_hash_free(iree_tokenizer_vocab_hash_t* hash) {
  if (!hash) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(hash->allocator, hash);
  IREE_TRACE_ZONE_END(z0);
}
