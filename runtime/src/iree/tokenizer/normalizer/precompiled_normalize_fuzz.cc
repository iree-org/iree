// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for precompiled normalizer input handling and streaming.
//
// Uses a valid precompiled trie built at startup (with common NFKC-like
// mappings) to focus testing on the normalization algorithm rather than
// format parsing.
//
// Tests the normalization pipeline against:
// - All Unicode code point ranges (ASCII, BMP, astral planes)
// - Invalid UTF-8 sequences (overlong, surrogate, truncated)
// - Very long inputs that exceed internal buffers
// - Streaming with varied chunk sizes (1 byte to full input)
// - Small output buffers that force pending buffer draining
// - Inputs that stress overlap buffer (patterns at chunk boundaries)
// - Rapid alternation between matching and non-matching sequences
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer/precompiled.h"

// Global normalizer instance built once at fuzzer startup.
static iree_tokenizer_normalizer_t* g_normalizer = NULL;

//===----------------------------------------------------------------------===//
// Trie Builder (minimal copy from precompiled_test.cc)
//===----------------------------------------------------------------------===//

// Builds a precompiled charsmap binary from key->replacement mappings.
static std::vector<uint8_t> BuildCharsmap(
    const std::map<std::string, std::string>& mappings) {
  // Phase 1: Build pool with NUL-terminated strings.
  std::vector<uint8_t> pool;
  std::map<std::string, uint32_t> pool_offsets;
  for (const auto& [key, replacement] : mappings) {
    pool_offsets[key] = static_cast<uint32_t>(pool.size());
    for (char c : replacement) {
      pool.push_back(static_cast<uint8_t>(c));
    }
    pool.push_back(0);  // NUL terminator.
  }

  // Phase 2: Build trie using XOR-based addressing.
  std::vector<uint32_t> trie;
  trie.resize(1024, 0);

  const uint32_t ROOT_OFFSET = 256;
  trie[0] = (ROOT_OFFSET << 10);  // Root unit.

  uint32_t next_leaf_slot = 512;

  for (const auto& [key, replacement] : mappings) {
    if (key.empty()) continue;

    uint32_t pos = ROOT_OFFSET;

    for (size_t i = 0; i < key.size(); ++i) {
      uint8_t c = static_cast<uint8_t>(key[i]);
      uint32_t child_pos = pos ^ c;

      while (child_pos >= trie.size() || next_leaf_slot >= trie.size()) {
        trie.resize(trie.size() * 2, 0);
      }

      bool is_last = (i == key.size() - 1);

      if (is_last) {
        uint32_t leaf_slot = next_leaf_slot++;
        while (leaf_slot >= trie.size()) {
          trie.resize(trie.size() * 2, 0);
        }
        uint32_t offset_to_leaf = child_pos ^ leaf_slot;
        trie[child_pos] = c | (1u << 8) | (offset_to_leaf << 10);
        trie[leaf_slot] = 0x80000000u | pool_offsets.at(key);
      } else {
        uint32_t next_base = child_pos + 256 * (i + 2);
        while (next_base + 256 >= trie.size()) {
          trie.resize(trie.size() * 2, 0);
        }
        uint32_t offset_to_next = child_pos ^ next_base;
        trie[child_pos] = c | (offset_to_next << 10);
        pos = next_base;
      }
    }
  }

  // Trim trailing zeros.
  while (!trie.empty() && trie.back() == 0) {
    trie.pop_back();
  }

  // Phase 3: Build final binary.
  std::vector<uint8_t> result;
  uint32_t trie_size = static_cast<uint32_t>(trie.size() * sizeof(uint32_t));
  result.push_back(trie_size & 0xFF);
  result.push_back((trie_size >> 8) & 0xFF);
  result.push_back((trie_size >> 16) & 0xFF);
  result.push_back((trie_size >> 24) & 0xFF);

  for (uint32_t unit : trie) {
    result.push_back(unit & 0xFF);
    result.push_back((unit >> 8) & 0xFF);
    result.push_back((unit >> 16) & 0xFF);
    result.push_back((unit >> 24) & 0xFF);
  }

  for (uint8_t b : pool) {
    result.push_back(b);
  }

  return result;
}

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  (void)argc;
  (void)argv;

  // Build a normalizer with common NFKC-like mappings to exercise trie lookups.
  std::map<std::string, std::string> mappings = {
      // Single-byte ASCII transformations.
      {"A", "a"},
      {"B", "b"},
      {"C", "c"},
      // Multi-byte UTF-8 sequences.
      {"\xC3\x80", "A"},        // À -> A (Latin capital A with grave)
      {"\xC3\xA9", "e"},        // é -> e
      {"\xE2\x84\xA6", "Ohm"},  // Ω (Ohm sign) -> "Ohm"
      {"\xEF\xBC\xA1", "A"},    // Ａ (fullwidth A) -> A
      // CJK compatibility.
      {"\xE3\x80\x80", " "},  // Ideographic space -> ASCII space
      // Ligatures.
      {"\xEF\xAC\x81", "fi"},  // ﬁ -> fi
      {"\xEF\xAC\x82", "fl"},  // ﬂ -> fl
      // Control characters.
      {"\x00", ""},          // NUL -> empty (remove)
      {"\x0D\x0A", "\x0A"},  // CRLF -> LF
      // Multi-codepoint sequences.
      {"\xE2\x80\x93", "-"},   // En dash -> hyphen
      {"\xE2\x80\x94", "--"},  // Em dash -> double hyphen
  };

  std::vector<uint8_t> charsmap = BuildCharsmap(mappings);
  iree_const_byte_span_t span =
      iree_make_const_byte_span(charsmap.data(), charsmap.size());

  iree_status_t status = iree_tokenizer_precompiled_normalizer_allocate(
      span, iree_allocator_system(), &g_normalizer);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }

  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (g_normalizer == NULL || size < 2) return 0;

  // First byte controls test mode and chunk sizing.
  uint8_t mode = data[0];
  data++;
  size--;

  iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  // Allocate state.
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(g_normalizer);
  void* state_buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_allocator_system(), state_size, &state_buffer);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  iree_tokenizer_normalizer_state_t* state = NULL;
  status = iree_tokenizer_normalizer_state_initialize(g_normalizer,
                                                      state_buffer, &state);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), state_buffer);
    iree_status_ignore(status);
    return 0;
  }

  // Output buffer - size controlled by mode to test pending buffer draining.
  size_t output_size = (mode & 0x03) == 0   ? 4      // Very small.
                       : (mode & 0x03) == 1 ? 64     // Small.
                       : (mode & 0x03) == 2 ? 256    // Medium.
                                            : 4096;  // Large.
  char* output = (char*)malloc(output_size);
  if (!output) {
    iree_allocator_free(iree_allocator_system(), state_buffer);
    return 0;
  }

  if ((mode & 0x80) == 0) {
    //===------------------------------------------------------------------===//
    // Mode A: Streaming with varied chunk sizes
    //===------------------------------------------------------------------===//
    size_t offset = 0;
    size_t chunk_index = 0;

    while (offset < size) {
      // Data-dependent chunk size.
      size_t chunk_base =
          (chunk_index < size) ? (data[chunk_index % size] & 0x1F) : 8;
      size_t chunk_size = chunk_base + 1;  // 1-32 bytes.
      if (chunk_size > size - offset) chunk_size = size - offset;

      iree_string_view_t chunk =
          iree_make_string_view(input.data + offset, chunk_size);
      iree_mutable_string_view_t out_view =
          iree_make_mutable_string_view(output, output_size);

      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t bytes_written = 0;

      status = iree_tokenizer_normalizer_state_process(
          state, chunk, out_view, IREE_TOKENIZER_NORMALIZER_FLAG_NONE,
          &bytes_consumed, &bytes_written);

      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        break;
      }

      offset += bytes_consumed;
      ++chunk_index;

      // Prevent infinite loops.
      if (bytes_consumed == 0 && chunk_size > 0) {
        // Output buffer might be full - try draining with finalize pattern.
        break;
      }
      if (chunk_index > 10000) {
        break;
      }
    }

    // Finalize to flush remaining data.
    iree_mutable_string_view_t final_view =
        iree_make_mutable_string_view(output, output_size);
    iree_host_size_t final_written = 0;
    status = iree_tokenizer_normalizer_state_finalize(state, final_view,
                                                      &final_written);
    iree_status_ignore(status);

  } else {
    //===------------------------------------------------------------------===//
    // Mode B: One-shot process + finalize
    //===------------------------------------------------------------------===//
    iree_mutable_string_view_t out_view =
        iree_make_mutable_string_view(output, output_size);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t bytes_written = 0;

    status = iree_tokenizer_normalizer_state_process(
        state, input, out_view, IREE_TOKENIZER_NORMALIZER_FLAG_NONE,
        &bytes_consumed, &bytes_written);
    iree_status_ignore(status);

    iree_mutable_string_view_t final_view =
        iree_make_mutable_string_view(output, output_size);
    iree_host_size_t final_written = 0;
    status = iree_tokenizer_normalizer_state_finalize(state, final_view,
                                                      &final_written);
    iree_status_ignore(status);
  }

  free(output);
  iree_allocator_free(iree_allocator_system(), state_buffer);

  return 0;
}
