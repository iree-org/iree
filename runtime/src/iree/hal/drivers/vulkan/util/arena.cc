// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/util/arena.h"

#include <cstdlib>

#include "iree/base/api.h"

namespace iree {

namespace {

// Rounds up to the next alignment value, if it is not already aligned.
template <typename T>
IREE_ATTRIBUTE_ALWAYS_INLINE constexpr T RoundToAlignment(
    T value, T alignment) noexcept {
  return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

Arena::Arena(size_t block_size) : block_size_(block_size) {}

Arena::~Arena() { Clear(); }

void Arena::Clear() {
  // Deallocate all memory.
  auto block_header = block_list_head_;
  while (block_header) {
    auto next_block = block_header->next_block;
    std::free(block_header);
    block_header = next_block;
  }
  block_list_head_ = nullptr;
  block_header = unused_block_list_head_;
  while (block_header) {
    auto next_block = block_header->next_block;
    std::free(block_header);
    block_header = next_block;
  }
  unused_block_list_head_ = nullptr;

  bytes_allocated_ = 0;
  block_bytes_allocated_ = 0;
}

void Arena::Reset() {
  // Move all blocks to the unused list and reset allocation count only.
  auto block_header = block_list_head_;
  while (block_header) {
    auto next_block = block_header->next_block;
    block_header->bytes_allocated = 0;
    block_header->next_block = unused_block_list_head_;
    unused_block_list_head_ = block_header;
    block_header = next_block;
  }
  block_list_head_ = nullptr;

  bytes_allocated_ = 0;
}

uint8_t* Arena::AllocateBytes(size_t length) {
  if (!length) {
    // Guarantee zero-length allocations return nullptr.
    return nullptr;
  }

  // Pad length allocated so we are machine word aligned.
  // This ensures the next allocation starts at the right boundary.
  size_t aligned_length = RoundToAlignment(length, sizeof(uintptr_t));

  if (aligned_length > block_size_) {
    // This allocation is larger than an entire block. That's bad.
    // We could allocate this with malloc (and then keep track of those to free
    // things), but for now let's just die.
    iree_abort();
    return nullptr;
  }

  if (!block_list_head_ ||
      block_list_head_->bytes_allocated + aligned_length > block_size_) {
    // Check to see if we have an existing unused block we can use.
    if (unused_block_list_head_) {
      // Move block from unused list to main list.
      auto block_header = unused_block_list_head_;
      unused_block_list_head_ = block_header->next_block;
      block_header->next_block = block_list_head_;
      block_header->bytes_allocated = 0;
      block_list_head_ = block_header;
    } else {
      // Allocate a new block.
      auto block_ptr = reinterpret_cast<uint8_t*>(
          std::malloc(sizeof(BlockHeader) + block_size_));
      auto block_header = reinterpret_cast<BlockHeader*>(block_ptr);
      block_header->next_block = block_list_head_;
      block_header->bytes_allocated = 0;
      block_list_head_ = block_header;
      block_bytes_allocated_ += sizeof(BlockHeader) + block_size_;
    }
  }

  BlockHeader* target_block = block_list_head_;
  auto data_ptr = reinterpret_cast<uint8_t*>(target_block) +
                  sizeof(BlockHeader) + target_block->bytes_allocated;
  target_block->bytes_allocated += aligned_length;

  bytes_allocated_ += length;

  return data_ptr;
}

}  // namespace iree
