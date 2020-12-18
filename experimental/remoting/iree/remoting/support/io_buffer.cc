// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "experimental/remoting/iree/remoting/support/io_buffer.h"

#include <cstdlib>
#include <cstring>

namespace iree {
namespace remoting {

//===----------------------------------------------------------------------===//
// IoBufferPool
//===----------------------------------------------------------------------===//

IoBufferPool::~IoBufferPool() = default;
IoBufferPool::Slab::~Slab() { std::free(data); }

void IoBufferPool::GrowBuffers() {
  size_t step_size = buffer_physical_size();
  size_t alloc_size = step_size * slab_buffer_count_;
  IREE_DLOG(INFO) << "IoBufferPool add slab (step_size=" << step_size
                  << ", alloc_size=" << alloc_size << ")";

  // If this ever becomes hot, this could be done in one alloc instead of two.
  // Or various other smarter things...
  slab_head_ = std::unique_ptr<Slab>(new Slab{
      std::malloc(alloc_size),
      std::move(slab_head_),
  });

  // Initialize each buffer and add to list.
  auto current = static_cast<std::uint8_t *>(slab_head_->data);
  for (size_t i = 0; i < slab_buffer_count_; ++i, current += step_size) {
    IREE_DLOG(INFO) << "  + Add buffer to pool: "
                    << static_cast<void *>(current);
    head_ = new (current) IoBuffer(this, head_);
  }
}

//===----------------------------------------------------------------------===//
// IoBufferVec
//===----------------------------------------------------------------------===//

IoBufferVec::~IoBufferVec() {
  clear();
  free(entries_);
  free(extras_);
}

IoBufferVec::Pool::~Pool() {
  IoBufferVec *current = head_;
  while (current) {
    assert(current->is_free_ == 1);
    IoBufferVec *next = current->link_.next;
    delete current;
    current = next;
  }
}

void IoBufferVec::clear() {
  // Explicitly release backing buffers.
  for (uint32_t i = 0, e = size(); i < e; ++i) {
    extras_[i].~Extra();
  }
  size_ = 0;
}

void IoBufferVec::Grow(uint32_t new_capacity) {
  if (new_capacity == 0) {
    capacity_ = std::max(2, capacity_ + capacity_ / 2);
  } else {
    capacity_ = std::max(new_capacity, capacity_);
  }

  entries_ = static_cast<portable_iovec_t *>(
      realloc(entries_, sizeof(portable_iovec_t) * capacity_));
  extras_ = static_cast<Extra *>(realloc(extras_, sizeof(Extra) * capacity_));
}

}  // namespace remoting
}  // namespace iree
