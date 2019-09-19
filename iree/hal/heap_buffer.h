// Copyright 2019 Google LLC
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

#ifndef IREE_HAL_HEAP_BUFFER_H_
#define IREE_HAL_HEAP_BUFFER_H_

#include <memory>

#include "iree/base/status.h"
#include "iree/hal/buffer.h"

namespace iree {
namespace hal {

// Factory for buffers that are allocated from the host heap (malloc/free).
// These buffers cannot be used by devices and will incur copies/transfers when
// used. Prefer device-specific allocators instead.
class HeapBuffer {
 public:
  // Allocates a zeroed host heap buffer of the given size.
  // Returns a buffer allocated with malloc and have MemoryType::kHostLocal
  // and will not be usable by devices without copies.
  static ref_ptr<Buffer> Allocate(MemoryTypeBitfield memory_type,
                                  BufferUsageBitfield usage,
                                  size_t allocation_size);
  static ref_ptr<Buffer> Allocate(BufferUsageBitfield usage,
                                  size_t allocation_size) {
    return Allocate(MemoryType::kHostLocal, usage, allocation_size);
  }

  // Allocates a host heap buffer with a copy of the given data.
  // Returns a buffer allocated with malloc and have MemoryType::kHostLocal
  // and will not be usable by devices without copies.
  static ref_ptr<Buffer> AllocateCopy(BufferUsageBitfield usage,
                                      const void* data, size_t data_length);
  static ref_ptr<Buffer> AllocateCopy(BufferUsageBitfield usage,
                                      MemoryAccessBitfield allowed_access,
                                      const void* data, size_t data_length);
  template <typename T>
  static ref_ptr<Buffer> AllocateCopy(BufferUsageBitfield usage,
                                      absl::Span<const T> data);
  template <typename T>
  static ref_ptr<Buffer> AllocateCopy(BufferUsageBitfield usage,
                                      MemoryAccessBitfield allowed_access,
                                      absl::Span<const T> data);

  // Wraps an existing host heap allocation in a buffer.
  // Ownership of the host allocation remains with the caller and the memory
  // must remain valid for so long as the Buffer may be in use.
  // Will have MemoryType::kHostLocal in most cases and may not be usable
  // by the device.
  static ref_ptr<Buffer> Wrap(MemoryTypeBitfield memory_type,
                              BufferUsageBitfield usage, const void* data,
                              size_t data_length);
  static ref_ptr<Buffer> WrapMutable(MemoryTypeBitfield memory_type,
                                     MemoryAccessBitfield allowed_access,
                                     BufferUsageBitfield usage, void* data,
                                     size_t data_length);
  template <typename T>
  static ref_ptr<Buffer> Wrap(MemoryTypeBitfield memory_type,
                              BufferUsageBitfield usage,
                              absl::Span<const T> data);
  template <typename T>
  static ref_ptr<Buffer> WrapMutable(MemoryTypeBitfield memory_type,
                                     MemoryAccessBitfield allowed_access,
                                     BufferUsageBitfield usage,
                                     absl::Span<T> data);
};

// Inline functions and template definitions follow:

template <typename T>
ref_ptr<Buffer> HeapBuffer::AllocateCopy(BufferUsageBitfield usage,
                                         absl::Span<const T> data) {
  return HeapBuffer::AllocateCopy(usage, MemoryAccess::kAll, data);
}

template <typename T>
ref_ptr<Buffer> HeapBuffer::AllocateCopy(BufferUsageBitfield usage,
                                         MemoryAccessBitfield allowed_access,
                                         absl::Span<const T> data) {
  return HeapBuffer::AllocateCopy(usage, allowed_access, data.data(),
                                  data.size() * sizeof(T));
}

template <typename T>
ref_ptr<Buffer> HeapBuffer::Wrap(MemoryTypeBitfield memory_type,
                                 BufferUsageBitfield usage,
                                 absl::Span<const T> data) {
  return HeapBuffer::Wrap(memory_type, usage, data.data(),
                          data.size() * sizeof(T));
}

template <typename T>
ref_ptr<Buffer> HeapBuffer::WrapMutable(MemoryTypeBitfield memory_type,
                                        MemoryAccessBitfield allowed_access,
                                        BufferUsageBitfield usage,
                                        absl::Span<T> data) {
  return HeapBuffer::WrapMutable(memory_type, allowed_access, usage,
                                 data.data(), data.size() * sizeof(T));
}

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HEAP_BUFFER_H_
