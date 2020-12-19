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

#ifndef IREE_HAL_CC_ALLOCATOR_H_
#define IREE_HAL_CC_ALLOCATOR_H_

#include <cstddef>
#include <memory>

#include "absl/types/span.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/cc/buffer.h"

namespace iree {
namespace hal {

// Allocates buffers for a particular device memory space.
//
// Buffers allocated are only guaranteed to work with the driver that the
// allocator services. Any attempt to use buffers on drivers they were not
// allocated from must first be checked with CanUseBuffer.
//
// Thread-safe.
class Allocator : public RefObject<Allocator> {
 public:
  virtual ~Allocator() = default;

  // Returns true if the device can use the given buffer for the provided usage.
  // For buffers allocated from this allocator it's expected that the result
  // will always be true. For buffers that originate from another allocator
  // there may be limited support for cross-device usage.
  //
  // Returning false indicates that the buffer must be transferred externally
  // into a buffer compatible with the device this allocator services.
  bool CanUseBuffer(Buffer* buffer,
                    iree_hal_buffer_usage_t intended_usage) const {
    return CanUseBufferLike(buffer->allocator(), buffer->memory_type(),
                            buffer->usage(), intended_usage);
  }
  virtual bool CanUseBufferLike(
      Allocator* source_allocator, iree_hal_memory_type_t memory_type,
      iree_hal_buffer_usage_t buffer_usage,
      iree_hal_buffer_usage_t intended_usage) const = 0;

  // Returns true if the allocator can allocate a buffer with the given
  // attributes.
  virtual bool CanAllocate(iree_hal_memory_type_t memory_type,
                           iree_hal_buffer_usage_t buffer_usage,
                           size_t allocation_size) const = 0;

  // Adjusts allocation parameters to be compatible with the allocator.
  // Certain allocators may require particular memory types to function. By
  // adjusting the parameters prior to allocation callers can be sure they are
  // able to successfully Allocate a buffer later on with the same parameters.
  virtual Status MakeCompatible(iree_hal_memory_type_t* memory_type,
                                iree_hal_buffer_usage_t* buffer_usage) const {
    return OkStatus();
  }

  // Allocates a buffer from the allocator.
  // Fails if the memory type requested for the given usage cannot be serviced.
  // Callers can use CanAllocate to decide their memory use strategy.
  //
  // The memory type of the buffer returned may differ from the requested value
  // if the device can provide more functionality; for example, if requesting
  // IREE_HAL_MEMORY_TYPE_HOST_VISIBLE but the memory is really host cached you
  // may get a buffer back with IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
  // IREE_HAL_MEMORY_TYPE_HOST_CACHED. The only requirement is that the buffer
  // satisfy the required bits.
  virtual StatusOr<ref_ptr<Buffer>> Allocate(
      iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t buffer_usage,
      size_t allocation_size) = 0;

  // Wraps an existing host heap allocation in a buffer.
  // Ownership of the host allocation remains with the caller and the memory
  // must remain valid for so long as the Buffer may be in use.
  // Will have IREE_HAL_MEMORY_TYPE_HOST_LOCAL in most cases and may not be
  // usable by the device.
  //
  // The inference optimizer makes assumptions about buffer aliasing based on
  // Buffer instances and because of this wrapping the same host buffer in
  // multiple Buffers will create potential memory aliasing issues that can be
  // difficult to track down. There's no checking as to whether a host buffer
  // has already been wrapped so it's best for callers to ensure this is never
  // possible (the simplest way being to never use Wrap and always just allocate
  // new Buffers).
  //
  // Fails if the allocator cannot access host memory in this way.
  StatusOr<ref_ptr<Buffer>> Wrap(iree_hal_memory_type_t memory_type,
                                 iree_hal_buffer_usage_t buffer_usage,
                                 const void* data, size_t data_length) {
    return WrapMutable(memory_type, IREE_HAL_MEMORY_ACCESS_READ, buffer_usage,
                       const_cast<void*>(data), data_length);
  }
  virtual StatusOr<ref_ptr<Buffer>> WrapMutable(
      iree_hal_memory_type_t memory_type,
      iree_hal_memory_access_t allowed_access,
      iree_hal_buffer_usage_t buffer_usage, void* data, size_t data_length) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Allocator does not support wrapping host memory";
  }
  template <typename T>
  StatusOr<ref_ptr<Buffer>> Wrap(iree_hal_memory_type_t memory_type,
                                 iree_hal_buffer_usage_t buffer_usage,
                                 absl::Span<const T> data);
  template <typename T>
  StatusOr<ref_ptr<Buffer>> WrapMutable(iree_hal_memory_type_t memory_type,
                                        iree_hal_memory_access_t allowed_access,
                                        iree_hal_buffer_usage_t buffer_usage,
                                        absl::Span<T> data);
};

// Inline functions and template definitions follow:

template <typename T>
StatusOr<ref_ptr<Buffer>> Allocator::Wrap(iree_hal_memory_type_t memory_type,
                                          iree_hal_buffer_usage_t buffer_usage,
                                          absl::Span<const T> data) {
  return Wrap(memory_type, buffer_usage, data.data(), data.size() * sizeof(T));
}

template <typename T>
StatusOr<ref_ptr<Buffer>> Allocator::WrapMutable(
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t buffer_usage, absl::Span<T> data) {
  return WrapMutable(memory_type, allowed_access, buffer_usage, data.data(),
                     data.size() * sizeof(T));
}

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CC_ALLOCATOR_H_
