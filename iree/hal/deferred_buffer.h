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

#ifndef IREE_HAL_DEFERRED_BUFFER_H_
#define IREE_HAL_DEFERRED_BUFFER_H_

#include <cstddef>
#include <memory>
#include <utility>

#include "iree/base/status.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"

namespace iree {
namespace hal {

// A Buffer that can have its underlying allocation changed at runtime.
// Unbound buffers act as a way to logically group dependent ranges of memory
// without needing to have allocated that memory yet.
//
// Usage:
//  // Setup two spans referencing ranges of a deferred buffer.
//  auto deferred_buffer = make_ref<DeferredBuffer>(..., 200);
//  IREE_ASSIGN_OR_RETURN(auto span0, Buffer::Subspan(deferred_buffer, 0, 100));
//  IREE_ASSIGN_OR_RETURN(auto span1, Buffer::Subspan(deferred_buffer, 100,
//  100));
//
//  // Attempting to access |deferred_buffer| or |span0| or |span1| will fail.
//  // ERROR: span0->Fill(false);
//
//  // Now allocate a real buffer to serve as storage for the data.
//  IREE_ASSIGN_OR_RETURN(auto allocated_buffer, Buffer::Allocate(..., 200));
//  IREE_RETURN_IF_ERROR(deferred_buffer->BindAllocation(
//      allocated_buffer, 0, kWholeBuffer));
//
//  // And now we can use the spans.
//  IREE_RETURN_IF_ERROR(span0->Fill(false));
//
//  // If at some point we want to detach the buffer from the allocation (so we
//  // can use a different allocation, reuse the memory, etc).
//  deferred_buffer->ResetAllocation();
//
// Thread-compatible. Attempting to rebind the allocation while other threads
// are using the buffer will lead to undefined behavior.
class DeferredBuffer : public Buffer {
 public:
  DeferredBuffer(Allocator* allocator, MemoryTypeBitfield memory_type,
                 MemoryAccessBitfield allowed_access, BufferUsageBitfield usage,
                 device_size_t byte_length);
  ~DeferredBuffer() override;

  // Grows the minimum allocation size of the buffer to |new_byte_length|.
  // Attempting to bind an allocation less than this size will fail. This must
  // only be called when the buffer is not bound to an allocation.
  Status GrowByteLength(device_size_t new_byte_length);

  // Binds or rebinds the deferred buffer to an allocated buffer.
  Status BindAllocation(ref_ptr<Buffer> allocated_buffer,
                        device_size_t byte_offset, device_size_t byte_length);

  // Resets the deferred buffer to have no binding.
  void ResetAllocation();

 private:
  // Resolves the allocated buffer that this subspan references into.
  // This will fail if the buffer has not yet been bound to an allocation or
  // the allocated buffer has not been committed.
  StatusOr<Buffer*> ResolveAllocation() const;

  Status FillImpl(device_size_t byte_offset, device_size_t byte_length,
                  const void* pattern, device_size_t pattern_length) override;
  Status ReadDataImpl(device_size_t source_offset, void* data,
                      device_size_t data_length) override;
  Status WriteDataImpl(device_size_t target_offset, const void* data,
                       device_size_t data_length) override;
  Status CopyDataImpl(device_size_t target_offset, Buffer* source_buffer,
                      device_size_t source_offset,
                      device_size_t data_length) override;
  Status MapMemoryImpl(MappingMode mapping_mode,
                       MemoryAccessBitfield memory_access,
                       device_size_t local_byte_offset,
                       device_size_t local_byte_length,
                       void** out_data) override;
  Status UnmapMemoryImpl(device_size_t local_byte_offset,
                         device_size_t local_byte_length, void* data) override;
  Status InvalidateMappedMemoryImpl(device_size_t local_byte_offset,
                                    device_size_t local_byte_length) override;
  Status FlushMappedMemoryImpl(device_size_t local_byte_offset,
                               device_size_t local_byte_length) override;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DEFERRED_BUFFER_H_
