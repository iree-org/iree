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

#ifndef IREE_HAL_METAL_METAL_BUFFER_H_
#define IREE_HAL_METAL_METAL_BUFFER_H_

#import <Metal/Metal.h>

#include "iree/hal/buffer.h"

namespace iree {
namespace hal {
namespace metal {

class MetalDirectAllocator;

// A buffer implementation for Metal that directly wraps a MTLBuffer.
class MetalBuffer final : public Buffer {
 public:
  // Creates a MetalBuffer instance with retaining the given id<MTLBuffer>.
  static StatusOr<ref_ptr<MetalBuffer>> Create(
      MetalDirectAllocator* allocator, iree_hal_memory_type_t memory_type,
      iree_hal_memory_access_t allowed_access, iree_hal_buffer_usage_t usage,
      iree_device_size_t allocation_size, iree_device_size_t byte_offset,
      iree_device_size_t byte_length, id<MTLBuffer> buffer,
      id<MTLCommandQueue> transfer_queue);

  // Creates a MetalBuffer instance without retaining the given id<MTLBuffer>.
  static StatusOr<ref_ptr<MetalBuffer>> CreateUnretained(
      MetalDirectAllocator* allocator, iree_hal_memory_type_t memory_type,
      iree_hal_memory_access_t allowed_access, iree_hal_buffer_usage_t usage,
      iree_device_size_t allocation_size, iree_device_size_t byte_offset,
      iree_device_size_t byte_length, id<MTLBuffer> buffer,
      id<MTLCommandQueue> transfer_queue);

  ~MetalBuffer() override;

  id<MTLBuffer> handle() const { return metal_handle_; }

 private:
  // Creates a MetalBuffer instance without retaining the given id<MTLBuffer>.
  MetalBuffer(MetalDirectAllocator* allocator,
              iree_hal_memory_type_t memory_type,
              iree_hal_memory_access_t allowed_access,
              iree_hal_buffer_usage_t usage, iree_device_size_t allocation_size,
              iree_device_size_t byte_offset, iree_device_size_t byte_length,
              id<MTLBuffer> buffer, id<MTLCommandQueue> transfer_queue);

  Status FillImpl(iree_device_size_t byte_offset,
                  iree_device_size_t byte_length, const void* pattern,
                  iree_device_size_t pattern_length) override;
  Status ReadDataImpl(iree_device_size_t source_offset, void* data,
                      iree_device_size_t data_length) override;
  Status WriteDataImpl(iree_device_size_t target_offset, const void* data,
                       iree_device_size_t data_length) override;
  Status CopyDataImpl(iree_device_size_t target_offset, Buffer* source_buffer,
                      iree_device_size_t source_offset,
                      iree_device_size_t data_length) override;

  Status MapMemoryImpl(MappingMode mapping_mode,
                       iree_hal_memory_access_t memory_access,
                       iree_device_size_t local_byte_offset,
                       iree_device_size_t local_byte_length,
                       void** out_data) override;
  Status UnmapMemoryImpl(iree_device_size_t local_byte_offset,
                         iree_device_size_t local_byte_length,
                         void* data) override;
  Status InvalidateMappedMemoryImpl(
      iree_device_size_t local_byte_offset,
      iree_device_size_t local_byte_length) override;
  Status FlushMappedMemoryImpl(iree_device_size_t local_byte_offset,
                               iree_device_size_t local_byte_length) override;

  // Returns true if we need to automatically invaliate/flush CPU caches to keep
  // memory hierarchy consistent.
  //
  // Note: this is needed when the buffer is requested with
  // IREE_HAL_MEMORY_TYPE_HOST_COHERENT bit but under the hood we are using
  // memory types that does not have that property natively, e.g.,
  // MTLStorageModeManaged. Under such circumstances, we need to perform the
  // invalidate/flush operation "automatically" for users.
  bool requires_autosync() const;

  // We need to hold an reference to the queue so that we can encode
  // synchronizeResource commands for synchronizing the buffer with
  // MTLResourceStorageModeManaged.
  id<MTLCommandQueue> metal_transfer_queue_;

  id<MTLBuffer> metal_handle_;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_BUFFER_H_
