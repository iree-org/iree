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

#ifndef IREE_HAL_HOST_BUFFER_H_
#define IREE_HAL_HOST_BUFFER_H_

#include <cstdint>

#include "iree/base/status.h"
#include "iree/hal/cc/buffer.h"

namespace iree {
namespace hal {

// A buffer type that operates on host pointers.
// This can be used by Allocator implementations when they support operating
// on host memory (or mapping their memory to host memory).
class HostBuffer : public Buffer {
 public:
  HostBuffer(Allocator* allocator, iree_hal_memory_type_t memory_type,
             iree_hal_memory_access_t allowed_access,
             iree_hal_buffer_usage_t usage, iree_device_size_t allocation_size,
             void* data, bool owns_data);

  ~HostBuffer() override;

  const void* data() const { return data_; }
  void* mutable_data() { return data_; }

 protected:
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

 private:
  void* data_ = nullptr;
  bool owns_data_ = false;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_BUFFER_H_
