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
#include "iree/hal/buffer.h"

namespace iree {
namespace hal {

// A buffer type that operates on host pointers.
// This can be used by Allocator implementations when they support operating
// on host memory (or mapping their memory to host memory).
class HostBuffer : public Buffer {
 public:
  HostBuffer(Allocator* allocator, MemoryTypeBitfield memory_type,
             MemoryAccessBitfield allowed_access, BufferUsageBitfield usage,
             device_size_t allocation_size, void* data, bool owns_data);

  ~HostBuffer() override;

 protected:
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

 private:
  void* data_ = nullptr;
  bool owns_data_ = false;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_BUFFER_H_
