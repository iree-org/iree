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

#include "iree/hal/host/host_buffer.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "iree/base/logging.h"
#include "iree/base/status.h"

namespace iree {
namespace hal {

class Allocator;

HostBuffer::HostBuffer(Allocator* allocator, MemoryTypeBitfield memory_type,
                       MemoryAccessBitfield allowed_access,
                       BufferUsageBitfield usage, device_size_t allocation_size,
                       void* data, bool owns_data)
    : Buffer(allocator, memory_type, allowed_access, usage, allocation_size, 0,
             allocation_size),
      data_(data),
      owns_data_(owns_data) {}

HostBuffer::~HostBuffer() {
  if (owns_data_ && data_) {
    std::free(data_);
    data_ = nullptr;
  }
}

Status HostBuffer::FillImpl(device_size_t byte_offset,
                            device_size_t byte_length, const void* pattern,
                            device_size_t pattern_length) {
  auto data_ptr = data_;
  switch (pattern_length) {
    case 1: {
      uint8_t* data = static_cast<uint8_t*>(data_ptr);
      uint8_t value_bits = *static_cast<const uint8_t*>(pattern);
      std::fill_n(data + byte_offset, byte_length, value_bits);
      break;
    }
    case 2: {
      uint16_t* data = static_cast<uint16_t*>(data_ptr);
      uint16_t value_bits = *static_cast<const uint16_t*>(pattern);
      std::fill_n(data + byte_offset / sizeof(uint16_t),
                  byte_length / sizeof(uint16_t), value_bits);
      break;
    }
    case 4: {
      uint32_t* data = static_cast<uint32_t*>(data_ptr);
      uint32_t value_bits = *static_cast<const uint32_t*>(pattern);
      std::fill_n(data + byte_offset / sizeof(uint32_t),
                  byte_length / sizeof(uint32_t), value_bits);
      break;
    }
    default:
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Unsupported scalar data size: " << pattern_length;
  }
  return OkStatus();
}

Status HostBuffer::ReadDataImpl(device_size_t source_offset, void* data,
                                device_size_t data_length) {
  auto data_ptr = static_cast<uint8_t*>(data_);
  std::memcpy(data, data_ptr + source_offset, data_length);
  return OkStatus();
}

Status HostBuffer::WriteDataImpl(device_size_t target_offset, const void* data,
                                 device_size_t data_length) {
  auto data_ptr = static_cast<uint8_t*>(data_);
  std::memcpy(data_ptr + target_offset, data, data_length);
  return OkStatus();
}

Status HostBuffer::CopyDataImpl(device_size_t target_offset,
                                Buffer* source_buffer,
                                device_size_t source_offset,
                                device_size_t data_length) {
  // This is pretty terrible. Let's not do this.
  // TODO(benvanik): a way for allocators to indicate transfer compat.
  IREE_ASSIGN_OR_RETURN(auto source_data,
                        source_buffer->MapMemory<uint8_t>(
                            MemoryAccess::kRead, source_offset, data_length));
  CHECK_EQ(data_length, source_data.size());
  auto data_ptr = static_cast<uint8_t*>(data_);
  std::memcpy(data_ptr + target_offset, source_data.data(), data_length);
  return OkStatus();
}

Status HostBuffer::MapMemoryImpl(MappingMode mapping_mode,
                                 MemoryAccessBitfield memory_access,
                                 device_size_t local_byte_offset,
                                 device_size_t local_byte_length,
                                 void** out_data) {
  auto data_ptr = static_cast<uint8_t*>(data_);
  *out_data = data_ptr + local_byte_offset;

  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (AnyBitSet(memory_access & MemoryAccess::kDiscard)) {
    std::memset(data_ptr + local_byte_offset, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  return OkStatus();
}

Status HostBuffer::UnmapMemoryImpl(device_size_t local_byte_offset,
                                   device_size_t local_byte_length,
                                   void* data) {
  // No-op? We still want error checking to make finding misuse easier.
  return OkStatus();
}

Status HostBuffer::InvalidateMappedMemoryImpl(device_size_t local_byte_offset,
                                              device_size_t local_byte_length) {
  // No-op? We still want error checking to make finding misuse easier.
  return OkStatus();
}

Status HostBuffer::FlushMappedMemoryImpl(device_size_t local_byte_offset,
                                         device_size_t local_byte_length) {
  // No-op? We still want error checking to make finding misuse easier.
  return OkStatus();
}

}  // namespace hal
}  // namespace iree
