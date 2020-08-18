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

#include "iree/hal/allocator.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {

bool Allocator::CanUseBuffer(Buffer* buffer,
                             BufferUsageBitfield intended_usage) const {
  return CanUseBufferLike(buffer->allocator(), buffer->memory_type(),
                          buffer->usage(), intended_usage);
}

StatusOr<ref_ptr<Buffer>> Allocator::AllocateConstant(
    BufferUsageBitfield buffer_usage, ref_ptr<Buffer> source_buffer) {
  if (AnyBitSet(source_buffer->usage() & BufferUsage::kConstant) &&
      CanUseBuffer(source_buffer.get(), buffer_usage)) {
    // Buffer can be used directly by the device.
    return source_buffer;
  }

  IREE_TRACE_SCOPE0("Allocator::AllocateConstant");

  // We need to map so we can copy into it.
  buffer_usage |= BufferUsage::kMapping;
  // It will be constant after we write it.
  buffer_usage |= BufferUsage::kConstant;

  MemoryTypeBitfield memory_type =
      MemoryType::kDeviceLocal | MemoryType::kHostVisible;
  IREE_ASSIGN_OR_RETURN(
      auto device_buffer,
      Allocate(memory_type, buffer_usage, source_buffer->byte_length()));
  IREE_ASSIGN_OR_RETURN(auto source_mapping,
                        source_buffer->MapMemory<uint8_t>(MemoryAccess::kRead));
  IREE_RETURN_IF_ERROR(device_buffer->WriteData(0, source_mapping.data(),
                                                source_mapping.byte_length()));
  return device_buffer;
}

StatusOr<ref_ptr<Buffer>> Allocator::Wrap(MemoryTypeBitfield memory_type,
                                          BufferUsageBitfield buffer_usage,
                                          const void* data,
                                          size_t data_length) {
  return WrapMutable(memory_type, MemoryAccess::kRead, buffer_usage,
                     const_cast<void*>(data), data_length);
}

StatusOr<ref_ptr<Buffer>> Allocator::WrapMutable(
    MemoryTypeBitfield memory_type, MemoryAccessBitfield allowed_access,
    BufferUsageBitfield buffer_usage, void* data, size_t data_length) {
  return UnimplementedErrorBuilder(IREE_LOC)
         << "Allocator does not support wrapping host memory";
}

}  // namespace hal
}  // namespace iree
