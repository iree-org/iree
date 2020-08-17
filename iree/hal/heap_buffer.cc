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

#include "iree/hal/heap_buffer.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/host/host_buffer.h"

namespace iree {
namespace hal {

namespace {

// An allocator that allocates or wraps host-only buffers.
// The resulting buffers are not usable by most devices without a copy and
// using a device allocator is strongly preferred.
class HeapAllocator : public Allocator {
 public:
  // Returns a singleton heap allocator that can provide buffers that have
  // MemoryType::kHostLocal and are allocated with malloc/free.
  // These buffers will not be usable by devices directly and may incur
  // additional copies.
  static Allocator* std_heap();

  // TODO(benvanik): specify custom allocator (not malloc/free).
  HeapAllocator();
  ~HeapAllocator() override;

  bool CanUseBufferLike(Allocator* source_allocator,
                        MemoryTypeBitfield memory_type,
                        BufferUsageBitfield buffer_usage,
                        BufferUsageBitfield intended_usage) const override;

  bool CanAllocate(MemoryTypeBitfield memory_type,
                   BufferUsageBitfield buffer_usage,
                   size_t allocation_size) const override;

  StatusOr<ref_ptr<Buffer>> Allocate(MemoryTypeBitfield memory_type,
                                     BufferUsageBitfield buffer_usage,
                                     size_t allocation_size) override;

  StatusOr<ref_ptr<Buffer>> WrapMutable(MemoryTypeBitfield memory_type,
                                        MemoryAccessBitfield allowed_access,
                                        BufferUsageBitfield buffer_usage,
                                        void* data,
                                        size_t data_length) override;
};

// static
Allocator* HeapAllocator::std_heap() {
  static Allocator* std_heap_allocator = new HeapAllocator();
  return std_heap_allocator;
}

HeapAllocator::HeapAllocator() = default;

HeapAllocator::~HeapAllocator() = default;

bool HeapAllocator::CanUseBufferLike(Allocator* source_allocator,
                                     MemoryTypeBitfield memory_type,
                                     BufferUsageBitfield buffer_usage,
                                     BufferUsageBitfield intended_usage) const {
  // The host can use anything with kHostVisible.
  if (!AnyBitSet(memory_type & MemoryType::kHostVisible)) {
    return false;
  }

  // Host currently uses mapping to copy buffers, which is done a lot.
  if (!AnyBitSet(buffer_usage & BufferUsage::kMapping)) {
    return false;
  }

  return true;
}

bool HeapAllocator::CanAllocate(MemoryTypeBitfield memory_type,
                                BufferUsageBitfield buffer_usage,
                                size_t allocation_size) const {
  // This host only allocator cannot serve device visible allocation as we
  // can't know which devices these buffers will be used with.
  return (memory_type & MemoryType::kHostLocal) == MemoryType::kHostLocal &&
         !AnyBitSet(memory_type & MemoryType::kDeviceLocal) &&
         !AnyBitSet(memory_type & MemoryType::kDeviceVisible);
}

StatusOr<ref_ptr<Buffer>> HeapAllocator::Allocate(
    MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
    size_t allocation_size) {
  IREE_TRACE_SCOPE0("HeapAllocator::Allocate");

  if (!CanAllocate(memory_type, buffer_usage, allocation_size)) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Allocation not supported; memory_type="
           << MemoryTypeString(memory_type)
           << ", buffer_usage=" << BufferUsageString(buffer_usage)
           << ", allocation_size=" << allocation_size;
  }

  void* malloced_data = std::calloc(1, allocation_size);
  if (!malloced_data) {
    return ResourceExhaustedErrorBuilder(IREE_LOC)
           << "Failed to malloc " << allocation_size << " bytes";
  }

  auto buffer =
      make_ref<HostBuffer>(this, memory_type, MemoryAccess::kAll, buffer_usage,
                           allocation_size, malloced_data, true);
  return buffer;
}

StatusOr<ref_ptr<Buffer>> HeapAllocator::WrapMutable(
    MemoryTypeBitfield memory_type, MemoryAccessBitfield allowed_access,
    BufferUsageBitfield buffer_usage, void* data, size_t data_length) {
  auto buffer = make_ref<HostBuffer>(this, memory_type, allowed_access,
                                     buffer_usage, data_length, data, false);
  return buffer;
}

}  // namespace

// static
ref_ptr<Buffer> HeapBuffer::Allocate(MemoryTypeBitfield memory_type,
                                     BufferUsageBitfield usage,
                                     size_t allocation_size) {
  auto buffer_or =
      HeapAllocator::std_heap()->Allocate(memory_type, usage, allocation_size);
  return std::move(buffer_or.value());
}

// static
ref_ptr<Buffer> HeapBuffer::AllocateCopy(BufferUsageBitfield usage,
                                         const void* data, size_t data_length) {
  return AllocateCopy(usage, MemoryAccess::kAll, data, data_length);
}

// static
ref_ptr<Buffer> HeapBuffer::AllocateCopy(BufferUsageBitfield usage,
                                         MemoryAccessBitfield allowed_access,
                                         const void* data, size_t data_length) {
  IREE_TRACE_SCOPE0("HeapBuffer::AllocateCopy");
  // Ensure we can map so that we can copy into it.
  usage |= BufferUsage::kMapping;
  auto buffer_or = HeapAllocator::std_heap()->Allocate(MemoryType::kHostLocal,
                                                       usage, data_length);
  auto buffer = std::move(buffer_or.value());
  buffer->WriteData(0, data, data_length).IgnoreError();
  buffer->set_allowed_access(allowed_access);
  return buffer;
}

// static
ref_ptr<Buffer> HeapBuffer::Wrap(MemoryTypeBitfield memory_type,
                                 BufferUsageBitfield usage, const void* data,
                                 size_t data_length) {
  auto buffer_or =
      HeapAllocator::std_heap()->Wrap(memory_type, usage, data, data_length);
  return std::move(buffer_or.value());
}

// static
ref_ptr<Buffer> HeapBuffer::WrapMutable(MemoryTypeBitfield memory_type,
                                        MemoryAccessBitfield allowed_access,
                                        BufferUsageBitfield usage, void* data,
                                        size_t data_length) {
  auto buffer_or = HeapAllocator::std_heap()->WrapMutable(
      memory_type, allowed_access, usage, data, data_length);
  return std::move(buffer_or.value());
}

}  // namespace hal
}  // namespace iree
