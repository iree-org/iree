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

#include "iree/hal/host/host_local_allocator.h"

#include <cstdlib>
#include <string>
#include <utility>

#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/host/host_buffer.h"

namespace iree {
namespace hal {
namespace host {

HostLocalAllocator::HostLocalAllocator() = default;

HostLocalAllocator::~HostLocalAllocator() = default;

bool HostLocalAllocator::CanUseBufferLike(
    Allocator* source_allocator, MemoryTypeBitfield memory_type,
    BufferUsageBitfield buffer_usage,
    BufferUsageBitfield intended_usage) const {
  // Must always have visibility to the device, which ensures we can test
  // against the host but have things work on devices with separate address
  // spaces.
  if (!AnyBitSet(memory_type & MemoryType::kDeviceVisible)) {
    return false;
  }

  // kHostVisible is required for mapping.
  if (AnyBitSet(intended_usage & BufferUsage::kMapping) &&
      !AnyBitSet(memory_type & MemoryType::kHostVisible)) {
    return false;
  }

  // Dispatch needs to be specified if we intend to dispatch.
  if (AnyBitSet(intended_usage & BufferUsage::kDispatch) &&
      !AnyBitSet(buffer_usage & BufferUsage::kDispatch)) {
    return false;
  }

  return true;
}

bool HostLocalAllocator::CanAllocate(MemoryTypeBitfield memory_type,
                                     BufferUsageBitfield buffer_usage,
                                     size_t allocation_size) const {
  // Host allows everything, pretty much, so long as it is device-visible (as
  // the host is the device here).
  return AnyBitSet(memory_type & MemoryType::kDeviceVisible);
}

Status HostLocalAllocator::MakeCompatible(
    MemoryTypeBitfield* memory_type, BufferUsageBitfield* buffer_usage) const {
  // Always ensure we are host-visible.
  *memory_type |= MemoryType::kHostVisible;

  // Host currently uses mapping to copy buffers, which is done a lot.
  // We could probably remove this restriction somehow.
  *buffer_usage |= BufferUsage::kMapping;

  // TODO(b/111372612): tensorflow needs transfer too, but shouldn't.
  *buffer_usage |= BufferUsage::kTransfer;

  return OkStatus();
}

StatusOr<ref_ptr<Buffer>> HostLocalAllocator::Allocate(
    MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
    size_t allocation_size) {
  IREE_TRACE_SCOPE0("HostLocalAllocator::Allocate");

  if (!CanAllocate(memory_type, buffer_usage, allocation_size)) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Allocation not supported; memory_type="
           << MemoryTypeString(memory_type)
           << ", buffer_usage=" << BufferUsageString(buffer_usage)
           << ", allocation_size=" << allocation_size;
  }

  // Make compatible with our requirements.
  RETURN_IF_ERROR(MakeCompatible(&memory_type, &buffer_usage));

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

}  // namespace host
}  // namespace hal
}  // namespace iree
