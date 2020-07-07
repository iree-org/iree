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

#ifndef IREE_HAL_HOST_LOCAL_ALLOCATOR_H_
#define IREE_HAL_HOST_LOCAL_ALLOCATOR_H_

#include <cstddef>
#include <memory>

#include "iree/base/status.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"

namespace iree {
namespace hal {
namespace host {

// An allocator implementation that allocates buffers from host memory.
// This can be used for drivers that do not have a memory space of their own.
//
// Buffers allocated will have be MemoryType::kHostLocal | kDeviceVisible as
// the 'device' in the case of a host-local queue *is* the host. To keep code
// written initially for a host-local queue working when other queues are used
// the allocator only works with buffers that are kDeviceVisible.
class HostLocalAllocator : public Allocator {
 public:
  HostLocalAllocator();
  ~HostLocalAllocator() override;

  bool CanUseBufferLike(Allocator* source_allocator,
                        MemoryTypeBitfield memory_type,
                        BufferUsageBitfield buffer_usage,
                        BufferUsageBitfield intended_usage) const override;

  bool CanAllocate(MemoryTypeBitfield memory_type,
                   BufferUsageBitfield buffer_usage,
                   size_t allocation_size) const override;

  Status MakeCompatible(MemoryTypeBitfield* memory_type,
                        BufferUsageBitfield* buffer_usage) const override;

  StatusOr<ref_ptr<Buffer>> Allocate(MemoryTypeBitfield memory_type,
                                     BufferUsageBitfield buffer_usage,
                                     size_t allocation_size) override;
};

}  // namespace host
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_LOCAL_ALLOCATOR_H_
