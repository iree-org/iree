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

#ifndef IREE_HAL_METAL_METAL_DIRECT_ALLOCATOR_H_
#define IREE_HAL_METAL_METAL_DIRECT_ALLOCATOR_H_

#import <Metal/Metal.h>

#include <memory>

#include "iree/base/status.h"
#include "iree/hal/allocator.h"

namespace iree {
namespace hal {
namespace metal {

class MetalBuffer;

// An allocator implementation for Metal that directly wraps a MTLDevice and
// requests all allocations on the device. This is not of great performance,
// but good for start.
class MetalDirectAllocator final : public Allocator {
 public:
  static std::unique_ptr<MetalDirectAllocator> Create(
      id<MTLDevice> device, id<MTLCommandQueue> transfer_queue);

  ~MetalDirectAllocator() override;

  bool CanUseBufferLike(Allocator* source_allocator,
                        iree_hal_memory_type_t memory_type,
                        iree_hal_buffer_usage_t buffer_usage,
                        iree_hal_buffer_usage_t intended_usage) const override;

  bool CanAllocate(iree_hal_memory_type_t memory_type,
                   iree_hal_buffer_usage_t buffer_usage,
                   size_t allocation_size) const override;

  Status MakeCompatible(iree_hal_memory_type_t* memory_type,
                        iree_hal_buffer_usage_t* buffer_usage) const override;

  StatusOr<ref_ptr<Buffer>> Allocate(iree_hal_memory_type_t memory_type,
                                     iree_hal_buffer_usage_t buffer_usage,
                                     size_t allocation_size) override;

  StatusOr<ref_ptr<Buffer>> WrapMutable(iree_hal_memory_type_t memory_type,
                                        iree_hal_memory_access_t allowed_access,
                                        iree_hal_buffer_usage_t buffer_usage,
                                        void* data,
                                        size_t data_length) override;

 private:
  explicit MetalDirectAllocator(id<MTLDevice> device,
                                id<MTLCommandQueue> transfer_queue);

  StatusOr<ref_ptr<MetalBuffer>> AllocateInternal(
      iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t buffer_usage,
      iree_hal_memory_access_t allowed_access, size_t allocation_size);

  id<MTLDevice> metal_device_;
  id<MTLCommandQueue> metal_transfer_queue_;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_DIRECT_ALLOCATOR_H_
