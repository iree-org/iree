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

#ifndef IREE_HAL_METAL_METAL_COMMAND_QUEUE_H_
#define IREE_HAL_METAL_METAL_COMMAND_QUEUE_H_

#import <Metal/Metal.h>

#include "iree/base/arena.h"
#include "iree/base/status.h"
#include "iree/base/time.h"
#include "iree/hal/cc/command_queue.h"

namespace iree {
namespace hal {
namespace metal {

// A command queue implementation for Metal that directly wraps a
// MTLCommandQueue.
//
// Thread-safe.
class MetalCommandQueue final : public CommandQueue {
 public:
  MetalCommandQueue(std::string name,
                    iree_hal_command_category_t supported_categories,
                    id<MTLCommandQueue> queue);
  ~MetalCommandQueue() override;

  id<MTLCommandQueue> handle() const { return metal_handle_; }

  Status Submit(absl::Span<const SubmissionBatch> batches) override;

  Status WaitIdle(Time deadline_ns) override;

 private:
  id<MTLCommandQueue> metal_handle_;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_COMMAND_QUEUE_H_
