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

#ifndef IREE_HAL_COMMAND_BUFFER_VALIDATION_H_
#define IREE_HAL_COMMAND_BUFFER_VALIDATION_H_

#include "iree/hal/allocator.h"
#include "iree/hal/command_buffer.h"

namespace iree {
namespace hal {

// Wraps an existing command buffer to provide in-depth validation during
// recording. This should be enabled whenever the command buffer is being driven
// by unsafe code or when early and readable diagnostics are needed.
ref_ptr<CommandBuffer> WrapCommandBufferWithValidation(
    Allocator* allocator, ref_ptr<CommandBuffer> impl);

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_H_
