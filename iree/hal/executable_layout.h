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

#include "iree/hal/resource.h"

#ifndef IREE_HAL_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_EXECUTABLE_LAYOUT_H_

namespace iree {
namespace hal {

// Defines the resource binding layout used by an executable.
//
// Executables can share the same layout even if they do not use all of the
// resources referenced by descriptor sets referenced by the layout. Doing so
// allows for more efficient binding as bound descriptor sets can be reused when
// command buffer executable bindings change.
//
// Maps to VkPipelineLayout:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPipelineLayout.html
class ExecutableLayout : public Resource {
 public:
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_EXECUTABLE_LAYOUT_H_
