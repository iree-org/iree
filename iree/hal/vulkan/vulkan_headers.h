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

#ifndef IREE_HAL_VULKAN_VULKAN_HEADERS_H_
#define IREE_HAL_VULKAN_VULKAN_HEADERS_H_

// We exclusively use Vulkan via queried function pointers. To ensure that there
// are no accidental calls to the linker-loaded implicit functions we just
// compile them all out.
//
// Code under iree/hal/vulkan/ *MUST NOT* directly include vulkan.h or any
// header that includes it without this first being set. This means that this
// iree/hal/vulkan/vulkan_headers.h file must usually be included first in all
// files using it.
//
// From there, use iree/hal/vulkan/dynamic_symbols.h to plumb the dynamically
// resolved symbols to any code that may need to make Vulkan calls. See that
// header for more information: in general we try to keep our required set of
// symbols minimal to avoid binary size/runtime memory/linker time so symbols
// are only added as needed.
//
// Other non-core code can choose not to disable the prototypes if they want.
// I don't suggest it though for anything beyond samples.
//
// There's a bunch of reasons to dynamically link against Vulkan like supporting
// platforms without Vulkan or with differing Vulkan versions where all symbols
// may not be available.
//
// See this article for more information:
// https://djang86.blogspot.com/2019/01/what-is-vknoprototypes.html
#define VK_NO_PROTOTYPES 1

#include <vulkan/vulkan.h>

#endif  // IREE_HAL_VULKAN_VULKAN_HEADERS_H_
