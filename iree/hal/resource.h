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

#ifndef IREE_HAL_RESOURCE_H_
#define IREE_HAL_RESOURCE_H_

#include <ostream>
#include <string>

#include "iree/base/ref_ptr.h"

namespace iree {
namespace hal {

// Abstract resource type whose lifetime is managed by a ResourceSet.
// Used mostly just to get a virtual dtor, though we could add nicer logging.
class Resource : public RefObject<Resource> {
 public:
  virtual ~Resource() = default;

  // Returns a longer debug string describing the resource and its attributes.
  virtual std::string DebugString() const { return DebugStringShort(); }
  // Returns a short debug string describing the resource.
  virtual std::string DebugStringShort() const {
    // TODO(benvanik): remove this when all resource types have custom logic.
    return std::string("resource_") + std::to_string(static_cast<uint64_t>(
                                          reinterpret_cast<uintptr_t>(this)));
  }
};

}  // namespace hal
}  // namespace iree

inline std::ostream& operator<<(std::ostream& stream,
                                const iree::hal::Resource& resource) {
  stream << resource.DebugStringShort();
  return stream;
}

#endif  // IREE_HAL_RESOURCE_H_
