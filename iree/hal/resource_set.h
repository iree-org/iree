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

#ifndef IREE_HAL_RESOURCE_SET_H_
#define IREE_HAL_RESOURCE_SET_H_

#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/resource.h"

namespace iree {
namespace hal {

// A set of resources that are treated as a single reference within a timeline.
// All resources inserted into the set will be retained by the set.
class ResourceSet : public RefObject<ResourceSet> {
 public:
  ResourceSet();
  ~ResourceSet();

  // Inserts resources into the set.
  Status Insert(ref_ptr<Resource> resource);

  // Unions this set with another resource set. The |other_set| will not be
  // modified.
  Status Union(const ResourceSet& other_set);
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_RESOURCE_SET_H_
