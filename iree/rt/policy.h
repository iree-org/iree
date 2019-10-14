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

#ifndef IREE_RT_POLICY_H_
#define IREE_RT_POLICY_H_

#include "iree/base/ref_ptr.h"

namespace iree {
namespace rt {

// Defines how invocation scheduling is to be performed.
// The policy instance is used by the scheduler to determine when submissions
// should be flushed to target queues.
//
// Thread-safe; the policy may be evaluated from arbitrary threads after an
// invocation has began processing.
class Policy : public RefObject<Policy> {
 public:
  virtual ~Policy() = default;

  // TODO(benvanik): constraints:
  // - max memory usage
  // - max delay
  // - max in-flight items/etc
  // - allowed device types
};

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_POLICY_H_
