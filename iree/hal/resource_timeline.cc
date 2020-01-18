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

#include "iree/hal/resource_timeline.h"

namespace iree {
namespace hal {

ResourceTimeline::ResourceTimeline() = default;

ResourceTimeline::~ResourceTimeline() = default;

Status ResourceTimeline::AttachSet(ref_ptr<Semaphore> semaphore, uint64_t value,
                                   ref_ptr<ResourceSet> resource_set) {
  // DO NOT SUBMIT
}

Status ResourceTimeline::AdvanceSemaphore(SemaphoreValue semaphore_value) {
  //
}

Status ResourceTimeline::AwakeFromIdle() {
  //
}

}  // namespace hal
}  // namespace iree
