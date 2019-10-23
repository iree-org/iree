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

#ifndef IREE_HAL_EVENT_H_
#define IREE_HAL_EVENT_H_

#include "hal/resource.h"

namespace iree {
namespace hal {

// Events are used for defining synchronization scopes within CommandBuffers.
// An event only exists within a single CommandBuffer and must not be used
// across CommandBuffers from the same device or others.
//
// See CommandBuffer::SignalEvent and CommandBuffer::WaitEvents for more info.
class Event : public Resource {
 public:
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_EVENT_H_
