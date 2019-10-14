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

#include "iree/vm/instance.h"

#include "absl/memory/memory.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"

namespace iree {
namespace vm {

// static
int Instance::NextUniqueId() {
  static int next_id = 0;
  return ++next_id;
}

Instance::Instance()
    : device_manager_(absl::make_unique<hal::DeviceManager>()) {}

Instance::~Instance() = default;

}  // namespace vm
}  // namespace iree
