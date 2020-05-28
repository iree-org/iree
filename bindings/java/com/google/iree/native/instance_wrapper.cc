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

#include "bindings/java/com/google/iree/native/instance_wrapper.h"

#include "iree/base/api_util.h"

namespace iree {
namespace java {

Status InstanceWrapper::Create() {
  return FromApiStatus(
      iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance_), IREE_LOC);
}

iree_vm_instance_t* InstanceWrapper::instance() const { return instance_; }

}  // namespace java
}  // namespace iree
