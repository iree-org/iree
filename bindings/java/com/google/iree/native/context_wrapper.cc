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

#include "bindings/java/com/google/iree/native/context_wrapper.h"

#include "iree/base/api.h"
#include "iree/base/logging.h"

namespace iree {
namespace java {

Status ContextWrapper::Create() {
  // TODO(jennik): Create the instance on the java side.
  iree_vm_instance_t* instance;
  auto instance_status =
      iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance);
  if (instance_status != IREE_STATUS_OK) {
    return Status(StatusCode(instance_status), "Could not create instance");
  }

  auto context_status =
      iree_vm_context_create(instance, IREE_ALLOCATOR_SYSTEM, &context_);
  if (context_status != IREE_STATUS_OK) {
    return Status(StatusCode(context_status), "Could not create context");
  }
  return OkStatus();
}

int ContextWrapper::GetId() { return iree_vm_context_id(context_); }

}  // namespace java
}  // namespace iree
